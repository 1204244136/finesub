"""Nemotron-3-Diarization on our own material, priced with the W2 / W2b ruler.

    python tools/diarization_explore/nemotron3_cost.py infer onetonton-01   # GPU, one file per process
    python tools/diarization_explore/nemotron3_cost.py score --corpus onetonton
    python tools/diarization_explore/nemotron3_cost.py score --corpus hina

Pre-registered 2026-09-24, committed before any file of either corpus was inferred:

* Primary metric: the owner's edit cost (`cost_metrics.edit_cost`, total), paired
  Nemotron-3 minus DiariZen per file, bootstrap 95% CI of the mean (`bootstrap_mean_ci`),
  per corpus: 姐姐咚咚 #1-#15 (W2) and the hina 19-file holdout (W2b).
* ADVANCE (worth integration work): CI entirely below 0 on at least one corpus, and CI lower
  bound <= 0 on the other (significantly better somewhere, significantly worse nowhere).
* REJECT on quality: CI entirely above 0 on either corpus.
* Otherwise TIE: quality indistinguishable; what remains is the licence (OpenMDW-1.1 vs
  DiariZen's CC BY-NC 4.0), which is the owner's call, not this script's.
* Reported, not gating: fixed, B-cubed F1, leading >= 10 rows (naming burden), label part,
  the priced terms. Sortformer v2.1 (production timeline) is printed for context.
* Frozen: model revision f667ed7, NeMo Speech cf724ac, offline config (340/40/40/300, cache
  264), production `segments_from_activity` at the native 10 ms, rasterised onto the 80 ms
  truth grid the way the Sortformer arm is. No knob is tuned after seeing a result.

⚠ hina's truth names ROLES (`eval_contract`): edit cost is fine there, identity claims are not.
⚠ Nemotron-3 caps at 8 tracks; hina runs five to eight people plus unlabelled game characters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))
# The W2 helpers resolve everything against lab.WT and the cwd; the material lives in the main
# checkout, so point both there before anything computes a path at import time.
import accuracy_lab as lab  # noqa: E402

lab.WT = n3.MAIN
ARMS = ("sortformer", "diarizen", "nemotron3")


def source(ident):
    if ident.startswith("hina-"):
        return n3.MAIN / "out" / "refined-2609" / ident / f"{ident}-vocal.ogg"
    return n3.MAIN / "out" / ident / f"{ident}-vocal.ogg"


def infer(ident):
    import vram_guard
    vram_guard.install()
    import soundfile as sf

    wave, sr = sf.read(source(ident), dtype="float32", always_2d=True)
    assert sr == 16000, sr
    _, probs = n3.infer(n3.load_model(), wave.mean(axis=1), sr)
    directory = n3.OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "probs.npz", probs=probs)
    print(ident, "frames", probs.shape, "peak GiB", round(vram_guard.peak_gib(), 2), flush=True)


def nemotron_raster(ident, frames):
    """10 ms probabilities -> production segments -> 0/1 on the 80 ms grid, like the Sortformer arm."""
    import backend_cost as bc
    from finesub.speech.diarization.segments import segments_from_activity

    probs = np.load(n3.OUT / ident / "probs.npz")["probs"]
    tracks = [{"start": t.start, "end": t.end, "track": t.track}
              for t in segments_from_activity(probs.tolist(), frame_seconds=0.01)]
    return bc.timeline_probs(tracks, frames)


def corpus_keys(corpus):
    """Point the W2 helpers at the main checkout and return the corpus's file keys."""
    os.chdir(n3.MAIN)  # relabel_eval.episode reads cwd-relative out/
    import backend_cost as bc
    import refined_multi as rm
    rm.RUN_ROOT = n3.MAIN / "out" / "refined-2609"
    rm.IDS = n3.MAIN / "tmp" / "refined-2609" / "ids.json"
    # ⚠ Set, not inherited: backend_cost froze V2_OUT from lab.WT when it was first imported, and
    # whichever module imported accuracy_v2 first decided whether that was the worktree or main.
    bc.V2_OUT = n3.MAIN / "tmp" / "diarization-accuracy-v2"
    if corpus == "onetonton":
        from accuracy_cross import configure
        configure()
        return list(range(1, 16))
    frozen = json.loads((HERE / "results" / "diarizen-v3-20260917.json").read_text(encoding="utf-8"))
    return frozen["groups"]["heldout_all"]


def load_episode(corpus, key):
    """(ident, stable, grid, people, starts, {arm: 80 ms probs}) or None; call corpus_keys first."""
    import backend_cost as bc
    import relabel_eval as ev

    ident = key if corpus == "hina" else f"onetonton-{key:02d}"
    if corpus == "hina":
        loaded = bc.hina_episode(key)
        if loaded is None:
            print("skip", ident, "(no cached run)")
            return None
        (stable, timeline, grid, people, _), cached, starts = loaded
    else:
        data = ev.episode(key)
        cached = bc.V2_OUT / ident / "diarizen.npz"
        if data is None or not cached.exists():
            print("skip", ident, "(no cached run)")
            return None
        stable, timeline, grid, people, _ = data
        starts = bc.refined_starts(key)
    if not (n3.OUT / ident / "probs.npz").exists():
        print("skip", ident, "(not inferred)")
        return None
    arms = {"sortformer": bc.timeline_probs(timeline["tracks"], len(grid)),
            "diarizen": np.load(cached)["probs"],
            "nemotron3": nemotron_raster(ident, len(grid))}
    return ident, stable, grid, people, starts, {a: p[: len(grid)] for a, p in arms.items()}


def score(corpus):
    import backend_cost as bc
    import cost_metrics as cm
    from accuracy_identity import identity_score
    from accuracy_report import bootstrap_mean_ci
    from eval_contract import announce

    keys = corpus_keys(corpus)
    announce(corpus)
    rows = []
    for key in keys:
        loaded = load_episode(corpus, key)
        if loaded is None:
            continue
        ident, stable, grid, people, starts, arms = loaded
        row = {"episode": ident, "people": len(people)}
        for name, probs in arms.items():
            metrics = lab.score(probs, stable, grid, people)
            row[name] = {"fixed": metrics["fixed"], "f1": identity_score(probs, stable, grid)["f1"],
                         "naming": cm.naming_cost(probs, stable, grid),
                         "leading_counts": cm.leading_counts(probs, stable, grid),
                         "cost": cm.edit_cost(probs, stable, grid, people, starts),
                         "terms": bc.cost_terms_of(probs, stable, grid, people, starts)}
        rows.append(row)
        print(f"{ident:>13} ppl {len(people)}  " + "  ".join(
            f"{a[:4]} edit {row[a]['cost']['total']:6.0f} fixed {100*row[a]['fixed']:5.1f} "
            f"lead {row[a]['naming']['leading']:>2}" for a in ARMS), flush=True)

    print(f"\npaired over {len(rows)} files; primary = edit cost, Nemotron-3 minus DiariZen")
    summary = {}
    for label, get, lower_is_better in (
        ("edit cost", lambda a: a["cost"]["total"], True),
        ("  label part", lambda a: a["cost"]["labels"], True),
        ("fixed (pp)", lambda a: a["fixed"] * 100, False),
        ("B3 F1 (pp)", lambda a: a["f1"] * 100, False),
        ("leading >= 10 rows", lambda a: a["leading_counts"][10], True),
    ):
        for other in ("diarizen", "sortformer"):
            diffs = [get(r["nemotron3"]) - get(r[other]) for r in rows]
            ci = bootstrap_mean_ci(diffs)
            wins = sum((d < 0) if lower_is_better else (d > 0) for d in diffs)
            band = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "(n < 5)"
            print(f"  {label:<20} vs {other:<11}{np.mean(diffs):+9.2f} {band:>22}   N3 better {wins}/{len(diffs)}")
            summary[f"{label.strip()} vs {other}"] = {"mean": float(np.mean(diffs)), "ci95": ci,
                                                     "better": wins, "n": len(diffs)}
    for term in ("move", "delete", "add", "duplicate", "merge"):
        means = {a: np.mean([float(r[a]["terms"][term]) for r in rows]) for a in ARMS}
        print(f"  term {term:<10} " + "  ".join(f"{a} {m:7.1f}" for a, m in means.items()))
    out = n3.OUT / f"cost-{corpus}.json"
    out.write_text(json.dumps({"rows": rows, "paired": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("infer").add_argument("ident")
    sub.add_parser("score").add_argument("--corpus", choices=("onetonton", "hina"), default="onetonton")
    args = p.parse_args()
    infer(args.ident) if args.cmd == "infer" else score(args.corpus)


if __name__ == "__main__":
    main()
