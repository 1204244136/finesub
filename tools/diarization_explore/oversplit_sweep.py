"""Over-split on purpose: does a looser vbx clustering turn 串轨 into cheaper over-splitting?

Owner 2026-10-04 (「宁可过拆」可以试一下). On small casts the default Nemotron-3 + vbx loses to MOSS on
PRECISION -- several similar voices in one track (hina-71: three women in one 736 s track). The edit
cost charges every such line a move (2), while an extra track costs one merge decision. The two vbx
knobs that set how many speakers survive are the AHC cut (`AHC_THRESHOLD` 0.6) and VBx's `Fb` (0.8);
lowering either keeps more clusters. Only the diarization stage moves: the ASR segments (`-stable.json`)
are reused as they are.

    python tools/diarization_explore/oversplit_sweep.py cache     # GPU once: Nemotron activity + voiceprints
    python tools/diarization_explore/oversplit_sweep.py sweep     # CPU: re-cluster and score every setting

`cache` runs the production backend's own steps (`frame_activity`, `windows(rasterised(...))`,
`diarizen_embed.embed`) and stores their outputs, so `sweep` can call `diarizen_cluster.cluster` with
other knobs and rebuild the timeline exactly as `run_diarization` would. The default setting must
reproduce the production timeline (checked against `abstain_profile.py`'s cache).

Material: development = the seven 星穹铁道 episodes of the 2026-09-26 model survey (its `owner` / B³
ruler, where MOSS Pro has readings); confirmation = 姐姐咚咚 15 (the W2 rulers). The gate is in
`docs/reports/2026-10-04-diarization-review.md`, written before the confirmation run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))
sys.path.insert(0, str(n3.WT))

OUT = n3.MAIN / "tmp" / "diarization-oversplit"
DEVELOPMENT = ("hina-72", "hina-70", "hina-71", "hina-67", "hina-64", "hina-73", "hina-68")
CONFIRMATION = tuple(f"onetonton-{n:02d}" for n in range(1, 16))
#: (ahc_threshold, fb); the first is production.
SETTINGS = ((0.6, 0.8), (0.5, 0.8), (0.4, 0.8), (0.6, 0.4), (0.5, 0.4), (0.4, 0.4), (0.6, 0.2))


def source(ident):
    import nemotron3_cost as nc
    return nc.source(ident)


def cache(idents):
    import vram_guard
    vram_guard.install()
    import soundfile
    import torch

    from finesub.speech.diarization import diarizen_embed, nemotron3_backend as nb, nemotron3_model
    from finesub.speech.diarization.stage import resolve_backend
    from finesub.speech.preprocessing.audio import TARGET_SR, resample_if_needed

    handle = resolve_backend(None).ensure(None, "cuda")
    for ident in idents:
        path = OUT / f"{ident}.npz"
        if path.exists():
            continue
        signal, rate = soundfile.read(str(source(ident)), dtype="float32", always_2d=True)
        waveform, _ = resample_if_needed(torch.from_numpy(signal.mean(axis=1)).unsqueeze(0), rate, TARGET_SR)
        probabilities = nb.frame_activity(waveform, handle=handle, device="cuda")
        local = nb.windows(nb.rasterised(probabilities, samples=int(waveform.shape[-1]),
                                         source_step=nemotron3_model.FRAME_SECONDS))
        embeddings = diarizen_embed.embed(handle.embedding, waveform, local, device="cuda", only_active=True)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, local=local, embeddings=embeddings)
        print(ident, local.shape, flush=True)


def timeline(ident, ahc, fb):
    """Production's track list for one setting: cluster, assemble at 20 ms, segment."""
    from finesub.speech.diarization import diarizen_assemble, diarizen_cluster
    from finesub.speech.diarization.nemotron3_backend import NATIVE_STEP
    from finesub.speech.diarization.segments import segments_from_activity
    from finesub.speech.diarization.stage import resolve_backend

    data = np.load(OUT / f"{ident}.npz")
    plda = resolve_backend(None).ensure(None, "cpu").plda_dir
    clustered = diarizen_cluster.cluster(data["embeddings"], data["local"], plda, ahc_threshold=ahc, fb=fb)
    native = diarizen_assemble.assemble(data["local"], clustered.hard).astype(np.float32)
    return [{"start": t.start, "end": t.end, "track": t.track}
            for t in segments_from_activity(native.tolist(), frame_seconds=NATIVE_STEP)]


def check_default():
    """The default setting must give the timeline `run_diarization` gave (abstain_profile's cache)."""
    for ident in CONFIRMATION[:3]:
        production = n3.MAIN / "tmp" / "diarization-abstain" / ident / "nemotron3-diar.json"
        if not production.exists() or not (OUT / f"{ident}.npz").exists():
            continue
        ours = timeline(ident, *SETTINGS[0])
        theirs = json.loads(production.read_text(encoding="utf-8"))["tracks"]
        same = len(ours) == len(theirs) and all(
            a["track"] == b["track"] and abs(a["start"] - b["start"]) < 1e-6 and abs(a["end"] - b["end"]) < 1e-6
            for a, b in zip(ours, theirs))
        print(f"A/A {ident}: {len(ours)} vs {len(theirs)} segments, identical={same}")


def sweep_development():
    from tools.ts_diar_explore import score as S

    print("\n== development (星穹铁道, survey ruler): owner% / B3 F1 (P, R) / labels")
    for ahc, fb in SETTINGS:
        cells = []
        for ident in DEVELOPMENT:
            if not (OUT / f"{ident}.npz").exists():
                continue
            # The survey's reference-arm shape (`arms._timeline`), so production and MOSS read the same way.
            arm = {"model": "n3vbx", "words": None, "segments": [
                {"start": t["start"], "end": t["end"], "speaker": f"T{t['track']}", "text": ""}
                for t in timeline(ident, ahc, fb)]}
            result = S.speaker(S.reference(ident), arm, 0.0, 1e9)
            cells.append((ident, result))
        if not cells:
            continue
        mean = lambda key: np.mean([r[key] for _, r in cells])  # noqa: E731
        print(f"ahc {ahc} fb {fb}: mean owner {100*mean('owner'):.1f} B3 {mean('b3_f1'):.3f}")
        for ident, r in cells:
            print(f"   {ident:9} owner {100*r['owner']:5.1f} B3 {r['b3_f1']:.3f} (P {r['b3_p']:.3f} R {r['b3_r']:.3f})"
                  f" labels {r['labels']}/{r['persons']}")


def sweep_confirmation():
    import backend_cost as bc
    import cost_metrics as cm
    import nemotron3_cost as nc
    import relabel_eval as ev
    from accuracy_identity import identity_score
    from accuracy_report import bootstrap_mean_ci

    nc.corpus_keys("onetonton")
    rows = {s: {} for s in SETTINGS}
    for key in range(1, 16):
        ident = f"onetonton-{key:02d}"
        data = ev.episode(key)
        if data is None or not (OUT / f"{ident}.npz").exists():
            continue
        stable, _, grid, people, _ = data
        starts = bc.refined_starts(key)
        for setting in SETTINGS:
            probs = bc.timeline_probs(timeline(ident, *setting), len(grid))[: len(grid)]
            lab = __import__("accuracy_lab").score(probs, stable, grid, people)
            rows[setting][ident] = dict(
                priced=cm.edit_cost(probs, stable, grid, people, starts, price_unlabelled=True)["total"],
                fixed=lab["fixed"], f1=identity_score(probs, stable, grid)["f1"],
                lead10=int(cm.leading_counts(probs, stable, grid)[10]))
    base = rows[SETTINGS[0]]
    print("\n== confirmation (姐姐咚咚, W2 rulers; paired against production)")
    for setting, table in rows.items():
        common = [i for i in table if i in base]
        if not common:
            continue
        line = f"ahc {setting[0]} fb {setting[1]}: n {len(common)}"
        for key in ("priced", "fixed", "f1", "lead10"):
            diffs = [table[i][key] - base[i][key] for i in common]
            ci = bootstrap_mean_ci(diffs) if setting != SETTINGS[0] else None
            line += f"  {key} {np.mean([table[i][key] for i in common]):.3f}" + (
                f" ({np.mean(diffs):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}])" if ci else "")
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cache", "check", "develop", "confirm"))
    args = parser.parse_args()
    if args.command == "cache":
        cache(DEVELOPMENT + CONFIRMATION)
    elif args.command == "check":
        check_default()
    elif args.command == "develop":
        sweep_development()
    else:
        sweep_confirmation()


if __name__ == "__main__":
    main()
