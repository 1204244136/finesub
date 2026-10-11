"""Open-source MOSS as the speaker source on 姐姐咚咚, on the W2 rulers (owner 2026-10-04: MOSS验证可以补).

The 2026-09-26 survey saw MOSS's small-cast advantage on four episodes of one show only. This scores
MOSS's whole-episode single pass (`moss_serve.py` output in the main checkout's
`data/ts-diar/episodes/onetonton-NN/moss-single-vocal.json`) the way production is scored: its turns
become a track timeline, production's own `assign_all` labels the same `-stable.json` segments, and
`edit_cost` / `accuracy_lab.score` / `identity_score` read it against the same grid. Paired against
the production Nemotron-3 timeline cached by `abstain_profile.py`.

    python tools/diarization_explore/moss_onetonton.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))

EPISODES = n3.MAIN / "data" / "ts-diar" / "episodes"
PRODUCTION = n3.MAIN / "tmp" / "diarization-abstain"


def moss_tracks(path):
    segments = json.loads(path.read_text(encoding="utf-8"))["segments"]
    names = sorted({s["speaker"] for s in segments if s.get("speaker")})
    index = {name: i for i, name in enumerate(names)}
    return [{"start": float(s["start"]), "end": float(s["end"]), "track": index[s["speaker"]]}
            for s in segments if s.get("speaker") and float(s["end"]) > float(s["start"])]


def coverage(tracks, grid):
    import backend_cost as bc
    active = bc.timeline_probs(tracks, len(grid))[: len(grid)].max(axis=1) > 0
    spoken = grid.any(axis=1)
    return float((active & spoken).sum() / max(1, spoken.sum()))


def main(span="all"):
    import accuracy_lab as lab
    import backend_cost as bc
    import cost_metrics as cm
    import nemotron3_cost as nc
    import relabel_eval as ev
    from accuracy_identity import identity_score
    from accuracy_report import bootstrap_mean_ci

    nc.corpus_keys("onetonton")
    rows = []
    for key in range(1, 16):
        ident = f"onetonton-{key:02d}"
        moss = EPISODES / ident / "moss-single-vocal.json"
        production = PRODUCTION / ident / "nemotron3-diar.json"
        data = ev.episode(key)
        if data is None or not moss.exists() or not production.exists():
            print("skip", ident)
            continue
        stable, _, grid, people, _ = data
        starts = bc.refined_starts(key)
        if span == "covered":
            # Descriptive only: both arms on the rows before MOSS's last segment, which separates
            # "the output stopped early" from "the speakers it did give are right".
            cut = max(s["end"] for s in json.loads(moss.read_text(encoding="utf-8"))["segments"])
            stable = [s for s in stable if s["end"] <= cut]
        arms = {"production": json.loads(production.read_text(encoding="utf-8"))["tracks"],
                "moss": moss_tracks(moss)}
        row = {"ident": ident}
        for arm, tracks in arms.items():
            probs = bc.timeline_probs(tracks, len(grid))[: len(grid)]
            row[arm] = dict(
                priced=cm.edit_cost(probs, stable, grid, people, starts, price_unlabelled=True)["total"],
                fixed=float(lab.score(probs, stable, grid, people)["fixed"]),
                f1=float(identity_score(probs, stable, grid)["f1"]),
                lead10=int(cm.leading_counts(probs, stable, grid)[10]),
                tracks=len({t["track"] for t in tracks}), coverage=coverage(tracks, grid))
        rows.append(row)
        p, m = row["production"], row["moss"]
        print(f"{ident}  priced {p['priced']:7.1f} -> {m['priced']:7.1f}  fixed {p['fixed']:.3f} -> {m['fixed']:.3f}"
              f"  B3 {p['f1']:.3f} -> {m['f1']:.3f}  tracks {p['tracks']} -> {m['tracks']}  "
              f"lead10 {p['lead10']} -> {m['lead10']}  MOSS coverage {m['coverage']:.1%}")

    def paired(subset, label):
        if len(subset) < 5:
            print(f"{label}: n={len(subset)}, too few for an interval")
            return
        parts = []
        for key in ("priced", "fixed", "f1", "lead10"):
            diffs = [r["moss"][key] - r["production"][key] for r in subset]
            low, high = bootstrap_mean_ci(diffs)
            parts.append(f"{key} {np.mean(diffs):+.3f} [{low:+.3f}, {high:+.3f}]")
        print(f"{label} (MOSS minus production, n={len(subset)}): " + "  ".join(parts))

    paired(rows, "all")
    paired([r for r in rows if r["moss"]["coverage"] >= 0.9], "coverage >= 90% (descriptive)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
