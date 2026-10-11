"""The right subject for the A-class three-question review: signed quality deltas, per file.

⚠ The first pass asked the three questions about the end-to-end label-change RATE, and that is
the wrong quantity. Question (1) presupposes a signed measure -- "is it the same sign on every
file" -- and a rate has no sign, so half the question could not be answered at all. That is the
tell. This branch's own precedent agrees: W4, also an A-class comparison, reviewed signed deltas
of `fixed`, B-cubed F1 and edit cost, not a movement rate.

A rate answers "how far did the answer move", which is what the noise band is for. The three
questions ask "did it get WORSE, and systematically". Those are different questions and want
different numbers.

So: port minus research path, per file, on the measures the owner's ruler is built from.

    python tools/diarizen_export/port_quality_delta.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

CACHE = REPO / "tmp" / "diarization-dz3"
TIMELINES = REPO / "tmp" / "diarizen-export" / "port-timelines"


def measures(probs, stable, grid, people, starts):
    """The four numbers W4 used, plus the track count the naming burden is priced on."""
    import accuracy_lab as lab
    import cost_metrics as cm

    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    probs = probs[: len(grid)]
    scored = lab.score(probs, stable, grid, people)
    tracks = segments_from_activity(probs.tolist())
    labelled = [{**row, "speaker": speaker.to_dict()}
                for row, speaker in zip(stable, assign_all(stable, tracks))]
    from accuracy_identity import identity_score

    identity = identity_score(probs, stable, grid)
    return {
        "fixed": scored["fixed"],
        "cost": cm.edit_cost(probs, stable, grid, people, starts)["total"],
        "leading10": cm.leading_counts(probs, stable, grid)[10],
        "f1": float(identity["f1"]),
        "rows": len(labelled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-export" / "quality-delta.json"))
    parser.add_argument("--prefix", default="", help="only idents starting with this")
    args = parser.parse_args()

    from check_port import episode_for

    rows = []
    for path in sorted(TIMELINES.glob(f"{args.prefix}*.npz")):
        ident = path.stem
        loaded = episode_for(ident)
        if loaded is None:
            print(f"{ident}: no axis or no reference, skipped")
            continue
        stable, grid, starts, reference, people = loaded

        ours = measures(np.load(path)["probs"].astype(np.float32), stable, grid, people, starts)
        theirs = measures(reference, stable, grid, people, starts)
        row = {"ident": ident,
               **{f"d_{k}": ours[k] - theirs[k] for k in ("fixed", "cost", "leading10", "f1")},
               "port": ours, "reference": theirs}
        rows.append(row)
        print(f"{ident}: d_cost {row['d_cost']:+8.2f}  d_fixed {row['d_fixed']:+.4f}  "
              f"d_leading10 {row['d_leading10']:+d}  d_f1 {row['d_f1']:+.4f}", flush=True)

    if not rows:
        raise SystemExit("nothing scored")

    print(f"\nsigned deltas, port minus research path, n={len(rows)}")
    print("  ⚠ negative edit cost = the port needs LESS hand fixing; positive = worse")
    for key in ("cost", "fixed", "leading10", "f1"):
        values = np.array([row[f"d_{key}"] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        positive = int((finite > 0).sum())
        negative = int((finite < 0).sum())
        zero = int((finite == 0).sum())
        print(f"  d_{key:>10}: median {np.median(finite):+8.4f}  mean {finite.mean():+8.4f}  "
              f"range [{finite.min():+.4f}, {finite.max():+.4f}]  "
              f"signs +{positive} / -{negative} / 0 {zero}")

    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
