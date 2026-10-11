"""Does the turn detector find real handovers, and does it leave overlap alone?

Three numbers, and ⚠ the third is the one that can veto the whole idea:

* **recall** -- of rows the truth says hold a handover, how many are found;
* **cut error** -- how far the proposed cut sits from where the truth hands over;
* **overlap safety** -- of rows the truth says are simultaneous speech, how many were wrongly
  called a handover. Splitting overlap destroys the multi-label answer, which is the one
  capability this line has over clustering, so this number has to be near zero or the detector
  is not usable at any recall.

⚠ Truth here is the refined per-line speaker axis, at its own 80 ms resolution, so the cut error
cannot be read below that. It is reported as a distribution, not a single number.

    python tools/diarizen_export/turns_bed.py --episodes 1 2 3
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

NATIVE = REPO / "tmp" / "diarizen-export" / "port-timelines-native"


def truth_turn(span, people) -> "tuple[str, str, float] | None":
    """`(before, after, seconds into the row)` if the truth hands over inside this row."""
    leading = []
    for frame in span:
        active = np.flatnonzero(frame)
        leading.append(people[active[0]] if len(active) == 1 else None)
    named = [(index, who) for index, who in enumerate(leading) if who]
    if len(named) < 2 or named[0][1] == named[-1][1]:
        return None
    first, last = named[0][1], named[-1][1]
    last_of_first = max(i for i, who in named if who == first)
    first_of_last = min(i for i, who in named if who == last)
    if first_of_last <= last_of_first:
        return None
    from finesub.speech.diarization.segments import FRAME_SECONDS

    return first, last, 0.5 * (last_of_first + first_of_last + 1) * FRAME_SECONDS


def truth_overlap(span) -> float:
    """Share of the row where the truth says two people speak at once."""
    return float((span.sum(axis=1) >= 2).mean()) if len(span) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(1, 16)))
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-bed" / "turns.json"))
    args = parser.parse_args()

    import backend_cost as bc

    from finesub.speech.diarization.evidence import ClusterEvidence  # noqa: F401
    from finesub.speech.diarization.segments import FRAME_SECONDS
    from turns import find_turn

    found = missed = errors_list = None
    stats = {"turn_rows": 0, "found": 0, "overlap_rows": 0, "overlap_called": 0,
             "plain_rows": 0, "plain_called": 0}
    errors = []

    for episode in args.episodes:
        data = bc.ev.episode(episode)
        path = NATIVE / f"onetonton-{episode:02d}.npz"
        if data is None or not path.exists():
            continue
        stable, _, grid, people, _ = data
        native = np.load(path)["probs"]

        for row in stable:
            low = int(float(row["start"]) / FRAME_SECONDS)
            high = max(low + 1, int(np.ceil(float(row["end"]) / FRAME_SECONDS)))
            span = grid[low:high]
            if not span.any():
                continue

            reference = truth_turn(span, people)
            overlap = truth_overlap(span)
            proposed = find_turn(row, native, step=0.02)

            if reference is not None:
                stats["turn_rows"] += 1
                if proposed is not None:
                    stats["found"] += 1
                    errors.append(abs((proposed.at - float(row["start"])) - reference[2]))
            elif overlap >= 0.5:
                stats["overlap_rows"] += 1
                stats["overlap_called"] += proposed is not None
            else:
                stats["plain_rows"] += 1
                stats["plain_called"] += proposed is not None

    del found, missed, errors_list
    print(f"rows the truth says hold a handover: {stats['turn_rows']}")
    if stats["turn_rows"]:
        print(f"  found: {stats['found']} = {stats['found'] / stats['turn_rows']:.1%} recall")
    if errors:
        e = np.array(errors)
        print(f"  cut error: median {np.median(e):.2f}s  P90 {np.percentile(e, 90):.2f}s  "
              f"max {e.max():.2f}s   (truth resolution is {FRAME_SECONDS}s)")
    print(f"\nrows the truth says are mostly OVERLAP: {stats['overlap_rows']}")
    if stats["overlap_rows"]:
        share = stats["overlap_called"] / stats["overlap_rows"]
        print(f"  ⚠ wrongly called a handover: {stats['overlap_called']} = {share:.1%}"
              f"   {'-- safe' if share <= 0.05 else '-- TOO HIGH, splitting overlap destroys the multi-label answer'}")
    print(f"\nrows with neither: {stats['plain_rows']}")
    if stats["plain_rows"]:
        print(f"  called a handover anyway: {stats['plain_called']} = "
              f"{stats['plain_called'] / stats['plain_rows']:.1%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {**stats, "cut_error_median": float(np.median(errors)) if errors else None,
         "cut_error_p90": float(np.percentile(errors, 90)) if errors else None},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
