"""B1: do the two proposed quantities separate a real turn from overlap, on the RIGHT truth?

The plan's idea, unchanged: what distinguishes 「这一行里两个人接着说」 from 「两个人同时说」 is
not whether a second track exists -- `turns_bed` established that it almost always does -- but
whether the two tracks' activity sits side by side or on top of each other. Two quantities say so:

    centroid_gap  |time centroid of A - time centroid of B| / row length
    co_active     frames where both are on / active frames of the SHORTER of the two

A real handover: centroids one after the other, overlap only at the seam -> gap high, co_active low.
Simultaneous speech: centroids coincide, both on for most of it -> gap low, co_active high.

⚠⚠ **Scored against `turn_shape_bed`, never against `turns_bed.truth_turn`.** The old truth reads
the refined subtitle rasterised onto an 80 ms grid, and measured 2026-09-20 the two rulers barely
select the same rows: 60.6% of what the grid calls a handover is a neighbour clipping the row's
rim, and the grid misses 76.9% of the rows whose TEXT holds two people. This bed imports the
classifier rather than restating it, so there is one definition of a splittable row.

⚠ **This decides whether to build a detector; it is not one.** Two distributions that overlap mean
stop -- that is the outcome the plan asked for explicitly, and it is cheaper than a detector whose
precision then has to be explained.

⚠ **Rows where the signal cannot be computed are the ceiling and are reported as such.** If our
timeline never puts two tracks inside the row, no quantity derived from two tracks can fire, no
matter how good the rule is.

    python tools/diarizen_export/turn_signal_bed.py --episodes 1 2 3
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
OUT = REPO / "tmp" / "diarizen-bed"

#: The predicted timeline's own step. ⚠ Native 20 ms, not the 80 ms product grid: words run
#: 100-300 ms and the product grid is reduced with a max that widens activity outward, which
#: would blur exactly the seam these two quantities are trying to see.
STEP = 0.02

#: A track on for less than this inside the row is not one of its two speakers.
MIN_TRACK_SECONDS = 0.2


def two_track_signal(row, native) -> tuple[float, float] | None:
    """`(centroid_gap, co_active)` for the row's two most active tracks, or None if it has < 2."""
    low = int(float(row["start"]) / STEP)
    high = max(low + 1, int(np.ceil(float(row["end"]) / STEP)))
    span = np.asarray(native[low:high]) > 0
    if not span.size or span.shape[0] < 2:
        return None
    counts = span.sum(axis=0)
    ranked = np.argsort(-counts)
    if len(ranked) < 2 or counts[ranked[1]] * STEP < MIN_TRACK_SECONDS:
        return None
    first, second = span[:, ranked[0]], span[:, ranked[1]]
    frames = np.arange(span.shape[0], dtype=np.float64)
    gap = abs(frames[first].mean() - frames[second].mean()) / max(span.shape[0] - 1, 1)
    both = int((first & second).sum())
    return float(gap), float(both / max(min(first.sum(), second.sum()), 1))


def auc(positive, negative) -> float | None:
    """Rank AUC -- the chance a random positive scores above a random negative. 0.5 = blind."""
    if not len(positive) or not len(negative):
        return None
    joined = np.concatenate([np.asarray(positive), np.asarray(negative)])
    order = joined.argsort().argsort().astype(np.float64) + 1
    # Average ranks over ties, or a signal that is constant on one side reads as informative.
    for value in np.unique(joined):
        where = joined == value
        order[where] = order[where].mean()
    rank_sum = order[: len(positive)].sum()
    count = len(positive) * len(negative)
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / count)


#: Which way each quantity is SUPPOSED to point if the hypothesis holds. A handover should show
#: centroids one after the other (a bigger gap) and overlap only at the seam (less co-activity).
#: ⚠ Recorded because an AUC below 0.5 is not a weak signal -- it is the hypothesis being wrong,
#: and printing `|auc - 0.5|` as "distance from blind" hides exactly that.
EXPECTED = {"gap": +1, "co": -1}


def report(name: str, positive, negative) -> dict:
    """One quantity's two distributions, how separable they are, and WHICH WAY it points."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    area = auc(positive, negative)
    if area is None:
        print(f"  {name}: no data")
        return {"auc": None}
    quarters = lambda values: np.percentile(values, [25, 50, 75])
    low, mid, high = quarters(positive)
    nlow, nmid, nhigh = quarters(negative)
    # AUC > 0.5 means splittable rows score HIGHER; the hypothesis wants that only when EXPECTED
    # is +1. Folding the two together is how a reversed signal reads as a strong one.
    agrees = (area - 0.5) * EXPECTED[name] > 0
    verdict = ("as predicted" if agrees else "⚠ BACKWARDS") if abs(area - 0.5) >= 0.03 else "blind"
    print(f"  {name:<5} splittable {mid:.3f} [{low:.3f}, {high:.3f}]   "
          f"other {nmid:.3f} [{nlow:.3f}, {nhigh:.3f}]   AUC {area:.3f}  {verdict}")
    return {"auc": area, "splittable_median": float(mid), "other_median": float(nmid),
            "agrees_with_hypothesis": bool(agrees)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(1, 16)))
    arguments = parser.parse_args()

    from turn_shape_bed import classify, episode_rows

    buckets: dict[str, dict[str, list]] = {"gap": {}, "co": {}}
    computable: dict[str, list] = {}
    for episode in arguments.episodes:
        loaded = episode_rows(episode)
        path = NATIVE / f"onetonton-{episode:02d}.npz"
        if loaded is None or not path.exists():
            continue
        stable, _, _, events = loaded
        native = np.load(path)["probs"]
        for row in stable:
            shape, _, _ = classify(row, events)
            computable.setdefault(shape, [0, 0])[1] += 1
            signal = two_track_signal(row, native)
            if signal is None:
                continue
            computable[shape][0] += 1
            buckets["gap"].setdefault(shape, []).append(signal[0])
            buckets["co"].setdefault(shape, []).append(signal[1])

    print("rows where our timeline puts two tracks inside the row at all "
          "(⚠ the ceiling -- no rule can fire where the signal does not exist):")
    for shape in ("splittable", "backchannel", "edge", "single", "empty"):
        have, total = computable.get(shape, [0, 0])
        print(f"  {shape:<12} {have}/{total} = {have / max(total, 1):>6.1%}")

    # ⚠ Two contrasts, and the second is the one that matters. Half of all rows are `single`, and
    # telling a two-speaker row from a one-speaker row is not the question -- the timeline already
    # answers it by putting one track there. The decision is made among rows where a second person
    # IS audible: cut, or leave it. Pooling `single` into the negatives makes any quantity look
    # good by answering the easy half.
    contrasts = {"vs everything else": ("backchannel", "edge", "single", "empty"),
                 "⭐ vs a second person who is really there": ("backchannel", "edge")}
    record = {}
    for label, others in contrasts.items():
        print(f"\n{label}:")
        negatives = {name: [value for shape in others for value in buckets[name].get(shape, [])]
                     for name in ("gap", "co")}
        record[label] = {name: report(name, buckets[name].get("splittable", []), negatives[name])
                         for name in ("gap", "co")}

    real = record["⭐ vs a second person who is really there"]
    useful = [value for value in real.values()
              if value.get("auc") is not None and value.get("agrees_with_hypothesis")]
    best = max((abs(value["auc"] - 0.5) for value in useful), default=0.0)
    print(f"\n⇒ on the contrast that decides, the best quantity pointing the PREDICTED way is "
          f"{best:.3f} from blind. "
          + ("⚠ **STOP**: the plan said to stop here rather than build a detector, and this is "
             "that case -- at this separation the false positives cost more than the finds."
             if best < 0.10 else
             "Worth a detector, but pre-register the gate before building one."))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "turn-signal.json").write_text(json.dumps(
        {"episodes": arguments.episodes, "computable": computable, "contrasts": record,
         "min_track_seconds": MIN_TRACK_SECONDS, "step": STEP}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
