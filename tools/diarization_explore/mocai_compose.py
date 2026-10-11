"""Does the recursive split improve ATTRIBUTION, not just purity?

Findings 7: splitting 魔裁's big mixed tracks beats a time-block null on purity by +5.8 pp. Purity is a
proxy. What a consumer needs is attribution -- is the most active track over a line the right character's?

Builds three timelines per part and scores them the same way (first track == right character, duration-
weighted, each track's owner its whole-file majority -- what `-diar.json` gives a consumer):

  base    the original four tracks
  split   each split parent replaced by its children (from `mocai_split.py`'s saved assignment)
  blocks  the same frames and the same child sizes as `split`, but grouped into contiguous TIME runs -- so it
          keeps exactly what `split` dropped and kept, and differs only in how frames are grouped

⚠ `split > base` alone proves nothing: more tracks give every owner more room to fit, which raises attribution
mechanically -- the same arithmetic that made purity rise on 10/10 (findings 7). The split earns its keep
only by beating `blocks`, which gets the extra tracks and the scene structure for free.

    python tools/diarization_explore/mocai_compose.py 1 2 3 4 5
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import mocai_long as ML  # noqa: E402

FRAME = ML.FRAME


def owner_map(truth, activity, n_people):
    """Whole-file majority character per track column."""
    out = {}
    for k in range(activity.shape[1]):
        sel = activity[:, k] & (truth >= 0)
        if sel.any():
            out[k] = int(np.bincount(truth[sel], minlength=n_people).argmax())
    return out


def attribution(cues, activity, owner, people):
    right = total = 0.0
    for a, b, w in cues:
        lo, hi = int(a / FRAME), max(int(a / FRAME) + 1, int(np.ceil(b / FRAME)))
        act = activity[lo:hi].sum(axis=0)
        total += b - a
        if act.any():
            right += (b - a) * (owner.get(int(act.argmax())) == people.index(w))
    return right / total if total else float("nan")


def compose(track_of, split, grouping):
    """Replace each split parent column by child columns, grouped by `grouping(frames, assign) -> assign`."""
    parents = [int(k) for k in split["tracks"]]
    keep = [k for k in range(track_of.shape[1]) if k not in parents]
    columns = [track_of[:, k] for k in keep]
    for k in parents:
        frames, assign = split[f"frames_{k}"], split[f"assign_{k}"]
        assign = grouping(frames, assign)
        for j in sorted(set(assign.tolist()) - {-1}):
            col = np.zeros(len(track_of), dtype=bool)
            col[frames[assign == j]] = True
            columns.append(col)
    return np.stack(columns, axis=1)


def as_is(frames, assign):
    return assign


def time_blocks(rng):
    """Keep which frames are dropped (-1) and each child's size; regroup the kept ones into contiguous runs."""
    def group(frames, assign):
        kept = np.flatnonzero(assign >= 0)                      # positions of kept frames, in time order
        sizes = [int((assign == j).sum()) for j in sorted(set(assign.tolist()) - {-1})]
        order = np.roll(kept, int(rng.integers(len(kept)))) if len(kept) else kept
        out = np.full_like(assign, -1)
        cursor = 0
        for j, size in enumerate(sizes):
            out[order[cursor:cursor + size]] = j
            cursor += size
        return out
    return group


def main() -> None:
    parts = [int(p) for p in sys.argv[1:]] or [1, 2, 3, 4, 5]
    rng = np.random.default_rng(7)
    print(f"{'part':>5}{'tracks b/s':>11}{'base':>8}{'split':>8}{'blocks':>9}{'split-base':>12}{'split-blocks':>14}")
    rows = []
    for p in parts:
        path = WT / "out" / f"mocai-in{p}" / f"mocai-in{p}-split.npz"
        if not path.exists():
            print(f"{p:>5}  no saved split -- run mocai_split.py {p} first")
            continue
        split = dict(np.load(path))
        _, cues, people, truth, track_of = ML.build(p)
        n = len(people)
        base = attribution(cues, track_of, owner_map(truth, track_of, n), people)
        sp = compose(track_of, split, as_is)
        s = attribution(cues, sp, owner_map(truth, sp, n), people)
        blocks = []
        for _ in range(20):
            bl = compose(track_of, split, time_blocks(rng))
            blocks.append(attribution(cues, bl, owner_map(truth, bl, n), people))
        b = float(np.mean(blocks))
        hours = len(truth) * FRAME / 3600
        rows.append((p, hours, base, s, b))
        print(f"{p:>5}{f'4/{sp.shape[1]}':>11}{100 * base:>7.1f}%{100 * s:>7.1f}%{100 * b:>8.1f}%"
              f"{100 * (s - base):>+12.1f}{100 * (s - b):>+14.1f}")
    if rows:
        w = np.array([r[1] for r in rows])
        base, s, b = (np.average([r[i] for r in rows], weights=w) for i in (2, 3, 4))
        print(f"{'ALL':>5}{'':>11}{100 * base:>7.1f}%{100 * s:>7.1f}%{100 * b:>8.1f}%"
              f"{100 * (s - base):>+12.1f}{100 * (s - b):>+14.1f}")
        beats = sum(r[3] > r[4] for r in rows)
        print(f"\nsplit beats the time-block control on attribution in {beats}/{len(rows)} parts")


if __name__ == "__main__":
    main()
