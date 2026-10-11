"""Is the recursive split's purity gain real, or arithmetic?

⚠ `mocai_split.py` reports "purity on character frames: parent -> children (weighted)", and on all ten parent
tracks across 魔裁's five parts it goes up (+0.5 to +15.3 pp). **That cannot be read as the split working.**
Partitioning a set never lowers its weighted majority share: each child's top-character count is at least its
count of the PARENT's top character, and those counts sum to the parent's top count. A random partition
"improves" every track too.

So this compares the real split against two nulls with the same child sizes, on the parent's own frames:

  shuffle  children are random samples of the parent's frames -- the weak null
  blocks   children are CONTIGUOUS time runs of the parent's frames -- the strong null. Characters come in
           scenes, so a partition by time alone already gains purity from WHEN people talk. A split that only
           beats `shuffle` is rediscovering scene structure; it is separating voices only to the extent it beats
           `blocks`.

Child sizes come from `mocai_split.py`'s printed output (tmp/mocai-split*.log), so no GPU run is needed.

    python tools/diarization_explore/mocai_split_null.py tmp/mocai-split-input1.log tmp/mocai-split.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mocai_long as ML  # noqa: E402

FRAME = ML.FRAME
PARENT = re.compile(r"^=== input(\d+) parent track ===")
TRACK = re.compile(r"^s(\d+)\s+([\d.]+) h\s+outside cues\s+([\d.]+)%\s+purity\s+([\d.]+)%")
CHILD = re.compile(r"^\s+s\d+\.\d+\s+([\d.]+) h\s+outside cues\s+([\d.]+)%\s+purity\s+([\d.]+)%")
RESULT = re.compile(r"parent ([\d.]+)% -> children \(weighted\) ([\d.]+)%")


def parse(paths):
    """[(part, track, [(child_char_frames, child_purity), ...], parent_purity, children_purity)]"""
    out, cur = [], None
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            if m := PARENT.match(line):
                cur = {"part": int(m[1]), "children": []}
            elif cur is not None and (m := TRACK.match(line)) and "track" not in cur:
                cur["track"] = int(m[1])
            elif cur is not None and (m := CHILD.match(line)):
                hours, outside = float(m[1]), float(m[2]) / 100
                cur["children"].append(int(round(hours * (1 - outside) * 3600 / FRAME)))
            elif cur is not None and (m := RESULT.search(line)):
                cur["parent"], cur["split"] = float(m[1]), float(m[2])
                out.append(cur)
                cur = None
    return out


def weighted_purity(labels, sizes):
    """Sum of each group's top-label count over all frames -- the same statistic mocai_split reports."""
    top, cursor = 0, 0
    for size in sizes:
        group = labels[cursor:cursor + size]
        if len(group):
            top += np.bincount(group).max()
        cursor += size
    return top / max(cursor, 1)


def main() -> None:
    rows = parse(sys.argv[1:])
    cache = {}
    rng = np.random.default_rng(7)
    print(f"{'part/track':<11}{'parent':>8}{'split':>8}{'shuffle':>9}{'blocks':>8}{'split-blocks':>14}")
    gains = []
    for r in rows:
        if r["part"] not in cache:
            cache[r["part"]] = ML.build(r["part"])
        _, _, people, truth, track_of = cache[r["part"]]
        frames = np.flatnonzero(track_of[:, r["track"]] & (truth >= 0))   # parent's character frames, time order
        labels = truth[frames]
        sizes = r["children"]
        scale = len(labels) / max(sum(sizes), 1)                            # children exclude unassigned/tiny
        sizes = [max(1, int(round(s * scale))) for s in sizes]
        sizes[-1] += len(labels) - sum(sizes)
        shuffle = np.mean([weighted_purity(rng.permutation(labels), sizes) for _ in range(200)])
        # strong null: contiguous runs, at 50 random rotations of where the first run starts
        blocks = np.mean([weighted_purity(np.roll(labels, int(rng.integers(len(labels)))), sizes) for _ in range(50)])
        split = r["split"] / 100
        gains.append(split - blocks)
        print(f"in{r['part']}/s{r['track']:<6}{r['parent']:>7.1f}%{r['split']:>7.1f}%{100 * shuffle:>8.1f}%"
              f"{100 * blocks:>7.1f}%{100 * (split - blocks):>+13.1f}")
    g = np.array(gains)
    print(f"\nsplit beats the STRONG null (time blocks) on {int((g > 0).sum())}/{len(g)} tracks; "
          f"mean margin {100 * g.mean():+.1f} pp, median {100 * np.median(g):+.1f} pp")


if __name__ == "__main__":
    main()
