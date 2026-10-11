"""W8: does running the separator change the mocai timeline, once the format confound is gone?

findings 4.3 reads five unseparated eleven-hour parts, and its first premise is that feeding the
diarizer un-separated audio is close enough to what production does. Checking that needs two arms
that differ in ONE thing, and the obvious pair does not:

    -diar-sep.json       separator -> 16 kHz mono ogg -> diarize
    -diar-nosep.json     44.1 kHz stereo FLAC -> diarize          <- two differences, not one

The second arm changes both whether separation ran AND the decode path, and a 3e-6 input change
already re-draws this model's attractor (findings 5.1), so at four files that confound is not
ignorable. `tmp/mocai_nosep.sh` produces the third arm, routing the un-separated source through
`encode_asr_delivery` so it arrives at the same spec and path separation would have produced:

    -diar-nosep-ogg.json un-separated -> 16 kHz mono ogg -> diarize  <- differs from sep in one thing

⚠ mocai has no ground truth here, so nothing below is an accuracy. These are AGREEMENT readings:
how much of the timeline moves. The comparison that answers the premise is `sep vs nosep-ogg`; the
pair `nosep vs nosep-ogg` measures the format confound alone, and is the yardstick the first number
has to be read against -- exactly as W4 reads segmenting against a meaningless noise nudge.

    python tools/diarization_explore/w8_separation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402

HOURS = (2, 3, 6, 7)
ARMS = {"sep": "-diar-sep.json", "nosep": "-diar-nosep.json", "nosep-ogg": "-diar-nosep-ogg.json"}


def activity(path: Path, frames: int | None = None) -> np.ndarray:
    """`(frames, tracks)` boolean activity from a `-diar.json`'s flat spans."""
    spans = json.loads(path.read_text(encoding="utf-8"))["tracks"]
    width = max((span["track"] for span in spans), default=0) + 1
    length = frames or int(np.ceil(max(span["end"] for span in spans) / lab.FRAME))
    grid = np.zeros((length, width), dtype=bool)
    for span in spans:
        low = int(span["start"] / lab.FRAME)
        high = min(length, max(low + 1, int(np.ceil(span["end"] / lab.FRAME))))
        if low < length:
            grid[low:high, span["track"]] = True
    return grid


def leading(grid: np.ndarray) -> np.ndarray:
    """Per frame: the lowest-numbered active track, or -1 when silent.

    The timelines carry no probabilities, so "leading" can only mean "the one the
    artifact lists first" -- which is what a consumer reading `tracks[0]` sees.
    """
    active = grid.any(axis=1)
    return np.where(active, grid.argmax(axis=1), -1)


def align(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Permute `other`'s tracks onto `reference`'s by best one-to-one overlap."""
    from scipy.optimize import linear_sum_assignment

    width = max(reference.shape[1], other.shape[1])

    def pad(grid):
        if grid.shape[1] >= width:
            return grid
        return np.hstack([grid, np.zeros((len(grid), width - grid.shape[1]), dtype=bool)])

    a, b = pad(reference), pad(other)
    length = min(len(a), len(b))
    table = a[:length].T.astype(np.int64) @ b[:length].astype(np.int64)
    rows, columns = linear_sum_assignment(-table)
    mapping = dict(zip(columns, rows))
    out = np.zeros_like(b)
    for source in range(width):
        out[:, mapping.get(source, source)] |= b[:, source]
    return out


def disagreement(first: np.ndarray, second: np.ndarray) -> dict:
    """Share of frames whose leading track differs, over frames either arm calls speech."""
    second = align(first, second)
    length = min(len(first), len(second))
    a, b = leading(first[:length]), leading(second[:length])
    either = (a >= 0) | (b >= 0)
    if not either.any():
        return {"frames": 0, "changed_pct": float("nan")}
    return {
        "frames": int(either.sum()),
        "changed_pct": 100.0 * float((a[either] != b[either]).sum()) / int(either.sum()),
        "tracks": (int(first.shape[1]), int(second.shape[1])),
    }


def shape(grid: np.ndarray) -> dict:
    """The structural quantities findings 4.3's conclusions actually rest on.

    Frame-level identity is the wrong resolution for that premise: this chain re-draws its
    attractor on any input change at all (findings 5.1), so two arms disagreeing per frame says little.
    What findings 4.3 claims is about SHAPE -- four slots carrying fourteen-odd characters, two big mixed
    tracks and small pure ones -- and shape survives a re-draw that identity does not.
    """
    seconds = grid.sum(axis=0) * lab.FRAME
    total = float(seconds.sum())
    ordered = sorted((float(value) for value in seconds), reverse=True)
    return {
        "speech_seconds": total,
        "tracks_active": int((seconds > 0).sum()),
        "largest_share": ordered[0] / total if total else float("nan"),
        "top_two_share": sum(ordered[:2]) / total if total else float("nan"),
        "overlap_share": float((grid.sum(axis=1) > 1).sum()) * lab.FRAME / total if total else float("nan"),
    }


def main() -> None:
    rows = []
    for hour in HOURS:
        directory = lab.WT / "out" / f"mocai-h{hour}"
        paths = {name: directory / f"mocai-h{hour}{suffix}" for name, suffix in ARMS.items()}
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            print(f"hour {hour}: missing {', '.join(missing)}")
            continue
        grids = {name: activity(path) for name, path in paths.items()}
        rows.append({
            "hour": hour,
            # The question: did separation change anything, format held equal?
            "separation": disagreement(grids["sep"], grids["nosep-ogg"]),
            # The yardstick: how much moves when ONLY the decode path changes.
            "format": disagreement(grids["nosep"], grids["nosep-ogg"]),
            # What the confounded pair would have shown -- both changes at once.
            "confounded": disagreement(grids["sep"], grids["nosep"]),
            "shape": {name: shape(grid) for name, grid in grids.items()},
        })
    if not rows:
        raise SystemExit("no hours scored; run tmp/mocai_nosep.sh first")

    print(f"\nW8, mocai {len(rows)} 小时片段 —— 首选轨逐帧不一致率（无真值，只是一致性）\n")
    print(f"{'小时':>6}{'分离 vs 不分离':>18}{'只换格式':>14}{'两者混在一起':>16}")
    for row in rows:
        print(f"{row['hour']:>6}{row['separation']['changed_pct']:>17.2f}%"
              f"{row['format']['changed_pct']:>13.2f}%{row['confounded']['changed_pct']:>15.2f}%")
    for key, label in (("separation", "分离的效应"), ("format", "格式的效应（尺子）")):
        values = np.array([row[key]["changed_pct"] for row in rows])
        print(f"\n  {label}: 中位 {np.median(values):.2f}%  区间 {values.min():.2f}–{values.max():.2f}%")
    separation = np.array([row["separation"]["changed_pct"] for row in rows])
    format_only = np.array([row["format"]["changed_pct"] for row in rows])
    ratio = np.median(separation) / max(np.median(format_only), 1e-9)
    print(f"\n  ⇒ 分离的效应是格式效应的 {ratio:.2f}×"
          f"（n={len(rows)}，**不出区间**——4 份是个案集合，不是抽样分布）")
    print()
    print("结构量（第 6 条的结论靠的是这些，不是逐帧身份）")
    print()
    print(f"{'小时':>6}{'臂':>12}{'发声 s':>10}{'活跃轨':>8}{'最大轨占比':>12}{'前二占比':>11}{'重叠占比':>11}")
    for row in rows:
        for name in ARMS:
            s = row["shape"][name]
            print(f"{row['hour']:>6}{name:>12}{s['speech_seconds']:>10.0f}{s['tracks_active']:>8}"
                  f"{s['largest_share']*100:>11.1f}%{s['top_two_share']*100:>10.1f}%{s['overlap_share']*100:>10.1f}%")
    for key in ("largest_share", "top_two_share", "overlap_share", "speech_seconds"):
        sep = np.array([row["shape"]["sep"][key] for row in rows])
        nos = np.array([row["shape"]["nosep-ogg"][key] for row in rows])
        scale = 100.0 if key.endswith("share") else 1.0
        unit = " pp" if key.endswith("share") else " s"
        print(f"  {key:16} 分离 − 不分离：逐份 "
              f"{[round(float(x) * scale, 1) for x in (sep - nos)]}{unit}")

    (lab.WT / "tmp" / "diarization-w8-separation.json").write_text(
        json.dumps({"rows": rows, "script": lab.digest(__file__)}, indent=2), encoding="utf-8")
    print(f"\nwrote {lab.WT / 'tmp' / 'diarization-w8-separation.json'}")


if __name__ == "__main__":
    main()
