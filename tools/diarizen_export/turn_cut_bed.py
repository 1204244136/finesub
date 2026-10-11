"""C.5d, the other half: when the model says "split here", is the cut in the right place?

Owner 2026-09-20: 「旋钮判断该拆，但它怎么拆呢？LLM 才能拆吧？」 -- which lands on a real hole in
C.5d. The threshold knob it was benchmarked against can only ever say WHICH rows hold a turn; it
cannot produce the cut, so it is not a substitute for the model at all. The detection comparison
was measuring the one end where a free knob happens to compete.

The replies already carry the other end: every `A` answer came with `cut_after`, the last few
characters of the first speaker's part. Nothing scored them. This does, from the logs, with no
new calls.

⚠ **The cut is scored in SECONDS, through the word timestamps** -- `-stable.json` carries per-word
`start`/`end`, which is also what S4 would use to perform the split in production. Converting a
character position by proportion of the row would assume a constant speaking rate, and this bed's
sibling already recorded what that assumption buys (`turn_shape_bed.MIN_TURN_SECONDS`).

⚠ Truth for the cut is the refined subtitle's own speaker boundary inside the row: the midpoint
between where the first person stops and the second starts. It is only defined where the row
really does hold a handover, so **rows the model called `A` wrongly are reported separately** --
a cut placed inside a row that should never have been cut has no correct position to be near.

    python tools/diarizen_export/turn_cut_bed.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = REPO / "tmp" / "diarizen-bed" / "turn-llm"
BLOCK = re.compile(r"<splits>(.*?)</splits>", re.S)

#: A cut this close to the truth boundary is right. ⚠ Generous on purpose: the refined axis is a
#: fansub's line timing, not a forced alignment, so its boundary carries its own error.
TOLERANCE = 0.5


def row_words(row):
    """`[(char index just past this word, end time)]`, reconstructing the row's own text."""
    found, cursor = [], 0
    for word in row.get("words") or []:
        if word.get("space_before"):
            cursor += 1
        cursor += len(str(word.get("word", "")))
        found.append((cursor, float(word.get("end", 0.0))))
    return found


def model_cut_time(row, cut_after: str):
    """Where in time the model's `cut_after` falls, or None if it is not in the row's text."""
    text = str(row.get("text", ""))
    if not cut_after or cut_after not in text:
        return None
    index = text.find(cut_after) + len(cut_after)
    for cursor, end in row_words(row):
        if cursor >= index:
            return end
    words = row_words(row)
    return words[-1][1] if words else None


def truth_cut_time(row, events):
    """The refined axis's own speaker boundary inside the row, or None if it has none.

    The first place the speaker changes among the refined lines overlapping the row, taken as the
    midpoint of the gap between them. ⚠ Not the biggest gap and not the middle of the row: the
    question is where the handover is, and a row with two handovers is answered by the first.
    """
    start, end = float(row["start"]), float(row["end"])
    inside = sorted(((max(a, start), min(b, end), who) for a, b, who, _ in events
                     if min(b, end) - max(a, start) > 0),
                    key=lambda item: item[0])
    for earlier, later in zip(inside, inside[1:]):
        if earlier[2] != later[2]:
            return 0.5 * (earlier[1] + later[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=["audio-agy", "text-agy"])
    parser.add_argument("--episodes", nargs="+", type=int, default=[1, 2, 3])
    arguments = parser.parse_args()

    from turn_shape_bed import episode_rows

    loaded = {}
    for episode in arguments.episodes:
        found = episode_rows(episode)
        if found is not None:
            loaded[episode] = (found[0], found[3])

    for arm in arguments.arms:
        path = OUT / f"{arm}.jsonl"
        if not path.exists():
            print(f"{arm}: no log")
            continue
        errors, unlocatable, no_truth, wrong_row = [], 0, 0, 0
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()):
            stable, events = loaded.get(record["episode"], (None, None))
            if stable is None:
                continue
            match = BLOCK.search(record["raw"] or "")
            if not match:
                continue
            by_start = {round(float(row["start"]), 1): row for row in stable}
            for line in match.group(1).splitlines():
                try:
                    payload = json.loads(line.strip())
                except (json.JSONDecodeError, ValueError):
                    continue
                if str(payload.get("answer", "")).upper() != "A":
                    continue
                key = round(float(payload["time"]), 1)
                if record["truth"].get(str(key)) != "splittable":
                    wrong_row += 1
                    continue
                row = by_start.get(key)
                if row is None:
                    continue
                # v1 calls it `cut_after`, v2 calls it `first`. ⚠ Both mean "the last few
                # characters of the first speaker, quoted", so they score identically -- but a
                # scorer that knew only v1's name would read every v2 cut as unlocatable and
                # report the variant as having rewritten the text.
                quoted = payload.get("cut_after") or payload.get("first") or ""
                said = model_cut_time(row, str(quoted))
                truth = truth_cut_time(row, events)
                if said is None:
                    unlocatable += 1
                elif truth is None:
                    no_truth += 1
                else:
                    errors.append(abs(said - truth))

        print(f"\n{arm}: {len(errors) + unlocatable + no_truth} cuts on rows that really "
              f"should be cut  (+{wrong_row} on rows that should not)")
        if unlocatable:
            print(f"  ⚠ {unlocatable} `cut_after` strings are not in the row's own text -- the "
                  f"model rewrote instead of quoting, so there is no position to score")
        if no_truth:
            print(f"  {no_truth} rows have no speaker boundary on the refined axis")
        if errors:
            array = np.asarray(errors)
            print(f"  cut error: median {np.median(array):.2f}s  P90 "
                  f"{np.percentile(array, 90):.2f}s  max {array.max():.2f}s")
            print(f"  within {TOLERANCE}s: {float((array <= TOLERANCE).mean()):.0%} "
                  f"({int((array <= TOLERANCE).sum())}/{len(array)})")
        else:
            print("  no scorable cut")


if __name__ == "__main__":
    main()
