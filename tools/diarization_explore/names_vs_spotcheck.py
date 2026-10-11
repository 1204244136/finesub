"""Score per-track names from the role call against the human nepolabo spot-check.

Spot-check rows carry a trailing 1-4 (or mix) written by ear. Mapping taken from
the table header's order: 1 ねね, 2 ポルカ, 3 ぼたん, 4 ラミィ -- an inference, flagged.
"""

import json
import re
from pathlib import Path

# Paths: DATA = <repo>/data/diarization (hand-kept, main checkout), WORK =
# <repo>/tmp/diarization_explore (scratch; the three .ts go in WORK/export_v2).
import os as _os
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(_os.environ.get("DIAR_EXPLORE_DATA", ROOT / "data" / "diarization"))
WORK = Path(_os.environ.get("DIAR_EXPLORE_WORK", ROOT / "tmp" / "diarization_explore"))
SPOT = DATA / "nepolabo-spotcheck.md"
MAPPING = {"1": "ねね", "2": "ポルカ", "3": "ぼたん", "4": "ラミィ"}
ALIASES = {"ねね": ("ねね", "桃鈴"), "ポルカ": ("ポルカ", "尾丸"), "ぼたん": ("ぼたん", "獅白", "シシロ", "ししろ"),
           "ラミィ": ("ラミィ", "雪花")}


def spot_rows():
    rows = []
    for line in SPOT.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\| (\d+):([\d.]+) \| ([\d.]+)s \|.*\|\s*([^|]*?)\s*$", line)
        if not match:
            continue
        label = match.group(4).strip()
        if label not in MAPPING:
            continue  # blank or mix
        start = int(match.group(1)) * 60 + float(match.group(2))
        rows.append((start, start + float(match.group(3)), MAPPING[label]))
    return rows


def timeline():
    return json.loads((WORK / "roles" / "nepolabo-diar.json").read_text(encoding="utf-8"))["tracks"]


def dominant_track(tracks, start, end):
    cover = {}
    for seg in tracks:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > 0:
            cover[seg["track"]] = cover.get(seg["track"], 0) + overlap
    if not cover:
        return None, {}
    return max(cover, key=cover.get), cover


def canonical(name):
    for key, forms in ALIASES.items():
        if name and any(form in name for form in forms):
            return key
    return None


def tally(names, rows, tracks):
    """names: {"s0": raw name or None}. Returns (right, wrong, unnamed) over the spot-check rows."""
    names = {track: canonical(name) for track, name in names.items()}
    right = wrong = unnamed = 0
    for start, end, who in rows:
        track, _ = dominant_track(tracks, start, end)
        said = names.get(f"s{track}")
        if said is None:
            unnamed += 1
        elif said == who:
            right += 1
        else:
            wrong += 1
    return right, wrong, unnamed


def main():
    rows, tracks = spot_rows(), timeline()
    print(f"{len(rows)} labelled rows")
    truth_by_track = {}
    for start, end, who in rows:
        track, cover = dominant_track(tracks, start, end)
        truth_by_track.setdefault(track, []).append(who)
        print(f"{start:7.1f} {who:<4} -> s{track}  {({f's{k}': round(v, 1) for k, v in cover.items()})}")
    print("\ntruth per track:", {f"s{k}": v for k, v in truth_by_track.items()})

    for log in ("answers-v1.jsonl", "answers.jsonl"):
        for line in (WORK / "roles" / log).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["file"] != "nepolabo":
                continue
            names = {row["track"]: row["name"] for row in record["roles"]}
            right, wrong, unnamed = tally(names, rows, tracks)
            shown = {k: canonical(v) for k, v in names.items() if canonical(v)}
            print(f"{log} #{record['attempt']}: names={shown} right={right} wrong={wrong} unnamed={unnamed}")


if __name__ == "__main__":
    main()
