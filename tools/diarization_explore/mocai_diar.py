"""Score a Sortformer timeline against 魔裁's script-derived speaker truth.

Plan 5r.2: this is the >4-speaker instrument. 魔裁 carries 11-15 distinct speakers per hour against the
model's four slots, which is far past anything else we own, and its truth needs no ears (`mocai_truth.py`).

Unlike `refined_multi.py` this scores the TIMELINE DIRECTLY and never touches stable.json, so it needs no
ASR run: for every truth cue we ask which track was most active over it and whether that track belongs to the
right character. That also means the separated and un-separated arms differ in exactly one thing.

  python mocai_diar.py TIMELINE.json [--offset SECONDS] [--part lower1] [--min-track 20]

⚠ **Three limits, all structural, none of them fixable by more data.**
1. **The cue span is an upper bound, not the speech interval** (owner 2026-09-15): the OCR subtitle contains
   the speech but may carry blank padding. Coverage-based numbers are therefore diluted by an unknown amount
   and are only comparable BETWEEN ARMS, never against 希娜's or 姐姐咚咚's absolute levels.
2. **The streamer is not in the truth at all.** Only game voices carry a script tag, and the streamer talks
   over the whole recording -- so a track that belongs to them has no right answer and shows up as low
   `truth_cover`. `--min-track` drops tracks too quiet to judge; the streamer is reported, not scored.
3. **`Narrative` is a tag, not a person**, and may be unvoiced. It is excluded from the speaker set and
   reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mocai_truth as MT  # noqa: E402

FRAME = 0.08


def truth_cues(part: str):
    """(start, end, character) for cues whose text maps to exactly one script speaker."""
    idx, _ = MT.script_index()
    out = []
    for a, b, text in MT.ass_rows(MT.BASE / "game_text" / f"{part}_jp.ass"):
        who = idx.get(MT.norm(text))
        if who and len(who) == 1:
            name = next(iter(who))
            if name != "Narrative":
                out.append((a, b, name))
    return out


def score(timeline_path: Path, part: str, offset: float, min_track: float):
    tracks = json.loads(timeline_path.read_text(encoding="utf-8"))["tracks"]
    span_end = max(t["end"] for t in tracks)
    cues = [(a - offset, b - offset, w) for a, b, w in truth_cues(part)]
    cues = [(a, b, w) for a, b, w in cues if b > 0 and a < span_end]
    if not cues:
        raise SystemExit(f"no truth cues inside 0..{span_end:.0f}s -- wrong --offset?")
    people = sorted({w for *_, w in cues})
    frames = int(span_end / FRAME) + 2
    grid = np.zeros((frames, len(people)), dtype=bool)
    for a, b, w in cues:
        grid[max(0, int(a / FRAME)):min(frames, int(np.ceil(b / FRAME))), people.index(w)] = True

    # a track's owner: the character covering most of its active frames
    per_track = defaultdict(Counter)
    track_frames = Counter()
    for t in tracks:
        lo, hi = max(0, int(t["start"] / FRAME)), min(frames, int(np.ceil(t["end"] / FRAME)))
        track_frames[t["track"]] += hi - lo
        span = grid[lo:hi].sum(axis=0)
        for i, c in enumerate(span):
            per_track[t["track"]][people[i]] += int(c)
    owner, purity, cover = {}, {}, {}
    for k, counts in per_track.items():
        total = track_frames[k] or 1
        claimed = sum(counts.values())
        cover[k] = claimed / total          # how much of this track is inside ANY truth cue
        if counts and claimed:
            who, n = counts.most_common(1)[0]
            owner[k], purity[k] = who, n / claimed
    judged = {k for k in owner if track_frames[k] * FRAME >= min_track}

    # per cue: which track was most active over it, and is it the right character's?
    right = total = 0.0
    inset = 0.0
    for a, b, w in cues:
        lo, hi = max(0, int(a / FRAME)), min(frames, max(int(a / FRAME) + 1, int(np.ceil(b / FRAME))))
        active = Counter()
        for t in tracks:
            s, e = max(lo, int(t["start"] / FRAME)), min(hi, int(np.ceil(t["end"] / FRAME)))
            if e > s:
                active[t["track"]] += e - s
        if not active:
            total += b - a
            continue
        lead = active.most_common(1)[0][0]
        d = b - a
        total += d
        right += d * (owner.get(lead) == w)
        inset += d * any(owner.get(k) == w for k in active)
    # The ceiling four slots allow at all: one slot per character, best four. Anything below this is the
    # model's problem; the gap above it is arithmetic and no amount of tuning touches it. ⚠ The streamer is
    # not in the truth but does occupy a slot, so the honest ceiling for a stream is the three-slot one.
    spoken = Counter()
    for a, b, w in cues:
        spoken[w] += b - a
    whole = sum(spoken.values()) or 1.0
    ceiling = {n: sum(s for _, s in spoken.most_common(n)) / whole for n in (3, 4)}

    return {
        "tracks": len(track_frames), "people": len(people), "cues": len(cues),
        "first_is_human": right / total, "human_in_set": inset / total, "ceiling": ceiling,
        "owner": owner, "purity": purity, "cover": cover, "judged": judged,
        "track_seconds": {k: v * FRAME for k, v in track_frames.items()},
        "speaker_seconds": {p: float(grid[:, i].sum()) * FRAME for i, p in enumerate(people)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--part", default="lower1")
    parser.add_argument("--offset", type=float, default=0.0, help="seconds of the source this timeline starts at")
    parser.add_argument("--min-track", type=float, default=20.0)
    args = parser.parse_args()
    r = score(args.timeline, args.part, args.offset, args.min_track)
    print(f"{args.timeline}  |  {r['tracks']} tracks vs {r['people']} characters, {r['cues']} truth cues")
    print(f"  first track == right character : {r['first_is_human']:.1%}  (by cue duration)")
    print(f"  right character among active   : {r['human_in_set']:.1%}")
    print(f"  ceiling, 4 slots one each      : {r['ceiling'][4]:.1%}   "
          f"(3 slots, if the streamer takes one: {r['ceiling'][3]:.1%})")
    print(f"  -> reached {r['first_is_human'] / r['ceiling'][3]:.0%} of the three-slot ceiling")
    print(f"\n  {'track':<7}{'seconds':>9}{'owner':>12}{'purity':>9}{'truth_cover':>13}")
    for k in sorted(r["track_seconds"], key=lambda k: -r["track_seconds"][k]):
        mark = "" if k in r["judged"] else "   (too short to judge)"
        print(f"  s{k:<6}{r['track_seconds'][k]:>9.1f}{r['owner'].get(k, '-'):>12}"
              f"{r['purity'].get(k, float('nan')):>9.1%}{r['cover'].get(k, 0):>13.1%}{mark}")
    owned = {v for k, v in r["owner"].items() if k in r["judged"]}
    print(f"\n  characters with a track of their own: {len(owned)}/{r['people']}  {sorted(owned)}")
    missing = sorted(set(r["speaker_seconds"]) - owned, key=lambda p: -r["speaker_seconds"][p])
    print(f"  without one (by truth seconds): " +
          ", ".join(f"{p} {r['speaker_seconds'][p]:.0f}s" for p in missing[:8]))


if __name__ == "__main__":
    main()
