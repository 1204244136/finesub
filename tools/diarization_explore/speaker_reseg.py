"""T4 in the shape a fansub axis uses: one speaker per cue, interjections as their own overlapping cues.

Owner preview 2026-09-12. T4 (plan 5h.17, 5o.7) puts each production word on the track sounding under it
and is the only S4 arm that keeps every word (retention 99.6-99.8%). Its open fault is where it cuts:
it split a sentence at every track change (`今 / それも / ネネが全部頑張ってく`, plan 5o.11). Human axes
keep the main clause on one line and give an interjection its own line that overlaps it in time
(plan 5p.3, corrected in 5q). This rewrites a diarized `-stable.json` that way, before the LLM stage.

Rules, per production segment (a segment is never merged with its neighbours):
  1. a word goes to the track with the most active frames under it; none active -> the nearest track
     within NEAR seconds; still none -> the segment's own leader;
  2. a run of at most JITTER_CHARS characters and JITTER_S seconds at a track change is boundary jitter
     and joins its neighbour (the previous run, else the next);
  3. a run of another track sitting between two runs of one track, no longer than INTERJECT_S and
     INTERJECT_CHARS, becomes its own cue; the two sides merge into one cue spanning it;
  4. any other change of track starts a new cue.

Descriptive values, not calibrated -- this is a preview, not an arm with a verdict.

  python speaker_reseg.py OUT_DIR    # rewrites <stem>-stable.json, keeps <stem>-stable.pre-reseg.json
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np

FRAME = 0.08
NEAR = 0.5
JITTER_CHARS, JITTER_S = 2, 0.4
INTERJECT_CHARS, INTERJECT_S = 10, 1.6
CUT_GAP, HOLD_CHARS = 0.12, 4
BREAKS = tuple("、。，,.!?！？…」")


def grid_of(tracks, end):
    ids = sorted({t["track"] for t in tracks})
    active = np.zeros((int(end / FRAME) + 2, len(ids)), dtype=bool)
    for t in tracks:
        active[int(t["start"] / FRAME):int(np.ceil(t["end"] / FRAME)), ids.index(t["track"])] = True
    return ids, active


def track_of(word, ids, active, fallback):
    a = int(word["start"] / FRAME)
    b = max(a + 1, int(np.ceil(word["end"] / FRAME)))
    counts = active[a:b].sum(axis=0)
    if counts.any():
        return ids[int(counts.argmax())]
    reach = int(NEAR / FRAME)
    counts = active[max(0, a - reach):b + reach].sum(axis=0)
    return ids[int(counts.argmax())] if counts.any() else fallback


def hold_to_a_boundary(words, assigned, stats):
    """Move a track change to the next place the speech itself breaks.

    Whisper's Japanese words are characters or short tokens, so a change of track under them lands
    mid-word (`聞き取 | れないそれ`, seen on the first dry run). A change stays where it is only when a
    pause of CUT_GAP or punctuation precedes it; otherwise it moves forward to the first such break
    within HOLD_CHARS characters, and if there is none it stays -- a real turn with no pause is kept.
    """
    out = list(assigned)
    i = 1
    while i < len(words):
        if out[i] != out[i - 1] and not breaks_before(words, i):
            j, reach = i, 0
            while j < len(words) and reach <= HOLD_CHARS and not breaks_before(words, j):
                reach += len(words[j]["word"].strip())
                j += 1
            if j < len(words) and breaks_before(words, j) and j > i:
                for k in range(i, j):
                    out[k] = out[i - 1]
                stats["held"] += j - i
                i = j
                continue
        i += 1
    return out


def breaks_before(words, i):
    return words[i]["start"] - words[i - 1]["end"] >= CUT_GAP or words[i - 1]["word"].strip()[-1:] in BREAKS


def chars(run):
    return sum(len(w["word"].strip()) for w in run["words"])


def seconds(run):
    return run["words"][-1]["end"] - run["words"][0]["start"]


def runs_of(words, tracks_of_words):
    runs = []
    for word, track in zip(words, tracks_of_words):
        if runs and runs[-1]["track"] == track:
            runs[-1]["words"].append(word)
        else:
            runs.append({"track": track, "words": [word]})
    return runs


def merge_adjacent(runs):
    out = []
    for run in runs:
        if out and out[-1]["track"] == run["track"]:
            out[-1]["words"].extend(run["words"])
        else:
            out.append({"track": run["track"], "words": list(run["words"])})
    return out


def cues_for(runs, stats):
    # 2. boundary jitter
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if chars(run) <= JITTER_CHARS and seconds(run) <= JITTER_S:
                target = runs[i - 1] if i > 0 else runs[i + 1]
                if target["track"] != run["track"]:
                    run["track"] = target["track"]
                    stats["jitter"] += 1
                    runs = merge_adjacent(runs)
                    changed = True
                    break
    # 3. interjections
    main, side = [], []
    i = 0
    while i < len(runs):
        run = runs[i]
        if (0 < i < len(runs) - 1 and runs[i - 1]["track"] == runs[i + 1]["track"] != run["track"]
                and chars(run) <= INTERJECT_CHARS and seconds(run) <= INTERJECT_S):
            side.append(run)
            stats["interjections"] += 1
        else:
            main.append(run)
        i += 1
    main = merge_adjacent(main)
    stats["splits"] += len(main) - 1
    return main + side


def rewrite(out_dir):
    out_dir = Path(out_dir)
    stem = out_dir.name
    stable_path = out_dir / f"{stem}-stable.json"
    backup = out_dir / f"{stem}-stable.pre-reseg.json"
    if not backup.exists():
        shutil.copyfile(stable_path, backup)
    payload = json.loads(backup.read_text(encoding="utf-8"))
    timeline = json.loads((out_dir / f"{stem}-diar.json").read_text(encoding="utf-8"))
    end = max([t["end"] for t in timeline["tracks"]] + [s["end"] for s in payload["segments"]])
    ids, active = grid_of(timeline["tracks"], end)
    stats = {"segments_in": len(payload["segments"]), "held": 0, "jitter": 0, "interjections": 0, "splits": 0, "moved_words": 0}
    segments = []
    for segment in payload["segments"]:
        words = [w for w in segment.get("words") or [] if w.get("word", "").strip()]
        speaker = segment.get("speaker") or {}
        leader = speaker.get("track")
        if not words or leader is None:
            segments.append(segment)
            continue
        assigned = hold_to_a_boundary(words, [track_of(w, ids, active, leader) for w in words], stats)
        stats["moved_words"] += sum(t != leader for t in assigned)
        for cue in cues_for(runs_of(words, assigned), stats):
            new = {k: copy.deepcopy(v) for k, v in segment.items() if k not in ("words", "text", "start", "end", "speaker")}
            new["start"] = round(cue["words"][0]["start"], 3)
            new["end"] = round(max(w["end"] for w in cue["words"]), 3)
            new["words"] = cue["words"]
            new["text"] = "".join((" " if w.get("space_before") else "") + w["word"] for w in cue["words"]).strip()
            a, b = int(new["start"] / FRAME), max(int(new["start"] / FRAME) + 1, int(np.ceil(new["end"] / FRAME)))
            share = float(active[a:b, ids.index(cue["track"])].mean()) if cue["track"] in ids else 1.0
            new["speaker"] = {"labels": [], "block": speaker.get("block", 0), "track": cue["track"],
                              "tracks": [cue["track"]], "coverage": {str(cue["track"]): round(share, 4)}}
            segments.append(new)
    segments.sort(key=lambda s: (s["start"], s["end"]))
    stats["segments_out"] = len(segments)
    payload["segments"] = segments
    payload.setdefault("metadata", {})["speaker_reseg"] = {"source": backup.name, **stats,
                                                          "rules": {"near": NEAR, "jitter": [JITTER_CHARS, JITTER_S],
                                                                    "interject": [INTERJECT_CHARS, INTERJECT_S]}}
    stable_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stem, stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    rewrite(parser.parse_args().out_dir)


if __name__ == "__main__":
    main()
