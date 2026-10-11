"""Reduce a multi-speaker line's labels without cutting the line (owner 2026-09-12: "可以试试改标签").

The resegmenter (`speaker_reseg.py`) split sentences, and the preview showed that cost more than it
saved: without the LLM, unsplit read better; with it, the LLM merged most fragments back. This keeps
every production segment and its text as they are and only rewrites the `speaker` field of segments
labelled with two or more tracks:

  1. each word goes to a track exactly as in `speaker_reseg` (most active frames, nearest track, the
     change moved to the next pause or punctuation);
  2. a track's share is its characters over the segment's characters;
  3. if the leading share reaches THETA the segment keeps that one track; otherwise it keeps its
     tracks, ordered by share.

Because text and line breaks do not change, an LLM stage that already ran stays valid: `--csv` also
rewrites the `spk` column of `-annotated.csv` from the relabelled stable (roles from the cached file,
no model call).

  python speaker_relabel.py OUT_DIR [--theta 0.8] [--csv]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import speaker_reseg as rs  # noqa: E402

THETA = 0.8


def relabel_segments(segments, timeline, theta=THETA, mode="words"):
    """mode:
      words     re-rank by character share, reduce to the leader when it reaches THETA (the first try;
                it lowered first-track agreement on 姐姐咚咚 #1-#6 from 88.4% to 86%, re-ranking alone
                included -- word-level track assignment is a worse leader signal than coverage);
      lead      keep the production leader; drop the other labels when its character share >= THETA;
      coverage  keep the production leader; keep another label only while its coverage >= THETA x the
                leader's (no words at all -- production's `second_share` is the same rule at 0.2).
    """
    end = max([t["end"] for t in timeline["tracks"]] + [s["end"] for s in segments] + [1.0])
    ids, active = rs.grid_of(timeline["tracks"], end)
    stats = Counter()
    out = []
    for segment in segments:
        speaker = dict(segment.get("speaker") or {})
        tracks = speaker.get("tracks") or []
        words = [w for w in segment.get("words") or [] if w.get("word", "").strip()]
        if len(tracks) < 2 or (not words and mode != "coverage"):
            out.append(segment)
            continue
        stats["multi"] += 1
        if mode == "coverage":
            coverage = speaker.get("coverage") or {}
            lead = tracks[0]
            kept = [lead] + [t for t in tracks[1:] if coverage.get(str(t), 0.0) >= theta * coverage.get(str(lead), 0.0)]
            new = dict(segment)
            speaker.update(track=lead, tracks=kept)
            if len(kept) == 1:
                speaker.pop("review", None)
                stats["reduced"] += 1
            else:
                stats["kept"] += 1
            new["speaker"] = speaker
            out.append(new)
            continue
        hold = Counter()
        assigned = rs.hold_to_a_boundary(words, [rs.track_of(w, ids, active, speaker.get("track")) for w in words], hold)
        chars = Counter()
        for word, track in zip(words, assigned):
            chars[track] += len(word["word"].strip())
        total = sum(chars.values()) or 1
        shares = {t: chars.get(t, 0) / total for t in set(tracks) | set(chars)}
        ranked = sorted(shares, key=lambda t: -shares[t])
        lead = tracks[0] if mode == "lead" else ranked[0]
        if mode == "lead":
            ranked = [lead] + [t for t in tracks[1:]]
        new = dict(segment)
        if shares[lead] >= theta:
            speaker.update(track=lead, tracks=[lead], review=False)
            speaker.pop("review", None)
            stats["reduced"] += 1
            stats["reduced_changed_lead"] += lead != tracks[0]
        else:
            kept = [t for t in ranked if t in tracks]  # reorder only: never add a label the line lacked
            speaker.update(track=kept[0], tracks=kept)
            stats["kept"] += 1
            stats["kept_reordered"] += kept[0] != tracks[0]
        speaker["char_share"] = {str(t): round(v, 3) for t, v in shares.items() if v}
        new["speaker"] = speaker
        out.append(new)
    return out, stats


def rewrite_csv(out_dir, stem):
    from finesub.llm.stages.correction.run import _speaker_cells
    import preview_ass as pa

    csv = out_dir / f"{stem}-annotated.csv"
    backup = out_dir / f"{stem}-annotated.pre-relabel.csv"
    if not backup.exists():
        shutil.copyfile(csv, backup)
    cell = _speaker_cells(out_dir / f"{stem}-stable.json", out_dir / f"{stem}.srt", None)
    changed = 0
    lines = []
    for line in backup.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or cell is None:
            lines.append(line)
            continue
        head = line.split("|", 8)
        if len(head) < 9:
            lines.append(line)
            continue
        spk, _, note = head[8].partition("|")
        new = cell(pa.ids_in(head[1]))
        changed += new != spk
        lines.append("|".join(head[:8] + [new]) + "|" + note)
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--theta", type=float, default=THETA)
    parser.add_argument("--mode", choices=("words", "lead", "coverage"), default="words")
    parser.add_argument("--csv", action="store_true")
    arguments = parser.parse_args()
    out_dir = Path(arguments.out_dir)
    stem = out_dir.name
    stable = out_dir / f"{stem}-stable.json"
    backup = out_dir / f"{stem}-stable.pre-relabel.json"
    if not backup.exists():
        shutil.copyfile(stable, backup)
    payload = json.loads(backup.read_text(encoding="utf-8"))
    timeline = json.loads((out_dir / f"{stem}-diar.json").read_text(encoding="utf-8"))
    payload["segments"], stats = relabel_segments(payload["segments"], timeline, arguments.theta, arguments.mode)
    payload.setdefault("metadata", {})["speaker_relabel"] = {"mode": arguments.mode, "theta": arguments.theta,
                                                            "source": backup.name, **stats}
    stable.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    message = f"{stem} {arguments.mode} theta {arguments.theta}: {dict(stats)}"
    if arguments.csv:
        message += f"; annotated spk cells changed {rewrite_csv(out_dir, stem)}"
    print(message)


if __name__ == "__main__":
    main()
