"""Where do subtitle rows that hold two speakers come from (owner 2026-10-07): Whisper's own segment, or the DP splitter
merging two ASR segments?

Production's rows (`-stable.json`) come from the global DP splitter (docs/segmentation-split.md), which may cut inside
an ASR (Whisper) segment and may merge neighbouring ones; the first word of every ASR segment keeps
`whisper_segment_start`. A row is mixed when the refined reference has lines of two or more people over it, each
overlapping it by at least a share of the row and a number of seconds (LOOSE, STRICT). For each mixed row:

  overlap     the two people's lines overlap each other by half the shorter one or more, in one of two shapes:
              inside  the second person's line lies within a line of the first (an interjection, talk at once);
              tail    the first person's line runs on into the second's -- the reference's line ends run late
                      (findings 11.3), so part of these may be one-after-the-other with a padded end
  edge        one after the other, but the change lies outside the row or within NEAR seconds of its edge: most likely
              the splitter cut in the right place and the reference's timing is off; not counted as either below
  dp_merge    the change is inside the row, and an ASR segment seam lies within NEAR seconds of it: Whisper had cut
              there, the splitter merged across it
  in_segment  the change is inside the row, no seam near it: one Whisper segment spans it; reported with whether the
              words near the change leave a gap of GAP seconds or more (a place the splitter could have cut) -- rows with
              no word boundary near the change are counted apart

The change time is the start of the first line, inside the row, of a person other than the row's first speaker. The
reference's own timing is approximate (about NEAR). Run at two thresholds for "two people over the row" (LOOSE, STRICT).

    python tools/diarization_explore/mixed_row_origin.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf  # noqa: E402
import p2_review_video as pv  # noqa: E402

LOOSE, STRICT = (0.15, 0.3), (0.3, 0.5)  # (share of the row, seconds) each person's lines must cover
NEAR, GAP = 0.5, 0.2


def classify(row, lines, share, seconds):
    s, e = float(row["start"]), float(row["end"])
    over = [(a, b, who) for a, b, who, _ in lines if min(b, e) - max(a, s) >= max(seconds, share * (e - s))]
    people = {who for _, _, who in over}
    if len(people) < 2:
        return None
    over.sort()
    first = over[0][2]
    second = next((a, b, who) for a, b, who in over if who != first)
    firsts = [(a, b) for a, b, who in over if who == first]
    shorter = min(max(b - a for a, b in firsts), second[1] - second[0])
    together = max(min(b, second[1]) - max(a, second[0]) for a, b in firsts)
    if together >= 0.5 * shorter:
        inside = any(a <= second[0] and second[1] <= b for a, b in firsts)
        return "overlap", "inside" if inside else "tail"
    change = second[0]
    if not s + NEAR < change < e - NEAR:
        return "edge", None
    words = row.get("words") or []
    seams, gaps = [], []
    for k in range(len(words) - 1):
        at = (float(words[k]["end"]) + float(words[k + 1]["start"])) / 2
        if abs(at - change) > NEAR:
            continue
        if words[k + 1].get("whisper_segment_start"):
            seams.append(at)
        gaps.append(float(words[k + 1]["start"]) - float(words[k]["end"]))
    if seams:
        return "dp_merge", None
    return "in_segment", (None if not gaps else max(gaps) >= GAP)


def main():
    rows, _ = pf.load_rows("covered")
    files = sorted(rows, key=lambda i: i.startswith("onetonton"))  # ⚠ hina first, as P2 requires
    total, merged, cut = 0, 0, 0
    kinds = {name: Counter() for name in ("loose", "strict")}
    gapped = {name: Counter() for name in ("loose", "strict")}
    for ident in files:
        stable = pf.reference(ident)[0]
        lines = pv.reference_lines(ident)
        for row in stable:
            words = row.get("words") or []
            if not str(row.get("text", "")).strip() or not words:
                continue
            total += 1
            merged += any(w.get("whisper_segment_start") for w in words[1:])
            cut += not words[0].get("whisper_segment_start")
            for name, (share, seconds) in (("loose", LOOSE), ("strict", STRICT)):
                got = classify(row, lines, share, seconds)
                if got is not None:
                    kinds[name][got[0]] += 1
                    if got[0] == "overlap":
                        kinds[name][f"overlap_{got[1]}"] += 1
                    if got[0] == "in_segment":
                        gapped[name][got[1]] += 1
    print(f"{len(files)} files, {total} rows; rows holding an ASR seam (the splitter merged two ASR segments) {merged / total:.1%}, "
          f"rows starting inside an ASR segment (the splitter cut one) {cut / total:.1%}")
    for name, (share, seconds) in (("loose", LOOSE), ("strict", STRICT)):
        k = kinds[name]
        mixed = sum(v for key, v in k.items() if not key.startswith("overlap_"))
        inside = k["dp_merge"] + k["in_segment"]
        g = gapped[name]
        print(f"{name} (each person >= {share:.0%} of the row and {seconds} s): two people over {mixed} rows ({mixed / total:.1%}); "
              f"overlap {k['overlap']} ({k['overlap'] / mixed:.0%}: inside {k['overlap_inside']}, tail {k['overlap_tail']}), edge {k['edge']}, change inside the row {inside}: "
              f"dp_merge {k['dp_merge']} ({k['dp_merge'] / max(1, inside):.1%}), in_segment {k['in_segment']}; of these, "
              f"no gap >= {GAP} s near the change {g[False]} / {g[False] + g[True]} ({g[False] / max(1, g[False] + g[True]):.0%}), "
              f"no word boundary near it {g[None]}")


if __name__ == "__main__":
    main()
