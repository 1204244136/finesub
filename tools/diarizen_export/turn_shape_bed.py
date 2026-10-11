"""B1 step 0: what a "row holds a handover" row actually LOOKS like, before any detector.

Owner 2026-09-20: 「先看清 b 的形状，想清楚尺子，因为 b 的 ASR 本身就可能带错误，
比如只有一人，或者第二人不全。」

That is a truth-side objection and it lands. Every S4 reading so far -- 7.5% recall, 49.4%
precision, "the timeline says overlap and the truth says handover" -- was scored against
`turns_bed.truth_turn`, which reads the refined subtitle **rasterised onto an 80 ms grid** and
asks whether the leading person changes inside the row's time span. The text is thrown away at
the raster step, so that truth cannot tell these three apart:

  1. two people each say a sentence inside the row  -> genuinely two rows; a cut exists in the TEXT
  2. the second person says 「うん」 or laughs       -> the grid changes leader, and there is
                                                        NOTHING in the row's text to cut
  3. the second person spoke and the ASR never wrote it down -> same as 2 for our purposes, and
                                                        splitting would invent a boundary

⚠ **S4 delivers a cut in the text, so the ruler has to be about the text.** A row is splittable
only if its own transcript holds words from two people. Whether two people made noise during its
time span is a different question, and it is the one every previous reading answered.

This bed answers the first question instead. It classifies every stable row against the refined
subtitle WITH its text kept, and cross-tabulates that against the grid truth the old readings
used, so the size of the disagreement is a number rather than a worry.

⚠ **It builds no detector and proposes no cut.** Deciding the ruler comes first; the owner asked
for the shape, and a detector scored against a ruler nobody has inspected is how this line already
spent one pass.

    python tools/diarizen_export/turn_shape_bed.py --episodes 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = REPO / "tmp" / "diarizen-bed"

#: A refined line with fewer characters than this is a backchannel.
#:
#: ⚠⚠ **This was justified against the wrong language and the readings are very sensitive to it.**
#: The comment used to say 「うん」「はい」「へえ」 are two to three characters -- true of Japanese,
#: but onetonton's refined axis is a Chinese FANSUB (1153 of 1154 dialogue lines carry no kana),
#: where the same backchannels are 「嗯」「对」, one character. So 4 demotes genuinely meaningful
#: short Chinese utterances to `backchannel`, and the count moves a long way with it:
#:
#:     threshold   4      3      2      1
#:     splittable  1428   2121   2872   3377   (of 13215 rows)
#:
#: 2 is the language-appropriate analogue of "a token, not an utterance". ⚠ It is NOT simply
#: switched: every reading published from this bed used 4, and re-picking a constant after seeing
#: results is the drift `eval_contract` exists to stop. Both are reported side by side wherever
#: it matters (`speaker-identity-plan.md` C.5b/C.5d), and the C.5d verdict keeps its sign under
#: either -- what changes is the level, not the direction. Settle it before quoting a number.
#:
#: ⚠ Note the characters are PRORATED (`len(text) * overlap / duration`), so this is not a clean
#: character count in either language; seconds decide the class, this only demotes.
BACKCHANNEL_CHARS = 4

#: And this much of the row's credited characters, below which a second person is a fragment
#: however many characters it is -- a long row can hold four characters of somebody else and
#: still be one person's line for delivery purposes.
MINOR_SHARE = 0.15

#: Speech this short inside the row is a noise, not a turn: no words to move into a second row.
#: ⚠ **Seconds, not characters, decide the class.** Two character rules were tried first and both
#: broke. Prorating a line's characters by time overlap credited `0.1c` to a neighbour clipping
#: the row's edge; crediting whole lines instead let one LONG refined line covering a SHORT row
#: hand that row all of its characters, which is what made `splittable` rows look as though the
#: ASR had transcribed half of them. Overlap seconds are exact and assume nothing about speech
#: rate, so they decide; characters stay, prorated, as a DIAGNOSTIC where a ratio is all that is
#: being read.
MIN_TURN_SECONDS = 0.4


def refined_events(episode: int):
    """`(start, end, person, text)` for every refined dialogue line of one episode."""
    import refined_ass as ra

    path = ra.episodes()[episode - 1]
    return [(a, b, w, t) for a, b, w, t in ra.events(path) if w != ra.NOTE and t]


def speech_inside(row_start: float, row_end: float, events) -> tuple[dict, dict]:
    """`(seconds per person, estimated characters per person)` inside this row's span.

    Seconds are exact overlap and decide everything. Characters are `len(text) * overlap /
    duration` and are read only as a ratio against what the ASR wrote, never to classify a row --
    see `MIN_TURN_SECONDS` for the two ways a character rule got this wrong.
    """
    seconds: dict[str, float] = {}
    chars: dict[str, float] = {}
    for start, end, who, text in events:
        overlap = min(row_end, end) - max(row_start, start)
        if overlap <= 0:
            continue
        seconds[who] = seconds.get(who, 0.0) + overlap
        chars[who] = chars.get(who, 0.0) + len(text) * min(1.0, overlap / max(end - start, 1e-6))
    return seconds, chars


def grid_says_handover(row, grid, people) -> bool:
    """The OLD truth, reproduced exactly: does the sole-active person change inside the row?"""
    from finesub.speech.diarization.segments import FRAME_SECONDS
    from turns_bed import truth_turn

    low = int(float(row["start"]) / FRAME_SECONDS)
    high = max(low + 1, int(np.ceil(float(row["end"]) / FRAME_SECONDS)))
    return truth_turn(grid[low:high], people) is not None


def classify(row, events) -> tuple[str, dict, set]:
    """Which shape this row has, judged on TEXT rather than on who made noise when.

    * `splittable`  -- two people, both with a real turn inside the row: a cut exists
    * `backchannel` -- a second person held the floor long enough but said a token of text
    * `edge`        -- a second person is audible only at the row's rim (a clipped neighbour)
    * `single`      -- one person, full stop
    * `empty`       -- no refined line touches this row at all
    """
    seconds, chars = speech_inside(float(row["start"]), float(row["end"]), events)
    if not seconds:
        return "empty", chars, seconds
    ranked = sorted(seconds.items(), key=lambda item: -item[1])
    if len(ranked) == 1:
        return "single", chars, seconds
    total = sum(seconds.values())
    runner_seconds = ranked[1][1]
    runner_chars = chars.get(ranked[1][0], 0.0)
    if runner_seconds < MIN_TURN_SECONDS or runner_seconds / total < MINOR_SHARE:
        return "edge", chars, seconds
    if runner_chars < BACKCHANNEL_CHARS:
        return "backchannel", chars, seconds
    return "splittable", chars, seconds


def episode_rows(episode: int):
    """`(stable rows, grid, people, refined events)` for one episode, or None."""
    import relabel_eval as ev

    data = ev.episode(episode)
    if data is None:
        return None
    stable, _, grid, people, _ = data
    return stable, grid, people, refined_events(episode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(1, 16)))
    parser.add_argument("--examples", type=int, default=6,
                        help="print this many rows where the two rulers disagree")
    arguments = parser.parse_args()

    table = Counter()
    edges = Counter()
    coverage: dict[str, list] = {}
    disagreements = []
    for episode in arguments.episodes:
        loaded = episode_rows(episode)
        if loaded is None:
            continue
        stable, grid, people, events = loaded
        for row in stable:
            shape, chars, seconds = classify(row, events)
            grid_turn = grid_says_handover(row, grid, people)
            table[(shape, grid_turn)] += 1
            if len(seconds) > 1:
                edges[(shape, grid_turn)] += 1
            # ⚠ The owner's other case: the ASR may simply not have written the second person
            # down. Comparing the row's own transcript against the refined text credited to it
            # is what would show that -- a row far shorter than its truth is missing speech.
            written = len(str(row.get("text", "")))
            if chars:
                coverage.setdefault(shape, []).append(written / max(sum(chars.values()), 1))
            if grid_turn and shape != "splittable":
                disagreements.append((episode, shape, row, chars, seconds))

    total = sum(table.values()) or 1
    print(f"{total} stable rows over {len(arguments.episodes)} episode(s)\n")
    print(f"{'text shape':<12} {'grid: handover':>15} {'grid: no':>10} {'total':>8} {'share':>7}"
          f"  {'2+ people audible at all':>25}")
    for shape in ("single", "splittable", "backchannel", "edge", "empty"):
        yes, no = table[(shape, True)], table[(shape, False)]
        clipped = edges[(shape, True)] + edges[(shape, False)]
        print(f"{shape:<12} {yes:>15} {no:>10} {yes + no:>8} {(yes + no) / total:>6.1%}"
              f"  {clipped:>25}")

    turns = sum(table[(s, True)] for s in
                ("single", "splittable", "backchannel", "edge", "empty"))
    real = table[("splittable", True)]
    print(f"\nrows the GRID truth calls a handover: {turns} ({turns / total:.1%} of all rows)")
    if turns:
        print(f"  ⭐ of those, the TEXT holds two people's words: {real} ({real / turns:.1%})")
        for shape in ("backchannel", "edge", "single", "empty"):
            count = table[(shape, True)]
            print(f"     {shape:<12} {count:>5} ({count / turns:>5.1%}) -- nothing to cut")
    splittable = sum(table[("splittable", g)] for g in (True, False))
    print(f"\nrows the TEXT calls splittable: {splittable} ({splittable / total:.1%} of all rows)"
          f"  ⭐ of which the grid truth misses {table[('splittable', False)]}"
          f" ({table[('splittable', False)] / max(splittable, 1):.1%})")

    print("\nhow much of the row's truth the ASR actually wrote down (written chars / truth chars):")
    for shape, values in sorted(coverage.items()):
        array = np.asarray(values)
        print(f"  {shape:<12} median {np.median(array):.2f}   under 0.6: "
              f"{float((array < 0.6).mean()):.1%}   over 1.4: {float((array > 1.4).mean()):.1%}")

    if disagreements:
        print(f"\nfirst {arguments.examples} rows where the grid says handover and the text does not:")
        for episode, shape, row, chars, seconds in disagreements[: arguments.examples]:
            summary = ", ".join(f"{who} {seconds[who]:.2f}s/{chars.get(who, 0.0):.1f}c"
                                for who in sorted(seconds, key=lambda name: -seconds[name]))
            print(f"  ep{episode:02d} {float(row['start']):7.2f}-{float(row['end']):7.2f} [{shape}] "
                  f"{summary}\n      {str(row.get('text', ''))[:70]}")

    OUT.mkdir(parents=True, exist_ok=True)
    record = {"episodes": arguments.episodes, "rows": total,
              "backchannel_chars": BACKCHANNEL_CHARS, "minor_share": MINOR_SHARE,
              "min_turn_seconds": MIN_TURN_SECONDS,
              "table": {f"{shape}|{'turn' if turn else 'no'}": count
                        for (shape, turn), count in sorted(table.items())},
              "asr_coverage_median": {shape: float(np.median(values))
                                      for shape, values in sorted(coverage.items())}}
    (OUT / "turn-shape.json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
