"""Rows that contain a speaker turn, and where the cut would go.

Measured on 25 files: of the subtitle rows whose speaker attribution is wrong, about three
quarters span two people. Asking "which speaker is this line" is the wrong question for those --
the answer is that the line holds a turn and should be two lines. That is the single largest
category of speaker error, and nothing looks for it today.

⚠ **Spanning two people is NOT the same as holding a turn.** Only about 39% of mixed rows do; the
rest are **overlap**, two voices at once, which must stay multi-label and must never be split --
that capability is the one thing this line has over clustering. So the rule here separates the two
before it proposes anything.

⚠⚠ **NOT WIRED INTO THE PIPELINE.** Nothing in production calls this; only
`tools/diarizen_export/turns_bed.py` and its tests do. That is deliberate and follows from the
measurement below -- at one false positive per true one, putting these in front of a proofreader
would add more noise than it removes. It lives here rather than under `tools/` because it is
production-shaped code that becomes useful the moment the timeline improves, and because the
measurement beside it is the baseline anything better has to beat.

⚠ **Reports, never mutates.** The single-track SRT is a finished product (`CLAUDE.md`), so this
records a cut point for the speaker layer and for a proofreader. Whether a row may actually be
split is a delivery-shape decision that is not ours to make here.

⚠ Reads the NATIVE 20 ms activity from the evidence sidecar, not the 80 ms product grid: words run
100-300 ms, and the product grid is reduced with a max that widens activity outward -- too coarse
to place a cut.

⚠⚠ **Measured, and it is not good enough to act on** (2026-09-20, 5 onetonton episodes, 575 rows
the truth calls a handover): 7.5% recall at 49.4% precision -- one false positive for every true
one. It is *safe* (1.0% of overlap rows and 1.1% of plain rows wrongly called), just nearly blind.

**Where the ceiling is, corrected.** The first reading said the timeline "never saw the second
person" on 52.5% of handover rows. ⚠ That was wrong, and the rule was to blame for the wording:
`_leading_per_frame` marks any multi-active frame `-1`, so a speaker who appears ONLY in overlap
is invisible to it. Of those rows, **75.2% do hold a second track** -- 73.5% of them overlapping.
Only **13.0% of handover rows** genuinely have one track.

⚠ But a rule that keeps the overlap frames does not do better either: counting them for both
tracks and asking which one leads each half gives **5.4% recall at 47.0% precision**, against the
exclusive rule's 7.5% and 49.4%.

⇒ The accurate statement is the third one: **the timeline sees the second speaker and calls the
row OVERLAP where the truth calls it a HANDOVER.** The two disagree about what kind of row it is,
not about who is in it. And a detector that must never split overlap -- which is the one capability
this line has over clustering -- cannot act on those rows whichever way it reads them. Whether the
model smears a fast exchange, or the 80 ms truth axis is too coarse to show real overlap, is not
settled here. Kept because the code is correct and the measurement is what says it is
under-powered.

⚠ **This module is not wired into anything, and it lives in `tools/` for that reason**
(2026-09-20). It was written against the production seam and reads like production code, but
nothing calls it and the measurement below says nothing should. A never-called module inside
`src/` is the kind of thing the 0.6.0 close-out spent a pass deleting, so it sits beside
`turns_bed.py`, which is what measures it. Its tests moved with it and are therefore no longer
collected by the default suite. If the timeline ever starts distinguishing a fast exchange from
an overlap, this is the code to bring back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: A turn needs both speakers to hold the floor for at least this long, or it is a stutter in the
#: activity rather than someone taking over.
MIN_SIDE_SECONDS = 0.35

#: If the two tracks are simultaneously active for more than this share of the row, it is overlap.
#: ⚠ Overlap is a capability, not a defect: splitting it would destroy the multi-label answer.
MAX_SIMULTANEOUS = 0.35


@dataclass(frozen=True)
class Turn:
    """A row that holds a handover, and the word boundary to cut it at."""

    before: int
    after: int
    #: Time of the cut, snapped to the nearest word boundary inside the row.
    at: float
    #: How much of the row the two tracks were active at the same time.
    simultaneous: float


def _leading_per_frame(activity, low: int, high: int):
    """The single active track per frame, `-1` where none or several are."""
    import numpy as np

    window = np.asarray(activity[low:high]) > 0
    counts = window.sum(axis=1)
    leading = np.full(len(window), -1, dtype=np.int16)
    single = counts == 1
    if single.any():
        leading[single] = np.argmax(window[single], axis=1)
    return leading, counts


def find_turn(
    row: Mapping[str, Any],
    activity,
    *,
    step: float,
    min_side: float = MIN_SIDE_SECONDS,
    max_simultaneous: float = MAX_SIMULTANEOUS,
) -> "Turn | None":
    """The handover inside one subtitle row, or None.

    ⚠ The cut lands on a **word boundary**, never mid-word: a cut inside a word produces two rows
    that each hold half of it, which is worse than the row it replaced.
    """
    import numpy as np

    start, end = float(row.get("start", 0.0)), float(row.get("end", 0.0))
    if end <= start:
        return None
    low, high = int(start / step), int(np.ceil(end / step))
    high = min(high, len(activity))
    if high - low < 2:
        return None

    leading, counts = _leading_per_frame(activity, low, high)
    simultaneous = float((counts >= 2).mean())

    named = leading[leading >= 0]
    if len(named) < 2:
        return None
    first, last = int(named[0]), int(named[-1])
    if first == last:
        return None

    # Both sides must actually hold the floor, and not merely flicker.
    before = float((leading == first).sum()) * step
    after = float((leading == last).sum()) * step
    if before < min_side or after < min_side:
        return None
    if simultaneous > max_simultaneous:
        return None  # overlap, not a handover

    # The handover is where the last `first` frame gives way to the first `last` frame.
    last_of_first = np.flatnonzero(leading == first)[-1]
    first_of_last = np.flatnonzero(leading == last)[0]
    if first_of_last <= last_of_first:
        return None  # interleaved: the two take turns repeatedly, which is not one cut
    boundary = start + 0.5 * (last_of_first + first_of_last + 1) * step

    words = row.get("words") or []
    cut = _snap_to_word(boundary, words, start, end)
    if cut is None:
        return None
    return Turn(before=first, after=last, at=round(cut, 3), simultaneous=round(simultaneous, 4))


def _snap_to_word(boundary: float, words: Sequence[Mapping[str, Any]],
                  start: float, end: float) -> "float | None":
    """The word boundary nearest `boundary`, or None when there is nowhere sensible to cut."""
    edges = []
    for word in words:
        try:
            edges.append(float(word["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    # ⚠ Only interior boundaries: cutting at the row's own start or end produces an empty half.
    edges = [edge for edge in edges if start < edge < end]
    if not edges:
        return None
    return min(edges, key=lambda edge: abs(edge - boundary))


def find_turns(rows: Sequence[Mapping[str, Any]], activity, *, step: float) -> dict[int, Turn]:
    """`{row index: Turn}` for every row that holds one."""
    found = {}
    for index, row in enumerate(rows):
        turn = find_turn(row, activity, step=step)
        if turn is not None:
            found[index] = turn
    return found
