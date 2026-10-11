"""Turn detection: the two things it must never do, and the one it must.

⚠ The detector is measured at 7.5% recall (see the module docstring), so these do not claim it
works well. They pin the properties that make a low-recall detector *safe* -- splitting overlap
would destroy the multi-label answer, and a cut inside a word is worse than the row it replaces.

⚠ **Not collected by the default suite.** These moved to `tools/` with the module they guard
(2026-09-20) and `testpaths` does not include `tools/`, so they run only when someone runs them
here. That is the standing cost of a tools module; `verify_guards.py` beside them still puts each
bug back and checks the guard fires.
"""

from __future__ import annotations

import numpy as np

from turns import find_turn


def _row(start, end, word_edges):
    return {
        "start": start,
        "end": end,
        "words": [{"start": start, "end": edge} for edge in word_edges],
    }


def _activity(frames, spans):
    """`(frames, tracks)` with each `(track, lo, hi)` span switched on."""
    data = np.zeros((frames, 4), dtype=np.int8)
    for track, low, high in spans:
        data[low:high, track] = 1
    return data


def test_a_clean_handover_is_found_and_cut_on_a_word_boundary():
    # 0-2 s track 1, 2-4 s track 2, at 20 ms per frame.
    activity = _activity(300, [(1, 0, 100), (2, 100, 200)])
    row = _row(0.0, 4.0, [0.5, 1.4, 2.3, 3.1])

    turn = find_turn(row, activity, step=0.02)
    assert turn is not None
    assert (turn.before, turn.after) == (1, 2)
    assert turn.at in {0.5, 1.4, 2.3, 3.1}, "the cut is a word boundary, never mid-word"
    assert abs(turn.at - 2.0) <= 1.0


def test_simultaneous_speech_is_never_called_a_handover():
    """⚠ Splitting overlap destroys the multi-label answer, which is this line's whole advantage.

    ⚠ The row has to hold BOTH exclusive frames and heavy simultaneity, or the simultaneity rule
    never runs. An earlier version used two tracks on for the whole row: with no exclusive frame
    anywhere, the function returned None several checks earlier and the test passed without ever
    reaching the guard it names. Injecting the bug is what showed that.
    """
    # Track 1 alone, then both together at length, then track 2 alone: exclusive frames at each
    # end so the function gets as far as the simultaneity test, and 60% of the row overlapping.
    activity = _activity(500, [(1, 0, 40), (1, 40, 160), (2, 40, 160), (2, 160, 200)])
    row = _row(0.0, 4.0, [1.0, 2.0, 3.0])

    assert find_turn(row, activity, step=0.02) is None

    # And with the same shape but little overlap, it IS a handover -- otherwise the test above
    # would pass just as well if the function always said None.
    sparse = _activity(500, [(1, 0, 90), (1, 90, 110), (2, 90, 110), (2, 110, 200)])
    assert find_turn(row, sparse, step=0.02) is not None


def test_a_flicker_is_not_a_handover():
    """One frame of another track is activity noise, not somebody taking the floor."""
    activity = _activity(200, [(1, 0, 200), (2, 100, 103)])
    row = _row(0.0, 4.0, [1.0, 2.0, 3.0])

    assert find_turn(row, activity, step=0.02) is None


def test_repeated_alternation_is_not_one_cut():
    """Two people trading the floor several times cannot be fixed by a single split."""
    activity = _activity(400, [(1, 0, 50), (2, 50, 100), (1, 100, 150), (2, 150, 200)])
    row = _row(0.0, 8.0, [2.0, 4.0, 6.0])

    assert find_turn(row, activity, step=0.02) is None


def test_a_row_with_no_interior_word_boundary_is_left_alone():
    """Cutting at the row's own edge produces an empty half."""
    activity = _activity(300, [(1, 0, 100), (2, 100, 200)])
    row = _row(0.0, 4.0, [4.0])          # the only boundary is the row's end

    assert find_turn(row, activity, step=0.02) is None


def test_the_step_decides_which_frames_the_row_covers():
    """⚠ The same array read at the wrong rate covers a different span of it.

    A 0-4 s row over a 20 ms array is frames 0-200, which contains the handover at frame 100. Read
    as if the array were 80 ms, the same row is frames 0-50 -- entirely inside the first speaker,
    so there is nothing to find. That is why `step` is passed rather than assumed, and why the
    native grid has to come from the sidecar that recorded its own rate.
    """
    activity = _activity(300, [(1, 0, 100), (2, 100, 200)])
    row = _row(0.0, 4.0, [2.0])

    assert find_turn(row, activity, step=0.02) is not None, "the handover is inside frames 0-200"
    assert find_turn(row, activity, step=0.08) is None, "and outside frames 0-50"
