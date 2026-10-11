"""W5's contract: the overlap reading must not move when a track is split in two.

That is the whole point of deduplicating simultaneous tracks by owner. The raw `>=2 tracks` reading
does move, and this pins that too -- it is why the raw pair is a secondary number and not the one
the verdict rests on.

    python -m pytest -q -p xdist -n0 tools/diarization_explore/test_w5_overlap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import w5_overlap as w5  # noqa: E402

FRAME = w5.FRAME


def grid_of(spans, people, frames):
    grid = np.zeros((frames, people), dtype=bool)
    for person, start, end in spans:
        grid[int(start / FRAME):int(np.ceil(end / FRAME)), person] = True
    return grid


def activity_of(spans, tracks, frames):
    out = np.zeros((frames, tracks), dtype=bool)
    for track, start, end in spans:
        out[int(start / FRAME):int(np.ceil(end / FRAME)), track] = True
    return out


def scene(frames=int(200 / FRAME)):
    """A talks 0-100 s, B talks 60-140 s: 40 s of real overlap, the rest single."""
    return grid_of([(0, 0.0, 100.0), (1, 60.0, 140.0)], 2, frames)


def test_splitting_a_track_leaves_the_deduplicated_readings_unchanged():
    frames = int(200 / FRAME)
    grid = scene(frames)
    whole = activity_of([(0, 0.0, 100.0), (1, 60.0, 140.0)], 2, frames)
    # The same two people, but A is carried by two tracks that sound together throughout.
    split = activity_of([(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 60.0, 140.0)], 3, frames)

    before, after = w5.measure(whole, grid), w5.measure(split, grid)
    for key in ("detect_id", "detect_any", "false_id"):
        assert before[key] == after[key], (key, before[key], after[key])


def test_the_raw_reading_is_exactly_what_splitting_inflates():
    """Kept as a secondary number for one reason -- reproducing findings 6.1 -- not as evidence."""
    frames = int(200 / FRAME)
    grid = scene(frames)
    whole = activity_of([(0, 0.0, 100.0), (1, 60.0, 140.0)], 2, frames)
    split = activity_of([(0, 0.0, 100.0), (1, 0.0, 100.0), (2, 60.0, 140.0)], 3, frames)

    assert w5.measure(split, grid)["raw_false"] > w5.measure(whole, grid)["raw_false"], (
        "one person on two simultaneous tracks must inflate the undeduplicated reading")


def test_two_tracks_owned_by_the_wrong_people_are_detected_but_not_credited():
    """`detect_id` asks whether the pair is the pair that is actually speaking; `detect_any` does not."""
    frames = int(200 / FRAME)
    # A 0-70, B 60-70 (the only overlap), C 100-200.
    grid = grid_of([(0, 0.0, 70.0), (1, 60.0, 70.0), (2, 100.0, 200.0)], 3, frames)
    # Two tracks: A's, and one that is C's by majority (100 s of C) but leaks into the overlap.
    active = activity_of([(0, 0.0, 70.0), (1, 60.0, 70.0), (1, 100.0, 200.0)], 2, frames)
    m = w5.measure(active, grid)
    assert m["detect_any"] > 0, "two distinct owners are active during the overlap"
    assert m["detect_id"] == 0, "but the second one is C, who is not speaking there"


def test_a_file_without_overlap_reports_nan_rather_than_zero():
    """A missing denominator must not read as a perfect miss and drag a mean down."""
    frames = int(200 / FRAME)
    grid = grid_of([(0, 0.0, 100.0)], 1, frames)
    m = w5.measure(activity_of([(0, 0.0, 100.0)], 1, frames), grid)
    assert np.isnan(m["detect_id"]) and np.isnan(m["detect_any"])


def test_label_change_does_not_move_when_track_numbers_are_permuted():
    """W4's headline quantity must measure re-drawing, not renumbering.

    `label_change` aligns the two timelines before comparing, so relabelling track 0 as track 3
    has to read as zero change. If it did not, every W4 number would be part permutation -- and a
    permutation is exactly the shape a re-drawn attractor takes, so the error would look like the
    effect being measured (`fixed` is blind to it for the same reason; audit 2.1).
    """
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import w4_segmented as w4

    frames = int(120 / w4.lab.FRAME)
    probs = np.zeros((frames, 3), dtype=np.float32)
    probs[: frames // 3, 0] = 1.0
    probs[frames // 3: 2 * frames // 3, 1] = 1.0
    probs[2 * frames // 3:, 2] = 1.0
    grid = np.zeros(frames, dtype=bool)
    grid[:] = True
    stable = [{"start": float(i), "end": float(i) + 1.0, "text": "x"} for i in range(120)]

    shuffled = probs[:, [2, 0, 1]]
    assert w4.label_change(probs, shuffled, stable, grid) == 0.0, (
        "renumbering the tracks changed the reading; W4 would then be measuring permutations")
    assert w4.label_change(probs, probs, stable, grid) == 0.0
