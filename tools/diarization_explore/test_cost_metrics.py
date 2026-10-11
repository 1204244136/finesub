"""W1's contracts: what the naming cost must do under splitting, and what the two old rulers miss.

Run explicitly -- `tools/` is not in `testpaths`:

    python -m pytest -q -p xdist -n0 tools/diarization_explore/test_cost_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import accuracy_lab as lab  # noqa: E402
import cost_metrics as cm  # noqa: E402
import refined_multi as rm  # noqa: E402
from accuracy_identity import identity_score  # noqa: E402
from accuracy_report import MIN_FILES_FOR_INTERVAL, bootstrap_mean_ci  # noqa: E402

FRAME = lab.FRAME


def activity(spans, n_tracks, frames):
    """(frames, tracks) 0/1 map from (track, start_s, end_s) spans."""
    probs = np.zeros((frames, n_tracks), dtype=np.float32)
    for track, start, end in spans:
        probs[int(start / FRAME):int(np.ceil(end / FRAME)), track] = 1.0
    return probs


def rows(spans):
    return [{"id": str(i), "start": a, "end": b, "text": "x"} for i, (a, b) in enumerate(spans, 1)]


def reference(spans, people, frames):
    grid = np.zeros((frames, len(people)), dtype=bool)
    for person, start, end in spans:
        grid[int(start / FRAME):int(np.ceil(end / FRAME)), people.index(person)] = True
    return grid


# --- the naming cost under splitting -------------------------------------------------------

def test_splitting_a_track_never_lowers_the_two_counts_that_are_costs():
    """Plan closeout W1: a random split must not look like an improvement.

    `covering` and `leading` are the counts a person pays for -- how many tracks hold the speech,
    and how many ever reach the `spk` column. Cutting one track in half cannot reduce either, so
    an arm cannot win on this ruler by fragmenting.
    """
    frames = int(60 / FRAME) + 2
    stable = rows([(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)])
    grid = reference([("A", 0.0, 60.0)], ["A"], frames)

    whole = activity([(0, 0.0, 60.0)], 1, frames)
    split = activity([(0, 0.0, 30.0), (1, 30.0, 60.0)], 2, frames)
    four = activity([(0, 0.0, 15.0), (1, 15.0, 30.0), (2, 30.0, 45.0), (3, 45.0, 60.0)], 4, frames)

    costs = [cm.naming_cost(p, stable, grid) for p in (whole, split, four)]
    assert [c["covering"] for c in costs] == [1, 2, 4]
    assert [c["leading"] for c in costs] == [1, 2, 3], "three rows can lead at most three tracks"
    for finer, coarser in zip(costs[1:], costs):
        assert finer["covering"] >= coarser["covering"]
        assert finer["leading"] >= coarser["leading"]


def test_over_floor_is_descriptive_and_may_drop_when_a_track_is_split():
    """The exception, pinned so nobody later treats it as a cost.

    `over_floor` counts tracks holding at least 30 s. Cut a 40 s track in half and both halves fall
    under the floor, so the count goes 1 -> 0 -- fragmenting "improves" it. It says how many tracks
    are substantial, which is worth reporting; it is not a cost and must not be summed as one.
    """
    frames = int(60 / FRAME) + 2
    stable = rows([(0.0, 40.0)])
    grid = reference([("A", 0.0, 40.0)], ["A"], frames)
    whole = activity([(0, 0.0, 40.0)], 1, frames)
    halves = activity([(0, 0.0, 20.0), (1, 20.0, 40.0)], 2, frames)
    assert cm.naming_cost(whole, stable, grid)["over_floor"] == 1
    assert cm.naming_cost(halves, stable, grid)["over_floor"] == 0


def test_a_track_that_never_leads_a_row_costs_no_naming_decision():
    """Why raw track count is the wrong number (audit 2.2): extra tracks off the reference are free.

    The second track here sounds only where no subtitle row sits, so a proofreader never sees it in
    the `spk` column -- `tracks` counts it, `leading` does not.
    """
    frames = int(60 / FRAME) + 2
    stable = rows([(0.0, 10.0)])
    grid = reference([("A", 0.0, 10.0)], ["A"], frames)
    probs = activity([(0, 0.0, 10.0), (1, 30.0, 50.0)], 2, frames)
    cost = cm.naming_cost(probs, stable, grid)
    assert cost["tracks"] == 2 and cost["leading"] == 1


# --- what fixed misses and B-cubed catches -------------------------------------------------

def test_one_person_split_over_two_tracks_leaves_fixed_flat_and_drops_bcubed_recall():
    """Audit 2.1/2.2 made executable: the two rulers answer different questions.

    `fixed` names every track after its own best-matching reference person, so splitting one
    speaker across two tracks keeps every row "correct". B-cubed sees one person's rows land in two
    clusters and its recall falls. Anything that reports only `fixed` cannot see fragmentation.
    """
    frames = int(40 / FRAME) + 2
    stable = rows([(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)])
    people = ["A"]
    grid = reference([("A", 0.0, 40.0)], people, frames)

    whole = activity([(0, 0.0, 40.0)], 1, frames)
    split = activity([(0, 0.0, 20.0), (1, 20.0, 40.0)], 2, frames)

    assert lab.score(whole, stable, grid, people)["fixed"] == 1.0
    assert lab.score(split, stable, grid, people)["fixed"] == 1.0, "fixed is blind to the split"

    before, after = identity_score(whole, stable, grid), identity_score(split, stable, grid)
    assert before["recall"] == 1.0
    assert after["recall"] < before["recall"], "B-cubed recall is what notices"
    assert after["precision"] == before["precision"] == 1.0, "both partitions are pure"


# --- intervals ------------------------------------------------------------------------------

def test_no_interval_below_five_files():
    """Audit 2.9: n=2 gives three distinct resample means, n=3 gives ten. That is not a 95% interval."""
    assert MIN_FILES_FOR_INTERVAL == 5
    for n in range(0, MIN_FILES_FOR_INTERVAL):
        assert bootstrap_mean_ci([0.01] * n) is None, n
    interval = bootstrap_mean_ci([0.01, 0.02, 0.03, 0.04, 0.05])
    assert interval is not None and interval[0] <= interval[1]


def test_the_breakdown_sums_to_the_number_of_record():
    """`cost_terms` may not drift from `strong_cost`; the latter is what the documents quote.

    The decomposition re-walks the same minimisation, so an edit to either side that changes the
    arithmetic shows up here rather than as two published totals that disagree.

    ⚠ The fixture carries an UNLABELLED row, and both sides are asked the same question about it.
    Without one this test could not see `price_unlabelled` at all -- and the two functions now
    default to opposite answers, so a fixture that cannot express the difference would pin them
    together while they disagreed on every real file.
    """
    import numpy as np

    import refined_multi as rm

    # Four segments: one clean, one with a spurious second label, one with a missed speaker, and
    # one the arm declined to label at all although the reference covers it.
    grid = np.zeros((int(40 / lab.FRAME), 2), dtype=bool)
    grid[: int(10 / lab.FRAME), 0] = True
    grid[int(10 / lab.FRAME): int(20 / lab.FRAME), 1] = True
    grid[int(20 / lab.FRAME): int(30 / lab.FRAME), 0] = True
    grid[int(30 / lab.FRAME):, 1] = True
    people = ["A", "B"]
    owner = {0: "A", 1: "B"}
    starts = [(10.0, "B"), (20.0, "A"), (30.0, "B")]
    labelled = [
        {"start": 0.0, "end": 10.0, "speaker": {"track": 0, "extras": {"tracks": [0]}}},
        {"start": 10.0, "end": 20.0, "speaker": {"track": 1, "extras": {"tracks": [1, 0]}}},
        {"start": 20.0, "end": 30.0, "speaker": {"track": 1, "extras": {"tracks": [1]}}},
        {"start": 30.0, "end": 40.0},
    ]
    for priced in (False, True):
        total, merges = rm.strong_cost(labelled, grid, people, owner, starts, 2, merge=3,
                                       price_unlabelled=priced)
        terms = cm.cost_terms(labelled, grid, people, owner, starts, add=2, merge=3,
                              price_unlabelled=priced)
        assert terms["total"] == float(total), priced
        assert terms["merge"] == 3.0 * merges, priced


def test_a_row_nobody_labelled_is_not_free():
    """⚠ Deleting every label used to make the edit cost ZERO, so abstention was a discount.

    Harmless where this ruler came from -- `relabel_eval --strong` compares theta arms over one
    fixed segmentation -- and wrong the moment W2 used it to compare two backends whose coverage
    differs. The price is not a new rule: with no labels the same minimisation charges `add` per
    person the reference says is in the row.
    """
    import numpy as np

    grid = np.zeros((int(20 / lab.FRAME), 2), dtype=bool)
    grid[: int(10 / lab.FRAME), 0] = True
    grid[int(10 / lab.FRAME):, 1] = True
    people, starts = ["A", "B"], [(10.0, "B")]
    blank = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}]

    free, _ = rm.strong_cost(blank, grid, people, {}, starts, 2, merge=3)
    priced, _ = rm.strong_cost(blank, grid, people, {}, starts, 2, merge=3, price_unlabelled=True)
    assert free == 0, "the historical reading, kept for the theta arms it was written for"
    assert priced == 4, "two rows, one missed person each, at the owner's price of 2"


def _scene(seconds=120.0, people=("A", "B")):
    """Two speakers alternating, with a stretch where both are on."""
    frames = int(seconds / lab.FRAME)
    grid = np.zeros((frames, len(people)), dtype=bool)
    grid[: int(80 / lab.FRAME), 0] = True
    grid[int(60 / lab.FRAME):, 1] = True
    probs = np.zeros((frames, 2), dtype=np.float32)
    probs[: int(80 / lab.FRAME), 0] = 1.0
    probs[int(60 / lab.FRAME):, 1] = 1.0
    stable = [{"start": float(i), "end": float(i) + 1.0, "text": "x"} for i in range(0, int(seconds))]
    starts = [(60.0, "B"), (0.0, "A")]
    return probs, stable, grid, list(people), sorted(starts)


def test_the_top_one_arm_actually_removes_labels():
    """A silent no-op here reads as "second labels are free", which is a conclusion.

    The first run of the ablation produced a table of exact zeros because the stripping wrote a
    nested `extras` dict while `tracks_of` reads `speaker["tracks"]` at the top level -- `to_dict`
    flattens `extras` into the payload. Nothing failed; the answer was just wrong.
    """
    import relabel_eval as ev

    probs, stable, grid, people, starts = _scene()
    full = cm.labelled_segments(probs, stable)
    top1 = cm.labelled_segments(probs, stable, top_one=True)

    assert max(len(ev.tracks_of(row)) for row in top1) <= 1, "the top-1 arm still carries a second label"
    assert sum(len(ev.tracks_of(row)) for row in full) > sum(len(ev.tracks_of(row)) for row in top1), (
        "stripping removed nothing -- the ablation would report a difference of exactly zero")


def test_splitting_a_track_by_time_lowers_the_edit_cost():
    """Pin the direction of the ruler's known bias rather than trusting it is small.

    `owner_of_tracks` maps many tracks to one person, so cutting a track in two can only help it:
    each half gets its own best owner. That is why W2/W2b need a time-block null, and why a
    backend that emits more tracks cannot be credited on this ruler without one.
    """
    import refined_multi as rm

    probs, stable, grid, people, starts = _scene()
    owner = cm.owner_of_tracks(probs, grid, people)
    whole, _ = rm.strong_cost(cm.labelled_segments(probs, stable), grid, people, owner, starts, 2, merge=3)

    doubled = cm.time_block_split(probs)
    split_owner = cm.owner_of_tracks(doubled, grid, people)
    split, _ = rm.strong_cost(cm.labelled_segments(doubled, stable), grid, people, split_owner,
                              starts, 2, merge=3)
    assert split <= whole, (
        f"splitting by time raised the cost ({whole} -> {split}); the null arm is supposed to be "
        "free money, so a rise means the split or the mapping is not doing what it claims")


def test_the_crossfit_mapping_never_sees_the_rows_it_scores():
    """The rows a fold prices must be disjoint from the minutes its mapping was fitted on.

    The first version asserted only that the two folds' tracks covered the whole set -- which is
    equally true of the in-sample mapping, so it passed while `w2_bias` priced EVERY row under each
    fold's mapping and averaged the two, leaving half of each pass in-sample and calling the result
    a crossfit (fable 2026-09-19). Disjointness is the property that was missing, so it is the
    property this asserts.
    """
    import w2_bias

    probs, stable, grid, people, starts = _scene(seconds=240.0)
    first = cm.owner_of_tracks_folded(probs, grid, people, fold=0)
    second = cm.owner_of_tracks_folded(probs, grid, people, fold=1)
    assert set(first) | set(second) == set(cm.owner_of_tracks(probs, grid, people)), (
        "a track vanished from both folds")
    assert first and second, "one fold is empty -- the interleave is not interleaving"

    # ⚠ Assert through `cost_with` itself. Filtering the rows here and then checking they are in
    # the fold proves nothing about the function -- it would stay green if `cost_with` went back to
    # pricing every row, which is the bug this test exists for (fable 2026-09-19, and it is the
    # same "necessary condition instead of the property" mistake the previous version made).
    mapping = cm.owner_of_tracks(probs, grid, people)
    whole = w2_bias.cost_with(probs, stable, grid, people, starts, mapping)
    halves = [w2_bias.cost_with(probs, stable, grid, people, starts, mapping, fold=fold)
              for fold in (0, 1)]
    assert whole > 0, "the scene costs nothing, so it cannot tell the two behaviours apart"
    # `strong_cost` sums over segments, so a partition of the rows must sum to the whole. This is
    # the assertion that bites: were `fold` ignored, both calls would return `whole` and the sum
    # would be twice it. ⚠ Not asserting each half is non-zero -- which fold carries the cost is a
    # property of the scene, not of the folding, and a scene-shaped assertion is how the last two
    # versions of this test ended up proving nothing.
    assert sum(halves) == whole, (
        f"the folds do not partition the timeline ({halves} against {whole}); a sum of "
        f"{2 * whole} would mean `fold` is being ignored and half of each pass is in-sample")


def test_changing_only_held_out_truth_cannot_move_the_fitted_mapping():
    """The property the disjointness test above does NOT have, and the leak it missed.

    Folds were decided per segment, by its start, and then the WHOLE of `grid[lo:hi]` went into
    the mapping -- so a segment running 59-65 s was called training by its start and carried five
    seconds of held-out truth with it. Not a constructed corner: 36 to 39 segments cross a minute
    boundary on a real episode.

    ⚠ The assertion is the definition rather than a symptom. Change ONLY the held-out fold's truth
    and the mapping fitted on the other fold must not notice.
    """
    frames = 1502
    people = ["A", "B"]
    grid = reference([("A", 0, 60), ("B", 60, 120)], people, frames)
    boundary = activity([(0, 59, 65)], 1, frames)

    moved = grid.copy()
    moved[750:, 0], moved[750:, 1] = True, False   # 60 s onwards: held out when fold=1

    before = cm.owner_of_tracks_folded(boundary, grid, people, fold=1)
    after = cm.owner_of_tracks_folded(boundary, moved, people, fold=1)
    assert before == after == {0: "A"}, (
        f"the held-out truth reached the training mapping ({before} -> {after})")


# --- units -------------------------------------------------------------------------------------

def test_naming_reads_probs_on_the_grid_they_are_on():
    """A native 20 ms timeline read as 80 ms squeezes every track into a quarter of its time.

    Two tracks of one second each, one subtitle row apiece: two tracks lead. Read on the default
    80 ms grid, both collapse into the first half-second and one track leads both rows -- which is
    what `score()` reported for native timelines until it passed its step through (Codex audit
    2026-09-21). A3 called `leading_counts` directly and was not affected.
    """
    native = 0.02
    frames = int(2.0 / native)
    probs = np.zeros((frames, 2), dtype=np.float32)
    probs[: frames // 2, 0] = 1.0
    probs[frames // 2 :, 1] = 1.0
    stable = rows([(0.0, 1.0), (1.0, 2.0)])
    grid = reference([("A", 0.0, 1.0), ("B", 1.0, 2.0)], ["A", "B"], int(2.0 / FRAME) + 1)

    assert cm.naming_cost(probs, stable, grid, frame_seconds=native)["leading"] == 2
    assert lab.score(probs, stable, grid, ["A", "B"], frame_seconds=native)["naming"]["leading"] == 2
