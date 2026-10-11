"""Identity assignment contracts; no model imports or GPU work."""
import itertools

import numpy as np
import pytest

from accuracy_dz3 import active_assignment, overlap_assignment


def test_reuse_identity_for_disjoint_locals_but_preserve_overlap():
    scores = np.array([[[.9, .2], [.8, .1], [.7, .6], [np.nan, np.nan]]])
    activity = np.zeros((1, 10, 4))
    activity[0, :5, 0] = 1
    activity[0, 5:, 1] = 1
    activity[0, :, 2] = 1
    assert overlap_assignment(scores, activity)[0, :3].tolist() == [0, 0, 1]


def test_inactive_slot_cannot_steal_only_global_identity():
    scores = np.array([[[.1], [.9]]])
    activity = np.array([[[1., 0.]]])
    assert active_assignment(scores, activity).tolist() == [[0, -2]]


def test_assignment_is_optimal_under_coactivity_constraints():
    rng = np.random.default_rng(90217)
    for _ in range(12):
        scores = rng.random((1, 4, 5))
        activity = rng.integers(0, 2, size=(1, 8, 4)).astype(float)
        conflict = activity[0].T @ activity[0] > 0
        pairs = [(i, j) for i in range(4) for j in range(i+1, 4) if conflict[i, j]]
        actual = overlap_assignment(scores, activity)[0]
        assert all(actual[i] != actual[j] for i, j in pairs)
        optimum = max(sum(scores[0, i, c[i]] for i in range(4))
                      for c in itertools.product(range(5), repeat=4)
                      if all(c[i] != c[j] for i, j in pairs))
        assert scores[0, np.arange(4), actual].sum() == pytest.approx(optimum)


def test_insufficient_clusters_does_not_invent_an_identity():
    scores = np.ones((1, 4, 2))
    result = overlap_assignment(scores, np.ones((1, 5, 4)))
    assert result.shape == (1, 4)
    assert np.all((result >= 0) & (result < 2))


def test_four_speaker_overlap_requires_all_four_correct_identities():
    from accuracy_overlap import score_intervals
    reference = [(0., 2., str(i)) for i in range(4)]
    hypothesis = [(0., 2., str(i)) for i in range(3)] + [(0., 1., "3")]
    score = score_intervals(reference, hypothesis, 2.)
    assert score["four_plus_seconds"] == 2
    assert score["four_plus_all_recall"] == .5
    assert score["der"] == .125


def test_rare_speaker_cannot_reuse_a_majority_speakers_identity():
    from accuracy_dz3_diagnose import speaker_recall
    rows = speaker_recall([(0., 3., "A"), (3., 4., "B")], [(0., 4., "X")], 4.)
    assert rows[0]["recall"] == 1.
    assert rows[1]["matched_hypothesis"] is None
    assert rows[1]["recall"] == 0.
    assert rows[1]["best_temporal_overlap_track"] == "X"
    assert rows[1]["best_temporal_overlap_recall"] == 1.
