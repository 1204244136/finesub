"""Scoring invariants independent of model implementation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from accuracy_overlap import score_intervals


def test_permutation_duplicate_union_and_overlap():
    ref = [(0., 2., "A"), (1., 3., "B"), (0., 1., "A")]
    hyp = [(0., 2., "Y"), (1., 3., "X")]
    s = score_intervals(ref, hyp, 3.)
    assert s["der"] == 0
    assert s["overlap_seconds"] == 1
    assert s["reference_speaker_seconds"] == 4
    assert s["overlap_all_speakers_recall"] == 1


def test_missed_overlapping_speaker_and_false_alarm():
    ref = [(0., 2., "A"), (1., 2., "B")]
    hyp = [(0., 3., "X")]
    s = score_intervals(ref, hyp, 3.)
    assert s["miss"] == pytest.approx(1/3)
    assert s["false_alarm"] == pytest.approx(1/3)
    assert s["confusion"] == 0
    assert s["overlap_recall"] == 0


def test_fragmentation_is_confusion_not_per_track_oracle_credit():
    s = score_intervals([(0., 4., "A")], [(0., 2., "X"), (2., 4., "Y")], 4.)
    assert s["confusion"] == .5
    assert s["der"] == .5


def test_empty_hypothesis_and_uem_clipping():
    s = score_intervals([(-1., 4., "A")], [], 2.)
    assert s["reference_speaker_seconds"] == 2
    assert s["der"] == 1


def test_cross_model_merge_cannot_link_and_unexplained_activity():
    import numpy as np
    from accuracy_consensus import consolidate
    a = np.zeros((400, 3))
    a[:200, 0] = a[200:, 1] = a[180:220, 2] = 1
    w = np.ones((400, 1))
    _, groups = consolidate(a, w, support=10, cannot_link=10)
    assert groups == [[0, 1], [2]]
    # Sparse evidence cannot approve a merge by ignoring unexplained activity.
    w[100:300] = 0
    _, groups = consolidate(a, w, support=10, cannot_link=10)
    assert groups == [[0], [1], [2]]


def test_continuous_der_matches_independent_pyannote_metric():
    import numpy as np
    metrics = pytest.importorskip("pyannote.metrics.diarization")
    from pyannote.core import Annotation, Segment, Timeline
    rng = np.random.default_rng(42)
    metric = metrics.DiarizationErrorRate(collar=0, skip_overlap=False)
    for _ in range(20):
        sets, annotations = [], []
        for speakers in (4, 5):
            rows = [(float(t), float(t+d), str(s)) for t, d, s in zip(
                rng.uniform(0, 8, 20), rng.uniform(.1, 2, 20), rng.integers(0, speakers, 20))]
            annotation = Annotation()
            for j, (a, b, speaker) in enumerate(rows):
                annotation[Segment(a, b), j] = speaker
            sets.append(rows)
            # Match our documented union of duplicate same-speaker intervals.
            annotations.append(annotation.support())
        actual = score_intervals(*sets, 10.)["der"]
        expected = metric(*annotations, uem=Timeline([Segment(0, 10)]))
        assert actual == pytest.approx(expected, abs=1e-10)


def test_union_alignment_preserves_unmatched_identity_and_uncertain_fallback():
    import numpy as np
    from accuracy_vote import align_union, arbitrate
    a = np.zeros((300, 2), dtype=np.float32)
    a[:100, 0] = a[100:200, 1] = 1
    b = np.pad(a[:, ::-1], ((0, 0), (0, 1)))
    b[200:, 2] = 1
    x, y = align_union(a, b)
    assert x.shape == y.shape == (300, 3)
    np.testing.assert_array_equal(x[:, :2], a)
    np.testing.assert_array_equal(y[200:, 2], np.ones(100))
    result = arbitrate(a, b, np.zeros((300, 4)))
    np.testing.assert_array_equal(result, x)


def test_diarizen_empty_activity_skips_clustering_and_restores_decoder():
    pytest.importorskip("pyannote.core")
    import numpy as np
    import torchaudio
    from types import SimpleNamespace
    from accuracy_diarizen import infer

    class Pipeline:
        def get_segmentations(self):
            return SimpleNamespace(data=np.zeros((2, 20, 4)))

        def __call__(self, *args, **kwargs):
            self.get_segmentations()
            raise AssertionError("empty speech must not enter clustering")

    pipeline = Pipeline()
    decoder = torchaudio.load
    original = pipeline.get_segmentations
    assert not infer(pipeline, np.zeros(320, dtype=np.float32))
    assert torchaudio.load is decoder
    assert pipeline.get_segmentations == original
