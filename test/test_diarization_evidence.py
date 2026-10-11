"""The evidence sidecar: what it holds, where it lives, and how it fails.

⚠ Every consumer of this file is an extra -- a hint, a veto's input, a proofreading mark -- so a
missing or corrupt sidecar must cost that extra and never the run. These pin that, and the
asymmetry that Sortformer produces no evidence at all.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from finesub.speech.diarization import stage as diarization
from finesub.speech.diarization.evidence import (
    SUFFIX,
    ClusterEvidence,
    evidence_path,
    read_if_present,
)


def _sample():
    return ClusterEvidence(
        soft=np.array([[[0.9, 0.1], [0.2, 0.8]]], dtype=np.float32),
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        native_activity=np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
        native_step=0.02,
    )


def test_a_round_trip_keeps_every_field(tmp_path):
    original = _sample()
    original.save(tmp_path / "x.npz")
    loaded = ClusterEvidence.load(tmp_path / "x.npz")

    assert np.array_equal(loaded.soft, original.soft)
    assert np.array_equal(loaded.centroids, original.centroids)
    assert np.array_equal(loaded.native_activity, original.native_activity)
    assert loaded.native_step == original.native_step


def test_the_native_activity_is_stored_as_int8(tmp_path):
    """⚠ It is 0/1 over 120k frames for a 40-minute file; float32 would be four times the size."""
    _sample().save(tmp_path / "x.npz")
    with np.load(tmp_path / "x.npz") as data:
        assert data["native_activity"].dtype == np.int8


def test_the_sidecar_sits_beside_the_timeline():
    assert evidence_path("/out/clip/clip-diar.json").name == "clip" + SUFFIX
    assert evidence_path("/out/clip/clip-diar.json").parent.as_posix().endswith("/out/clip")


def test_a_missing_or_unreadable_sidecar_reads_as_absent(tmp_path):
    """Fail-soft: the consumers are extras, and deleting the file is how you recompute it."""
    assert read_if_present(tmp_path / "nothing.npz") is None

    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not an npz at all")
    assert read_if_present(broken) is None


def test_sortformer_reports_no_evidence_rather_than_a_substitute(tmp_path, monkeypatch):
    """⚠ It has no embeddings. "No claim" is the correct answer, not a weaker signal.

    Reaching for a timeline-shaped substitute is exactly what sank the first single-speaker veto.
    """
    import torch

    class FakeRunner:
        def __init__(self, directory, device="cuda", geometry=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            pass

        def diarize(self, waveform):
            return torch.zeros((4, 2))

    from finesub.speech.diarization import runner as runner_module

    monkeypatch.setattr(runner_module, "SortformerRunner", FakeRunner)
    produced = diarization.BACKENDS["sortformer"].activity(None, handle=tmp_path, device="cpu")

    assert produced.evidence is None
    assert produced.activity.shape == (4, 2)


def test_the_evidence_file_is_removable_not_a_record():
    """It is reproducible from the audio, unlike the timeline beside it."""
    from finesub_bootstrap.artifacts import RECORD_SUFFIXES, REMOVABLE_SUFFIXES

    assert SUFFIX in REMOVABLE_SUFFIXES
    assert SUFFIX not in RECORD_SUFFIXES
    assert "-diar.json" in RECORD_SUFFIXES, "while the timeline it belongs to is a record"


# --- one clean sample per track ---------------------------------------------------------------

def test_a_sample_is_exclusive_and_long_enough():
    """⚠ A stretch where two people talk at once is the worst possible voice sample."""
    from finesub.speech.diarization.samples import pick_samples

    activity = np.zeros((1000, 2), dtype=np.int8)
    activity[0:200, 0] = 1
    activity[150:400, 1] = 1          # 150-200 is shared, so s0's clean run ends at 150
    picks = {sample.track: sample for sample in pick_samples(activity, step=0.02, seconds=2.0)}

    assert picks[0].end <= 150 * 0.02, "the shared frames are excluded"
    assert picks[1].start >= 200 * 0.02
    assert all(sample.seconds >= 2.0 for sample in picks.values())


def test_a_track_that_never_sounds_alone_is_omitted_not_faked():
    """That is a real finding about the material -- usually someone who only speaks over others."""
    from finesub.speech.diarization.samples import pick_samples

    activity = np.zeros((500, 2), dtype=np.int8)
    activity[:, 0] = 1
    activity[:, 1] = 1                # neither track is ever alone
    assert pick_samples(activity, step=0.02, seconds=2.0) == []

    activity[300:, 1] = 0             # now track 0 gets a clean tail
    assert [s.track for s in pick_samples(activity, step=0.02, seconds=2.0)] == [0]


def test_closeness_is_read_through_the_mapping_not_by_track_id():
    """⚠ The sentinel renumbering means track id is not a centroid index.

    Caught in production: the sidecar held four track columns against three centroids, so indexing
    `soft` by track id silently read a neighbour's column -- which picked measurably worse samples,
    not just a wrong number.
    """
    from finesub.speech.diarization.samples import pick_samples

    activity = np.zeros((400, 2), dtype=np.int8)
    activity[0:150, 0] = 1
    activity[200:400, 1] = 1
    evidence = ClusterEvidence(
        soft=np.full((5, 2, 1), 1.9, dtype=np.float32),
        centroids=np.zeros((1, 4), dtype=np.float32),
        native_activity=activity,
        native_step=0.02,
        centroid_of_track=np.array([-1, 0], dtype=np.int16),
    )
    picks = {s.track: s for s in pick_samples(activity, step=0.02, evidence=evidence, seconds=2.0)}

    assert picks[0].closeness is None, "a track with no centroid reports no closeness"
    assert picks[1].closeness == pytest.approx(0.1), "and one with a centroid reads ITS column"


def test_the_window_that_wins_is_the_window_that_is_delivered():
    """⚠ The pick used to score whole runs and then cut ten seconds out of the winner's middle.

    So a long run could win on evidence the cut discarded. The fixture is the shape that exposes
    it: a 40 s run whose good evidence sits only at its very start, against a 10 s run that is
    better throughout. Scoring runs delivered 15-25 s at closeness 0.90 and threw away
    50-60 s at 0.20.
    """
    from finesub.speech.diarization.samples import pick_samples

    native = np.zeros((3100, 1), dtype=np.int8)
    native[:2000] = 1          # 0-40 s
    native[2500:3000] = 1      # 50-60 s
    soft = np.full((40, 1, 1), 1.1, dtype=np.float32)
    soft[0] = 1.99             # the good evidence, only at the very start of the long run
    soft[32] = 1.8             # 51.2 s -- inside the short run, and better than 1.1
    evidence = ClusterEvidence(soft=soft, centroids=np.zeros((1, 2), dtype=np.float32),
                               native_activity=native, native_step=0.02)

    pick = pick_samples(native, step=0.02, evidence=evidence)[0]
    assert pick.closeness == pytest.approx(0.01), "the best window's own score, not a run's"
    assert pick.start < 10.0, "and it is the window that scored, delivered unaltered"
    assert pick.seconds == pytest.approx(10.0)


def test_the_sample_picks_survive_the_sidecar_round_trip(tmp_path):
    evidence = ClusterEvidence(
        soft=np.zeros((1, 1, 1), dtype=np.float32),
        centroids=np.zeros((1, 2), dtype=np.float32),
        native_activity=np.zeros((2, 1), dtype=np.int8),
        native_step=0.02,
        centroid_of_track=np.array([0], dtype=np.int16),
        samples=({"track": 0, "start": 1.5, "end": 11.5, "closeness": 0.25},),
    )
    evidence.save(tmp_path / "e.npz")
    loaded = ClusterEvidence.load(tmp_path / "e.npz")

    assert loaded.samples == evidence.samples
    assert list(loaded.centroid_of_track) == [0]


def test_the_sidecar_refuses_a_timeline_it_did_not_come_from(tmp_path):
    """The columns are track ids, so beside another partition they are confidently wrong.

    ⚠ Reachable without anyone doing anything odd: the sidecar is registered as removable and
    `-diar.json` is reused on existence, so the two files expire independently.
    """
    from finesub.speech.diarization.evidence import timeline_key
    from finesub.speech.diarization.segments import TrackSegment

    tracks = [TrackSegment(0, 0.0, 1.0), TrackSegment(1, 1.0, 2.0), TrackSegment(0, 3.0, 4.0)]
    replace(_sample(), timeline=timeline_key(tracks)).save(tmp_path / "e.npz")

    assert read_if_present(tmp_path / "e.npz", tracks=tracks) is not None
    assert read_if_present(tmp_path / "e.npz", tracks=tracks[:2]) is None, \
        "a different partition must read as no evidence at all"
    assert read_if_present(tmp_path / "e.npz") is not None, \
        "a caller that does not offer a timeline still gets the file"


def test_a_sidecar_with_no_binding_is_not_trusted(tmp_path):
    """Nobody can check its columns, so it is treated as absent rather than as probably fine."""
    from finesub.speech.diarization.segments import TrackSegment

    _sample().save(tmp_path / "e.npz")  # saved without a timeline

    assert read_if_present(tmp_path / "e.npz") is not None
    assert read_if_present(tmp_path / "e.npz", tracks=[TrackSegment(0, 0.0, 1.0)]) is None


def test_the_sidecar_is_not_a_pickle(tmp_path):
    """These files travel between users inside `out/<stem>/`; reading one must not execute code."""
    import numpy as np

    replace(_sample(),
            samples=({"track": 0, "start": 1.0, "end": 2.0, "closeness": 0.5},)).save(
        tmp_path / "e.npz")

    with np.load(tmp_path / "e.npz", allow_pickle=False) as data:
        assert data["samples"].dtype.kind == "U", "the sample list must be text, not an object"


def test_the_log_bayes_total_travels_with_its_divisor(tmp_path):
    """It is a sum over the embeddings, so without the count it is unreadable across files."""
    replace(_sample(), single_speaker_log_bayes=240.0, log_bayes_embeddings=120).save(
        tmp_path / "e.npz")
    loaded = ClusterEvidence.load(tmp_path / "e.npz")

    assert loaded.single_speaker_log_bayes == 240.0
    assert loaded.log_bayes_embeddings == 120
