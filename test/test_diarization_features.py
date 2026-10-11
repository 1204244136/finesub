"""Which side of the featurize fork a file takes, and that the segmented side assembles correctly.

The fork exists because one call to the front end costs 2.50 GiB per hour of audio, so an eleven
hour file cannot be featurized whole. Two things about it are easy to break and silent when broken:

* **Every file is segmented, whatever its length, and nothing may reintroduce a second path.**
  A fork existed until 2026-09-19 (one call under 90 minutes) and was removed once W4 showed the
  re-draw is smaller than this chain's own noise. Two numerical paths chosen at runtime is the
  failure to guard against: were free VRAM ever to pick one, the same file would diarize
  differently depending on what else was on the card.
* **The assembly must land every frame on the whole file's grid.** A segment whose frames sit a
  fraction of a hop off still produces plausible features, and every one of them is wrong.

The front end here is a stand-in whose output is a linear function of its input, which makes the
segmented result *exactly* equal to the whole-file result -- so any index slip shows up as a
mismatch rather than as a small numeric difference.

⚠ The fixtures run more than ten minutes of (synthetic) audio on purpose. `feature_segments`
defaults to 600-second segments, so a shorter file takes the segmented path in ONE call and is
indistinguishable from the whole-file path by call count -- which is how the first version of these
tests managed to assert the opposite of what it checked.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from finesub.speech.diarization import runner as dr  # noqa: E402
from finesub.speech.diarization.chunking import (  # noqa: E402
    FEATURE_HOP_SAMPLES,
    SortformerGeometry,
    feature_frame_counts,
)

FEAT_DIM = SortformerGeometry().feat_dim


class CountingFrontEnd:
    """`(features, valid_lengths)` where each frame is the mean of its own hop of audio.

    Linear in the input, so a segment's frames equal the whole file's frames exactly.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, waveform, lengths):
        samples = int(lengths[0])
        self.calls.append(samples)
        x = waveform[0].detach().cpu().numpy()
        total, valid = feature_frame_counts(samples)
        out = np.zeros((1, FEAT_DIM, total), dtype=np.float32)
        if valid:
            hops = x[: valid * FEATURE_HOP_SAMPLES].reshape(valid, FEATURE_HOP_SAMPLES).mean(axis=1)
            out[0, :, :valid] = hops
        return torch.from_numpy(out), torch.tensor([valid])


def make_runner(front_end):
    runner = object.__new__(dr.SortformerRunner)
    runner._featurizer = front_end
    runner.device = "cpu"
    runner.geometry = SortformerGeometry()
    return runner


def ramp(samples: int):
    """A waveform whose value encodes its own position, so a misplaced frame is visible."""
    return torch.from_numpy(np.arange(samples, dtype=np.float32) / samples).unsqueeze(0)


def test_every_length_takes_the_segmented_path():
    """No fork: a short file goes through the same code as an eleven-hour one, in one segment."""
    rate = SortformerGeometry().sample_rate
    for seconds, calls in ((1, 1), (600, 1), (700, 2), (1800, 3)):
        front = CountingFrontEnd()
        samples = seconds * rate
        make_runner(front)._features(ramp(samples), samples)
        assert len(front.calls) == calls, (seconds, front.calls)
        assert front.calls[0] <= 610 * rate, "no call may cover more than one segment plus context"


def test_free_vram_is_never_consulted(monkeypatch):
    """The prose rule, as a test: asking the driver about memory would make the path non-deterministic."""

    def explode(*args, **kwargs):  # pragma: no cover - the point is that it is not called
        raise AssertionError("the featurize path must not depend on free VRAM")

    monkeypatch.setattr(torch.cuda, "mem_get_info", explode, raising=False)
    monkeypatch.setattr(torch.cuda, "memory_allocated", explode, raising=False)
    samples = 700 * SortformerGeometry().sample_rate
    front = CountingFrontEnd()
    make_runner(front)._features(ramp(samples), samples)
    assert len(front.calls) > 1, "this length must take more than one segment"


def test_the_segments_assemble_into_the_whole_file_result(monkeypatch):
    rate = SortformerGeometry().sample_rate
    samples = 700 * rate + 12345
    wave = ramp(samples)

    # The removed whole-file path, reconstructed: one segment covering everything is the single
    # call it used to make.
    def one_segment(n, **kwargs):
        from finesub.speech.diarization.chunking import FeatureSegment

        return [FeatureSegment(0, n, 0, 0, feature_frame_counts(n)[1])]

    monkeypatch.setattr(dr, "feature_segments", one_segment)
    whole, whole_total, whole_valid = make_runner(CountingFrontEnd())._features(wave, samples)

    monkeypatch.undo()
    front = CountingFrontEnd()
    pieces, total, valid = make_runner(front)._features(wave, samples)

    assert len(front.calls) > 1, "the fixture must exercise the segmented path"
    assert (total, valid) == (whole_total, whole_valid)
    assert torch.equal(pieces[:, :, :valid], whole[:, :, :valid].cpu())


def test_padding_past_the_audio_stays_zero():
    samples = 700 * SortformerGeometry().sample_rate + 7
    features, total, valid = make_runner(CountingFrontEnd())._features(ramp(samples), samples)
    assert total > valid, "this length must leave padding to check"
    assert torch.count_nonzero(features[:, :, valid:]) == 0


def test_a_front_end_that_disagrees_with_the_geometry_is_refused():
    """`feature_frame_counts` is pinned only up to about five minutes; long files extrapolate it."""

    class Drifting(CountingFrontEnd):
        def __call__(self, waveform, lengths):
            features, valid = super().__call__(waveform, lengths)
            return features, torch.tensor([int(valid[0]) - 1])

    samples = 700 * SortformerGeometry().sample_rate
    with pytest.raises(ValueError, match="valid frames"):
        make_runner(Drifting())._features(ramp(samples), samples)
