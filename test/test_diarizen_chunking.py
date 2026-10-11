"""Lazy batching must be the old batching, exactly -- contents AND partition.

⚠ Both halves matter, and the second is the one that would fail silently. Batch composition is
not neutral on this chain: batches of 8 and 32 flip 0.0117% of frames outright, because the hard
powerset argmax turns a last-bit float difference into a different winning subset. So a tail chunk
that used to share a batch with complete chunks has to keep sharing it -- yielding it alone would
be "the same chunks" and a different answer.
"""

from __future__ import annotations

import torch

from finesub.speech.diarization.diarizen_runner import chunk_batches, chunk_waveform


def partitions(samples: int, window: int, step: int, batch: int):
    """The lazy batches, and the batches slicing the materialised version would give."""
    waveform = torch.arange(samples, dtype=torch.float32)[None]
    lazy = list(chunk_batches(waveform, window_size=window, step_size=step, batch_size=batch))
    whole, _ = chunk_waveform(waveform, window_size=window, step_size=step)
    eager = [whole[start:start + batch] for start in range(0, int(whole.shape[0]), batch)]
    return lazy, eager


def test_lazy_batches_match_the_materialised_ones_bit_for_bit():
    """A tail that needs padding, and a batch size that does not divide the chunk count."""
    for samples, window, step, batch in (
        (100_000, 16_000, 1_600, 2),      # tail lands mid-batch
        (17_600, 16_000, 1_600, 2),       # exactly two complete chunks, no tail
        (100_000, 16_000, 1_600, 7),      # batch size coprime with the chunk count
        (8_000, 16_000, 1_600, 2),        # shorter than one window: only the padded tail
    ):
        lazy, eager = partitions(samples, window, step, batch)
        assert len(lazy) == len(eager), f"different number of batches for {samples}"
        for index, (a, b) in enumerate(zip(lazy, eager)):
            assert a.shape == b.shape, f"batch {index} has a different shape for {samples}"
            assert torch.equal(a, b), f"batch {index} differs for {samples}"


def test_the_padded_tail_still_shares_its_batch():
    """⚠ The regression this guards: yielding the tail alone is the same chunks, a different answer."""
    lazy, eager = partitions(100_000, 16_000, 1_600, 2)
    assert int(lazy[-1].shape[0]) == int(eager[-1].shape[0])
    # 100000 samples -> 53 complete chunks + a padded tail = 54, so the last batch holds two.
    assert int(lazy[-1].shape[0]) == 2, "the tail rides with the chunk before it, as it used to"


def test_lazy_batching_does_not_materialise_the_overlapped_copy():
    """The point of the change: peak is one batch, not ten times the audio."""
    waveform = torch.arange(320_000, dtype=torch.float32)[None]
    whole, _ = chunk_waveform(waveform, window_size=16_000, step_size=1_600)
    assert whole.numel() > 8 * waveform.numel(), "the materialised form really is ~10x"

    for batch in chunk_batches(waveform, window_size=16_000, step_size=1_600, batch_size=2):
        assert batch.numel() <= 2 * 16_000, "a batch is two windows, whatever the file's length"
