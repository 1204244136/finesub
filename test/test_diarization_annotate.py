"""Writing speaker labels onto a stable.json payload.

The two that matter: an unlabelled segment carries no `speaker` key at all (a
reader must not have to guess what an empty one means), and a stale label from an
earlier run is removed rather than left to look current.
"""

from __future__ import annotations

import pytest

from finesub.speech.diarization.annotate import (
    annotate_payload,
    annotate_segments,
    attach_metadata,
    speaker_metadata,
)
from finesub.speech.diarization.labels import SegmentSpeaker, SpeakerLabel

LABELLED = SegmentSpeaker(labels=(SpeakerLabel("streamer", "high"),), track=0, block=1)
PLACED = SegmentSpeaker(track=2, block=1)
UNPLACED = SegmentSpeaker()


def segment(start, end, text="x", **extra):
    return {"start": start, "end": end, "text": text, **extra}


def test_a_labelled_segment_gets_a_speaker_object():
    [row] = annotate_segments([segment(0, 1)], [LABELLED])
    assert row["speaker"]["labels"][0]["role"] == "streamer"
    assert row["speaker"]["track"] == 0


def test_an_unlabelled_segment_gets_no_key_at_all():
    [row] = annotate_segments([segment(0, 1)], [UNPLACED])
    assert "speaker" not in row, "absence is how the schema says 'not labelled'"


def test_a_track_without_a_role_is_still_worth_writing():
    # The acoustic layer places a segment long before anyone names the person.
    [row] = annotate_segments([segment(0, 1)], [PLACED])
    assert row["speaker"]["track"] == 2
    assert row["speaker"]["labels"] == []


def test_a_stale_label_from_an_earlier_run_is_removed():
    stale = segment(0, 1, speaker={"labels": [{"role": "guest", "conf": "high"}]})
    [row] = annotate_segments([stale], [UNPLACED])
    assert "speaker" not in row


def test_everything_else_on_the_segment_survives():
    original = segment(0, 1, confidence=0.97, tags=["a"], words=[{"w": "hi"}])
    [row] = annotate_segments([original], [LABELLED])
    for key, value in original.items():
        if key != "speaker":
            assert row[key] == value


def test_the_input_is_not_mutated():
    original = segment(0, 1)
    annotate_segments([original], [LABELLED])
    assert "speaker" not in original


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        annotate_segments([segment(0, 1)], [])


def test_metadata_records_that_the_timeline_is_not_reproducible():
    metadata = speaker_metadata(
        model="sortformer-4spk-v2.1", artifacts_revision="abc123", track_count=4, parameters={"enter": 0.5}
    )
    assert metadata["reproducible"] is False, "same audio, different hardware, ~7% of labels move"
    assert metadata["model"] == "sortformer-4spk-v2.1"


def test_metadata_lands_under_the_metadata_block():
    payload = {"segments": []}
    attach_metadata(payload, {"model": "m"})
    assert payload["metadata"]["speaker"]["model"] == "m"


def test_attaching_metadata_keeps_what_was_already_there():
    payload = {"metadata": {"asr": {"model": "large-v3-turbo"}}, "segments": []}
    attach_metadata(payload, {"model": "m"})
    assert payload["metadata"]["asr"]["model"] == "large-v3-turbo"


def test_annotate_payload_round_trips_a_whole_file():
    payload = {"metadata": {"asr": {}}, "segments": [segment(0, 1), segment(1, 2)]}
    result = annotate_payload(payload, [LABELLED, UNPLACED], speaker_metadata(
        model="m", artifacts_revision="r", track_count=2, parameters={}
    ))
    assert "speaker" in result["segments"][0]
    assert "speaker" not in result["segments"][1]
    assert result["metadata"]["speaker"]["track_count"] == 2
    assert payload["segments"][0].get("speaker") is None, "the original payload is untouched"


def test_a_payload_without_segments_is_an_error():
    with pytest.raises(TypeError):
        annotate_payload({"metadata": {}}, [])
