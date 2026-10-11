"""Lossless exports: per-speaker SRT tracks and a single ASS.

The invariant is the point: every source line must reach at least one track. A
misassigned line costs one deletion; a dropped line has to be noticed, found and
retyped. So the tests below care far more about "nothing vanished" than about
"the right speaker got it".
"""

from __future__ import annotations

import pytest

from finesub.speech.diarization.export import (
    FALLBACK_TRACK,
    check_completeness,
    group_by_speaker,
    render_ass,
    render_speaker_srts,
)
from finesub.speech.diarization.labels import SegmentSpeaker, SpeakerLabel


def line(start, end, text):
    return {"start": start, "end": end, "text": text}


def on_track(track, *, tracks=None, name=None):
    labels = (SpeakerLabel("guest", "high", name=name),) if name else ()
    return SegmentSpeaker(labels=labels, track=track, extras={"tracks": tracks or [track]})


UNPLACED = SegmentSpeaker()


def test_every_line_reaches_a_track():
    subtitles = [line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c")]
    speakers = [on_track(0), UNPLACED, on_track(1)]
    groups = group_by_speaker(subtitles, speakers)
    assert check_completeness(subtitles, groups) == []


def test_a_line_with_two_speakers_appears_in_both_tracks():
    subtitles = [line(0, 2, "both")]
    groups = group_by_speaker(subtitles, [on_track(0, tracks=[0, 1])])
    assert sorted(groups) == ["speaker0", "speaker1"]
    assert groups["speaker0"] == groups["speaker1"] == subtitles


def test_an_unplaced_line_lands_in_the_fallback_track():
    subtitles = [line(0, 1, "who?")]
    groups = group_by_speaker(subtitles, [UNPLACED])
    assert list(groups) == [FALLBACK_TRACK]


def test_a_named_speaker_names_the_track():
    groups = group_by_speaker([line(0, 1, "hi")], [on_track(1, name="miyako")])
    assert list(groups) == ["miyako"]


def test_completeness_reports_the_missing_line_rather_than_just_failing():
    subtitles = [line(0, 1, "kept"), line(1, 2, "lost")]
    groups = {"speaker0": [subtitles[0]]}
    assert check_completeness(subtitles, groups) == [1]


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        group_by_speaker([line(0, 1, "a")], [])


def test_srt_tracks_are_independently_numbered_from_one():
    subtitles = [line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c")]
    speakers = [on_track(0), on_track(1), on_track(0)]
    rendered = render_speaker_srts(subtitles, speakers)
    assert rendered["speaker0"].startswith("1\n")
    assert rendered["speaker1"].startswith("1\n")
    assert "\n2\n" in rendered["speaker0"], "the second line of that track, not of the file"


def test_srt_text_survives_the_round_trip():
    rendered = render_speaker_srts([line(0, 1, "こんにちは")], [on_track(0)])
    assert "こんにちは" in rendered["speaker0"]


def test_ass_keeps_overlapping_speech_as_two_events():
    subtitles = [line(0.0, 2.0, "A talks"), line(0.5, 2.5, "B talks")]
    text = render_ass(subtitles, [on_track(0), on_track(1)])
    assert text.count("Dialogue:") == 2
    assert "A talks" in text and "B talks" in text, "one file expresses both"


def test_ass_declares_a_style_per_speaker():
    text = render_ass([line(0, 1, "a"), line(1, 2, "b")], [on_track(0), on_track(1)])
    assert "Style: speaker0," in text
    assert "Style: speaker1," in text


def test_ass_events_are_ordered_by_time_across_speakers():
    subtitles = [line(5.0, 6.0, "late"), line(0.0, 1.0, "early")]
    text = render_ass(subtitles, [on_track(1), on_track(0)])
    assert text.index("early") < text.index("late")


def test_ass_timestamps_are_centiseconds():
    text = render_ass([line(3661.5, 3662.25, "x")], [on_track(0)])
    assert "1:01:01.50" in text
    assert "1:01:02.25" in text


def test_ass_never_drops_a_line_either():
    subtitles = [line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c")]
    text = render_ass(subtitles, [on_track(0), UNPLACED, on_track(0, tracks=[0, 1])])
    assert text.count("Dialogue:") == 4, "three lines, one of them in two tracks"
