"""Attaching speaker tracks to subtitle segments.

Two commitments live here, and both are easy to lose to a "pick the loudest"
shortcut: a subtitle that really has two people in it keeps both, and a subtitle
with no speech comes back unlabelled rather than as a guess.
"""

from __future__ import annotations

import pytest

from finesub.speech.diarization.assign import (
    AssignmentParameters,
    assign_all,
    assign_subtitle,
    track_coverage,
)
from finesub.speech.diarization.segments import TrackSegment


def timeline(*rows) -> list[TrackSegment]:
    return [TrackSegment(track=track, start=start, end=end) for track, start, end in rows]


def test_a_subtitle_inside_one_track_gets_that_track():
    speaker = assign_subtitle(1.0, 2.0, timeline((0, 0.0, 5.0)))
    assert speaker.track == 0
    assert speaker.extras["tracks"] == [0]
    assert speaker.review is False


def test_a_subtitle_spanning_two_speakers_keeps_both():
    # Half A, half B -- a merged line or a turn the VAD did not split.
    speaker = assign_subtitle(0.0, 2.0, timeline((0, 0.0, 1.0), (1, 1.0, 2.0)))
    assert sorted(speaker.extras["tracks"]) == [0, 1], "the second speaker must not be dropped"


def test_a_subtitle_over_genuine_overlap_keeps_both():
    speaker = assign_subtitle(0.0, 2.0, timeline((0, 0.0, 2.0), (1, 0.0, 2.0)))
    assert sorted(speaker.extras["tracks"]) == [0, 1]


def test_a_tie_is_flagged_for_review():
    assert assign_subtitle(0.0, 2.0, timeline((0, 0.0, 1.0), (1, 1.0, 2.0))).review is True


def test_a_clear_winner_is_not_flagged():
    speaker = assign_subtitle(0.0, 2.0, timeline((0, 0.0, 1.9), (1, 1.9, 2.0)))
    assert speaker.track == 0
    assert speaker.review is False


def test_a_brief_intruder_does_not_become_a_second_label():
    # 5% of the line: below min_share, so it is not a speaker of this subtitle.
    speaker = assign_subtitle(0.0, 2.0, timeline((0, 0.0, 2.0), (1, 1.9, 2.0)))
    assert speaker.extras["tracks"] == [0]


def test_silence_comes_back_unlabelled_not_unknown():
    speaker = assign_subtitle(10.0, 11.0, timeline((0, 0.0, 5.0)))
    assert speaker.is_labelled is False
    assert speaker.track is None, "no track at all, rather than a guessed one"


def test_a_subtitle_barely_touching_a_track_stays_unlabelled():
    speaker = assign_subtitle(0.0, 10.0, timeline((0, 9.9, 10.0)))
    assert speaker.track is None


def test_coverage_is_a_fraction_of_the_subtitle():
    coverage = track_coverage(0.0, 4.0, timeline((0, 0.0, 1.0), (1, 2.0, 4.0)))
    assert coverage[0] == pytest.approx(0.25)
    assert coverage[1] == pytest.approx(0.5)


def test_coverage_sums_repeated_visits_by_one_track():
    coverage = track_coverage(0.0, 4.0, timeline((0, 0.0, 1.0), (0, 3.0, 4.0)))
    assert coverage[0] == pytest.approx(0.5)


def test_a_zero_length_subtitle_is_not_an_error():
    assert track_coverage(1.0, 1.0, timeline((0, 0.0, 5.0))) == {}


def test_the_block_travels_onto_every_assignment():
    speakers = assign_all([{"start": 0.0, "end": 1.0}], timeline((0, 0.0, 1.0)), block=7)
    assert speakers[0].block == 7


def test_assign_all_preserves_subtitle_order():
    subtitles = [{"start": 2.0, "end": 3.0}, {"start": 0.0, "end": 1.0}]
    speakers = assign_all(subtitles, timeline((0, 0.0, 1.0), (1, 2.0, 3.0)))
    assert [speaker.track for speaker in speakers] == [1, 0]


def test_parameters_reject_a_share_outside_zero_to_one():
    with pytest.raises(ValueError):
        AssignmentParameters(min_share=1.5)
