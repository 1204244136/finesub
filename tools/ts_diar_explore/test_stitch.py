"""Window stitching keeps speech both windows transcribed exactly once, whatever their segment shapes."""

from tools.ts_diar_explore.episode import handover, stitch


def _seg(a, b, text, speaker="S01"):
    return {"start": a, "end": b, "text": text, "speaker": speaker}


def _text(left, right):
    parts = [(0, 300, {"words": None, "segments": left}), (240, 540, {"words": None, "segments": right})]
    return "".join(u["text"] for u in stitch(parts, do_link=False)["segments"])


def test_one_long_segment_left_two_short_right():
    left = [_seg(200, 259, "前の話です"), _seg(260, 280, "それからこの話")]
    right = [_seg(260, 275, "それから"), _seg(275, 280, "この話"), _seg(281, 300, "次の話題へ")]
    assert _text(left, right) == "前の話ですそれからこの話次の話題へ"


def test_two_short_segments_left_one_long_right():
    left = [_seg(200, 259, "前の話です"), _seg(260, 275, "それから"), _seg(275, 280, "この話")]
    right = [_seg(260, 280, "それからこの話"), _seg(281, 300, "次の話題へ")]
    assert _text(left, right) == "前の話ですそれからこの話次の話題へ"


def test_continuous_speech_without_a_gap_keeps_every_sentence_once():
    # No point in the overlap is free of speech; the left copy starts before the cut, the right copy spans it.
    left = [_seg(230, 262, "ずっと話し続けている"), _seg(255, 300, "これが肝心なところだ", "S02")]
    right = [_seg(240, 290, "これが肝心な", "S02"), _seg(290, 300, "ところだ", "S02"), _seg(262, 320, "続きの長い発言です")]
    text = _text(left, right)
    for sentence in ("ずっと話し続けている", "続きの長い発言です"):
        assert text.count(sentence) == 1
    assert text.count("肝心") == 1 and text.count("ところだ") == 1


def test_overlapping_speakers_at_the_cut_keep_the_short_line():
    # R8001 at 510 s: both windows have the line, each copy's midpoint on its own discarded side.
    left = [_seg(502, 509, "前期我感觉可以", "S02"), _seg(509.30, 511.04, "覆盖覆盖对", "S01")]
    right = [_seg(509.24, 510.22, "覆盖覆盖对", "S01"), _seg(511, 520, "跟不上那我觉得", "S01")]
    parts = [(240, 540, {"words": None, "segments": left}), (480, 780, {"words": None, "segments": right})]
    text = "".join(u["text"] for u in stitch(parts, do_link=False)["segments"])
    assert text.count("覆盖覆盖对") == 1


def test_handover_stays_near_target_without_a_gap():
    assert handover([_seg(240, 300, "continuous")], 240, 300, 270) == 270


def test_separate_utterances_with_the_same_words_are_both_kept():
    # Both windows transcribe two separate "はい" correctly; they are 0.4 s apart, not copies of each other.
    for second in ("S01", "S02"):  # the same person repeating, or someone else answering
        left = [_seg(200, 259, "前の話です"), _seg(269.0, 269.8, "はい"), _seg(270.2, 271.0, "はい", second)]
        right = [_seg(269.0, 269.8, "はい"), _seg(270.2, 271.0, "はい", second), _seg(281, 300, "次の話題へ")]
        assert _text(left, right).count("はい") == 2


def test_same_utterance_timed_slightly_apart_is_kept_once():
    # The two windows time one utterance 0.6 s apart -- more than half of it, within their jitter.
    left = [_seg(200, 259, "前の話です"), _seg(269.0, 270.0, "はいそうです")]
    right = [_seg(269.25, 270.25, "はいそうです"), _seg(281, 300, "次の話題へ")]
    assert _text(left, right).count("はいそうです") == 1
