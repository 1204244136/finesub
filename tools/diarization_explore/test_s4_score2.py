"""Contract tests for the plan 5o.4 scorer (not in the default suite -- `testpaths` is `test/`).

    python -m pytest tools/diarization_explore/test_s4_score2.py -p no:xdist

Each test is one sentence of 5o.4's "打分器的契约测试".
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]

import s4_score2 as s  # noqa: E402

REFS = [{"id": "a", "norm": "今天我们去公园散步", "spk": "A"},
        {"id": "b", "norm": "好的明天见", "spk": "B"}]
OWNER = {"a": 0, "b": 1}


def score(lines):
    content, copies, kinds, _ = s.window(lines, REFS, OWNER.get)
    status = {key: max((o[0] for o in options), key=s.b.RANK.get) for key, options in content.items()}
    return copies, Counter(kind for kind, _ in kinds), status


def content(status):
    return sum(value in s.CORRECT for value in status.values())


def test_splitting_or_joining_lines_changes_neither_content_nor_made_up():
    whole = [((0,), "今天我们去公园散步好的明天见")]
    split = [((0,), "今"), ((0,), "天我们去"), ((0,), "公园散步好"), ((0,), "的明天见")]
    _, kinds_whole, status_whole = score(whole)
    _, kinds_split, status_split = score(split)
    assert status_whole == status_split
    assert kinds_whole["nothing near"] == kinds_split["nothing near"] == 0


def test_regrouping_lines_across_labels_changes_neither_content_nor_made_up():
    # The teeth the first test lacks: it splits inside ONE label, and the per-label pass only
    # runs when a window holds more than one group, so nothing there could ever have moved.
    # Same characters in the same order, regrouped across labels -- 编造 must not budge
    # (audit 5q, B: it did, because the decoy count was read off the per-label alignments).
    text = "今天我们去公园散步好的明天见啊啊啊"
    one = [((0,), text)]
    two = [((0,), text[:9]), ((1,), text[9:])]
    three = [((0,), text[:5]), ((1,), text[5:9]), ((2,), text[9:])]
    counts = [score(lines)[1] for lines in (one, two, three)]
    assert {c["nothing near"] for c in counts} == {3}, [dict(c) for c in counts]
    assert {c["repeat"] for c in counts} == {0}, [dict(c) for c in counts]


def test_a_character_the_joined_string_left_unmatched_stays_unmatched_whatever_the_labels():
    # The case with teeth (Fable, 5q round 2): red on 07308ca1, where the per-label pass rescued
    # the second 去 as a repeat once the two halves sat under different labels. The joined string
    # is identical in both layouts, so "nothing near" must be too.
    one_label = [((0,), "今天我们去去工园散步")]
    two_labels = [((0,), "今天我们去"), ((1,), "去工园散步")]
    assert score(one_label)[1]["nothing near"] == score(two_labels)[1]["nothing near"] == 1
    assert score(two_labels)[1]["repeat"] == 0


def test_splitting_a_misheard_line_changes_nothing_either():
    whole = [((0,), "今天我门去工园散步")]
    split = [((0,), "今天我"), ((0,), "门去"), ((0,), "工园散步")]
    assert score(whole)[2] == score(split)[2]
    assert score(whole)[1] == score(split)[1]


def test_a_gap_does_not_reach_into_the_next_persons_line():
    # Unrestricted, "去公园散步" would pair with 嗯嗯嗯嗯嗯 on s1's line as a mishearing
    # of speaker A's sentence -- a made-up copy on the wrong line.
    _, kinds, status = score([((0,), "今天我们"), ((1,), "嗯嗯嗯嗯嗯")])
    assert all(status.get(("a", pos)) is None for pos in range(4, 9))
    assert kinds["near-miss"] == kinds["wrong"] == 0
    assert kinds["nothing near"] == 5


def test_a_wrong_character_never_raises_correct_recall():
    _, _, right = score([((0,), "今天我们")])
    _, _, wrong = score([((0,), "今天我们庚辛壬癸")])
    assert content(wrong) == content(right)


def test_a_repeat_is_a_repeat_and_costs_the_reference_side_nothing():
    once = [((0,), "今天我们去公园散步")]
    twice = [((0,), "今天我们去公园散步"), ((1,), "我们去公园")]
    _, kinds_once, status_once = score(once)
    copies, kinds_twice, status_twice = score(twice)
    assert status_twice == status_once
    assert kinds_twice["repeat"] == len("我们去公园")
    assert kinds_twice["nothing near"] == kinds_once["nothing near"] == 0


def test_a_repeat_in_the_right_line_rescues_the_attribution():
    # The first copy sits on the other person's line; the repeat on the own one wins.
    copies, kinds, _ = score([((1,), "今天我们去"), ((0,), "今天我们去")])
    best = {key: max((o[1] for o in options), key=s.ATTR_RANK.get) for key, options in copies.items()}
    assert set(best.values()) == {"strict"}
    assert kinds["repeat"] == len("今天我们去"), "the other person's copy is what has to be deleted"


def test_the_own_copy_counts_even_where_only_its_gaps_line_up():
    # The other person's line has the clean copy and wins the joined alignment; the own
    # line's copy is half misheard, so its ends are only gap pairs -- still the own copy.
    copies, _, _ = score([((1,), "今天我们去公园散步"), ((0,), "今添我们去公园三布")])
    best = {key: max((o[1] for o in options), key=s.ATTR_RANK.get) for key, options in copies.items() if key[0] == "a"}
    assert set(best.values()) == {"strict"}


def test_one_output_character_stands_for_one_reference_character():
    lines = [((0, 1), "今天我们去公园散步好的明天见"), ((0,), "公园散步")]
    _, copies, kinds, _ = s.window(lines, REFS, OWNER.get)
    standing = [rid for _, rid in kinds if rid is not None]
    assert len(standing) == len(copies), "every covered reference character has exactly one stand-in"
    assert Counter(kind for kind, _ in kinds)["repeat"] == len("公园散步")


def test_unlabelled_text_is_counted_apart_not_as_the_wrong_speaker():
    copies, kinds, _ = score([((), "今天我们去公园散步")])
    assert kinds["unlabelled"] == len("今天我们去公园散步")
    assert kinds["wrong"] == 0


def test_t4_keeps_every_character_even_where_no_track_sounds():
    import numpy as np

    class Session:
        spans = [{"start": 10.0, "end": 11.0}]
        chars = [(9.5, "甲", 0), (10.2, "乙", 1), (10.4, "丙", 1)]
        labels = [(), ()]

    active = np.zeros((200, 2), dtype=bool)  # nothing sounds anywhere
    lines = s.tc.word_lines(Session(), active)[(10.0, 11.0)]
    assert "".join(text for _, text, _ in lines) == "甲乙丙"
    assert all(tracks == () for tracks, _, _ in lines)
