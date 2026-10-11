"""Scoring a grouping of labels through `pairs_from` equals relabelling the arm and scoring it."""

from itertools import product

from tools.ts_diar_explore import score as S


def _ref(a, b, person):
    return {"start": a, "end": b, "speaker": person, "text": ""}


def _seg(a, b, label):
    return {"start": a, "end": b, "speaker": label, "text": ""}


# Lines A, A, B. u1 and u2 each cover one A line; on the B line u1 and u2 each overlap 0.3 s and u3 0.5 s,
# so grouping u1 with u2 lets the group win the B line (0.6 s) -- fixing winners before grouping would not.
REF = [_ref(0, 1, "A"), _ref(2, 3, "A"), _ref(4, 5, "B")]
ARM = {"segments": [_seg(0, 1, "u1"), _seg(2, 3, "u2"), _seg(4.0, 4.3, "u1"), _seg(4.3, 4.6, "u2"), _seg(4.5, 5.0, "u3")]}


def test_grouped_winner_matches_relabelled_arm():
    lines, overlaps = S.line_overlaps(REF, ARM, 0, 1e9)
    for groups in product("xyz", repeat=3):
        group = dict(zip(("u1", "u2", "u3"), groups))
        relabelled = {"segments": [{**s, "speaker": group[s["speaker"]]} for s in ARM["segments"]]}
        assert S.b3(S.pairs_from(lines, overlaps, group)[0])[2] == S.speaker(REF, relabelled, 0, 1e9)["b3_f1"]


def test_best_grouping_of_the_counterexample_is_0_8():
    lines, overlaps = S.line_overlaps(REF, ARM, 0, 1e9)
    best = max(S.b3(S.pairs_from(lines, overlaps, dict(zip(("u1", "u2", "u3"), g)))[0])[2] for g in product("xyz", repeat=3))
    assert abs(best - 0.8) < 1e-9
