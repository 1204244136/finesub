"""Contract tests for the session-mode judging collectors (``rank_session_ab``, ``probe_session_ab``).

A judge's malformed answer must stop the tally, never score quietly: an unknown
verdict word counted as wrong, a label it forgot shrinking that run's
denominator, or a label ranked twice shifting every place after it would each
move a reported number without a trace.

Lives beside its tool (tools/ tests are not in the default suite). Run explicitly:

    python -m pytest tools/bench/test_session_ab_collect.py -o addopts=""
"""

from __future__ import annotations

import pytest

from tools.bench.probe_session_ab import _entity_offs, _probe_scores, main as probe_main
from tools.bench.rank_session_ab import _ranks

KEY = {"A": "per-window/run01", "B": "resume/run01"}
STRENGTH = {"P1": "strong", "P2": "weak"}
ONE = {"P1": "strong"}


def test_ties_take_the_mid_rank() -> None:
    assert _ranks([["A", "B"], ["C"]]) == {"A": 1.5, "B": 1.5, "C": 3}


def test_a_label_ranked_twice_stops_the_tally() -> None:
    with pytest.raises(SystemExit, match="more than once"):
        _ranks([["A"], ["B", "A"]])


def test_primary_and_weak_probes_are_scored_apart() -> None:
    scores = _probe_scores(
        [
            {"id": "P1", "verdicts": {"A": "对", "B": "半对"}},
            {"id": "P2", "verdicts": {"A": "错", "B": "对"}},
        ],
        KEY, STRENGTH, allow_unscorable=False,
    )

    assert scores["probes"] == {"per-window/run01": 1.0, "resume/run01": 0.5}
    assert scores["probes_weak"] == {"per-window/run01": 0.0, "resume/run01": 1.0}


def test_an_unknown_verdict_word_stops_the_tally() -> None:
    with pytest.raises(SystemExit, match="unknown verdict"):
        _probe_scores([{"id": "P1", "verdicts": {"A": "梗", "B": "对"}}], KEY, ONE, allow_unscorable=False)


def test_an_unknown_word_can_be_dropped_and_is_listed() -> None:
    scores = _probe_scores([{"id": "P1", "verdicts": {"A": "梗", "B": "对"}}], KEY, ONE, allow_unscorable=True)

    assert scores["probes"] == {"per-window/run01": None, "resume/run01": 1.0}
    assert scores["unscorable"] == [{"probe": "P1", "run": "per-window/run01", "verdict": "梗"}]


def test_a_probe_outside_the_frozen_list_stops_the_tally() -> None:
    with pytest.raises(SystemExit, match="frozen probe list"):
        _probe_scores([{"id": "P9", "verdicts": {"A": "对", "B": "对"}}], KEY, STRENGTH, allow_unscorable=False)


def test_collect_requires_the_frozen_list() -> None:
    """Without it every probe would land in the primary bucket -- the
    all-probes mean the pre-registration ruled out -- and nothing would say so."""

    with pytest.raises(SystemExit):
        probe_main(["collect", "some-dir"])


def test_an_entity_missing_a_label_stops_the_tally() -> None:
    with pytest.raises(SystemExit, match="every label once"):
        _entity_offs([{"id": "E1", "runs": {"A": {"off_majority": 0}}}], KEY)


def test_entity_counts_add_up_per_run() -> None:
    offs = _entity_offs(
        [{"id": "E1", "runs": {"A": {"off_majority": 2, "matches_reference": True}, "B": {"off_majority": 0}}}], KEY
    )

    assert offs == {
        "per-window/run01": {"off_majority": 2, "matches_reference": 1},
        "resume/run01": {"off_majority": 0, "matches_reference": 0},
    }


def test_a_frozen_probe_without_a_verdict_stops_the_tally() -> None:
    with pytest.raises(SystemExit, match="no verdict"):
        _probe_scores([{"id": "P1", "verdicts": {"A": "对", "B": "对"}}], KEY, STRENGTH, allow_unscorable=False)


def test_a_label_the_judge_left_out_stops_the_tally() -> None:
    with pytest.raises(SystemExit, match="every label once"):
        _probe_scores([{"id": "P1", "verdicts": {"A": "对"}}], KEY, ONE, allow_unscorable=False)
