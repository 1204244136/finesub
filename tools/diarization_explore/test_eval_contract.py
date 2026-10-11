"""The evaluation contract's own guards.

Run explicitly -- `tools/` is not in `testpaths`:

    python -m pytest -q -p xdist -n0 tools/diarization_explore/test_eval_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import cost_metrics as cm  # noqa: E402
import eval_contract as contract  # noqa: E402
import refined_multi as rm  # noqa: E402


def test_an_identity_question_is_refused_on_truth_that_names_roles():
    """The gate that would have stopped Q12's headline and the "merging is closed off" retraction.

    Both came from hina, whose `同人` population holds one game recording replayed and role
    placeholders shared by different voices. The measurement was fine; calling it identity was not.
    """
    with pytest.raises(ValueError) as failure:
        contract.require_people_truth("hina", "are these two tracks the same person")
    message = str(failure.value)
    assert "roles" in message, "it has to say what the labels actually name"
    assert "派蒙" in message or "嘉宾" in message, "and hand over the evidence, not just a verdict"
    assert "roles_ok" in message, "and name the way through for a deliberate contrast run"


def test_a_deliberate_roles_run_is_allowed_but_has_to_say_why(capsys):
    """Refusing outright would delete a reading the branch actually needed.

    Characterising what the measurement looks like on role labels is how the contamination got
    understood. It just may not be silent.
    """
    found = contract.require_people_truth("hina", "same person?", roles_ok="the contrast case")
    assert found.truth is contract.TruthKind.ROLES
    printed = capsys.readouterr().out
    assert "NOT about identity" in printed
    assert "the contrast case" in printed


def test_people_truth_passes_and_carries_its_evidence():
    found = contract.require_people_truth("onetonton", "same person?")
    assert found.truth is contract.TruthKind.PEOPLE
    assert found.line_level and found.why, "a classification with no evidence is the thing to avoid"


def test_spot_checked_material_is_not_people_truth():
    """⚠ nepolabo IS four real humans -- and a 70% spot check cannot settle an identity question.

    Declaring it `PEOPLE` because the humans are real is the mistake this value exists to block.
    """
    assert contract.corpus("nepolabo").truth is contract.TruthKind.UNKNOWN
    assert contract.corpus("nepolabo").line_level is False
    with pytest.raises(ValueError):
        contract.require_people_truth("nepolabo", "same person?")


def test_an_undeclared_corpus_raises_rather_than_defaulting():
    """A silent default puts the property back out of sight, which is the whole problem."""
    with pytest.raises(KeyError) as failure:
        contract.corpus("some-new-show")
    assert "declared truth kind" in str(failure.value)


def test_every_declaration_carries_its_evidence():
    for name, found in contract.CORPORA.items():
        assert found.name == name
        assert len(found.why) > 20, f"{name} is classified with no evidence behind it"


def test_the_cost_ruler_takes_its_defaults_from_the_policy():
    """One place to read, one to change.

    ⚠ `strong_cost` is NOT read through the policy, and its default is pinned separately below.
    Today the two happen to agree -- both skip unlabelled rows, since owner 2026-09-20 ruled those
    cost 0 -- but for different reasons: `strong_cost` skips because `relabel_eval`'s theta arms
    were built on that and stay comparable only if it does, the backend ruler skips because of the
    ruling. If the ruling is ever revisited, the policy moves and `strong_cost` must not move with
    it, which is exactly the kind of coupling that gets "tidied" into one line by someone who has
    not read why.
    """
    import inspect

    for function in (cm.edit_cost, cm.cost_terms):
        defaults = inspect.signature(function).parameters
        assert defaults["price_unlabelled"].default is contract.DEFAULT_POLICY.price_unlabelled
        assert defaults["add"].default == contract.DEFAULT_POLICY.add
        assert defaults["merge"].default == contract.DEFAULT_POLICY.merge
    assert inspect.signature(cm.owner_of_tracks_folded).parameters["seconds"].default == (
        contract.DEFAULT_POLICY.fold_seconds)
    assert inspect.signature(rm.strong_cost).parameters["price_unlabelled"].default is False, (
        "strong_cost keeps skipping by default; relabel_eval's theta arms depend on it")


def test_the_two_fold_lengths_are_one_number():
    """`w2_bias` splits the ROWS and `owner_of_tracks_folded` holds out the FRAMES.

    Two copies of the fold length is two chances for them to disagree about where a fold ends,
    which is a leak that looks like nothing.
    """
    import w2_bias

    assert w2_bias.FOLD_SECONDS == contract.DEFAULT_POLICY.fold_seconds
