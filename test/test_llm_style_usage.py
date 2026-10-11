"""Style usage booking (kb-followups batch 2, item 2).

Pinned: a task books `exposed` / `opportunity="style_injected"` for the style
its choice was fixed on, at the point the choice becomes final -- the
orchestrator's pin (an `auto` pick, an explicit `--style`, the static
`default_style`), the fast batched request turn, the fast agentic window's
first valid pick -- once per task and style; reading a candidate without
injecting it does not count; usage is the number of distinct tasks; a closed
signal gate books nothing and raises nothing.
"""

from __future__ import annotations

from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.signals import (
    STYLE_INJECTED_OPPORTUNITY,
    book_style_injected,
    log_exposed_nodes,
    style_usage_counts,
)
from finesub.llm.knowledge.style import load_style_catalog, record_style_injected
from test.test_llm_auto_style import _run_auto, _store
from test.test_llm_auto_style_window import DIRECT_DELIVERY, FULL_READ, _delivered, _request, _run


def _usage(root) -> dict[str, int]:
    KnowledgeRepo.forget(root)
    repo = KnowledgeRepo.open(root)
    counts = style_usage_counts(repo.store)
    out = {}
    for subject in repo.subjects("style"):
        key = subject.payload.get("surface")
        out[key] = counts.get(subject.local_id, 0)
    return out


def test_the_event_shape_the_report_reads(tmp_path) -> None:
    root = _store(tmp_path)
    repo = KnowledgeRepo.open(root)
    assert book_style_injected(repo, ["style/甲组"], task_id="run-1") == 1
    row = repo.store.conn.execute(
        "SELECT kind, opportunity, task_id, window_id, subject_id, node_id FROM events"
        " WHERE opportunity=?",
        (STYLE_INJECTED_OPPORTUNITY,),
    ).fetchone()
    subject = repo.resolve_qualified("style/甲组", repo.rev).subject_id
    assert tuple(row) == ("exposed", "style_injected", "run-1", None, subject, subject)
    # Once per task and style, whatever the revision of a later call.
    assert book_style_injected(repo, ["style/甲组"], task_id="run-1", rev=0) == 0
    assert book_style_injected(repo, ["style/甲组"], task_id="run-2") == 1
    assert _usage(root) == {"甲组": 2, "default_style": 0}



def test_the_report_reads_what_the_correction_side_books(tmp_path) -> None:
    """The two halves of batch 2 meet here: `book_style_injected` writes the
    usage, `knowledge.report`'s style section reads it -- and the per-node
    exposed count above it does not."""

    from finesub.llm.knowledge.report import build_report

    root = _store(tmp_path)
    repo = KnowledgeRepo.open(root)
    book_style_injected(repo, ["style/甲组"], task_id="run-1")
    book_style_injected(repo, ["style/甲组"], task_id="run-2")
    lines = build_report(repo.store)
    assert "### 甲组（使用率：2 个任务）" in lines
    assert not any(line.startswith("### 甲组") and "exposed" in line for line in lines)

def test_reading_a_candidate_is_not_using_it(tmp_path) -> None:
    root = _store(tmp_path)
    repo = KnowledgeRepo.open(root)
    subject = repo.resolve_qualified("style/甲组", repo.rev).subject_id
    log_exposed_nodes(repo.store, [(subject, subject)], task_id="run-1")  # a kb_read, `context`
    assert _usage(root)["甲组"] == 0


def test_a_normal_run_books_its_pinned_pick(tmp_path, monkeypatch) -> None:
    chosen = {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "主播A"}
    _run_auto(tmp_path, monkeypatch, saved_selection=chosen, task_id="run-1")
    assert _usage(tmp_path / "kb") == {"甲组": 1, "default_style": 0}


def test_an_explicit_and_the_static_default_style_are_booked_too(tmp_path, monkeypatch) -> None:
    root = _store(tmp_path)
    _run_auto(tmp_path, monkeypatch, saved_selection=None, root=root, style="甲组", task_id="run-1")
    (tmp_path / "out" / "clip.srt").unlink()
    _run_auto(tmp_path, monkeypatch, saved_selection=None, root=root, style="default_style", task_id="run-2")
    assert _usage(root) == {"甲组": 1, "default_style": 1}


def test_the_fast_batched_pick_is_booked_in_its_request_turn(tmp_path, monkeypatch) -> None:
    root, _catalog, _art = _run(tmp_path, monkeypatch, [_request(), DIRECT_DELIVERY])
    assert _usage(root)["甲组"] == 1


def test_the_fast_agentic_pick_is_booked_when_it_pins(tmp_path, monkeypatch) -> None:
    root, _catalog, _art = _run(
        tmp_path, monkeypatch, [(_delivered(), [FULL_READ])], strategy="agentic"
    )
    assert _usage(root) == {"甲组": 1, "default_style": 0}


def test_a_closed_signal_gate_books_nothing_and_raises_nothing(tmp_path, monkeypatch) -> None:
    import finesub.llm.knowledge.base as base

    root = _store(tmp_path)
    monkeypatch.setattr(base, "knowledge_signals_enabled", lambda _root: False)
    record_style_injected(root, ("style/甲组",), task_id="run-1")
    record_style_injected(root, ("style/没有这套",), task_id="run-1")  # unresolvable: silent
    monkeypatch.undo()
    assert _usage(root)["甲组"] == 0
    assert load_style_catalog(root)  # the store is untouched otherwise
