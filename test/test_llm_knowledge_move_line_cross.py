"""Cross-entry `move_line` (owner 2026-10-02): a merge moves the lines worth
keeping instead of re-typing them, so the node -- and its human-version mark,
now naming the entry it came from -- travels with it."""

from __future__ import annotations

import json

from finesub.llm.knowledge.node.edit import edit_subject
from finesub.llm.knowledge.node.human_marks import EDITED_LABEL
from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.render import HandleMap, render_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo

TERM = "アリス|爱丽丝||主角"


def _repo(tmp_path) -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        for sid, surface in (("X", "游戏X"), ("Y", "游戏Y")):
            txn.create_node(
                sid, "subject",
                {"surface": surface, "intro": "一个游戏", "category": "common", "section_order": ["角色"]},
            )
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "desc": "主角"})
        txn.create_membership("M", "X", "T", "角色", 0)
        txn.create_node("N", "note", {"text": "她总是迟到"})
        txn.create_membership("M2", "X", "N", "角色", 1)
        txn.create_node("U", "term", {"surface": "ボブ", "zh": "鲍勃", "desc": "配角"})
        txn.create_membership("M3", "Y", "U", "角色", 0)
        txn.create_node("Z", "subject", {"surface": "某字幕组", "intro": "一套风格", "category": "style"})
    # a human edits the line, so it has a human-endorsed value to travel with
    text = render_subject(repo.store, "X", mode="human", preview="full")
    report = edit_subject(repo, repo.store.node("X"), text.replace("|主角", "|女主角", 1), refresh=False)
    assert report.changed
    return repo


def _handles(repo: KnowledgeRepo) -> tuple[HandleMap, dict[str, str]]:
    handles = HandleMap()
    for sid in ("X", "Y", "Z"):
        render_subject(repo.store, sid, mode="prompt", handles=handles)
    by_node = {bound[0]: handle for handle, bound in handles.nodes.items()}
    return handles, by_node


def _apply(repo: KnowledgeRepo, handles: HandleMap, *proposals: dict):  # type: ignore[no-untyped-def]
    return apply_model_proposals(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )


def test_a_line_moves_to_another_entry_keeping_its_node(tmp_path) -> None:
    repo = _repo(tmp_path)
    handles, h = _handles(repo)
    report = _apply(repo, handles, {"op": "move_line", "id": h["T"], "to_entry": h["Y"], "reason": "并入"})
    assert report.rev is not None and not report.skipped
    assert [m.parent_id for m in repo.store.parents("T")] == ["Y"]
    assert "アリス" in repo.entry_text("Y") and "アリス" not in repo.entry_text("X")


def test_a_merge_moves_then_retires_and_the_mark_names_the_source(tmp_path) -> None:
    repo = _repo(tmp_path)
    handles, h = _handles(repo)
    report = _apply(
        repo, handles,
        {"op": "move_line", "id": h["T"], "to_entry": h["Y"], "reason": "并入"},
        {"op": "update", "id": h["T"], "line": "アリス|爱丽丝||AI 改写的描述", "reason": "r"},
        {"op": "retire_entry", "entry": h["X"], "merged_into": h["Y"], "reason": "已并入游戏Y"},
    )
    assert report.rev is not None and not report.skipped
    assert repo.store.node("X") is None  # retired
    assert repo.store.node("T") is not None  # the moved line lives on, under Y
    marks = [line for line in repo.entry_text("Y").splitlines() if line.startswith(f"<!-- {EDITED_LABEL}")]
    assert len(marks) == 1
    assert "迁自「游戏X」" in marks[0] and "女主角" in marks[0]


def test_an_unchanged_moved_line_carries_no_mark(tmp_path) -> None:
    repo = _repo(tmp_path)
    handles, h = _handles(repo)
    _apply(repo, handles, {"op": "move_line", "id": h["T"], "to_entry": h["Y"], "reason": "并入"})
    # the value is what the human endorsed: nothing to show, wherever it sits
    assert "女主角" in repo.entry_text("Y")
    assert not any(line.startswith(f"<!-- {EDITED_LABEL}") for line in repo.entry_text("Y").splitlines())


def test_cross_entry_moves_are_refused_where_they_cannot_land(tmp_path) -> None:
    repo = _repo(tmp_path)
    handles, h = _handles(repo)
    report = _apply(
        repo, handles,
        # a style entry is outside a knowledge update's write grant
        {"op": "move_line", "id": h["N"], "to_entry": h["Z"], "section": "约定", "reason": "r"},
        {"op": "move_line", "id": h["T"], "to_entry": "不存在的条目", "reason": "r"},
    )
    reasons = " ".join(r.reason for r in report.skipped)
    assert "not allowed here" in reasons and "does not exist" in reasons
    assert [m.parent_id for m in repo.store.parents("N")] == ["X"]
    assert [m.parent_id for m in repo.store.parents("T")] == ["X"]


def test_moving_into_the_entry_it_is_already_under_is_an_ordinary_move(tmp_path) -> None:
    repo = _repo(tmp_path)
    handles, h = _handles(repo)
    report = _apply(repo, handles, {"op": "move_line", "id": h["T"], "to_entry": h["X"], "reason": "r"})
    assert report.rev is None or report.skipped  # no section change: nothing to do
    assert [m.parent_id for m in repo.store.parents("T")] == ["X"]


def test_a_label_already_in_the_target_section_skips_only_that_move(tmp_path) -> None:
    """A merge's two entries often share a label; the engine would roll the
    whole block back on the duplicate, so the move is refused per op and the
    rest of the block still lands (acceptance review 2026-10-02)."""

    repo = _repo(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("LX", "note", {"text": "X 的设定", "label": "设定"})
        txn.create_membership("MLX", "X", "LX", "角色", 5)
        txn.create_node("LY", "note", {"text": "Y 的设定", "label": "设定"})
        txn.create_membership("MLY", "Y", "LY", "角色", 5)
    handles, h = _handles(repo)
    report = _apply(
        repo, handles,
        {"op": "move_line", "id": h["LX"], "to_entry": h["Y"], "reason": "并入"},
        {"op": "move_line", "id": h["T"], "to_entry": h["Y"], "reason": "并入"},
        {"op": "retire_entry", "entry": h["X"], "merged_into": h["Y"], "reason": "已并入游戏Y"},
    )
    assert report.rev is not None
    assert any("已有 [设定]" in r.reason for r in report.skipped)
    assert [m.parent_id for m in repo.store.parents("T")] == ["Y"]
    assert repo.store.node("X") is None


def test_one_entry_named_two_ways_keeps_one_label_ledger(tmp_path) -> None:
    """An `append_lines` naming the entry by key and a `move_line` naming it by
    handle land in the same section: the ledger is the entry's, however it was
    spelled, and labels compare as the engine compares them (acceptance
    review 2026-10-02, round 5)."""

    repo = _repo(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("LX", "note", {"text": "X 的设定", "label": "设定"})
        txn.create_membership("MLX", "X", "LX", "角色", 5)
        txn.create_node("RX", "note", {"text": "X 的备注", "label": "Ｎｏｔｅ"})
        txn.create_membership("MRX", "X", "RX", "角色", 6)
    handles, h = _handles(repo)
    report = _apply(
        repo, handles,
        {"op": "append_lines", "entry": "游戏Y", "category": "common", "section": "角色",
         "content": "[设定] Y 新写的设定\n[note] Y 新写的备注", "reason": "r"},
        {"op": "move_line", "id": h["LX"], "to_entry": h["Y"], "reason": "并入"},
        {"op": "move_line", "id": h["RX"], "to_entry": h["Y"], "reason": "并入"},
        {"op": "move_line", "id": h["T"], "to_entry": h["Y"], "reason": "并入"},
    )
    assert report.rev is not None  # nothing rolled the block back
    refused = [r.reason for r in report.skipped]
    assert sum("已有 [" in reason for reason in refused) == 2, refused
    assert [m.parent_id for m in repo.store.parents("T")] == ["Y"]
    assert [m.parent_id for m in repo.store.parents("LX")] == ["X"]


def test_a_line_moves_into_an_entry_created_in_the_same_batch(tmp_path) -> None:
    """`to_entry` names the new entry without a category; the batch's own
    `create_entry` registered it under one (review 2026-10-02)."""

    repo = _repo(tmp_path)
    handles, h = _handles(repo)
    report = _apply(
        repo, handles,
        {"op": "create_entry", "category": "common", "entry": "游戏W", "entry_type": "游戏",
         "intro": "从游戏X分出来的作品", "reason": "是另一部作品"},
        {"op": "move_line", "id": h["T"], "to_entry": "游戏W", "section": "角色", "reason": "属于游戏W"},
    )
    assert report.rev is not None and not report.skipped, [r.reason for r in report.skipped]
    resolved = repo.resolve("游戏W")
    assert resolved is not None
    assert [m.parent_id for m in repo.store.parents("T")] == [resolved.subject_id]


def test_one_batch_cannot_create_the_same_name_in_two_matchable_categories(tmp_path) -> None:
    """Matchable keys share one namespace, inside a batch too."""

    repo = _repo(tmp_path)
    handles, _h = _handles(repo)
    report = _apply(
        repo, handles,
        {"op": "create_entry", "category": "common", "entry": "同名", "entry_type": "游戏",
         "intro": "一个游戏", "reason": "r"},
        {"op": "create_entry", "category": "streamer", "entry": "同名", "intro": "一位主播", "reason": "r"},
    )
    assert [r.reason for r in report.skipped] == ["entry already exists (alias/script variant resolves to it)"]
