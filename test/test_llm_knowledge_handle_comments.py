"""A handle comment copied into a proposal is never content: every prompt
projection ends a line with `<!-- @k… -->` (or `<!-- @a… -->` on the `[别名]`
row), and a line copied whole and edited would otherwise store it -- in a
description, or as part of an alias."""

from __future__ import annotations

import json

from finesub.llm.knowledge.maintain import main as maintain_main
from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.render import HandleMap, subject_aliases
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _repo(tmp_path):  # type: ignore[no-untyped-def]
    assert maintain_main(
        ["--root", str(tmp_path), "new", "common", "游戏X", "--intro", "一个游戏", "--type", "游戏"]
    ) == 0
    KnowledgeRepo.forget(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    return repo, repo.subjects("common")[0].local_id


def _apply(repo, subject_id, *proposals):  # type: ignore[no-untyped-def]
    handles = HandleMap()
    text = repo.entry_prompt_text(subject_id, handles)
    block = "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals)
    report = apply_model_proposals(
        f"<knowledge_proposals>\n{block}\n</knowledge_proposals>",
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    return text, report


def test_appended_and_rewritten_lines_drop_a_copied_handle(tmp_path) -> None:
    repo, sid = _repo(tmp_path)
    _text, report = _apply(repo, sid, {
        "op": "append_lines", "entry": "@k1", "section": "待归类",
        "content": "ナヒーダ|纳西妲||角色  <!-- @k9 -->", "reason": "r",
    })
    assert report.rev is not None and not report.skipped
    text, _ = _apply(repo, sid)
    handle = next(line.split("<!-- ")[1].split(" -->")[0] for line in text.splitlines() if "ナヒーダ" in line)
    _text, report = _apply(repo, sid, {
        "op": "update", "id": handle, "line": f"ナヒーダ|纳西妲||草神  <!-- {handle} -->", "reason": "r",
    })
    assert report.rev is not None and not report.skipped
    stored = [n.payload for n in (repo.store.node(m.child_id) for m in repo.store.children(sid)) if n.payload.get("surface") == "ナヒーダ"]
    assert stored and stored[0]["desc"] == "草神"


def test_a_handle_comment_in_the_middle_of_the_alias_row_is_dropped(tmp_path) -> None:
    repo, sid = _repo(tmp_path)
    _text, report = _apply(repo, sid, {
        "op": "update", "id": "@a1", "line": "[别名] GX  <!-- @a1 -->、Game X", "reason": "r",
    })
    assert report.rev is not None and sorted(subject_aliases(repo.store, sid)) == ["GX", "Game X"]
