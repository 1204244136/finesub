"""Human-version marks on the human projection (owner 2026-10-01): wherever a
line's current value differs from the latest human-endorsed one, `entry_text`
shows that value in a comment; `knowledge ack` endorses the current value."""

from __future__ import annotations

from finesub.llm.knowledge.maintain import main
from finesub.llm.knowledge.node.edit import edit_subject, harvest_rendered_edits
from finesub.llm.knowledge.node.history import revert_revision
from finesub.llm.knowledge.node.human_marks import (
    ACKED_LABEL,
    EDITED_LABEL,
    AckError,
    ack_lines,
    line_marks,
)
from finesub.llm.knowledge.node.importer import parse_entry_text
from finesub.llm.knowledge.node.render import HandleMap, render_subject
from finesub.llm.knowledge.node.repo import RENDERED_DIRNAME, KnowledgeRepo
from finesub.llm.knowledge.node.schema import SCHEMA_VERSION
from finesub.llm.knowledge.node.store import KnowledgeStore

import pytest

TERM = "アリス|爱丽丝||主角"
NOTE = "她总是迟到"


def _repo(tmp_path):  # type: ignore[no-untyped-def]
    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        # S renders its 档案 (so its [别名] slot); S2 does not
        for sid, surface, order in (("S", "游戏X", ["档案", "角色"]), ("S2", "游戏Y", ["角色"])):
            txn.create_node(
                sid, "subject",
                {"surface": surface, "intro": "一个游戏", "category": "common", "section_order": order},
            )
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "desc": "主角"})
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_node("N", "note", {"text": NOTE})
        txn.create_membership("M2", "S", "N", "角色", 1)
        txn.create_node("U", "note", {"text": "Y 的一句"})
        txn.create_membership("M3", "S2", "U", "角色", 0)
    return repo


def _human(repo, old: str, new: str, subject: str = "S") -> int:  # type: ignore[no-untyped-def]
    """A human edit through the real round-trip, against the unmarked text."""

    text = render_subject(repo.store, subject, mode="human", preview="full")
    assert old in text
    report = edit_subject(repo, repo.store.node(subject), text.replace(old, new, 1), refresh=False)
    assert report.changed
    return report.rev


def _model(repo, node: str, **payload) -> int:  # type: ignore[no-untyped-def]
    with repo.store.begin("harness") as txn:
        txn.update_node(node, payload=payload)
        return txn.rev


def _model_retire(repo, node: str, membership: str) -> int:  # type: ignore[no-untyped-def]
    with repo.store.begin("harness") as txn:
        txn.tombstone_membership(membership)
        txn.tombstone_node(node)
        return txn.rev


def _marks(repo, subject: str = "S") -> list[str]:  # type: ignore[no-untyped-def]
    return [
        line for line in repo.entry_text(subject).splitlines()
        if line.startswith(f"<!-- {EDITED_LABEL}") or line.startswith(f"<!-- {ACKED_LABEL}")
    ]


def _dump(repo) -> str:  # type: ignore[no-untyped-def]
    return "\n".join(repo.store.conn.iterdump())


# ---- when a mark appears -------------------------------------------------------------


def test_human_then_model_edit_shows_both(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, TERM, "アリス|爱丽丝||女主角")
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="模型写的")
    text = repo.entry_text("S")
    assert "- アリス|爱丽丝||模型写的\n" in text
    assert f"<!-- {EDITED_LABEL} r{human}（行 T）：アリス|爱丽丝||女主角 -->" in text
    # a second model edit keeps the mark; only the current value moved
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="又改了")
    assert _marks(repo) == [f"<!-- {EDITED_LABEL} r{human}（行 T）：アリス|爱丽丝||女主角 -->"]
    # the model writing the human value back: nothing differs, no mark
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="女主角")
    assert _marks(repo) == []


def test_model_lines_and_unedited_imports_carry_no_mark(tmp_path) -> None:
    repo = _repo(tmp_path)
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="模型写的")
    assert _marks(repo) == []


def _alias_item(repo, node: str, value: str) -> str:  # type: ignore[no-untyped-def]
    return next(i.item_id for i in repo.store.items_of(node) if i.value == value)


def test_aliases_compare_as_a_set_that_may_only_grow(tmp_path) -> None:
    """Owner 2026-10-01: the model appending an alias (or the order moving)
    is no departure from the human's line; losing one of theirs is."""

    repo = _repo(tmp_path)
    human = _human(repo, TERM, "アリス|爱丽丝|Alice|主角")
    with repo.store.begin("harness") as txn:
        txn.create_item("I2", "T", "aliases", "艾莉丝")
    assert "- アリス|爱丽丝|Alice、艾莉丝|主角" in repo.entry_text("S")
    assert _marks(repo) == []
    # the order moves (Alice re-created after 艾莉丝): still the same set
    with repo.store.begin("harness") as txn:
        txn.tombstone_item(_alias_item(repo, "T", "Alice"))
        txn.create_item("I3", "T", "aliases", "Alice")
    assert "- アリス|爱丽丝|艾莉丝、Alice|主角" in repo.entry_text("S")
    assert _marks(repo) == []
    # losing the human's alias is a change; the mark gives their whole line
    with repo.store.begin("harness") as txn:
        txn.tombstone_item("I3")
    assert _marks(repo) == [f"<!-- {EDITED_LABEL} r{human}（行 T）：アリス|爱丽丝|Alice|主角 -->"]


def test_a_description_with_pipes_compares_without_reparsing(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, TERM, "アリス|爱丽丝|Alice|主角|配角")
    with repo.store.begin("harness") as txn:
        txn.create_item("I2", "T", "aliases", "艾莉丝")
    assert _marks(repo) == []
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="主角|路人")
    assert len(_marks(repo)) == 1


def test_a_human_appending_an_alias_is_still_a_human_edit(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, TERM, "アリス|爱丽丝|Alice|主角")
    second = _human(repo, "アリス|爱丽丝|Alice|主角", "アリス|爱丽丝|Alice、Ali|主角")
    with repo.store.begin("harness") as txn:
        txn.tombstone_item(_alias_item(repo, "T", "Ali"))
    assert _marks(repo) == [f"<!-- {EDITED_LABEL} r{second}（行 T）：アリス|爱丽丝|Alice、Ali|主角 -->"]


def test_an_acked_value_lets_the_model_append_too(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, TERM, "アリス|爱丽丝|Alice|主角")
    with repo.store.begin("harness") as txn:
        txn.tombstone_item(_alias_item(repo, "T", "Alice"))
        txn.create_item("I2", "T", "aliases", "艾莉丝")
    assert len(_marks(repo)) == 1
    ack_lines(repo.store, "S", "T")
    with repo.store.begin("harness") as txn:
        txn.create_item("I3", "T", "aliases", "Ally")
    assert _marks(repo) == []


def test_a_retired_human_line_leaves_its_comment_in_the_section(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, NOTE, "人写的句子")
    _model_retire(repo, "N", "M2")
    text = repo.entry_text("S")
    section = text.split("## 角色", 1)[1].split("## 元数据", 1)[0]
    assert f"<!-- {EDITED_LABEL} r{human}（行 N，已被删除）：人写的句子 -->" in section
    assert "- 人写的句子" not in text


def test_a_line_the_human_deleted_leaves_no_comment(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, NOTE, "人写的句子")
    _human(repo, "- 人写的句子\n", "")
    assert repo.store.node("N") is None
    assert _marks(repo) == []


def test_a_retired_line_whose_section_is_gone_names_it(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, NOTE, "人写的句子")
    _model_retire(repo, "N", "M2")
    _model_retire(repo, "T", "M")
    _model(repo, "S", surface="游戏X", intro="一个游戏", category="common", section_order=[])
    text = repo.entry_text("S")
    assert "## 角色" not in text
    assert f"<!-- {EDITED_LABEL} r{human}（行 N，已被删除，原在「角色」）：人写的句子 -->" in text


# ---- what endorses, and what does not ---------------------------------------------------


def test_share_mark_after_a_human_edit_keeps_the_mark(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, TERM, "アリス|爱丽丝||女主角")
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="模型写的")
    # what `share mark` writes: a kind=user revision that only flips visibility
    with repo.store.begin("user", note="share mark 游戏X") as txn:
        txn.update_node("T", visibility="shareable")
    assert _marks(repo) == [f"<!-- {EDITED_LABEL} r{human}（行 T）：アリス|爱丽丝||女主角 -->"]


def test_a_pull_that_only_moves_canonical_id_adds_no_mark(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, TERM, "アリス|爱丽丝||女主角")
    with repo.store.begin("pull") as txn:
        txn.update_node("T", canonical_id="C-T", accepted_rev=7)
    assert _marks(repo) == []


def test_revert_back_to_the_human_version_shows_no_mark(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, TERM, "アリス|爱丽丝||女主角")
    model = _model(repo, "T", surface="アリス", zh="爱丽丝", desc="模型写的")
    assert _marks(repo)
    revert_revision(repo.store, model)
    assert _marks(repo) == []


def test_revert_to_a_never_endorsed_value_is_marked(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, TERM, "アリス|爱丽丝||女主角")
    # revert is not a human endorsement: undoing the human edit restores the
    # imported value, which no human ever endorsed
    revert_revision(repo.store, human)
    assert repo.store.node("T").payload["desc"] == "主角"
    assert _marks(repo) == [f"<!-- {EDITED_LABEL} r{human}（行 T）：アリス|爱丽丝||女主角 -->"]


def test_whitespace_only_edits_write_nothing(tmp_path) -> None:
    repo = _repo(tmp_path)
    before = _dump(repo)
    text = render_subject(repo.store, "S", mode="human", preview="full")
    for old, new in (
        (f"- {TERM}", f"- {TERM}   "),       # trailing
        (f"- {NOTE}", f"-    {NOTE}\t"),      # leading after the bullet
        (TERM, "アリス |爱丽丝||主角"),         # inside one column
        ("一个游戏\n", "一个游戏 \n"),          # the intro
    ):
        report = edit_subject(repo, repo.store.node("S"), text.replace(old, new, 1))
        assert not report.changed, (old, new)
    assert _dump(repo) == before
    # and so no human version: a later model edit is not marked
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="模型写的")
    assert _marks(repo) == []


# ---- ack -------------------------------------------------------------------------------


def _two_marked_lines(repo) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    human = _human(repo, TERM, "アリス|爱丽丝||女主角")
    _human(repo, NOTE, "人写的句子")
    with repo.store.begin("harness") as txn:
        txn.update_node("T", payload={"surface": "アリス", "zh": "爱丽丝", "desc": "模型写的"})
        txn.update_node("N", payload={"text": "模型的句子"})
    return human, human + 1


def test_ack_one_line_clears_only_that_mark(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human_t, human_n = _two_marked_lines(repo)
    acked = ack_lines(repo.store, "S", "T")
    assert [mark.node_id for mark in acked] == ["T"]
    assert _marks(repo) == [f"<!-- {EDITED_LABEL} r{human_n}（行 N）：人写的句子 -->"]
    # a later version that changes no value does not bring it back
    with repo.store.begin("pull") as txn:
        txn.update_node("T", canonical_id="C-T")
    assert len(_marks(repo)) == 1
    # a later model edit marks the ACKED value
    at = repo.rev
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="再改")
    assert f"<!-- {ACKED_LABEL} r{at - 1}（行 T）：アリス|爱丽丝||模型写的 -->" in _marks(repo)


def test_ack_the_whole_entry_clears_its_marks_only(tmp_path) -> None:
    repo = _repo(tmp_path)
    _two_marked_lines(repo)
    _human(repo, "Y 的一句", "Y 人写的", subject="S2")
    _model(repo, "U", text="Y 模型写的")
    _model_retire(repo, "N", "M2")
    assert len(_marks(repo)) == 2 and len(_marks(repo, "S2")) == 1
    assert {mark.node_id for mark in ack_lines(repo.store, "S")} == {"T", "N"}
    assert _marks(repo) == []
    assert len(_marks(repo, "S2")) == 1
    assert ack_lines(repo.store, "S") == []


def test_ack_refuses_an_unmarked_or_ambiguous_line(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, TERM, "アリス|爱丽丝||女主角")
    with pytest.raises(AckError):
        ack_lines(repo.store, "S", "T")  # human value is current: nothing to ack
    with pytest.raises(AckError):
        ack_lines(repo.store, "S", "")


def test_the_cli_acks_and_refreshes_rendered(tmp_path, capsys) -> None:
    repo = _repo(tmp_path)
    _two_marked_lines(repo)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    assert "<!-- 人改版" in path.read_text(encoding="utf-8")
    assert main(["--root", str(tmp_path), "ack", "游戏X", "N"]) == 0
    assert "acked 1 line(s)" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8").count("<!-- 人改版") == 1
    assert main(["--root", str(tmp_path), "ack", "游戏X"]) == 0
    assert "<!-- 人改版" not in path.read_text(encoding="utf-8")
    assert main(["--root", str(tmp_path), "ack", "游戏X", "T"]) == 1


# ---- the comment never becomes content -------------------------------------------------


def test_a_value_with_a_comment_close_is_escaped_and_restorable(tmp_path) -> None:
    repo = _repo(tmp_path)
    original = "人写的 --> 箭头 --!> 也算"
    _human(repo, NOTE, original)
    _model(repo, "N", text="模型的句子")
    (mark,) = _marks(repo)
    assert mark.endswith(" -->") and mark.count("-->") == 1  # closes exactly once
    assert "反斜杠" in mark  # the comment says how to undo the escape
    # the documented restore: copy the value back as the line, minus the backslashes
    value = mark.split("）：", 1)[1].removesuffix(" -->").replace("\\>", ">")
    assert value == original
    text = repo.entry_text("S").replace("- 模型的句子", f"- {value}")
    assert edit_subject(repo, repo.store.node("S"), text).changed
    assert repo.store.node("N").payload["text"] == original
    assert _marks(repo) == []


def test_marks_are_never_harvested_back(tmp_path) -> None:
    repo = _repo(tmp_path)
    _two_marked_lines(repo)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    marked = path.read_text(encoding="utf-8")
    parsed = parse_entry_text(path, marked)
    assert not any("人写的" in raw for s in parsed.sections for _, raw, _ in s.lines)
    before = _dump(repo)
    # an untouched file is not harvested; the marked text itself round-trips as a no-op
    assert harvest_rendered_edits(repo) == []
    assert not edit_subject(repo, repo.store.node("S"), marked).changed
    assert _dump(repo) == before
    # an edit elsewhere in the same file applies without reverting the marked lines
    path.write_text(marked.replace("一个游戏", "一个好游戏"), encoding="utf-8")
    assert [row["status"] for row in harvest_rendered_edits(repo)] == ["applied"]
    assert repo.store.node("T").payload["desc"] == "模型写的"
    assert repo.store.node("N").payload["text"] == "模型的句子"
    assert len(_marks(repo)) == 2


def test_prompt_projections_never_carry_marks(tmp_path) -> None:
    repo = _repo(tmp_path)
    _two_marked_lines(repo)
    assert len(_marks(repo)) == 2
    injection = repo.entry_injection_text("S")
    prompt = repo.entry_prompt_text("S", HandleMap())
    assert "人写的" not in injection and "人写的" not in prompt
    assert injection == render_subject(repo.store, "S", mode="prompt")
    with pytest.raises(ValueError):
        render_subject(repo.store, "S", mode="prompt", marks=line_marks(repo.store, "S"))


def test_pinned_reads_see_the_marks_of_their_rev(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, TERM, "アリス|爱丽丝||女主角")
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="模型写的")
    assert "<!-- 人改版" not in repo.entry_text("S", human)
    assert "<!-- 人改版" in repo.entry_text("S")


# ---- schema ----------------------------------------------------------------------------


def test_a_v5_store_gains_the_ack_table(tmp_path) -> None:
    path = tmp_path / "kb.sqlite"
    with KnowledgeStore(path) as store:
        store.conn.execute("DROP TABLE line_acks")
        store.conn.execute("UPDATE meta SET value='5' WHERE key='schema_version'")
    with KnowledgeStore(path) as store:
        assert store.meta("schema_version") == str(SCHEMA_VERSION) == "6"
        assert store.conn.execute("SELECT COUNT(*) FROM line_acks").fetchone()[0] == 0


# ---- the entry head (owner 2026-10-01) -------------------------------------------------


def _head(repo, subject: str = "S") -> list[str]:  # type: ignore[no-untyped-def]
    return [line for line in _marks(repo, subject) if "（条目头 " in line]


def _model_head(repo, **changes) -> int:  # type: ignore[no-untyped-def]
    payload = {**repo.store.node("S").payload, **changes}
    return _model(repo, "S", **payload)


def test_the_head_is_one_unit_marked_under_the_intro(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, "一个游戏\n", "人写的简介\n")
    _model_head(repo, intro="模型写的简介")
    expected = f"<!-- {EDITED_LABEL} r{human}（条目头 S）：H1「游戏X」；简介「人写的简介」；别名「」 -->"
    lines = repo.entry_text("S").splitlines()
    assert lines[:3] == ["# 游戏X", "模型写的简介", expected]
    # the marked text round-trips as a no-op: the comment is not a second intro line
    assert not edit_subject(repo, repo.store.node("S"), repo.entry_text("S")).changed
    # a rename by the model is the same unit, still one mark
    _model_head(repo, surface="游戏Z")
    assert _head(repo) == [expected]
    # back to the human triple: no mark
    _model_head(repo, surface="游戏X", intro="人写的简介")
    assert _head(repo) == []


def test_entry_aliases_compare_as_a_set_that_may_only_grow(tmp_path) -> None:
    repo = _repo(tmp_path)
    human = _human(repo, "- [别名]\n", "- [别名] GX\n")
    with repo.store.begin("harness") as txn:
        txn.create_item("SA", "S", "aliases", "GY")
    assert _head(repo) == []
    with repo.store.begin("harness") as txn:
        txn.tombstone_item(_alias_item(repo, "S", "GX"))
    assert _head(repo) == [
        f"<!-- {EDITED_LABEL} r{human}（条目头 S）：H1「游戏X」；简介「一个游戏」；别名「GX」 -->"
    ]


def test_unrendered_subject_fields_never_mark_the_head(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, "一个游戏\n", "人写的简介\n")
    _model_head(repo, updated_date="2099-01-01")
    _model_head(repo, recategorized_reason="移过来的", section_order=["角色", "别的"])
    assert _head(repo) == []
    # nor do the user revisions that only bump them (every line edit does)
    _human(repo, TERM, "アリス|爱丽丝||女主角")
    assert _head(repo) == []
    _model_head(repo, category="streamer")
    assert _head(repo) == []


def test_the_head_follows_the_line_rules_for_ack_and_revert(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, "一个游戏\n", "人写的简介\n")
    model = _model_head(repo, intro="模型写的简介")
    revert_revision(repo.store, model)
    assert _head(repo) == []
    _model_head(repo, intro="模型写的简介")
    assert [mark.node_id for mark in ack_lines(repo.store, "S", "S")] == ["S"]
    assert _head(repo) == []
    at = repo.rev
    _model_head(repo, intro="又改了")
    assert _head(repo) == [
        f"<!-- {ACKED_LABEL} r{at}（条目头 S）：H1「游戏X」；简介「模型写的简介」；别名「」 -->"
    ]


def test_a_line_moved_to_another_entry_and_retired_there_is_mourned_once(tmp_path) -> None:
    repo = _repo(tmp_path)
    _human(repo, NOTE, "人写的句子")
    with repo.store.begin("harness") as txn:
        txn.move_membership("M2", parent_id="S2", section="角色", order_key=1)
    assert _marks(repo) == [] and _marks(repo, "S2") == []
    _model_retire(repo, "N", "M2")
    assert _marks(repo) == []
    assert [line for line in _marks(repo, "S2") if "已被删除" in line] != []


def test_a_line_retired_under_two_entries_is_remembered_in_both(tmp_path) -> None:
    repo = _repo(tmp_path)
    with repo.store.begin("pull") as txn:
        txn.create_membership("M4", "S2", "N", "角色", 1)
    _human(repo, NOTE, "人写的句子")
    with repo.store.begin("harness") as txn:
        txn.tombstone_membership("M2")
        txn.tombstone_membership("M4")
        txn.tombstone_node("N")
    for subject in ("S", "S2"):
        assert [line for line in _marks(repo, subject) if "人写的句子" in line and "已被删除" in line]


def test_the_head_leaves_out_aliases_the_projection_does_not_show(tmp_path) -> None:
    repo = _repo(tmp_path)
    # S2 does not render its 档案, so it has no [别名] slot to show them in
    _human(repo, "一个游戏\n", "人写的简介\n", subject="S2")
    with repo.store.begin("harness") as txn:
        txn.create_item("S2A", "S2", "aliases", "GY")
    assert _head(repo, "S2") == []
    # S does, until a line of its own in 档案 carries the 别名 label: the core
    # slot is then not rendered
    _human(repo, "一个游戏\n", "人写的简介\n")
    with repo.store.begin("harness") as txn:
        txn.create_node("L", "note", {"text": "GX", "label": "别名"})
        txn.create_membership("ML", "S", "L", "档案", 0)
    with repo.store.begin("harness") as txn:
        txn.create_item("SA", "S", "aliases", "GX")
    assert _head(repo) == []


def test_a_hidden_alias_slot_takes_the_aliases_out_of_the_comparison(tmp_path) -> None:
    """The human endorsed GX while the slot showed; once a 档案 line carries
    the 别名 label the slot is gone, and losing GX is nothing anyone sees."""

    repo = _repo(tmp_path)
    _human(repo, "- [别名]\n", "- [别名] GX\n")
    with repo.store.begin("harness") as txn:
        txn.create_node("L", "note", {"text": "GX", "label": "别名"})
        txn.create_membership("ML", "S", "L", "档案", 0)
    with repo.store.begin("harness") as txn:
        txn.tombstone_item(_alias_item(repo, "S", "GX"))
    assert _head(repo) == []


def test_a_style_entry_head_has_no_alias_part(tmp_path) -> None:
    repo = _repo(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "ST", "subject",
            {"surface": "某字幕组", "intro": "一套习惯", "category": "style", "section_order": []},
        )
    _human(repo, "一套习惯\n", "人写的习惯\n", subject="ST")
    with repo.store.begin("harness") as txn:
        txn.create_item("STA", "ST", "aliases", "某组")
    assert _head(repo, "ST") == []


def test_a_node_retired_without_closing_its_membership_is_remembered(tmp_path) -> None:
    """A pull that meets an upstream retirement only tombstones the node; a
    membership under a local parent can stay open. The line still hung here
    when it was retired."""

    repo = _repo(tmp_path)
    _human(repo, NOTE, "人写的句子")
    with repo.store.begin("pull") as txn:
        txn.tombstone_node("N")
    assert repo.store.conn.execute(
        "SELECT 1 FROM membership_versions WHERE membership_id='M2' AND valid_to_rev IS NULL"
    ).fetchone() is not None
    assert [line for line in _marks(repo) if "人写的句子" in line and "已被删除" in line]


def test_a_section_holding_only_tentative_lines_is_not_rendered_for_the_probe(tmp_path) -> None:
    """S2 has no 档案 in its order; a tentative line there does not make the
    section (and its alias slot) rendered, exactly as load_subject_tree."""

    repo = _repo(tmp_path)
    _human(repo, "一个游戏\n", "人写的简介\n", subject="S2")
    with repo.store.begin("pull") as txn:
        txn.create_node("P", "note", {"text": "待定"}, maturity="tentative")
        txn.create_membership("MP", "S2", "P", "档案", 0)
    with repo.store.begin("harness") as txn:
        txn.create_item("S2A", "S2", "aliases", "GY")
    assert "## 档案" not in repo.entry_text("S2")
    assert _head(repo, "S2") == []


def test_acks_written_in_an_older_value_format_still_render(tmp_path) -> None:
    """Before the structured values, `line_acks.value` held a plain line or a
    head triple. An ack endorses the value AT its rev, recomputed from the
    history, so the stored string is never parsed and an old row cannot break
    a refresh."""

    repo = _repo(tmp_path)
    _human(repo, TERM, "アリス|爱丽丝||女主角")
    _human(repo, "一个游戏\n", "人写的简介\n")
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="模型写的")
    _model_head(repo, intro="模型写的简介")
    at = repo.rev
    for node_id, old_value in (
        ("T", "アリス|爱丽丝||模型写的"),
        ("S", '["游戏X", "模型写的简介", ""]'),
    ):
        repo.store.conn.execute(
            "INSERT INTO line_acks(node_id, rev, value, created_at) VALUES (?, ?, ?, '')",
            (node_id, at, old_value),
        )
    assert _marks(repo) == []
    repo.refresh_rendered()  # the whole-library pass that one bad row used to sink
    _model(repo, "T", surface="アリス", zh="爱丽丝", desc="再改")
    _model_head(repo, intro="又改了")
    assert _marks(repo) == [
        f"<!-- {ACKED_LABEL} r{at}（条目头 S）：H1「游戏X」；简介「模型写的简介」；别名「」 -->",
        f"<!-- {ACKED_LABEL} r{at}（行 T）：アリス|爱丽丝||模型写的 -->",
    ]
