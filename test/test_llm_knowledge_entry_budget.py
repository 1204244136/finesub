"""Knowledge entries in an injected block (plan kb-entry-size, step 1).

Each entry is capped at its category's `max_entry_tokens`; a full block is
shared out longest-first (short entries stay whole); an entry is cut by whole
lines, keeping its head and metadata; the block never exceeds its limit.
Counted one token per character so the arithmetic is readable.
"""

from __future__ import annotations

from finesub.llm.injection_budget import (
    NOTICE_RESERVE_TOKENS,
    fit_entry_text,
    render_knowledge_entries_block,
    retain_injected_handles,
)


def chars(text: str) -> int:
    return len(text)


def entry(name: str, lines: int, *, width: int = 20, section: str = "角色", extra: str = "") -> str:
    body = [f"[{name}{i:03d}] " + "x" * width for i in range(lines)]
    return "\n".join(
        [f"# {name}", "一句简介", "", "## 档案", "", f"[本名] {name}", "", f"## {section}", "", *body]
        + ([""] + extra.split("\n") if extra else [])
        + ["", "## 元数据", "", "最近更新日期: 2026-10-02"]
    )


def render(details, limit, **kwargs):
    return render_knowledge_entries_block(
        details, count_tokens=chars, heuristic_count=chars, block_limit=limit, **kwargs
    )


def test_everything_fits_and_goes_in_untouched() -> None:
    details = {"甲": entry("甲", 3), "乙": entry("乙", 3)}
    block = render(details, 10_000, entry_limit=6000)
    assert block.included == ("甲", "乙") and not block.truncated and not block.dropped
    assert block.text == details["甲"] + "\n\n" + details["乙"]


def test_a_full_block_cuts_only_the_longest_entry() -> None:
    details = {"短": entry("短", 3), "长": entry("长", 80), "中": entry("中", 6)}
    whole = sum(len(text) for text in details.values())
    block = render(details, whole // 2, entry_limit=None)
    assert set(block.included) == {"短", "中"} and block.truncated == ("长",)
    assert block.causes == {"长": "block"}
    kept = block.kept["长"]["角色"]
    assert 0 < kept < 80
    assert f"[长{kept - 1:03d}]" in block.text and f"[长{kept:03d}]" not in block.text
    assert block.tokens <= whole // 2


def test_a_cut_entry_keeps_its_head_and_metadata_and_says_what_is_missing() -> None:
    block = render({"长": entry("长", 80)}, 1200, entry_limit=None)
    assert "# 长" in block.text and "[本名] 长" in block.text and "最近更新日期" in block.text
    assert "「角色」节后" in block.text and "行" in block.text.rsplit("（", 1)[1]


def test_the_category_cap_cuts_even_an_entry_alone_in_the_block() -> None:
    block = render({"长": entry("长", 80)}, 100_000, entry_limit=1000)
    assert block.truncated == ("长",) and block.causes == {"长": "cap"}
    assert len(block.text.split("（注入预算说明")[0]) <= 1000


def test_an_entry_whose_head_overflows_its_share_keeps_only_the_head() -> None:
    from finesub.llm.injection_budget import _cost, _split_entry

    details = {f"条目{i}": entry(f"条目{i}", 20) for i in range(4)}
    floors = sum(_cost(_split_entry(k, v), chars, False).floor + 2 for k, v in details.items())
    block = render(details, floors + NOTICE_RESERVE_TOKENS, entry_limit=None)
    assert not block.dropped
    assert all(not kept for kept in block.kept.values())
    assert "只注入了条目头" in block.text


def test_heads_that_do_not_fit_drop_whole_entries_from_the_tail() -> None:
    details = {f"条目{i}": entry(f"条目{i}", 20) for i in range(6)}
    floor = len(entry("条目0", 0))
    block = render(details, 3 * floor + NOTICE_RESERVE_TOKENS, entry_limit=None)
    assert block.dropped and block.dropped[0] == "条目5"
    assert "被整体丢弃" in block.text
    assert block.tokens <= 3 * floor + NOTICE_RESERVE_TOKENS


def test_the_block_never_exceeds_its_limit() -> None:
    for lines in (1, 7, 40, 200):
        for count in (1, 3, 12):
            details = {f"条目{i}": entry(f"条目{i}", lines * (i + 1)) for i in range(count)}
            for limit in (900, 3000, 20_000):
                block = render(details, limit, entry_limit=6000)
                assert block.tokens <= limit and len(block.text) <= limit


def test_pinned_entries_are_served_first_up_to_half_the_block() -> None:
    style = entry("风格", 40)
    others = {f"别的{i}": entry(f"别的{i}", 40) for i in range(3)}
    limit = int(len(style) * 2.2) + NOTICE_RESERVE_TOKENS
    # shared alike, all four are cut ...
    alike = render({"风格": style, **others}, limit, entry_limit=None)
    assert "风格" in alike.truncated
    # ... pinned, the style goes in whole and the others share the rest
    block = render({"风格": style, **others}, limit, entry_limit=None, pinned=["风格"])
    assert "风格" in block.included and set(block.truncated) == set(others)
    # pinned entries get at most half the block: one that does not fit is cut
    # with the rest, it does not squeeze the others
    big = render({"风格": entry("风格", 300), "别的": entry("别的", 10)}, 4000, entry_limit=None, pinned=["风格"])
    assert big.truncated == ("风格",) and "别的" in big.included


def test_pinned_entries_go_in_whole_in_order_or_join_the_rest() -> None:
    """Two pinned entries that together want more than half the block: the
    first goes in whole, the second is cut with the rest -- never both cut to
    a shared level (peer review, step 2)."""

    first, second = entry("一", 40), entry("二", 40)
    limit = int(len(first) * 2.4) + NOTICE_RESERVE_TOKENS
    block = render({"一": first, "二": second, "别的": entry("别的", 40)}, limit, entry_limit=None, pinned=["一", "二"])
    assert "一" in block.included and "二" in block.truncated


def test_the_notice_stays_inside_its_reserve() -> None:
    details = {f"很长的条目名字{i:02d}": entry(f"很长的条目名字{i:02d}", 50, section=f"节{i}") for i in range(30)}
    block = render(details, 20_000, entry_limit=None)
    notice = block.text[block.text.rindex("（注入预算说明"):]
    assert len(notice) <= NOTICE_RESERVE_TOKENS


def test_guidance_comments_stay_with_their_heading() -> None:
    text = entry("甲", 30, extra="").replace(
        "## 角色\n\n", "## 角色\n\n<!-- 用途：作品内的固定名词\n收录纪律：只收专名 -->\n"
    )
    block = render({"甲": text}, 700, entry_limit=None)
    assert block.truncated == ("甲",)
    assert "<!-- 用途" in block.text and "收录纪律：只收专名 -->" in block.text


def test_handles_of_cut_lines_are_unbound() -> None:
    from finesub.llm.knowledge.node.render import HandleMap

    handles = HandleMap()
    for i in range(1, 41):
        handles.nodes[f"@k{i}"] = (f"node{i}", 1)
        handles._reverse[f"k:node{i}"] = f"@k{i}"
    lines = [f"[行{i:02d}] {'x' * 20}  <!-- @k{i + 2} -->" for i in range(38)]
    text = "\n".join(
        ["# 甲  <!-- @k1 -->", "", "## 档案", "", "[本名] 甲  <!-- @k2 -->", "", "## 角色", "", *lines,
         "", "## 元数据", "", "最近更新日期: 2026-10-02"]
    )
    block = render({"甲": text}, 800, entry_limit=None)
    retain_injected_handles(handles, block.text)
    kept = block.kept["甲"]["角色"]
    assert set(handles.nodes) == {"@k1", "@k2", *(f"@k{i + 2}" for i in range(kept))}
    assert "k:node40" not in handles._reverse


def test_fit_entry_text_cuts_one_entry_and_returns_the_note() -> None:
    text = entry("风格", 200)
    body, note = fit_entry_text("风格", text, 1500, chars)
    assert len(body) <= 1500 and note.startswith("（注入预算说明") and "风格" in note
    assert fit_entry_text("风格", entry("风格", 2), 1500, chars) == (entry("风格", 2), "")


def test_exposure_counts_only_the_lines_that_went_in(tmp_path) -> None:
    """`injected_node_ids` replays the cut: the head section and the first N
    lines of each cut section, nothing after."""

    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.knowledge.node.signals import injected_node_ids

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "甲", "intro": "一个游戏", "category": "common",
                                         "section_order": ["档案", "角色", "地区"]})
        txn.create_node("I", "note", {"text": "甲", "label": "本名"})
        txn.create_membership("MI", "S", "I", "档案", 0)
        for section, prefix, count in (("角色", "C", 5), ("地区", "R", 3)):
            for i in range(count):
                txn.create_node(f"{prefix}{i}", "term", {"surface": f"{prefix}{i}名", "zh": "译", "desc": "描述"})
                txn.create_membership(f"M{prefix}{i}", "S", f"{prefix}{i}", section, i)
    rev = repo.store.current_rev()
    assert injected_node_ids(repo.store, "S", rev, None) and len(injected_node_ids(repo.store, "S", rev, None)) == 10
    shown = injected_node_ids(repo.store, "S", rev, {"角色": 2})
    assert shown == ["S", "I", "C0", "C1"]


def test_without_the_local_counter_the_callers_counter_is_never_asked() -> None:
    """The caller's counter may be a countTokens round trip; the heuristic
    upper bound that allocated the block also checks it (peer review)."""

    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text)

    block = render_knowledge_entries_block(
        {"长": entry("长", 80), "短": entry("短", 2)}, count_tokens=counter, block_limit=1500, entry_limit=None
    )
    assert block.truncated and calls == []


def test_fit_entry_text_never_returns_an_entry_that_did_not_fit() -> None:
    huge_head = entry("风格", 5).replace("一句简介", "长" * 2000)
    body, note = fit_entry_text("风格", huge_head, 20, chars)
    assert body == "" and "被整体丢弃" in note


def test_a_reordered_identity_section_is_head_for_the_cut_too() -> None:
    text = "\n".join(
        ["# 甲", "简介", "", "## 角色", "", *(f"[角{i:02d}] " + "x" * 20 for i in range(40)),
         "", "## 档案", "", "[本名] 甲", "", "## 元数据", "", "最近更新日期: 2026-10-02"]
    )
    block = render({"甲": text}, 600, entry_limit=None)
    assert "[本名] 甲" in block.text and block.truncated == ("甲",)


def test_the_notice_stays_inside_its_reserve_with_many_long_sections() -> None:
    names = [f"一个相当长的小节名字第{i}号" * 2 for i in range(4)]
    def many(name: str) -> str:
        parts = [f"# {name}", "", "## 档案", "", f"[本名] {name}"]
        for section in names:
            parts += ["", f"## {section}", "", *(f"[{section[:2]}{i}] " + "x" * 30 for i in range(15))]
        return "\n".join(parts + ["", "## 元数据", "", "最近更新日期: 2026-10-02"])

    details = {f"很长的条目名字很长的条目名字{i:02d}": many(f"条目{i}") for i in range(12)}
    block = render(details, 6000, entry_limit=None)
    notice = block.text[block.text.rindex("（注入预算说明"):]
    assert len(notice) <= NOTICE_RESERVE_TOKENS, len(notice)
