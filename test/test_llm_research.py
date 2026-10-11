from __future__ import annotations

from dataclasses import replace
import json

import pytest

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.client import LLMCallResult
from finesub.llm.routing.config import LLMRole
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.prompts import ContextPack
from finesub.llm.research import (
    check_research_input_limit,
    load_research_context,
    merge_research_general_contexts,
    parse_round1_output,
    parse_round2_output,
    plan_research_chunks,
    render_research_transcript,
    resolve_round1_entries,
)
from finesub.llm.web_search import QuerySearchResult, SearchResultItem, QueryExtractResult


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))



def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "第一句。"),
        SubtitleSegment("2", 1.2, 2.0, "第二句还没完"),
        SubtitleSegment("3", 2.0, 3.0, "现在结束。"),
        SubtitleSegment("4", 3.2, 4.0, "第四句。"),
        SubtitleSegment("5", 4.2, 5.0, "第五句。"),
    ]


class FakeSearchClient:
    def __init__(self, results: list[QuerySearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[tuple[str, ...], int | None]] = []

    def search_many(self, queries, *, max_queries=None):
        requested = [getattr(item, "query", item) for item in queries]
        self.calls.append((tuple(requested), max_queries))
        if not self.results:
            return []
        # Like the real client, echo each request's query into its result
        # (priority decrement matches rendered labels against request queries).
        return [
            replace(self.results[index % len(self.results)], query=query)
            for index, query in enumerate(requested)
        ]

    def extract_many(self, requests, *, max_urls=None):
        return []


def test_render_research_transcript_marks_windows_without_duplicates() -> None:
    segments = _segments()
    windows = plan_correction_windows(
        segments,
        counter=FakeTokenCounter(),
    )
    transcript = render_research_transcript(segments, windows)

    for window in windows:
        assert f"--- window {window.chunk_id} ---" in transcript
    for segment in segments:
        assert transcript.count(f"{segment.id}|") == 1


def test_parse_round1_output_reads_tag_blocks() -> None:
    text = (
        "好的，以下是结果。\n"
        "<requested_entries>\n主播A\n- 游戏B\n</requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n1. 游戏B 剧情\n游戏B 角色名\n</search_queries>\n"
    )

    result = parse_round1_output(text)

    assert result.requested_entries == ("主播A", "游戏B")
    assert result.keep_entries == ()
    assert result.search_queries == ("游戏B 剧情", "游戏B 角色名")
    assert result.analysis_notes == ""


def test_resolve_round1_entries_caps_channels_and_merges_keep_first(
    tmp_path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text(
        "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
    )
    (knowledge_root / "common" / "index.md").write_text(
        "- 游戏B [游戏] | Game B | 测试游戏\n", encoding="utf-8"
    )
    (knowledge_root / "streamer" / "主播A.md").write_text(
        "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    (knowledge_root / "common" / "游戏B.md").write_text(
        "# 游戏B\n游戏资料。\n\n## 档案\n本名: 游戏B\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=["Game B", "不存在"],
        keep_names=["エーちゃん", "Game B"],
        visible_keep_keys=["主播A"],
        max_requested_entries=1,
        max_keep_entries=1,
        max_total_entries=1,
    )

    assert list(selected) == ["主播A"]  # keep channel wins shared cap
    assert missing == ["不存在"]
    assert ignored == ["Game B"]  # indexed, but not visible as a preinjection
    assert dropped == ["游戏B"]


def test_resolve_round1_entries_enforces_eight_each_and_twelve_total(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    streamer = knowledge_root / "streamer"
    common = knowledge_root / "common"
    streamer.mkdir(parents=True)
    common.mkdir(parents=True)
    kept_keys = [f"保留{i}" for i in range(10)]
    requested_keys = [f"请求{i}" for i in range(10)]
    (streamer / "index.md").write_text(
        "".join(f"- {key} | keep-{i} | 测试\n" for i, key in enumerate(kept_keys)),
        encoding="utf-8",
    )
    (common / "index.md").write_text(
        "".join(
            f"- {key} [游戏] | request-{i} | 测试\n"
            for i, key in enumerate(requested_keys)
        ),
        encoding="utf-8",
    )
    for key in [*kept_keys, *requested_keys]:
        target = streamer if key.startswith("保留") else common
        (target / f"{key}.md").write_text(
            f"# {key}\n内容。\n\n## 档案\n本名: {key}\n\n## 元数据\n最近更新日期: 2026-08-01\n",
            encoding="utf-8",
        )

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=requested_keys,
        keep_names=kept_keys,
        visible_keep_keys=kept_keys,
    )

    assert list(selected) == [*kept_keys[:8], *requested_keys[:4]]
    assert missing == []
    assert ignored == []
    assert dropped == [*kept_keys[8:], *requested_keys[4:]]


def test_resolve_round1_entries_does_not_report_cross_channel_duplicate_as_dropped(
    tmp_path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    streamer = knowledge_root / "streamer"
    common = knowledge_root / "common"
    streamer.mkdir(parents=True)
    common.mkdir(parents=True)
    (streamer / "index.md").write_text("- 主播A | エーちゃん | 测试\n", encoding="utf-8")
    (common / "index.md").write_text("", encoding="utf-8")
    (streamer / "主播A.md").write_text(
        "# 主播A\n内容。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=["エーちゃん"],
        keep_names=["主播A"],
        visible_keep_keys=["主播A"],
        max_requested_entries=0,
        max_keep_entries=1,
        max_total_entries=1,
    )

    assert list(selected) == ["主播A"]
    assert missing == []
    assert ignored == []
    assert dropped == []


def test_parse_round1_output_reads_optional_analysis_notes() -> None:
    text = (
        "<analysis_notes>\n主播在玩游戏B（待定）。\n</analysis_notes>\n"
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>"
    )

    result = parse_round1_output(text)

    assert result.analysis_notes == "主播在玩游戏B（待定）。"


def test_parse_round1_output_requires_both_blocks() -> None:
    with pytest.raises(ValueError, match="search_queries"):
        parse_round1_output(
            "<requested_entries>\n主播A\n</requested_entries>\n"
            "<keep_entries></keep_entries>"
        )


def test_parse_round1_output_accepts_empty_blocks() -> None:
    result = parse_round1_output(
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n</search_queries>"
    )

    assert result.requested_entries == ()
    assert result.search_queries == ()


def test_parse_round2_output_reads_context_pack_tag() -> None:
    payload = {
        "general_context": {"global_summary": "摘要"},
        "window_contexts": [{"window_id": "0001", "context": "背景一"}],
    }
    text = f"<context_pack>\n{json.dumps(payload, ensure_ascii=False)}\n</context_pack>"

    pack = parse_round2_output(text)

    assert pack.general_context["global_summary"] == "摘要"
    assert pack.unbound_window_contexts == {"0001": "背景一"}


def test_parse_round2_output_rejects_bare_json() -> None:
    payload = {"general_context": {"global_summary": "摘要"}, "window_contexts": []}

    with pytest.raises(ValueError, match="exactly one <context_pack>"):
        parse_round2_output(json.dumps(payload, ensure_ascii=False))


def test_check_research_input_limit_raises_on_overflow() -> None:
    messages = [{"role": "user", "content": "字" * 10_000}]

    with pytest.raises(ValueError, match="Split the audio"):
        check_research_input_limit(
            messages,
            round_name="round 1",
            limits=type("L", (), {"prompt_input_limit": 10})(),
            counter=FakeTokenCounter(),
        )


def _knowledge_root(tmp_path, *, with_index: bool):
    root = tmp_path / "knowledge"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    if with_index:
        (root / "streamer" / "index.md").write_text(
            "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
        )
        (root / "streamer" / "主播A.md").write_text(
            "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
        )
        (root / "common" / "index.md").write_text("", encoding="utf-8")
    return root


def test_a_note_for_a_window_outside_the_plan_warns_instead_of_crashing(tmp_path, monkeypatch) -> None:
    """The preparation task may name a window the plan does not have; the
    note is dropped with a warning and the run goes on (the warning call once
    passed a keyword the reporter does not take and killed the whole task)."""

    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.research import run_research_stage

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    responses = [
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>",
        '<context_pack>{"general_context": {}, "window_contexts": '
        '[{"window_id":"9999","context":"背景"}]}</context_pack>',
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0), role=role, model="fake", fallback_used=False, raw_response={}
            )

    monkeypatch.setattr("finesub.llm.research.RoleClient", FakeClient)
    context_path = tmp_path / "clip-research-context.json"
    run_research_stage(
        stable_json=stable_json,
        context_path=context_path,
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
    )

    payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert payload["window_context_bind_report"]["unplaceable_window_ids"] == ["9999"]


def test_run_research_stage_writes_planning_metadata(tmp_path, monkeypatch) -> None:
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.prompts import PROMPT_VERSION
    from finesub.llm.routing.config import WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    from finesub.llm.research import run_research_stage

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)

    responses = [
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>",
        '<context_pack>{"general_context": {}, "window_contexts": '
        '[{"window_id":"0001","context":"背景"}]}</context_pack>',
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    monkeypatch.setattr("finesub.llm.research.RoleClient", FakeClient)

    context_path = tmp_path / "clip-research-context.json"
    run_research_stage(
        stable_json=stable_json,
        context_path=context_path,
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
    )

    payload = json.loads(context_path.read_text(encoding="utf-8"))
    planning = payload["planning"]
    assert planning["prompt_version"] == PROMPT_VERSION
    assert planning["context_reserve_tokens"] == WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    assert planning["profile_id"] == (
        "correction_media=audio,retrieval=local,"
        "difficulty=quality,continuity=serial"
    )
    assert planning["output_scale"] == 1.0
    assert planning["stable_json_hash"].startswith("sha256:")
    assert planning["extra_info_hash"].startswith("sha256:")
    assert planning["knowledge_inputs_hash"].startswith("sha256:")
    assert planning["knowledge_enabled"] is True
    assert planning["collect_task_feedback"] is False
    assert payload["context_pack"]["window_contexts"] == [
        {
            "window_id": "0001",
            "first_source_id": "1",
            "last_source_id": "1",
            "context": "背景",
        }
    ]


def test_knowledge_tri_state_remains_in_research_audit_metadata(tmp_path) -> None:
    """Generation settings remain observable even though they do not gate reuse."""

    from finesub.llm.research import planning_metadata

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    knowledge_root = _knowledge_root(tmp_path, with_index=True)
    common = dict(
        stable_json=stable,
        knowledge_root=knowledge_root,
        extra_info="",
        max_window_subtitle_tokens=None,
    )
    off = planning_metadata(
        resolve_profile("text", "none", "quality"),
        knowledge_enabled=False,
        collect_task_feedback=False,
        **common,
    )
    on = planning_metadata(
        resolve_profile("text", "none", "quality"),
        knowledge_enabled=True,
        collect_task_feedback=True,
        **common,
    )

    assert off != on
    assert off["knowledge_enabled"] is False and on["knowledge_enabled"] is True
    assert off["knowledge_inputs_hash"] == ""
    assert on["knowledge_inputs_hash"].startswith("sha256:")


def test_difficulty_is_normalized_in_research_audit_metadata(tmp_path) -> None:
    """Difficulty is not a research input; other generation axes stay auditable."""

    from finesub.llm.research import planning_metadata, research_reuse_key

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    common = dict(
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )
    high = planning_metadata(resolve_profile("audio", "local", "quality"), **common)
    med = planning_metadata(resolve_profile("audio", "local", "intermediate"), **common)
    other_axis = planning_metadata(
        resolve_profile("text", "local", "quality"), **common
    )

    # The full vector stays recorded, so a mixed-difficulty run is auditable...
    assert high["profile_id"] != med["profile_id"]
    # ...but difficulty is not a reuse key: it picks the correction prompt
    # variant and the thinking knob, neither of which the research rounds read.
    assert research_reuse_key(high) == research_reuse_key(med)
    # Correction media changes geometry, not what research means.
    assert research_reuse_key(other_axis) == research_reuse_key(high)


def test_every_planning_field_is_classified_as_gate_or_audit(tmp_path) -> None:
    """L3's answer to `_task_fingerprint`'s include-list assertion.

    `research_reuse_key` picks named fields, so a field added to
    `planning_metadata` without a decision would be silently ignored -- an old
    context reused past something that should have invalidated it. Classifying
    is the whole point of the whitelist; forgetting must fail here, loudly,
    instead of in someone's subtitles.
    """

    from finesub.llm.research import (
        RESEARCH_REUSE_AUDIT_ONLY,
        RESEARCH_REUSE_GATES,
        planning_metadata,
        research_reuse_key,
    )

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    planning = planning_metadata(
        resolve_profile("audio", "local", "quality"),
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )

    classified = set(RESEARCH_REUSE_GATES) | set(RESEARCH_REUSE_AUDIT_ONLY)
    assert set(planning) == classified
    assert not set(RESEARCH_REUSE_GATES) & set(RESEARCH_REUSE_AUDIT_ONLY)
    assert set(research_reuse_key(planning)) == set(RESEARCH_REUSE_GATES)


def test_retrieval_mode_gates_research_reuse(tmp_path, monkeypatch) -> None:
    """`research_semantics_id` (= retrieval) is what separates a context whose
    preparation could fetch nothing from one that searched."""

    from finesub.llm.research import planning_metadata, research_reuse_key

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    common = dict(
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )
    none = planning_metadata(resolve_profile("audio", "none", "quality"), **common)
    native = planning_metadata(resolve_profile("audio", "native", "quality"), **common)
    local = planning_metadata(resolve_profile("audio", "local", "quality"), **common)

    assert research_reuse_key(none) != research_reuse_key(native)
    assert research_reuse_key(none) != research_reuse_key(local)


def test_native_sources_are_flagged_when_the_backend_reports_no_urls() -> None:
    """A `sources` list that nothing can check must not read like a citation.

    The native prompt asks the model to cite what it verified. On a backend
    that reports no result URLs the reply comes back with site-level "sources"
    the harness cannot corroborate -- and in the 2026-08-15 A/B the one
    checkable claim carried that way was a wrong official title.
    """

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "global_summary": "x",
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}],
        }
    )
    agy = [
        {
            "backend": "local_agent",
            "search_events": [{"tool": "search_web", "query": "q", "urls": []}],
        }
    ]

    marked = _mark_unverified_sources(pack, agy)

    assert marked.general_context["sources"][0]["verified"] is False
    assert "无法核对" in marked.general_context["sources_note"]


def test_native_sources_are_left_alone_when_urls_were_recorded() -> None:
    """Codex/Claude keep a per-call URL ledger, so there is something to check."""

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}]
        }
    )
    grounded = [
        {
            "backend": "local_agent",
            "search_events": [{"tool": "web_search", "urls": ["https://example.org/a"]}],
        }
    ]

    marked = _mark_unverified_sources(pack, grounded)

    assert marked is pack
    assert "verified" not in marked.general_context["sources"][0]
    assert "sources_note" not in marked.general_context


def test_marking_skips_rounds_that_are_not_context_packs() -> None:
    from finesub.llm.research import _mark_unverified_sources

    round1 = parse_round1_output(
        "<requested_entries></requested_entries>"
        "<keep_entries></keep_entries>"
        "<search_queries></search_queries>"
    )
    assert _mark_unverified_sources(round1, []) is round1


def test_marking_reaches_a_pack_wrapped_in_a_preparation_reply() -> None:
    """The preparation task parses into its own reply type, not a bare pack --
    and the mark is what stops a native agent's self-reported `verified=true`
    from reaching the windows unchallenged. A request turn carries no pack and
    is handed back untouched."""

    from finesub.llm.research import _mark_unverified_sources
    from finesub.llm.stages.prepare_task import PrepareReply

    agy = [{"backend": "local_agent", "search_events": [{"tool": "search_web", "urls": []}]}]
    reply = PrepareReply(
        raw="",
        context_pack=ContextPack(
            general_context={"sources": [{"title": "Some Wiki", "verified": True}]}
        ),
    )

    marked = _mark_unverified_sources(reply, agy)

    assert marked is not reply
    assert marked.context_pack.general_context["sources"][0]["verified"] is False
    assert "无法核对" in marked.context_pack.general_context["sources_note"]
    assert _mark_unverified_sources(PrepareReply(raw="x"), agy).context_pack is None


def test_gemini_native_sources_are_not_stamped_unverified() -> None:
    """The shipped default's native path is Gemini `google_search`, not an agent.

    Since the client parses `groundingMetadata`, this path exits at the URL
    check with something real to show. The backend list still has to hold: an
    older Gemini reply with no grounding row must not be stamped either --
    saying "the backend reported nothing" about a provider that does report is
    untrue. `test_llm_switch_matrix` asserts that route exists.
    """

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}]
        }
    )
    grounded = [
        {
            "backend": "gemini_rest",
            "search_events": [
                {
                    "tool": "google_search",
                    "queries": ["q"],
                    "urls": ["https://example.org/a"],
                }
            ],
        }
    ]
    assert _mark_unverified_sources(pack, grounded) is pack

    bare = [{"backend": "gemini_rest", "return_code": 200}]
    marked = _mark_unverified_sources(pack, bare)

    assert marked is pack
    assert "sources_note" not in marked.general_context


# --- chunked research for ultra-long material (plan W7 / P8) -----------------


def _many_windows():
    segments = [
        SubtitleSegment(str(n), float(n), float(n) + 0.8, f"第{n}句的内容文本。")
        for n in range(1, 13)
    ]
    windows = plan_correction_windows(
        segments,
        counter=FakeTokenCounter(),
        max_window_subtitle_tokens=30,
    )
    assert len(windows) >= 3, "fixture must yield several research windows"
    return segments, windows


def test_plan_research_chunks_packs_whole_windows() -> None:
    segments, windows = _many_windows()
    counter = FakeTokenCounter()
    chunks = plan_research_chunks(
        windows, count_tokens=counter.count_text, max_tokens=60
    )
    assert len(chunks) >= 2
    # Every window lands in exactly one chunk, order preserved, never split.
    flattened = [window.chunk_id for chunk in chunks for window in chunk]
    assert flattened == [window.chunk_id for window in windows]
    # 0 disables chunking outright.
    assert plan_research_chunks(
        windows, count_tokens=counter.count_text, max_tokens=0
    ) == [list(windows)]
    # A cap smaller than any single window still never splits one.
    tiny = plan_research_chunks(windows, count_tokens=counter.count_text, max_tokens=1)
    assert [len(chunk) for chunk in tiny] == [1] * len(windows)


def test_merged_general_context_is_deterministic_and_capped() -> None:
    counter = FakeTokenCounter()
    merged = merge_research_general_contexts(
        {"c02": {"b": "2"}, "c01": {"a": "1"}}, count_tokens=counter.count_text
    )
    assert list(merged) == ["合并说明", "c01", "c02"]  # instruction first, stable order
    assert "证据更充分" in merged["合并说明"]  # the conflict rule rides the pack
    assert merge_research_general_contexts({}, count_tokens=counter.count_text) == {}

    # The cap drops whole trailing chunk blocks, never truncates mid-JSON.
    big = {"填充": "x" * 400}
    capped = merge_research_general_contexts(
        {"c01": big, "c02": big, "c03": big},
        count_tokens=counter.count_text,
        max_tokens=counter.count_text(json.dumps({"合并说明": "x" * 200, "c01": big}, ensure_ascii=False)),
    )
    kept = [key for key in capped if key != "合并说明"]
    assert kept and kept == sorted(kept) and len(kept) < 3
    assert kept[0] == "c01"  # earlier material survives


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", 100_000),  # default ON (plan W7)
        ("research_transcript_max_tokens = 0\n", 0),
        ("research_transcript_max_tokens = 50000\n", 50_000),
    ],
)
def test_research_transcript_cap_reads_config(tmp_path, monkeypatch, body, expected) -> None:
    from finesub import config as app_config
    from finesub import settings

    path = tmp_path / "config.toml"
    path.write_text("[chunking]\n" + body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()

    assert settings.RESEARCH_TRANSCRIPT_MAX_TOKENS.resolve() == expected


@pytest.mark.parametrize(
    "value",
    ["-1", "true", "2.5", '"50000"'],
    ids=["negative", "bool", "float", "string"],
)
def test_a_malformed_chunking_cap_is_an_error_not_a_guess(tmp_path, monkeypatch, value) -> None:
    """`int(raw)` used to turn `true` into a 1-token cap, `2.5` into 2 and
    `"50000"` into 50000, all silently. Each is now an error naming the key."""

    from finesub import config as app_config
    from finesub import settings

    path = tmp_path / "config.toml"
    path.write_text(
        f"[chunking]\nresearch_transcript_max_tokens = {value}\n"
        f"max_window_subtitle_tokens = {value}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()

    with pytest.raises(ValueError, match="research_transcript_max_tokens"):
        settings.RESEARCH_TRANSCRIPT_MAX_TOKENS.resolve()
    with pytest.raises(ValueError, match="max_window_subtitle_tokens"):
        settings.MAX_WINDOW_SUBTITLE_TOKENS.resolve()


def test_chunked_research_merges_packs_and_replays_finished_chunks(
    tmp_path, monkeypatch
) -> None:
    """Plan W7: every chunk runs the full rounds over its own slice, notes
    bind to disjoint source intervals, keep entries union in chunk order, and
    a rerun replays finished chunks off their per-chunk files."""

    import re

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import DEFAULT_PROFILE

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:3]]
    calls: list[str] = []

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        calls.append(chunk_label)
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        return {
            "context_pack": {
                "general_context": {"专名": f"来自{chunk_label}"},
                "window_contexts": [
                    {"window_id": window_id, "context": f"note-{chunk_label}"}
                ],
            },
            "keep_entries": ["共有词条", f"词条{chunk_label}"],
            "rounds": {"round1": True, "round2": True},
            "injected_entries": [f"词条{chunk_label}"],
            "token_report": {"totals": {"total_tokens": 10}},
        }

    kwargs = dict(
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=DEFAULT_PROFILE,
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        research_runner=fake_run_research,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
    )
    payload = research_mod._run_chunked_research(chunk_plan, **kwargs)

    assert calls == ["c01", "c02", "c03"]
    pack = ContextPack.from_dict(payload["context_pack"])
    assert len(pack.window_contexts) == 3
    assert all(note.first_source_id and note.last_source_id for note in pack.window_contexts)
    assert list(pack.general_context) == ["合并说明", "c01", "c02", "c03"]
    # Union in chunk order, deduplicated.
    assert payload["keep_entries"][:2] == ["共有词条", "词条c01"]
    assert len(payload["keep_entries"]) == len(set(payload["keep_entries"]))
    for record in payload["research_chunks"]:
        assert (tmp_path / record["file"]).exists()
        assert record["replayed"] is False
    assert payload["token_report"]["totals"]["total_tokens"] == 30

    # A rerun must replay every finished chunk without a single LLM call.
    def exploding_run_research(**_kwargs):
        raise AssertionError("a finished chunk was re-run")

    replayed = research_mod._run_chunked_research(
        chunk_plan, **{**kwargs, "research_runner": exploding_run_research}
    )
    assert [record["replayed"] for record in replayed["research_chunks"]] == [True] * 3
    assert replayed["keep_entries"] == payload["keep_entries"]
    assert replayed["context_pack"] == payload["context_pack"]


def _vote(key: str) -> dict:
    return {"intent": "auto", "outcome": "selected" if key else "none", "key": key, "reason": "r", "rev": 3}


def _chunked_with_votes(tmp_path, votes, *, knowledge_root=None, catalog=None, seen=None, context_name="research-context.json"):
    import finesub.llm.research as research_mod
    from finesub.llm.knowledge.style import StyleCatalog
    from finesub.llm.routing.profiles import DEFAULT_PROFILE

    segments, windows = _many_windows()
    order = iter(votes)

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        if seen is not None:
            seen.append({"chunk_label": chunk_label, **kwargs})
        return {
            "context_pack": {"general_context": {}, "window_contexts": []},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {"total_tokens": 1}},
            "style_selection": _vote(next(order)),
        }

    catalog = catalog or StyleCatalog.from_keys(["甲组", "乙组", "default_style"], rev=3)
    return research_mod._run_chunked_research(
        [[window] for window in windows[: len(votes)]],
        segments=segments,
        context_file=tmp_path / context_name,
        extra_info="",
        knowledge_root=knowledge_root if knowledge_root is not None else tmp_path,
        knowledge_enabled=False,
        profile=DEFAULT_PROFILE,
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        research_runner=fake_run_research,
        research_runner_kwargs={"style_catalog": catalog, "extra_style": "口语一些"},
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
    )


def test_chunked_research_picks_the_style_by_votes(tmp_path) -> None:
    """auto-style-gaps plan B (owner 2026-10-01): every chunk sees the catalog
    and votes with its own forced pick; the most votes win, and the reason
    lists the tally."""

    seen: list[dict] = []
    payload = _chunked_with_votes(tmp_path, ["甲组", "乙组", "甲组"], seen=seen)

    assert seen and all(kwargs["style_catalog"] for kwargs in seen)
    assert all(kwargs["extra_style"] == "口语一些" for kwargs in seen)
    choice = payload["style_selection"]
    assert choice["outcome"] == "selected" and choice["key"] == "甲组" and choice["rev"] == 3
    assert "甲组 2 票（c01、c03）" in choice["reason"] and "乙组 1 票（c02）" in choice["reason"]


def test_a_chunk_saved_without_a_vote_runs_again(tmp_path) -> None:
    """A chunk finished before chunks voted (or under a run without auto) has
    no vote; replaying it would lose one, so that chunk -- only that one --
    runs again. Chunks that carry a vote replay."""

    seen: list[dict] = []
    _chunked_with_votes(tmp_path, ["甲组", "乙组"], seen=seen)
    stale = tmp_path / "research-context-c01.json"
    saved = json.loads(stale.read_text(encoding="utf-8"))
    del saved["style_selection"]
    stale.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    seen.clear()
    payload = _chunked_with_votes(tmp_path, ["乙组"], seen=seen)
    assert [kwargs["chunk_label"] for kwargs in seen] == ["c01"]
    assert payload["style_selection"]["key"] == "乙组"  # c01 now votes 乙组, c02 replays 乙组


def test_a_tie_breaks_by_usage_then_revision_then_name() -> None:
    from finesub.llm.knowledge.style import StyleCatalog, StyleChoice, StyleVoteFacts, tally_style_votes

    catalog = StyleCatalog.from_keys(["甲组", "乙组", "丙组"], rev=9)

    def votes(*keys):
        return [(f"c{i + 1:02d}", StyleChoice(outcome="selected", key=k, rev=9)) for i, k in enumerate(keys)]

    by_usage = tally_style_votes(
        votes("甲组", "乙组"), catalog, StyleVoteFacts(usage={"甲组": 1, "乙组": 4}, last_rev={"甲组": 8, "乙组": 2})
    )
    assert by_usage.key == "乙组" and "使用率：乙组 4、甲组 1" in by_usage.reason

    by_rev = tally_style_votes(
        votes("甲组", "乙组"), catalog, StyleVoteFacts(usage={"甲组": 2, "乙组": 2}, last_rev={"甲组": 8, "乙组": 2})
    )
    assert by_rev.key == "甲组" and "最近修订（rev）：乙组 2、甲组 8" in by_rev.reason

    unreadable = tally_style_votes(
        votes("乙组", "甲组"), catalog, StyleVoteFacts(usage=None, last_rev={"甲组": 5, "乙组": 5})
    )
    assert unreadable.key == "乙组" and "使用率读不到，跳过" in unreadable.reason  # 乙 < 甲 by code point
    assert "按名字字典序" in unreadable.reason
    # Deterministic: the same votes and facts, the same answer, in any order.
    again = tally_style_votes(
        votes("甲组", "乙组"), catalog, StyleVoteFacts(usage=None, last_rev={"甲组": 5, "乙组": 5})
    )
    assert again.key == unreadable.key


def test_a_style_beats_none_on_a_tie() -> None:
    from finesub.llm.knowledge.style import StyleCatalog, StyleChoice, StyleVoteFacts, tally_style_votes

    catalog = StyleCatalog.from_keys(["甲组"], rev=1)  # no default_style: "none" is a legal vote
    choice = tally_style_votes(
        [("c01", StyleChoice(outcome="none", rev=1)), ("c02", StyleChoice(outcome="selected", key="甲组", rev=1))],
        catalog,
        StyleVoteFacts(usage=None),
    )
    assert choice.outcome == "selected" and choice.key == "甲组"
    assert "与「不用风格」打平时选风格" in choice.reason
    none_wins = tally_style_votes(
        [("c01", StyleChoice(outcome="none", rev=1)), ("c02", StyleChoice(outcome="none", rev=1)),
         ("c03", StyleChoice(outcome="selected", key="甲组", rev=1))],
        catalog,
        StyleVoteFacts(usage=None),
    )
    assert none_wins.outcome == "none" and none_wins.key == ""


def test_the_tie_break_reads_usage_and_revision_off_the_store(tmp_path) -> None:
    """The facts come from the store as of the catalog's revision (a later
    edit does not change a replay's answer) and from the `style_injected`
    usage ledger; both are written into the reason."""

    from finesub.llm.knowledge.maintain import main as maintain_main
    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.knowledge.node.signals import book_style_injected
    from finesub.llm.knowledge.style import load_style_catalog

    root = tmp_path / "kb"
    maintain_main(["--root", str(root), "new", "style", "甲组", "--intro", "甲"])
    maintain_main(["--root", str(root), "new", "style", "乙组", "--intro", "乙"])
    KnowledgeRepo.forget(root)
    repo = KnowledgeRepo.open(root)
    for task in ("t1", "t2"):
        book_style_injected(repo, ["style/乙组"], task_id=task)
    book_style_injected(repo, ["style/甲组"], task_id="t1")
    catalog = load_style_catalog(root)

    payload = _chunked_with_votes(tmp_path, ["甲组", "乙组"], knowledge_root=root, catalog=catalog)
    assert payload["style_selection"]["key"] == "乙组"
    assert "使用率：乙组 2、甲组 1" in payload["style_selection"]["reason"]


def test_chunked_research_fans_out_under_parallel_continuity(tmp_path, monkeypatch) -> None:
    """W7 follow-through: chunks are independent, so `continuity=parallel`
    dispatches them concurrently -- one chunk rides one worker, the shared
    writers (exchange logger / checkpoint store / search client) are single
    instances, and the merge stays index-ordered whatever finishes first."""

    import re
    import threading

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:3]]
    seen: dict[str, dict] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=10)
    overlapped = threading.Event()

    def fake_run_research(*, transcript, chunk_label, exchange_logger,
                          checkpoint_store, **kwargs):
        if not overlapped.is_set():
            try:
                barrier.wait()  # proves at least two chunks run concurrently
                overlapped.set()
            except threading.BrokenBarrierError:
                pass
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        with lock:
            seen[chunk_label] = {
                "exchange_logger": exchange_logger,
                "checkpoint_store": checkpoint_store,
            }
        return {
            "context_pack": {
                "general_context": {"专名": f"来自{chunk_label}"},
                "window_contexts": [
                    {"window_id": window_id, "context": f"note-{chunk_label}"}
                ],
            },
            "keep_entries": [f"词条{chunk_label}"],
            "rounds": {"round1": True},
            "injected_entries": [],
            "token_report": {"totals": {"total_tokens": 1}},
        }

    payload = research_mod._run_chunked_research(
        chunk_plan,
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=resolve_profile(continuity="parallel"),
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        research_runner=fake_run_research,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
        parallel_windows=2,
    )
    assert sorted(seen) == ["c01", "c02", "c03"]
    assert overlapped.is_set()  # the barrier really saw two chunks in flight
    # One shared instance each, whichever worker served the chunk.
    loggers = {id(entry["exchange_logger"]) for entry in seen.values()}
    stores = {id(entry["checkpoint_store"]) for entry in seen.values()}
    assert len(loggers) == 1 and len(stores) == 1
    # Assembly is index-ordered regardless of completion order.
    assert [r["chunk_id"] for r in payload["research_chunks"]] == ["c01", "c02", "c03"]
    assert payload["keep_entries"] == ["词条c01", "词条c02", "词条c03"]


def test_chunked_research_stays_serial_on_a_conversational_backend(tmp_path, monkeypatch) -> None:
    """Plan W6, research face: a person's own agent has ONE queue, so fanning
    chunks out just stacks N assignments behind it. The correction stage forces
    serial for this reason but only for its own role and only later, which left
    this phase dispatching against the same single agent."""

    import threading

    import finesub.llm.research as research_mod
    from finesub.reporting import NullReporter, reporting_to
    from finesub.llm.routing.profiles import resolve_profile

    class ConversationalClient:
        asked: list = []

        def routes_to_conversational(self, role, task_group="", difficulty=""):
            ConversationalClient.asked.append((role, task_group))
            return True

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:3]]
    active = 0
    peak = 0
    lock = threading.Lock()
    warnings: list[str] = []

    def fake_run_research(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            active -= 1
        return {
            "context_pack": {"general_context": {}, "window_contexts": []},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    class _Reporter(NullReporter):
        def warning(self, code, message, *, impact="", action="") -> None:
            warnings.append(code)

    with reporting_to(_Reporter()):
        research_mod._run_chunked_research(
            chunk_plan,
            segments=segments,
            context_file=tmp_path / "research-context.json",
            extra_info="",
            knowledge_root=tmp_path,
            knowledge_enabled=False,
            profile=resolve_profile(continuity="parallel"),
            test_profile=True,
            task_artifact_dir=None,
            task_id="t",
            research_runner=fake_run_research,
            token_counter=FakeTokenCounter(),
            collect_task_feedback=False,
            resume=True,
            client=ConversationalClient(),
            transcript_cap=1000,
            parallel_windows=4,
        )
    assert peak == 1
    assert warnings == ["conversational-forced-serial"]
    # asked with the research rounds' own cell, not the correction one
    assert ConversationalClient.asked and ConversationalClient.asked[0][1] == "research"


def test_chunked_research_stays_serial_without_parallel_continuity(tmp_path, monkeypatch) -> None:
    """Default runs (`continuity=serial`) keep chunk execution serial: the
    fan-out follows the task's existing intra-task opt-in, not a new knob."""

    import threading

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:2]]
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run_research(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        with lock:
            active -= 1
        return {
            "context_pack": {"general_context": {}, "window_contexts": []},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    research_mod._run_chunked_research(
        chunk_plan,
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=resolve_profile(continuity="serial"),
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        research_runner=fake_run_research,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
        parallel_windows=4,
    )
    assert peak == 1


def test_chunk_resume_honours_the_stage_level_reuse_gates(tmp_path, monkeypatch) -> None:
    """Reviewer 2026-08-30 P1-2: the chunk files are an L3-adjacent cache, so
    they obey the same gates -- prompt_version or test_profile changing must
    re-run the chunk; difficulty changing must NOT (it is deliberately exempt:
    switching to intermediate exists to stop burning quota, not to redo it)."""

    import re
    from dataclasses import replace as dc_replace

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[windows[0]]]
    calls: list[str] = []

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        calls.append(chunk_label)
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        return {
            "context_pack": {"general_context": {},
                             "window_contexts": [{"window_id": window_id, "context": "n"}]},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    profile = resolve_profile()

    def run(**overrides):
        kwargs = dict(
            segments=segments,
            context_file=tmp_path / "research-context.json",
            extra_info="",
            knowledge_root=tmp_path,
            knowledge_enabled=False,
            profile=profile,
            test_profile=True,
            task_artifact_dir=None,
            task_id="t",
            research_runner=fake_run_research,
            token_counter=FakeTokenCounter(),
            collect_task_feedback=False,
            resume=True,
            client=object(),
            transcript_cap=1000,
        )
        kwargs.update(overrides)
        return research_mod._run_chunked_research(chunk_plan, **kwargs)

    run()
    assert calls == ["c01"]
    run()  # unchanged inputs: replayed off the chunk file
    assert calls == ["c01"]
    run(test_profile=False)  # a smoke chunk must never serve a real run
    assert calls == ["c01", "c01"]
    monkeypatch.setattr(research_mod, "PROMPT_VERSION", "test-bumped-version")
    run(test_profile=False)  # a prompt upgrade re-runs the chunk
    assert calls == ["c01", "c01", "c01"]
    # Difficulty is exempt from the gates: the finished chunk replays.
    run(test_profile=False, profile=dc_replace(profile, difficulty="intermediate"))
    assert calls == ["c01", "c01", "c01"]
