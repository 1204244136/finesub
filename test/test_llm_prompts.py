from __future__ import annotations

import re
from dataclasses import replace as dataclass_replace

import pytest

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.prompts import (
    ContextPack,
    PROMPT_TEMPLATE_DIR,
    build_correction_csv_messages,
    build_knowledge_update_messages,
)
from finesub.llm.prompt_artifacts import write_prompt_artifacts
from finesub.subtitles.metrics import weighted_char_count


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def _direct_input_block(content: str, tag: str) -> str:
    match = re.search(
        rf"(?ms)^[ \t]*<{re.escape(tag)}>[ \t]*\r?\n"
        rf"(.*?)^[ \t]*</{re.escape(tag)}>[ \t]*$",
        content,
    )
    return match.group(1).strip() if match else ""



def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "えっと、あの子ちゃんが来た。"),
        SubtitleSegment("2", 1.2, 2.0, "Haha やばい。"),
    ]


def test_correction_prompt_forbids_omitting_long_singles_or_translated() -> None:
    window = plan_correction_windows(_segments(), counter=FakeTokenCounter())[0]
    messages = build_correction_csv_messages(window=window)
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "不能因为预计输出较长而省略" in combined
    assert "<singles>" not in combined
    assert "translated 必须给出完整终稿" in combined
    assert "严格保持标签、header、9 列字段与顺序" in combined


def test_context_pack_matches_source_coverage_after_split() -> None:
    base = plan_correction_windows(_segments(), counter=FakeTokenCounter())[0]
    split = type(base)(
        chunk_id="0001-a",
        segments=[base.segments[0]],
        overlap_segments=[],
        boundary_reason="test",
        budget=base.budget,
    )
    pack = ContextPack.from_dict(
        {
            "general_context": {},
            "window_contexts": [{
                "window_id": "0001",
                "first_source_id": "1",
                "last_source_id": "2",
                "context": "窗口背景",
            }],
        }
    ).with_source_order(["1", "2"])

    assert pack.window_context_for(split, counter=FakeTokenCounter()) == "窗口背景"


def test_context_pack_from_dict_accepts_window_context_list() -> None:
    pack = ContextPack.from_dict(
        {
            "general_context": {"global_summary": "摘要"},
            "window_contexts": [
                {"window_id": "0001", "context": "背景一"},
                {"window_id": "0002", "context": "背景二"},
            ],
        }
    )

    assert pack.unbound_window_contexts == {"0001": "背景一", "0002": "背景二"}
    assert pack.window_contexts == ()


def test_bind_window_ranges_reports_ids_it_could_not_place() -> None:
    """A pack that loses every note must not look like a pack that had none.

    Binding drops notes whose window id is not in the plan. With no report,
    a round 2 that gets the ids systematically wrong writes a well-formed
    artifact holding zero notes -- and the loader's unbound-note check, which
    exists to refuse half a pack, sees nothing wrong with it.
    """

    segments = [
        SubtitleSegment(str(index), float(index), float(index) + 0.5, str(index))
        for index in range(1, 7)
    ]
    windows = plan_correction_windows(segments, counter=FakeTokenCounter())
    planned_id = windows[0].chunk_id

    report: dict[str, object] = {}
    bound = ContextPack.from_dict(
        {
            "general_context": {"global_summary": "摘要"},
            "window_contexts": [
                {"window_id": planned_id, "context": "能放下的笔记"},
                {"window_id": "chunk-1", "context": "模型自己编的 id"},
            ],
        }
    ).bind_window_ranges(windows, report_sink=report)

    assert [note.context for note in bound.window_contexts] == ["能放下的笔记"]
    assert report["unplaceable_window_ids"] == ["chunk-1"]
    assert report["bound"] == 1

    # Every id wrong -> zero notes, and the report is the only evidence.
    all_wrong: dict[str, object] = {}
    empty = ContextPack.from_dict(
        {"window_contexts": [{"window_id": "chunk-1", "context": "笔记"}]}
    ).bind_window_ranges(windows, report_sink=all_wrong)
    assert empty.window_contexts == ()
    assert not empty.has_unbound_window_contexts
    assert all_wrong["unplaceable_window_ids"] == ["chunk-1"]


def test_context_pack_concatenates_contained_notes_and_drops_crossing_notes() -> None:
    segments = [
        SubtitleSegment(str(index), float(index), float(index) + 0.5, str(index))
        for index in range(1, 7)
    ]
    base = plan_correction_windows(segments, counter=FakeTokenCounter())[0]
    window = dataclass_replace(
        base,
        chunk_id="new",
        segments=segments[1:5],
        overlap_segments=[segments[1]],
    )
    pack = ContextPack.from_dict(
        {
            "window_contexts": [
                {"window_id": "left", "first_source_id": "2", "last_source_id": "3", "context": "越界"},
                {"window_id": "a", "first_source_id": "3", "last_source_id": "3", "context": "三"},
                {"window_id": "b", "first_source_id": "4", "last_source_id": "5", "context": "四五"},
            ]
        }
    ).with_source_order([segment.id for segment in segments])

    assert pack.window_context_for(window, counter=FakeTokenCounter()) == "三\n\n四五"


def test_basic_a_prompt_locks_singles_and_style_rules() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        context_pack=ContextPack.from_dict(
            {
                "general_context": {"global_summary": "主播正在玩游戏。", "must": ["角色名：小明"]},
                "window_contexts": [{
                    "window_id": window.chunk_id,
                    "first_source_id": "1",
                    "last_source_id": "2",
                    "context": "本窗口在打 BOSS。",
                }],
            }
        ).with_source_order(["1", "2"]),
        audio_file_label="clip.wav",
        previous_advice="BOSS 名固定译为「王」。",
        variant="basicA",
    )
    system = messages[0]["content"]

    assert "<singles>...</singles>" in system or "<singles>" in system
    assert "<translated>...</translated>" in system or "<translated>" in system
    assert "不要 Markdown 代码块" in system
    assert "不要无标签散文" in system
    assert "一次性完整输出" in system
    assert "<asr_result>" in system
    assert "Haha -> 哈哈" in system
    assert "やばい" in system
    assert "酱" in system
    assert "音似的假名" in system
    assert "原始音频" in system
    assert "不擅自添加原文没有的人称" in system
    assert "保留原文语气、情感和句式" in system
    assert "微调语序" in system
    assert "保守 1:1 策略" in system
    assert "仅词中切断的接回" in system
    assert "<singles>" in system
    assert "<translated>" in system
    assert "不要写 `plan|" in system or "不要输出 `plan|" in system
    assert "\nplan|" not in system  # no plan example rows (v34)
    assert "两阶段 oneshot" in system or "一一对应" in system
    assert "宜与前一句合并" in system
    assert "视情况可向后合并" in system
    assert "宜独立" in system
    assert "宜丢弃" in system
    assert "五选一取舍结论" in system
    assert "五选一邻接结论" not in system
    # Main oneshot must use the forward-looking verdict.
    oneshot = system.split("两阶段完整 oneshot", 1)[1]
    oneshot_singles = oneshot.split("<singles>", 1)[1].split("</singles>", 1)[0]
    oneshot_translated = oneshot.split("<translated>", 1)[1].split(
        "</translated>", 1
    )[0]
    assert "视情况可向后合并" in oneshot_singles
    assert "（待与后" not in oneshot_singles
    assert "等与后句" not in oneshot_singles
    assert "与后句合流" not in oneshot_singles
    assert "莉奈娅" in system or "リンネ" in system
    assert "ルビルビルビルビ" in system
    assert "sub|1|" in system and "sub|43|" in system
    assert len([line for line in oneshot_singles.splitlines() if line.startswith("sub|")]) == 43
    assert all(
        len(line.split("|")) == 10
        for line in oneshot_singles.splitlines()
        if line.startswith("sub|")
    )
    verdicts = (
        "宜与前一句合并",
        "宜与前两句合并",
        "视情况可向后合并",
        "宜独立",
        "宜丢弃",
    )
    assert all(
        line.endswith(verdicts)
        for line in oneshot_singles.splitlines()
        if line.startswith("sub|")
    )
    assert all(
        len(line.split("|")) == 10
        for line in oneshot_translated.splitlines()
        if line.startswith("sub|")
    )
    # BasicA 主 oneshot 保持 1:1，唯一多源示例是独立的词中接回短例。
    assert "sub|10,11|" not in oneshot_translated
    assert "sub|1,2|" in system
    assert "三源及以上禁止" in system or "禁止三源及以上" in system
    assert "（…同样的“啊”连续共 12 条…）" not in system
    assert "1,2|1.8|good morning|你好" not in system
    assert "分行重试" not in system
    assert "|high|" in system and "|median|" in system and "|low|" in system
    assert "char_count" in system
    assert "拉丁字母、数字和标点每个计 0.5" in system
    for line in system.splitlines():
        if not line.startswith(("sub|", "insert|")):
            continue
        fields = line.split("|")
        assert len(fields) == 10, line
        assert float(fields[8]) == weighted_char_count(fields[6]), line
    assert "不要为了保持逐行语义对齐而强行合并成长字幕" in system
    assert r"\n" not in system
    assert "注入的搜索结果" in system
    assert "<search_results>" in system
    assert "google_search" not in system
    assert "<next_advice>" in system
    assert "task_update_feedback" not in system
    assert "特殊翻译风格要求" not in system
    assert "主播正在玩游戏。" not in system

    user = messages[1]["content"]
    assert user.index("通用背景知识和术语") < user.index("本窗口待处理 ASR")
    assert "<search_results>" in user
    assert "<pre_round_notes>" not in user  # empty in the task flow: not shown (cold read P044)
    assert "此前所有窗口的累积建议" in user
    assert (
        "type|position|start|duration|gap|corrected_text|translation|conf|char_count|note"
        in user
    )
    assert "共有 2 条字幕" in user
    assert "恰好有 **2 条字幕行**" in user
    assert "<singles>" in user
    assert "不要输出 `plan|" in user or "plan|" in user  # forbid plan in recap
    assert "translation 为中文语序小幅前后错位" in user or "允许 translation 为中文语序小幅前后错位" in user
    assert "主播正在玩游戏。" in user
    assert "角色名：小明" in user
    assert "本窗口在打 BOSS。" in user
    assert "BOSS 名固定译为「王」。" in user
    assert "<next_advice>" in user
    assert "本窗口易忘要求" in user
    assert '"current_asr_csv"' not in user
    assert '"budget"' not in user
    assert '"chunk_id"' not in user
    assert "overlap_source_ids" not in user
    # First window: the direct preceding block exists but is empty.
    assert _direct_input_block(user, "preceding_context") == ""
    asr = _direct_input_block(user, "asr_result")
    assert "\\n2|" not in asr
    # CSV local times are relative to the window's audio clip start.
    window = plan_correction_windows(_segments(), counter=FakeTokenCounter())[0]
    assert asr.splitlines()[0] == "local_id|start|duration|gap|text"
    first_line = asr.splitlines()[1]
    expected_local_start = window.segments[0].start - window.clip_start
    assert first_line.split("|")[1] == f"{expected_local_start:.1f}"


def test_correction_prompt_renders_preceding_context_with_negative_times() -> None:
    from finesub.llm.chunking import SubtitleWindow

    base = plan_correction_windows(_segments(), counter=FakeTokenCounter())[0]
    preceding = [
        SubtitleSegment("p1", base.clip_start - 12.4, base.clip_start - 11.1, "前文一"),
        SubtitleSegment("p2", base.clip_start - 10.6, base.clip_start - 9.6, "前文二"),
    ]
    window = SubtitleWindow(
        chunk_id=base.chunk_id,
        segments=base.segments,
        overlap_segments=base.overlap_segments,
        boundary_reason=base.boundary_reason,
        budget=base.budget,
        clip_start=base.clip_start,
        clip_end=base.clip_end,
        preceding_segments=preceding,
    )

    messages = build_correction_csv_messages(window=window)
    user = messages[1]["content"]

    block = _direct_input_block(user, "preceding_context")
    assert block.splitlines()[0] == "-1|-12.4|1.3|0.5|前文一"
    assert "0|-10.6|1.0|0.0|前文二" in block
    # Preceding ids never leak into the translatable input CSV.
    assert "p1" not in _direct_input_block(user, "asr_result")


def test_correction_prompt_injects_window_notes_when_provided() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        window_notes="本窗口疑似在打BOSS，BOSS名待定。",
    )
    user = messages[1]["content"]

    assert "本窗口疑似在打BOSS，BOSS名待定。" in user
    assert user.index("<pre_round_notes>") < user.index("<search_results>")


def test_correction_prompt_injects_extra_style_when_provided() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        extra_style="所有语气词保留日语原文。",
    )
    system = messages[0]["content"]

    assert "特殊翻译风格要求" in system
    assert "所有语气词保留日语原文。" in system


def test_correction_prompt_can_request_task_update_feedback() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        task_update_feedback=True,
    )
    system = messages[0]["content"]

    assert "任务反馈采集" in system
    assert "<task_update_feedback>" in system
    assert "knowledge_hints" in system
    assert "new_entry|replace_section|append_lines" in system
    assert "本窗口局部字幕序号" in system
    assert "harness 会映射回稳定源字幕行号" in system
    assert "不要提出翻译风格、prompt 或 harness 修改建议" in system


def test_prompt_templates_are_loaded_from_src_package() -> None:
    assert PROMPT_TEMPLATE_DIR.name == "prompt_templates"
    assert PROMPT_TEMPLATE_DIR.parent.name == "llm"
    expected = {
        "correction_main_v1.md",
        "correction_user_v2.md",
        "fragment_corr_role_audio_v1.md",
        "fragment_corr_role_text_v1.md",
        "fragment_corr_role_video_v1.md",
        "fragment_output_contract_v1.md",
        "fragment_hallucination_v1.md",
        "fragment_merge_rules_basic_v1.md",
        "fragment_translated_common_v1.md",
        "fragment_native_search_v1.md",
        "fragment_effort_low_v1.md",
        "fragment_effort_deep_v1.md",
        "fragment_search_queries_output_v1.md",
        "fragment_query_style_v1.md",
        "fragment_search_results_usage_v1.md",
        "knowledge_update_artifacts_only_v1.md",
        "knowledge_update_refined_v1.md",
        "knowledge_update_user_v1.md",
        "fragment_knowledge_structure_v1.md",
        "fragment_knowledge_output_v1.md",
        "fragment_knowledge_update_inputs_v1.md",
        "fragment_task_feedback_schema_v3.md",
        "correction_task_update_feedback_v2.md",
        "research_task_feedback_v1.md",
    }

    assert expected.issubset({path.name for path in PROMPT_TEMPLATE_DIR.glob("*.md")})


def test_knowledge_update_artifacts_only_prompt_never_mentions_mistakes() -> None:
    messages = build_knowledge_update_messages(
        refined=False,
        task_summary="测试任务",
        window_packs="--- window 0001 [0.0s – 10.0s] ---\n<raw_csv>\n1|0.0|1.0|0.0|a\n</raw_csv>",
        general_context='{"global_summary":"摘要"}',
        research_feedback='{"knowledge_hints":[]}',
        kb_entries="### streamer/星野灯\n档案内容",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "无精修模式" in system
    assert "streamer" in system and "common" in system
    assert "append_lines" in system and "update" in system
    assert "retire_entry" in system and "create_entry" in system
    assert "edit_lines" not in system and "replace_section" not in system
    assert "<knowledge_proposals>" in system
    assert "宁缺毋滥" in system
    # Design G: the artifacts_only prompt must not define/mention the mistake
    # block at all (the harness ignores a stray one and never applies it).
    assert "mistake_proposals" not in system
    assert "add_mistake" not in system and "add_mistake" not in user
    # harness_notes has been removed from knowledge update; prompt iteration
    # is now fully owned by session_replay.
    assert "<harness_notes>" not in system
    assert "--- window 0001" in user
    assert "星野灯" in user
    assert "<common_mistakes>" not in user
    assert "<good_examples>" not in user


def test_knowledge_update_refined_prompt_covers_style_and_noise() -> None:
    """The refined variant proposes STYLE conventions (plan §3 step 3): the
    mistake ledger and its second output block are gone, so a style
    convention is an ordinary entry update travelling in the one block."""

    messages = build_knowledge_update_messages(
        refined=True,
        task_summary="测试任务",
        window_packs="--- window 0001 ---\n<refined_csv>\n0.0|1.0|你好\n</refined_csv>",
        chunk_index=2,
        multi_chunk=True,
        window_range="0003–0004",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "精修对照模式" in system
    # one output block, and the style rules live in it
    assert "<mistake_proposals>" not in system and "add_mistake" not in system
    assert "翻译风格条目（category `style`）" in system
    assert "只动本次注入的那个 style 条目里的行" in system  # the retire scope (§2.4)
    assert "不要为此新建条目" in system  # a style is created by a human, not here
    assert "有且仅有一个 `<knowledge_proposals>` 块" in user
    # harness_notes removed; prompt iteration owned by session_replay.
    assert "<harness_notes>" not in system
    # Refined-noise disclosure (design J): annotations, split/merge, offsets.
    assert "非音频内容" in system
    assert "明确对应" in system
    assert "已按开始时间重排" in system
    # Chunk notice (multi-chunk, no total stated).
    assert "第 2 块" in user
    assert "0003–0004" in user
    assert "<common_mistakes>" not in user
    assert "<good_examples>" not in user
    assert "<common_mistakes>" not in system


def test_correction_prompt_injects_the_style_block() -> None:
    """The slot that used to carry the curated mistake list now carries the
    style entry the run named (`docs/knowledge.md`, 「style 类别」)."""

    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    with_block = build_correction_csv_messages(
        window=window,
        style_block='本任务的翻译风格约定：\n<style name="某字幕组">\n[自称转换] 用「我」\n</style>',
    )
    without_block = build_correction_csv_messages(window=window)

    assert "[自称转换] 用「我」" in with_block[0]["content"]
    assert "翻译风格约定" not in without_block[0]["content"]
    assert "$style_block" not in without_block[0]["content"]


def test_write_prompt_artifacts_writes_plan_and_prompts(tmp_path) -> None:
    artifacts = {
        "model_limits": {"output_limit": 1},
        "research_messages": [
            [
                {"role": "system", "content": "准备系统"},
                {"role": "user", "content": "准备用户"},
            ],
        ],
        "correction_messages": [
            [
                {"role": "system", "content": "纠错系统"},
                {"role": "user", "content": "纠错用户"},
            ],
        ],
        "correction_basic_example_messages": [
            {"role": "system", "content": "basic纠错系统"},
            {"role": "user", "content": "basic纠错用户"},
        ],
    }

    write_prompt_artifacts(artifacts, tmp_path)

    assert (tmp_path / "plan.json").read_text(encoding="utf-8")
    assert "准备系统" not in (tmp_path / "plan.json").read_text(encoding="utf-8")
    assert "准备用户" in (tmp_path / "research-prepare.txt").read_text(encoding="utf-8")
    assert (tmp_path / "correction-0001.txt").read_text(encoding="utf-8")
    assert "basic纠错系统" in (
        tmp_path / "correction-0001-basic-tier.txt"
    ).read_text(encoding="utf-8")


def test_prompt_artifacts_carry_the_style_block(tmp_path) -> None:
    """Dry run is the product's default mode, so its artifact has to be the
    prompt the run would really send. With style on by default, a resolution
    that happened only on the execution path made the two differ.

    Rendered, not inspected: a source-level guard stays green while the
    rendering is broken (review 2026-09-02)."""

    import inspect
    import json

    from finesub.llm import correction_translation
    from finesub.llm.knowledge.maintain import main as maintain_main
    from finesub.llm.knowledge.node.edit import edit_subject
    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.prompt_artifacts import build_prompt_artifacts

    root = tmp_path / "knowledge"
    assert maintain_main(["--root", str(root), "new", "style", "default_style",
                          "--intro", "默认口味"]) == 0
    repo = KnowledgeRepo.open(root)
    edit_subject(repo, repo.subjects("style")[0], (
        "# default_style\n\n默认口味\n\n## 档案\n\n- [本名] default_style\n\n"
        "## 约定\n\n- [自称转换] 说话者用自己名字自指时改用「我」\n"
    ))

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    artifacts = build_prompt_artifacts(
        stable_json=stable_json,
        knowledge_root=root,
        counter=FakeTokenCounter(),
        style_names=("default_style",),
    )
    system = artifacts["correction_messages"][0][0]["content"]
    assert '<style name="default_style">' in system
    assert "[自称转换] 说话者用自己名字自指时改用「我」" in system

    # ...and the task resolves the selection BEFORE it renders the plan, so the
    # plan and the run get the same names
    source = inspect.getsource(correction_translation._run_task)
    assert source.index("style_selection = resolve_style_selection") < source.index(
        "build_prompt_artifacts("
    )


def test_the_dry_run_shows_the_preparation_prompt_this_run_would_send(
    tmp_path, monkeypatch
) -> None:
    """`build_prompt_artifacts` promises the prompt the run sends. The window
    half branched on the strategy; the preparation half did not -- and the
    production default (an agent preset plus `--media audio`) is exactly the
    split where they differ: a clip makes the window batched while the
    text-only preparation task can be agentic. A user reading the dry run then
    saw "ask for a batch and wait for the fill" for a task that really fetches
    for itself."""

    import json

    from finesub.llm import prompt_artifacts
    from finesub.llm.routing.profiles import resolve_profile

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    strategies = {"research": "agentic", "correction": "batched"}
    monkeypatch.setattr(
        prompt_artifacts,
        "resolve_execution_strategy",
        lambda profile, **kw: (strategies[kw.get("task_group", "correction")], "stub"),
    )
    artifacts = prompt_artifacts.build_prompt_artifacts(
        stable_json=stable_json,
        counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_indexes={},
        knowledge_enabled=False,
    )

    prepare_system = artifacts["research_messages"][0][0]["content"]
    assert "由你自行取得" in prepare_system
    assert "获取缺失输入（本任务没有回填轮" in prepare_system or "本任务没有回填轮" in prepare_system
    # ...while the window, which carries the clip, still shows the batched shape.
    window_system = artifacts["correction_messages"][0][0]["content"]
    assert "获取缺失输入" in window_system and "由你自行取得" not in window_system


@pytest.mark.parametrize("strategy", ["batched", "agentic"])
def test_dry_run_shows_the_none_window_the_run_sends(tmp_path, monkeypatch, strategy) -> None:
    """window-kb-request-channel plan, acceptance 7a: the dry run (the product
    default) and the run read the same `task_graph.window_flow`, so a `none`
    window with a readable base shows its entries-only channel -- or, agentic,
    the knowledge-only tool note -- and the preparation task its pruning keep
    at the request-capable cap."""

    import json

    from finesub.llm import prompt_artifacts
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages import task_graph

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        task_graph, "resolve_execution_strategy", lambda profile, **kw: (strategy, "stub")
    )
    monkeypatch.setattr(
        prompt_artifacts, "resolve_execution_strategy", lambda profile, **kw: ("batched", "stub")
    )
    artifacts = prompt_artifacts.build_prompt_artifacts(
        stable_json=stable_json,
        counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality"),
        knowledge_indexes={"streamer": "- 主播A | | 测试主播", "common": ""},
    )
    window_system = artifacts["correction_messages"][0][0]["content"]
    assert "搜索 query 输出规则" not in window_system
    assert "`web_search`" not in window_system
    if strategy == "batched":
        assert "获取缺失输入（同一任务内" in window_system and "<requested_entries>" in window_system
    else:
        assert "由你自行取得" in window_system and "`kb_search`" in window_system
    assert "<keep_entries>" in window_system
    prepare_system = artifacts["research_messages"][0][0]["content"]
    assert "<keep_entries>" in prepare_system and "单独上限 8 条" in prepare_system
