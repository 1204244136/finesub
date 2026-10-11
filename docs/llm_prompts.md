# Prompt 体系：模板、fragment 与组装

> **前向提示（2026-08-19）**：本文描述的是**今天**的形态。知识库若按
> [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§7 未做与闸门](llm_agent_tool_protocol.md#7-未做与闸门)（设计在维护者的本地归档里）改成「索引必读 + 词条自主
> query」，这里的预注入名额、`<keep_entries>` 透传链与那组注入上限会一并消失。**不要据此实施
> 新功能**，先确认那个改动落地没有。

本文是 prompt 层的现行参考：模板文件清单、`prompt_compose.py` 的组装规则与槽位选择表。运行时行为（注入内容、上限、重试）见 [`llm_harness_behavior.md`](llm_harness_behavior.md)；设计取舍见 [`llm_design_notes.md`](llm_design_notes.md)。

原则：

- **Prompt/harness 文本不硬编码在 Python 里**——全部模板在 `src/finesub/llm/prompt_templates/*.md`（主仓库跟踪），运行时经 `Template.safe_substitute` 加载填充；prompt 迭代不需要改代码。
- **reasoning 全局必须（v17）**：所有 system 模板经 `$reasoning_clause` 要求回复以 `<reasoning>` 块开头；v76 起措辞不再随 thinking 档缩放，BASIC 变体仍使用受限的 8 行版本；缺块不校验不重试。
- **变体优先做整文件 fragment**：结构性差异（有无音频、有无检索）用整个 fragment 文件切换；短语级差异用参数（`$csv_time_note`、`$verify_basis` 等，见 `prompt_compose._MEDIA_PARAMS` / `_TEXT_MEDIA_PARAMS` / `_VIDEO_PARAM_OVERRIDES`）。
- **示例 builder**（v69 起，`src/finesub/llm/example_builder.py` + `prompt_templates/examples/*.md`）：素材只写场景/译文/教学点，行用稳定 `label` 寻址；gap、duration、`char_count`、`{calc:span|dur|gap|cc:...}` 全部由 builder 计算，阈值经 `{thr:}` 指向 `finesub/llm/prompt_constants.py`（**阈值在仓库里只有这一处定义**）。列布局取自 `output_protocol` 的 header 常量。未解析的 `{ref:}`/`{thr:}`/`{calc:}` 是硬错误。**全部示例均由 builder 生成**：43 行主场景是单一素材，输出按 overlay 分组（`nosingles` / `basic`），capableC 的行间 `#` 注释挂在行上、只在该变体渲染；素材可用 `requires: media>=audio` 之类的开关谓词限定适用向量。三份手写 examples fragment、`_add_start_to_example_outputs` 与「9 列→10 列」`replace` 链均已删除。配套 golden 快照在 `test/data/prompt_goldens/`（8 组合 × 4 变体），`manifest.json` 记 `PROMPT_VERSION → {文件: sha256}`——同一版本下内容改了即测试失败，必须 bump。重生成：`pytest test/test_llm_prompt_goldens.py --regenerate-goldens`。
- **槽位注册表**（v68）：开关驱动的 fragment 选择集中在 `prompt_compose.CORRECTION_SLOTS`，一条记录 = `(谓词, fragment, requires)`，首个命中的规则生效。`requires` 是该 fragment 假定存在的注入块；`PromptContext.injected_blocks` 由开关向量推出，二者不符即抛 `PromptAssemblyError`。加档位只改表，不加 if。
- **按传输追加的 fragment（不进 `PROMPT_VERSION`）**：`fragment_conversational_correction_effort_v1.md`
  只在 conversational 路的**纠错** task 上被追加到 protocol 文档末尾（`client._run_conversational_call`）。
  判据与 agent bootstrap 模板相同——它不改输出契约（列、格式、覆盖要求一概不动），只说这条路上
  某一列值得花多少力气，因此既不 bump 版本也不进 `input_hash`。想让某句话对**所有** backend 生效，
  那就是输出契约的改动，要进 `fragment_output_contract_*` 并 bump（2026-08-25 试过一次，见
  [`prompt-iterate.md`](prompt-iterate.md) [§5 失效模式知识](prompt-iterate.md#5-失效模式知识) 的 v78/v78b/v78c 三条）。
- `PROMPT_VERSION`（当前 `zh-subtitle-correction-csv-v93`，定义于 `src/finesub/llm/prompt_compose.py`）：prompt 语义变化时递增。它写入 planning/exchange/session 元数据的 `prompt_version` 字段，并且是**三层恢复身份里唯一同时出现在三层的键**——L1 单次调用 checkpoint、L2 的 `WINDOW_INVALIDATION_INPUTS`、L3 的 `research_reuse_key`（见 `docs/llm_harness_behavior.md` §12）。因此 **bump 会作废已提交的纠错窗口与 research context，这是设计如此**：输出契约本身变了，混合新旧产物没有意义。只影响知识更新路径的 fragment（如 `fragment_knowledge_structure_v1.md`）不进纠错契约，改它不 bump。**准备任务的模板（`prepare_task_*` 与 `fragment_prepare_*`）同理不 bump**：它们只改准备阶段那条对话，纠错契约与 `<context_pack>` 的语义都不随它变；改动本身会因为 messages 变了而自然让 L1 checkpoint 落空，不需要连坐作废已提交的窗口。v65 删除不再使用的 capableA/BasicC 与 JSONL 输出支线；所有单窗口模型调用把目标行重编号为 `1..N`，只读前文按时间顺序编号为 `1-M..0`，validator 后由 harness 映射回稳定源序号；oneshot 同步使用该局部编号。v66 转到开关轴（`--media/--retrieval/--difficulty`）：撤回 text-med 的检索说明特判（它描述了一个永不到达的 `<search_results>`）、reasoning depth 规则化、`$verify_basis` 改为按轴拼接而非整句写死、difficulty 作为 prompt 上限封顶变体。v67 落地 §1.4 的轮结构拆分：研究轮两轮各自按轴组装（`research_round{1,2}_v2.md` 及其 user 模板取代 v1），depth 谓词改为「有无 context_pack」——`retrieval=native` 因此由 high 降 medium，只剩 `retrieval=none` 用 high；无查询轮的窗不再被要求输出 `<keep_entries>`（harness 强制全量透传）。v68 把纠错组装改为**声明式槽位注册表**（`prompt_compose.CORRECTION_SLOTS`）：每个开关驱动的槽位是一张「谓词 → fragment + `requires`」表，`requires` 声明该 fragment 假定会被注入的块，组装时对着 `PromptContext.injected_blocks` 校验——「prompt 提到不存在的输入」从此是组装期错误而非评审事项。同时 `media=video` 有了自己的参数层（不再是「audio 措辞 + 视频补充段」），role/goals 的输入清单与「严防背景污染」段改为按轴拼装（无检索就不写「背景资料」、无知识库就不写「知识库词条」、无音频或无可冲突材料就整段不出现）。v69 起 capableB/basicB 的小例改由 **builder 生成**（见下「示例 builder」）：派生数值与局部序号由机器算，注释里的阈值走 `{thr:}`——迁移过程中查出并修正了三处手算错误（一处 gap、一处 char_count、一处合并跨度）。v73：查询轮 user prompt 的知识输入段（双索引、`<carried_entries>`、剩余额度行）与词条请求规则同谓词——`--knowledge none` 或空库时整段撤除，不再对模型展示「（空）」索引（`fragment_query_{index,carried}_input_v1.md`）；同时 `fragment_merge_rules_nosingles_v1.md` 与 `fragment_output_contract_nosingles_reasoning_v1.md` 中的字面阈值改为 `${thr_*}` 占位符（渲染逐字节不变，纠错 prompt 本体与 v72 相同）。v74：fast round 1 同享该谓词——`--knowledge none` 或空库时，输入清单、背景条目、职责条目、`<requested_entries>`/`<keep_entries>` 输出块与 user 侧的双索引/预注入段整体撤除（背景/职责列表改由组装层按轴编号，`fragment_fast_{entry_blocks,knowledge_inputs}_v1.md`）；两个搜索规则 fragment 里「已注入的知识库条目…」句尾同谓词撤除（查询轮与 research r1 在 local+knowledge=none 下同受益）；knowledge 开启时全部渲染逐字节不变。v75：第四条开关轴 `continuity`（P7c）——`parallel` 下 advice 台账的输入段（`fragment_advice_input_v1.md` 化的 `<previous_advice>`）、`<next_advice>` 输出要求（system 的 advice 槽位、user reminders 第 5 条、feedback 附加块的锚定措辞）与 `<keep_entries>` 全体同谓词撤除；`serial`（含全部 golden 组合）渲染逐字节不变。 v76 将可见 `<reasoning>` 改为不随 thinking 档缩放的中性措辞；v77 增加“内部已推演时只写梳理后结论”的软要求。v78（2026-08-22，知识库真相源切换）：`<kb_entries>` 改为 prompt 投影并带 `@k` 句柄、`N| ` 行号渲染删除，知识提案 op 换 handle 形态（`docs/knowledge.md`）。v79（2026-08-26）：`<kb_entries>` 行内停渲染 membership 句柄 `@m`——现行 op 集合没有消费它的操作，只是逐行噪音 token。v80（2026-08-28，条目结构 §11.2）：term 行改三/四段（`表面形|中文|描述[|别名]`，读音列删除、别名列由 items 渲染），「常被误听」停止注入（仅匹配器消费），streamer preset 增「频道用语」节，知识结构 fragment 同步。v81（2026-08-28，kb-followups A1/A5）：term 行改**固定四列** `源语言|中文定名|别名|描述`（别名第三列恒在、desc 居末可含竖线、恰三列报错），ops 契约补行格式规范与 fewshot worked example（示例由守卫测试喂真实解析器钉住），repair 会话新增 `<candidate_verdicts>` 块。另：知识类 prompt 组装点全部走 `load_prompt_template(strict=True)`——占位符覆盖在**模板层**校验（`Template.get_identifiers()` 减传入键，非空即抛），不扫渲染产物（素材/条目内容里的 `$WORD` 合法），修复了 repair 曾把字面 `$reasoning_clause` 发给模型的静默失效。v82–v88（2026-09-07 起）是知识库类别可扩展与条目结构那一族的 bump（逐条见 `git log -G'csv-v8' -- src/finesub/llm/prompt_compose.py`）。v89（2026-09-30）：`retrieval=none|native` 有可读知识库时窗口也能请求词条，`<keep_entries>` 的槽与 user 侧提醒随之出现在这些格（判据换成 `task_graph.window_requests`）；local 与无知识库的格逐字节不变（纠错 golden 与 local 窗口的请求通道都是）。v90（2026-10-01，`--style auto` 补齐 A）：风格选择改成强制选（`fragment_prepare_style_selection_v1.md` 与新的 `fragment_style_*`），fast 窗口的请求通道在要选风格时改「不需要外部输入也不能直接交付」（`$no_request_clause`），只为选风格开的请求回合有自己的开场（`fragment_window_style_channel_v1.md`），agentic 窗口的交付要附选择；**没有风格目录的格逐字节不变**，纠错 golden 全部同 v89；bump 照常作废已提交的窗口与 checkpoint（设计如此），让带目录的旧请求回合不会按旧契约回放。v91（2026-10-01，知识库 followup 第二批）：只有搜索、没有词条通道的 batched 窗口，回填轮不再要求 `<keep_entries>`；分块准备时每块都带风格目录；知识更新与两个 repair 会话的 op 词表加 `set_intro`（`fragment_knowledge_output_v1.md`；共享冲突修复会话也列，只放行本轮有简介冲突的条目；同一模板还写明 H1 冲突用 `rename_entry` 答，只放行本轮有 H1 冲突的条目）；`move_line` 多一个 `to_entry`（跨条目挪行，合并条目时先挪再 `retire_entry`）；带精修字幕且钉了 style 的知识更新多一个 `<style_marks>` 交付块（`fragment_knowledge_style_marks_v1.md`，其余格逐字节不变）。纠错 golden 与 v90 逐字节相同。v92（2026-10-04，prompt 冷读首轮的修正）：回填轮的交付顺序按变体写（只有要整窗 singles 的变体才提 `<singles>`），请求与风格通道不再点名契约外的块；fast 窗口要选风格时，user 消息的易忘要求第 1 条与最后提醒先说首条回复要交什么（agentic 档说先 `kb_read` 读风格、`<translated>` 之后附选择，`fragment_style_pick_reminder_*`）；特批理由的位置按变体渲染成一句（`CorrectionVariant.approval_reason_slot`），prompt 里不再出现变体代号；capableC 示例的合并与丢弃行补上前置 `#` 注释；窗口请求块的 keep 来源写 `<entry_details>`、准备任务交付轮的 keep 来源含 `<preinjected_entries>`（agentic：预注入的或 `kb_read` 读过的），与实现对齐；「背景调查与前置轮听不到音频」改为只说背景调查；空的 `<pre_round_notes>` 槽整段不渲染。v93（2026-10-05，prompt 冷读第二轮的修正）：缩窄合并示例的时间改成两次合并都在 4 秒硬门槛内（原注释把 4.6 秒说成「在界内」，示例守卫 `test_a_merged_example_row_past_the_hard_span_says_so` 从此钉住）；请求与风格通道说明 `<reasoning>` 是留给模型自己想清楚的地方、想清楚的一两句带过（与 `<window_notes>` 的分工不在这句里明说：明说的那版在 lite 重放里把请求回合的合格率从约五成压到 1/20，owner 2026-10-05 选不写），要选风格时第 1 条把 `<style_selection>` 列进首条回复的块；触发请求回合的条件写成「需要由 harness 代为取得的输入」（native 档不再把联网搜索算进去）；「首条消息」写成「首条 user 消息」；agentic 档的 `<style_selection>` 紧接在 `</translated>` 之后；知识库约定那句挪进 `fragment_background_knowledge_convention_v1.md`。同批不进纠错契约的修正：知识写会话的句柄说明统一（H1 句柄指整个条目，可 `add_item` 登记条目名的误听），结构说明里「可以改用 `split_entry`」那句只在 op 表带 split_entry 时出现（`$split_clause`，冲突回喂与没有拆分提示的知识更新都不再提），agent 外壳只把 payload 里的素材当数据，共享审核写明纯新建术语包无佐证走 tentative、空的 `merge` / `external_evidence` 怎么写。**任务式流程（2026-09-09，阶段 A–E 与删旧流程的阶段 D）没有 bump**：纠错 prompt 本体逐字节不变，窗口的请求通道由 `request_prompts.py` 叠在它之上，准备任务另有自己的模板；v67/v73/v74 提到的研究两轮、查询轮与 fast 融合轮模板已随阶段 D 删除，这里只作历史。
- Prompt 中只用“拉丁字母、数字和标点计 0.5”简述字数规则，不展开控制字符等 Unicode 实现细节；运行时完整口径以 `finesub.subtitles.metrics.weighted_char_count` 及 `docs/llm_harness_behavior.md` 为准。
- **词条 key**：index 行首主 key = 条目 Markdown 文件一级标题（`# 源语言本名`）；`<requested_entries>` / `<keep_entries>` 每行写主 key 或别名即可（详见 `docs/knowledge.md`）。

## 模板清单（按用途分组）

```text
src/finesub/llm/prompt_templates/
├─ 纠错（骨架 + fragment 组装，见下节）
│  correction_main_v1.md                  # 纠错 system 骨架（全部开关向量 + fast 共用）
│  correction_user_v2.md                  # 纠错 user 模板
│  fragment_corr_role_{audio,text,video}_v1.md   # 角色/剪辑说明（video 为 media=video 增补段）
│  fragment_goals_correction_{audio,text}_v1.md  # 纠错目标（听音版/纯文本版）
│  fragment_goals_translation_v1.md       # 翻译与内容取舍（共享；口语颗粒三步判定：内容→保留/机械噪声→压缩/无残值→丢弃，见 llm_design_notes）
│  fragment_csv_input_v1.md               # CSV 输入格式（时间列措辞参数化）
│  fragment_output_contract_v1.md         # 输出契约：BasicA 为带 header、含 start 十列；动态条数只计算字幕行
│  fragment_output_contract_nosingles{,_reasoning}_v1.md # capableB/C：去 singles、带 header 九列；BasicB 复用并动态加 start
│  fragment_weighted_char_count_v1.md     # 字幕加权字数的简短共享说明（运行时公式见 src/finesub/subtitles/metrics.py）
│  fragment_hallucination_v1.md           # 幻觉与丢弃（套话特征、保守保留、丢弃取舍子句仅 audio；$hallucination_handling 分模态）
│  fragment_translated_common_v1.md       # translated 产出纪律（tier 无关：gap 方向、char_count 列纪律、列核对），恒定拼在合并策略片段之前
│  fragment_merge_rules_basic_v1.md         # basicA 保守 1:1 策略（仅词中接回）
│  examples/*.md                           # 所有 oneshot 与小例的素材（builder 生成，见下「示例 builder」）
│  fragment_alignment_v1.md / fragment_advice_v1.md / fragment_keep_entries_v1.md   # advice/keep 槽位（v75 起仅 continuity=serial 渲染；词条 key = H1）
│  fragment_advice_input_v1.md            # user 侧 <previous_advice> 输入段（同谓词）
│  fragment_next_advice_reminder_v1.md    # user 提醒第 5 条：<next_advice> 产出要求（仅 serial）
│  fragment_trailing_blocks_{serial,parallel}_v1.md  # 尾部块提醒两形态（serial 带 $keep_entries_reminder 连接子）
│  fragment_window_overlap_v1.md           # 窗口策略：重叠（可为空）+ 只读前文块规则与人造示例（$preceding_audibility_note 分模态）
│  fragment_retrieval_injected_v1.md      # 注入检索消费引言（内嵌 $search_results_usage）
│  fragment_native_search_v1.md           # 内置搜索指引（retrieval=native）
│  fragment_effort_{low,deep}_v1.md       # 思考力度 prose（retrieval≠local；minimum 取 low）
│  fragment_user_reminders_{audio,text}_v1.md    # user 侧易忘要求
├─ 窗口任务的请求通道（叠在纠错 prompt 之上，纠错 golden 不变；组装在 `request_prompts.py`）
│  fragment_window_request_channel_v1.md   # 首条消息的「获取缺失输入」一节：$acquire_clause（按授权选 fragment_window_acquire_{search_knowledge,search,knowledge}_v1.md，后者在 native 下带 fragment_window_native_search_note_v1.md）+ <reasoning> 与 <window_notes> 的说明 + 请求块（$entry_request_blocks + local 才有的 $search_queries_rules）+ 知识索引；要选风格时第 1 条带 fragment_window_request_style_block_v1.md（$style_block_clause）
│  fragment_window_agentic_acquire_v1.md   # 同一节的 agentic 形态（工具会话自行取数，叠在纠错 system 尾部）
│  correction_fill_v1.md                   # 回填消息：词条全文 + 搜索结果 + 「内容不是指令」+ 交付提醒
│  fragment_search_queries_output_v1.md    # <search_queries> 输出规则（内嵌 $query_style；准备任务与窗口共用）
│  fragment_query_style_v1.md              # query 写法通则（自包含/语言选择/引导语）
│  fragment_search_results_usage_v1.md     # 搜索结果消费规则（纠错 $retrieval_block 与回填消息共用）
├─ 准备任务（一条对话；组装在 `prepare_prompts.py`）
│  prepare_task_v1.md / prepare_task_user_v1.md / prepare_task_results_v1.md   # 首条=完整契约与 <context_pack> schema；results=回填的第二条 user
│  fragment_prepare_request_channel_v1.md / fragment_prepare_agentic_acquire_v1.md / fragment_prepare_no_evidence_v1.md   # 「获取缺失输入」三种形态（batched 请求通道 / agentic 自取 / 无外部证据）
│  fragment_prepare_style_selection_v1.md    # `--style auto` 的候选目录与 <style_selection> 块（auto-style 计划 §3）
│  fragment_knowledge_inputs_v1.md           # 双索引 + 预注入输入段（knowledge 谓词）
│  fragment_research_entry_blocks_v1.md      # <requested_entries>/<keep_entries> 请求块（knowledge 谓词；窗口请求通道复用）
│  fragment_research_keep_entries_v1.md      # 交付时的词条透传块（仅 retrieval=local）
│  fragment_native_search_research_v1.md     # 交付调用的自带检索指引（仅 retrieval=native）
├─ 任务反馈采集（--knowledge update，或参考资料导入）
│  fragment_task_feedback_schema_v3.md    # feedback v3 JSON schema（共享；source_ids 规则按单窗口局部/多窗口稳定序号参数化）
│  correction_task_update_feedback_v2.md  # 纠错窗口采集要求
│  research_task_feedback_v1.md           # 准备任务交付时的采集要求
├─ 统一知识更新（docs/knowledge.md）
│  knowledge_update_artifacts_only_v1.md  # 无精修模式 system
│  knowledge_update_refined_v1.md         # 精修对照模式 system
│  knowledge_update_user_v1.md            # 共用 user 模板（窗口包 + 全局块）
│  fragment_knowledge_update_inputs_v1.md # 两模式共用的输入说明（refined_csv bullet 与库描述参数化）
│  fragment_knowledge_structure_v1.md     # 知识库定位/结构 v2/更新原则（共享；含行文法、吸收判据、迁移指令）
│  fragment_knowledge_output_v1.md        # <knowledge_proposals> op schema（`MODEL_OPS` 里除 remove_item 外的十种）
│  fragment_knowledge_split_clause_v1.md  # 结构说明尺寸纪律里「可以改用 split_entry」那句；只在 op 表带 split_entry 时出现（$split_clause）
│  fragment_knowledge_term_lines_v1.md    # 四列术语行的写法（op schema 与共享冲突修复的 op 小节共用）
│  fragment_knowledge_style_marks_v1.md   # 精修对照模式 + 钉了 style 时的 <style_marks> 契约（knowledge/style_marks.py）
│  fragment_knowledge_style_marks_reminder_v1.md # 同一格最后提醒里补的那半句
│  knowledge_conflict_repair_v1.md        # B' 冲突回喂 user 模板（system 复用上面 artifacts_only 那份）
├─ 知识库维护会话（plan §11.3/§11.4；docs/knowledge.md）
│  fragment_kb_judgment_v1.md             # 每个维护会话都给的判断标准（第 0 条收录标准：只为 ASR 听对 + 字幕写对；同一实体判定；证据纪律）
│  fragment_kb_writing_v1.md              # 只给写条目的会话（修复/材料吸收/共享冲突修复）：有无参照物决定能否写误听、行的归属、描述纪律、标记命名
│  fragment_kb_task_material_v1.md        # 吸收任务的任务段（$material 素材正文 + $user_prompt_block 用户交代；不可信文本边界）
│  fragment_kb_task_candidates_v1.md      # 修复任务的任务段（$candidate_rows + $verdicts 裁定枚举）
│  kb_repair_v1.md                        # 修复任务：二次迁移候选或任意素材 → <knowledge_proposals>（$ops_contract 复用同一份 op schema，$writing 是写条目的规则）
│  kb_verify_v1.md                        # 校验任务：缺证 term/fact claim → <verify_results>（confirmed 必须带 URL / unverifiable 终态）
│  share_review_v1.md                     # 共享审核会话（share/review.py；<review_verdict>）
│  share_conflict_repair_v1.md            # 共享 pull 冲突的修复会话（share/conflict_repair.py；<conflict_verdicts>）
│  fragment_share_conflict_ops_v1.md      # 共享冲突修复的 op 小节：只有闸门放得过的 update/remove/add_item（不给整份 op schema）
│  fragment_share_conflict_{intro,surface}_v1.md # 本轮有简介 / H1 冲突时才插入的 set_intro / rename_entry 说明与行格式
├─ 说话人裁判（--diarize judge-text / judge-audio；组装在 `stages/speaker_judge/prompt.py`，从实验工具 `window_judge_r9.py` 移植；每窗一次调用）
│  speaker_judge_system_v2.md             # 裁判 system：$summary（准备任务的概要）、$marks（置信标说明，开发开关关掉时为空）、$audio（带音频时）
│  speaker_judge_user_v2.md               # 裁判 user：标题、候选人物（准备任务的 speakers、知识库主播、人物，最多 30 个）、轨迹画像（带音频时附布局表）、这一窗的行（▶ 标要判的行）
│  speaker_judge_{audio_note,marks_note,summary}_v1.md   # 上面三个槽位的正文
│  speaker_judge_audio_{layout,silent}_v1.md            # 带音频时 user 里的布局表与「没有声音样本的轨」
│  fragment_prepare_speakers_{duty,field}_v1.md  # 开了裁判时准备任务多交的 speakers（职责第 5 条与 JSON 字段；关着时两个槽为空）
│  # 轨号格、画像行、布局表的行这些逐行短语在 prompt.py 的 PHRASES（同 prompt_compose 的短语层）
└─ 修复轮（validation 失败后的同窗重试）
   fragment_repair_round_v1.md            # 会话无关：列出校验错误 + 要求重出完整结果（$validation_errors）
```

`fragment_repair_round_v1.md` 是唯一**不属于任何一个会话**的模板：它只谈输出契约、不提某个
阶段的块名，因此任何有输出校验的调用都能把自己的错误递回去。它由
`compose_repair_turns()` 渲染成 `assistant`（上一轮输出）+ `user`（错误清单）两轮，
在 `client.complete()` 里追加到已组装好的 messages 之后——**在附件之后**，这样剪辑仍留在
陈述任务的那一轮上。它不进 `PROMPT_VERSION`：attempt 0 的 prompt 一字未改，只有重试多了
东西，所以已提交的窗口和 research context 不作废。

## 组装器（`src/finesub/llm/prompt_compose.py`）

纠错的两个入口按 `TranslationProfile` 选 fragment 并填参；准备任务与窗口请求通道的组装分别在
`prepare_prompts.py` / `request_prompts.py`，按相同开关谓词叠加：

```python
compose_correction_system(profile, *, tier=CapabilityTier.CAPABLE, variant=None,
                          extra_style="", style_block="", knowledge_enabled=True)
compose_correction_user(profile, *, general_context_json, window_context, entry_details,
                        previous_advice, pre_round_notes, search_results,
                        preceding_context_csv, current_asr_csv,
                        current_asr_row_count, tier=CapabilityTier.CAPABLE)
build_window_task_messages(request_channel=..., knowledge_indexes=..., carried_entries=..., ...)   # request_prompts
build_prepare_task_messages(transcript=..., emits_queries=..., emits_entries=..., agentic=..., ...)  # prepare_prompts
```

更高层的 `build_correction_csv_messages` 在 `src/finesub/llm/prompts.py`，负责把 harness 注入内容填进 user 槽位（`build_window_task_messages` 包住它加请求通道与回填消息）；其中 `current_asr_row_count` 由实际窗口片段数计算：basicA 用它锁定 singles 行数，B/C 变体用它重申 translated 必须完整覆盖本窗。纠错窗（请求回合与交付轮是同一份首条消息）的 `<asr_result>` 使用 `local_id|start|duration|gap|text`：目标行每个执行窗口重置为 `1..N`；并列的 `<preceding_context>` 使用非正数，最近前文为 0。模型输出在 validator 后立即映射回稳定源序号，窗口拼接、时间轴、annotated CSV 与知识材料始终使用稳定源序号。

### 纠错 system 槽位选择表（`—` = 空槽塌缩）

槽位由**开关谓词**选，不再按 preset 枚举：

| 骨架槽位 | 谓词 |
| --- | --- |
| `$role_block` | `correction_media≥audio` → audio 版（`=video` 追加 video 增补）；否则 text 版 |
| `$goals_correction_block` | 同上（`correction_media≥audio` 二分） |
| `$retrieval_block` | `retrieval=local` → injected（讲 `<search_results>` 用法）；`retrieval=native` → native；`none` → — |
| `$hallucination_block` 丢弃取舍子句 | `correction_media≥audio`（能重听才谈重听） |
| 模态参数（`$correction_basis`/`$csv_time_note`/`$paren_rule` 等） | `correction_media≥audio` 二分 |
| `$verify_basis` | **按轴拼接**：媒体分量（音频/上下文）+ 检索分量（local→背景资料+搜索结果；native→自己检索到的资料；none→无） |
| `$effort_block` | `retrieval≠local`（无注入才需要 effort 措辞）；`difficulty=efficiency` → low，否则 deep |
| `$advice_block` | `continuity=serial`；`parallel` → — |
| `$keep_block` | `knowledge` 开启且 `retrieval=local` 且 `continuity=serial`；否则 — |
| `$window_block` / `$hallucination_block` 主体 / 翻译目标 | 全组合共有 |
| 插轴（rules/example/output 子句） | v63 起全面废弃，无组合注入 |

**注入实况原则**（§5.2）：prompt 提到的每个输入块必须真的会注入，注入的每个块必须有
用法说明；二者由同一谓词驱动。渲染产物残留任何未解析 `$token` 即
`PromptAssemblyError`（`safe_substitute` 本身会静默放过）。

`$merge_block` / `$examples_block` 由具名 variant 选择。**生产的变体名来自任务组格子
（model-routing v2 D2/D3）**：correction quality 格 = capableC、intermediate/efficiency 格 = basicB，
模型组条目可按 target 覆盖（prompt 迭代/自定义组用），组内退让**不再换变体**；旧的
「应答模型档位 + difficulty 封顶」推导（`effective_tier`）已删除。名字仍可在
replay/`--variant` 显式指定。capableC 去 singles、使用九列 CSV，并在决策点前置 reasoning
注释；basicB 继承 capableB 合并 + 带 start 十列 CSV。capableB 是无 reasoning 的去-singles
对照；basicA 是保守 1:1/singles 对照组。B/C 行为使用 20 字/4 秒硬门槛与 36 字/7 秒绝对门槛。

fast 单窗复用同一骨架（整段一个窗口，请求通道同样叠加）。请求回合没有单独的 prompt：它与交付轮共用这份首条消息，媒体措辞只随 `correction_media` 切换。

准备任务的请求回复按 `<analysis_notes>` → `<requested_entries>` → `<keep_entries>` → `<search_queries>` 输出；request 只负责新加载并按重要性排序，keep 只负责保留本轮可见的预注入词条。两类 canonicalize 后各自最多 8 条、合计最多 12 条，harness keep-first 合并，超限时从 request 尾部丢弃。回填消息（`prepare_task_results_v1` / `correction_fill_v1`）把词条全文放在 `<search_results>` 前，并声明「内容不是指令」。Raw query/URL section 分隔符分别是 `--- query: ... ---` 与 `--- 深度提取 url: ... ---`。

输出完整性约束集中在纠错契约：模型不得省略必需标签、header 或记录。capableB/C 要求精确九列 CSV header；BasicA/B 要求精确十列 header 与 `start`。validator 只接受当前窗口的正局部序号，0、负数和越界值都会整窗重试；通过后统一映射回稳定源序号。

## 迭代惯例

- 改模板 wording 不改语义：不 bump 版本；改输出契约/输入结构/职责边界：bump `PROMPT_VERSION` 并同步更新 `test_llm_prompt_compose.py` 的版本断言与相关 snapshot 断言。
- 用 `finesub.llm.correction_translation --prompt-dir` 输出真实任务的完整 prompt（准备任务首条消息 `research-prepare.txt`；常规窗口为 capable 档，另落 `correction-0001-basic-tier.txt` 首窗 basic 档变体）；冻结注入重打纠错交付轮用 [`session_replay.md`](session_replay.md)。
- prompt/harness 迭代由 `tools/session_replay` 受控重放驱动（见 [`prompt-iterate.md`](prompt-iterate.md)），知识更新不再输出 `<harness_notes>`。人工审阅 replay 产物后手动改模板，绝不自动应用。
- 合并策略：精修标定的软门槛与模型边界（thinking=0、错并代价）见 [`merge-calibration.md`](merge-calibration.md)；
  现行变体契约见 [`prompt-iterate.md`](prompt-iterate.md) [§4 当前合并口径（prompt 层）](prompt-iterate.md#4-当前合并口径prompt-层)。
- 设计过程草稿与过期实验日志在 `docs/archive/`：**在 `dev` 上被跟踪**（clone 与 worktree 都拿得到），由 `scripts/publish-main.ps1` 从公开快照剥掉，所以**不随仓库发布**。
