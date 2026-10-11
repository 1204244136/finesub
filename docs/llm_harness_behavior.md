# LLM 纠错翻译 Harness 行为说明

> **前向提示（2026-08-19）**：本文描述的是**今天**的形态。知识库若按
> [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§7 未做与闸门](llm_agent_tool_protocol.md#7-未做与闸门)（设计在维护者的本地归档里）改成「索引必读 + 词条自主
> query」，这里的预注入名额、`<keep_entries>` 透传链与那组注入上限会一并消失。**不要据此实施
> 新功能**，先确认那个改动落地没有。

状态：实验性实现。默认只生成计划和中文 prompt；真实 API 调用必须显式使用 `--execute`。


> **本文拆出了两块**（2026-08-16）：dev 侧的模型事实/池/路由链/思考档位/限流去了
> [`llm_harness_routing.md`](llm_harness_routing.md)，本地检索代理与背景调查去了
> [`llm_harness_research.md`](llm_harness_research.md)。其余仍在本文。

## 开关轴（--media / --retrieval / --difficulty / --continuity）

正交开关轴取代了原先的严格 6 档 preset（`--route`/`--level` 已退役；设计意图见
`docs/llm_design_notes.md`，未完成的标定见 [`llm_followups.md`](llm_followups.md)，实现在
`src/finesub/llm/routing/profiles.py`）。
**媒体只有一个开关**（2026-08 曾按 model-routing v2 D20 拆成 `--correction-media` / `--planning-media` 两个按任务的开关；
查询轮并入窗口任务后只剩纠错窗读媒体，2026-09-09 合回一个；代码里的字段仍叫 `correction_media`）。

| 轴 | 取值 | 含义 |
| --- | --- | --- |
| `--media` | `text` / `audio` / `video` | **纠错窗**（请求回合与交付轮同一条对话）看到什么；阶梯，video 蕴含 audio；窗口几何（剪辑占的输入 token）、纠错 prompt 的媒体措辞与 fragment 选择读它，输出系数不读 |
| `--retrieval` | `none` / `local` / `native` | `local` = 整套 harness 注入（准备任务的回填 + 每窗请求回合 + 本地搜索代理）；`native` = 模型自带搜索工具；`none` = 无检索 |
| `--difficulty` | `quality` / `intermediate` / `efficiency` | 按**想要什么**命名（2026-08-12 由 high/med/minimum 改名，与 thinking 档位区分开）：选该格的 prompt 变体与思考旋钮，预设还可按档位绑不同模型组；`efficiency` 是最省的可用形态（钉死两个 media 开关 = text、retrieval=none；knowledge 与风格封顶在 `read`：默认照常读，`update` 拒绝——owner 2026-10-01） |
| `--continuity` | `serial` / `parallel` | 窗口连续性（v75）：`serial`（默认）保留窗口间链式上下文（advice 台账、词条透传链）；`parallel` 放弃它们、并投纠错窗换墙钟——两阶段一屏障（全部请求回合并投 → 会话级词条集一次定死 → 全部纠错窗的交付轮并投 → 按 chunk_id 有序合并），配 `--parallel-windows`（默认 **1**，owner 2026-08-30 由 4 改：任务内并行有质量与 token 效率代价，任务间并行没有——见下方「任务级并行与 agent 槽位预算」；预设可带它的默认值，出厂的 `agy-fast` / `fast` 写 4，只在 parallel 下生效）。prompt 侧 `<previous_advice>`/`<next_advice>`/`<keep_entries>` 整体撤除；失败 drain-then-raise（所有跑完的窗都进缓存后才抛错）、同批 **3 次会话链耗尽**即熔断（计的是**链**不是窗口，2026-08-19；它并不保证一个批次待在日额度以内，为什么这样定见下方「重试与拼接」）；缓存记录带 `continuity` 与去词条核心哈希，**只允许 serial→parallel 方向复用**；key 的用法与 serial 同构（2026-08-19 起并行不再额外钉 key，只有带媒体的调用钉一把，见 [`llm_harness_routing.md`](llm_harness_routing.md)） |

`--knowledge {none,read,update}` 是独立的任务级三态开关，不属于
`TranslationProfile` 的四轴向量；它控制知识输入、反馈采集与任务后更新，详见
[`knowledge.md`](knowledge.md)。

输出系数按开关合成：`c = 3.5 + 1.0(retrieval≠none)`。媒体不进系数（owner 2026-09-30）：剪辑只占输入，
实测输出长度主要随模型与推理强度变化。
**difficulty 不再进系数**（model-routing v2：efficiency 取消 -1.5 减免，text/none/efficiency 的 c 从实测的
2.0 提到 3.5、窗口容量降至 57%）。系数不按开关组合逐个标定，也不在运行时提示（owner 2026-09-30）。
difficulty 只做两件事：选 prompt 变体 + thinking level。
「开关高于实际可用媒体」是入口处的配置错误（缺 `--audio`/`--video` 直接报错），不做静默降级；
运行期仅剩一级 video→audio 安全网（见「client 层」）。

退役 preset 的换算（仅用于读旧产物；`finesub.llm.routing.profiles.parse_profile_id` 仍能解析这些旧名）：

| 旧 preset | media | retrieval | difficulty | c |
| --- | --- | --- | --- | ---: |
| text-low | text | none | efficiency | 3.5（原实测 2.0，model-routing v2 取消该档减免） |
| text-med | text | none | quality | 3.5 |
| text-high | text | native | quality | 4.5 |
| mm-low | text | local | quality | 4.5 |
| mm-med（两个入口的默认） | audio | local | quality | 4.5 |
| mm-high | video | local | quality | 4.5 |

- 纠错窗归 `correction-mm` / `correction-text` 任务组（由 `correction_media` 选；role 只是产物标签）。**`retrieval=local`** 的定义特征是
  harness 侧外部注入：准备任务与每个纠错窗的请求通道里都有本地搜索（`media=text` 时
  无媒体但保留注入）。**`native`** 不用本地搜索代理：准备任务的交付调用与纠错调用自身开模型的
  搜索工具（Gemini 3 免费层 grounding 429）。**`none`** 两者都没有，准备任务只整理素材与词条。
  逐档的请求通道见下方「准备任务与请求通道」。native search 是**按调用请求的能力**而非角色（D4，v1 的独立 native 链已删）：任务组不变，`client.complete(..., native_search=True)` 在**当前绑定的模型组内**按 `supports_native_search` + target 是否声明搜索工具过滤。出厂预设由付费 3.7 Flash 接地（其 target 直接声明 `google_search`，非 native 调用不发该工具）；免费档 3.x 完全不能联网，唯一能联网的免费模型 2.5 Flash 低于纠错/知识下限，因此只作为打包的 `gemini-native-search` 组供显式绑定。过滤后为空即报错，不做静默降级。
- 预期输出估算 `k × c × csv_tokens`（k 为 `--output-scale`，默认 1.0；调大 k 切出更小窗口）；常规窗口须满足 `≤ 0.9 × 65536 − 5000 = 53982`。
- **insert/插轴已于 v63 全面废弃**（所有路线、所有变体均不再注入或接受 insert 行；校验层视 insert 为结构性错误、同窗重试）。
- 纠错调用 thinking：来自预设级思考旋钮（出厂实况见「LLM thinking effort」的表）。
- 纠错 `<reasoning>` 措辞**不再分档**（2026-08-12）：所有会话用同一句中性措辞。这段 prompt 的
  用途是「模型自己的思考没发生或退化时」（无原生思考的模型——现已近乎绝迹——以及 flash-lite
  系列跳过思考阶段），靠显式可见推理兜住质量；这个用途不随思考旋钮缩放，而旧的
  `(difficulty, retrieval)` 分档编码的是一个**内容**理由（没有 context pack 就多推理），且从未
  标定。`basicA` 的 `bounded` 变体保留——它守的是实测过的失效模式（弱模型用计划代替产出）。
  改成「按实际注入了什么」来变的措辞是 prompt 迭代项，见 [`llm_followups.md`](llm_followups.md)。
- 两版措辞都带一条**软要求**（2026-08-12）：「**若你已在内部思考（thinking）中推演过，块内
  不必重复推演，写梳理后的结论即可**」。可见块的职责是在思考未发生或退化时兜底，思考真发生了
  就没有把同一套推演再写一遍的理由；用「不必」而非禁止，因为没有原生思考的模型仍须在此推理。
- 「不要逐行预演输出」是**纠错家族已有**的要求，共四处：三个 output-contract fragment、
  correction user 侧的最后提醒，以及 `bounded` 版自己的禁止条款。它不进通用措辞——研究/judge/
  知识更新根本没有字幕行可预演。

### 会话清单（触发谓词 → 路由链）

一次 run 最多存在 **4 类** LLM 会话（2026-09-09 起的形态：
旧的研究两轮、搜索 loop、每窗查询轮与 fast 融合轮都已删除）。下表的「触发」是代码里的真实谓词而非经验描述；
会话按 **任务组 × difficulty** 选模（v2：预设把格子绑到模型组，未绑格难度回落，见「模型事实、模型组与预设」），
thinking 与变体来自格子，输出契约见 `finesub.llm.session_contract.SESSION_CONTRACTS`。

| 会话 | 触发谓词 | 任务组 | 媒体 | 次数 | 输出校验 |
| --- | --- | --- | --- | --- | --- |
| 准备任务（一条对话：请求批 → 回填 → 交付，或直接交付） | 非 fast 恒跑（`_prepare_stage_runs`） | `research`；`retrieval=native` 时按能力过滤组内成员 | 纯文本 | 每 run 1 条对话（1–2 次调用 + 解析重试）；超长素材按块各 1 条 | `prepare_task_request` / `prepare_task_delivery` 契约，按回复里有无 `<context_pack>` 分流 |
| 纠错窗（含请求回合） | 恒跑 | `correction-mm` / `correction-text`（按 `correction_media`） | 按 `correction_media`：`≥audio` 带本窗剪辑（`video` 带 mp4） | 每窗 1 条对话：`retrieval=local` 且策略 batched 时首条消息带请求通道（直接交付即 attempt 0，否则回填后交付），其余档位首条消息就是交付轮；+ 校验重试 | 请求回合 `correction_request` 契约；交付 `output_protocol`（不进 SESSION_CONTRACTS） |
| fast 单窗 | fast 启用 | 同「纠错窗」 | 同上（整段剪辑） | 1 条对话（唯一窗口，同样带请求通道；不跑准备任务） | 同上 |
| 统一知识更新 | `--knowledge update`（且 `--execute`） | `knowledge` | 纯文本 | 每 chunk 1 次（`knowledge-update-chunk<NN>`） | 自有 parser |

会话选模型的三条规则（v2）：

- **任务组由会话（与 media 开关）决定，模型组由预设的格子绑定决定**。角色
  （`audio_multimodal` / `general_capable`）只剩 artifact 标签。`retrieval=native` 是 per-call 能力（D4）：
  在选中的模型组内按 `supports_native_search`（且该 target 声明了搜索工具）过滤，
  **过滤后为空即报错**，不做静默降级，也没有独立的 native 链。出厂预设可用：付费
  3.7 Flash 的 target 直接声明 `google_search`（非 native 调用不发该工具），免费 3.x 不能
  联网，因此纯免费档跑 `--retrieval native` 会在启动校验处失败——要么开付费档，要么显式绑定
  打包的 `gemini-native-search` 组（含免费 2.5 Flash，但它低于纠错/知识下限）。
- **native 检索的出处记录**（2026-08-15）：Gemini 应答里的 `groundingMetadata` 会被解析成
  检索 ledger 的同一形状（`{tool, query, queries, urls}`），挂在应答的那次 execution
  attempt 的 `search_events` 上，因此 **task artifacts** 里能看到模型跑了哪些 query、
  哪些页面接地了答案。**exchange 里看不到**：`## Execution Attempts` 表是固定列的，
  `search_events` 不在其中——三家 agent 的检索事件同样只落 artifacts，这里保持一致。
  **每次调用一行、不按 query 拆**：Gemini 从不说哪个页面回答了哪个
  query，拆开等于替它做一个它没做的归属。三家本地 agent 里 Codex/Claude 有逐 call 的完整
  URL provenance，**agy 只报 query、不报 URL**——做 native/local 对照时这条不对称必须算
  进去，`research._mark_unverified_sources` 也只对 agy 这类零 URL 的后端加
  `verified: false` 标记。
- **prompt 变体来自格子（可被模型组条目覆盖）**。`correction-*` 的 quality 格是 capableC、
  intermediate/efficiency 格是 basicB——旧的「按应答模型档位 + difficulty 上限」推导
  （`effective_tier`）已删除：组内退让不再换 prompt，链内不再有静默降质；
  `retrieval=native` + `difficulty=quality` 也拿 capableC。逐窗缓存记录实际使用的变体名。
- **agent 靠成为模型组的成员参与，policy 只是 backend 闸门**（2026-08-14）。此前 policy 会按
  任务组把 agent 组 prepend 到每个格子上；现在它只能否掉某个组已经列出的 backend，永远不能加上
  一个组没列的。出厂绑定 agent 的预设是 `agy-hybrid`（`agy-capable` / `agy-basic`，免费 API 打头）
  与 `agy`（`agy-only-*`，名单里没有 API 成员）；Codex/Claude
  的 target 仍然声明着，用户列进自己的 `[llm.model_groups]`、或把格子直接绑到 target id 上
  （快速选模型）即可用。带剪辑的纠错窗（含 fast 单窗）仍会在能力过滤时跳过纯文本的
  agent 候选（Codex `supports_audio=false`，agy 的 Claude 同样是纯文本）。

### 准备任务与请求通道（按轴拆分）

准备阶段是**一条逻辑对话**（`stages/prepare_task.py`，方案 §10.3）：首条消息就是完整任务与交付契约，模型要么直接交付
`<context_pack>`（一次调用），要么先提**一批**请求；harness 执行后把结果作为下一条 user 消息回填（模型自己的请求回复作为
assistant 消息放回历史，REST 适配器因此仍只发一段合法历史），第二条回复必须交付——再提请求按解析失败处理、走同一重试预算，
**不会有第二次回填**（§2.2：有界、无依赖猜测）。能力只改**请求通道**，不删任务。下表的「有知识库索引」= `knowledge_index_available()`
= `--knowledge read|update` **且** 索引非空；`--knowledge` 默认为 `read`，所以默认 run 在库非空时走「是」行，空库或 `--knowledge none` 时走「否」行：

| retrieval | 有知识库索引 | 请求通道里有什么 | harness 回填 | 交付 |
| --- | --- | --- | --- | --- |
| `local` | 是 | `search_queries` + `requested_entries`/`keep_entries` | 一轮本地搜索 + 查 KB | `<context_pack>` |
| `local` | 否 | 仅 `search_queries` | 一轮本地搜索 | `<context_pack>` |
| `native` | 是 | 仅 `requested_entries`/`keep_entries` | 查 KB | `<context_pack>`；交付调用自身开原生搜索工具 |
| `native` | 否 | 无请求通道 | — | 一次交付（原生搜索） |
| `none` | 是 | 仅 `requested_entries`/`keep_entries` | 查 KB | `<context_pack>` |
| `none` | 否 | 无请求通道（prompt 要求 `sources` 留空、推断进 `uncertainties`） | — | 一次交付 |

`<analysis_notes>` 只在请求回复里要求（≤1500 token，它是回填后交付的起点，不注入纠错窗）；直接交付的回复只要 `<reasoning>` + `<context_pack>`。
`--extra-info` 里的 URL（≤8 个，`retrieval=local`）在首条消息前预提取成 `<note_url_extracts>`；备注里命中 index key/alias 的词条预注入（≤8 条）并可被 `keep_entries` 保留，直接交付时同样进种子。

**窗口怎么取缺的输入**由 `task_graph` 一处回答：`window_requests` = 有东西可请求（`retrieval=local` 的 harness 搜索，
**或**知识开启且索引非空），与执行策略无关；`style_pick` = fast 窗口要先选 `--style auto` 的风格（fast、auto、目录里有
`default_style` 之外的候选，由编排层回答，2026-10-01）。`window_flow` 在两者之上给出 `request_channel` = batched 且
（`window_requests` 或 `style_pick`）——harness 的请求回合——与 `agent_fetches` = agentic 且（`window_requests` 或 `style_pick`）
——执行者自己的工具会话。`window_requests` 本身就是「丢掉的词条能不能再要回来」：透传链（`CorrectionRun.window_requests`
直接取它，不从通道布尔量反推——只为选风格开的通道要不回词条）、准备任务的下游额度读它；fast 闸门读「`window_requests` 或
有风格目录」（一律按 batched 形态估、刻意不把路由拉进 planner），运行与 dry-run 读 `window_flow`。

**无请求能力的窗强制保留词条**：`window_requests` 为假时（没有可请求的东西），准备任务与纠错窗
都不再被要求输出 `<keep_entries>`（prompt 里根本没有这个块），harness 直接透传当前全集，透传上限也从
`KB_TRANSFER_MAX_ENTRIES`(8) 提到 `KB_WINDOW_TOTAL_ENTRIES`(12)——给「下一窗再补」留的格子在这条路线上纯属白丢词条。
同理准备任务的请求额度在这些档位下用 12 而非 8（`finesub.llm.prompts.r1_request_cap`）。2026-09-30 起 none/native
有可读知识库时也能请求（见下方「纠错窗口调用形态」），所以这一格与 local 一样是 8 + `<keep_entries>`（owner 2026-09-30）。
⚠ **agentic 是例外，在任何档位上都要 `<keep_entries>`**：harness 没有注入过任何东西可以透传——
执行者是自己用 `kb_read` 读的库——所以不由它点名，后续窗口就什么都拿不到。前提仍是索引可读：
`--knowledge none` 下没有这个块，harness 也不会去解析它（KB 关闭不读正文）。准备任务的 `emits_keep` 因此是
`(window_requests or 准备任务自己是 agentic) and 索引可读`。

⚠ **改这一族的准入与可见集判据之前**，先读这四条（2026-09-09 分支复审留下的）：

1. **复用一个判据就复用整条。** 同一个错在这一族犯了五次：往条件里加一个 `or`（`or agentic`、
   `or run.agent_fetches`），删掉的是它今天为假所靠的隐含前提——比如「KB 关着就没有可见集」。
   扩一个判据之前先问**它今天为假靠的是谁**；**挪一个判断的位置和改它的内容一样危险**
   （第五次是改了它与 fast 早退的先后，`git diff` 里看不出）。
2. **通则不能替代例外条款。** 驳回一条引用了某节计划的复审意见之前，先去读那一节。
3. **修复自己会长出缺陷**，而且集中在新加的机制里：三十条修复里有七条是上一次修复引入的。
   补丁比原代码更值得一条回归测试。
4. **被点名的症状往往只是一格。** 照症状打补丁，会把同一个洞留在没被点名的档位上。

**纠错窗口也是一个任务**（§10.4，形态见下方「纠错窗口调用形态」）：`request_channel` 为真时首条消息就是纠错契约加
「获取缺失输入」一节（`request_prompts.py` 叠在纠错 prompt 之上），模型要么直接交付终稿
（作为 attempt 0 验收，不再调用），要么一批请求；harness 查库（+ local 下一轮搜索）后回填，交付轮与修复轮都在这条对话里。
请求回合跑在**纠错 cell** 上（同一模型组、同一媒体）。`native` 的窗两轮都设 `native_search=True`（与准备任务一致，owner 2026-09-30）。

### 开关组合 → 实际会跑的会话

一次 run 的会话集合只由 fast 与 `--knowledge update` 决定（`task_graph.plan_task_graph`：普通 = `prepare` + `translate`，
fast = 单个 `whole` 的 `translate`，update 追加 `knowledge_update`）；`retrieval` × 有无索引只改各任务的**工具授权**
（请求通道里有什么），media 开关只决定纠错窗挂不挂剪辑，`difficulty`/`continuity` 只改 prompt 与调度，都不增删会话：

| retrieval | 索引 | 准备任务的请求通道 | 纠错窗（batched）的请求通道 |
| --- | --- | --- | --- |
| `local` | 有 | queries + entries | notes + entries + queries |
| `local` | 无 | 仅 queries | notes + queries |
| `native` | 有 | 仅 entries（原生搜索） | notes + entries（原生搜索） |
| `native` | 无 | 无（一次交付，原生搜索） | —（纠错调用自身原生搜索） |
| `none` | 有 | 仅 entries | notes + entries |
| `none` | 无 | 无（一次交付，无外部证据） | — |

`retrieval=none` + `--knowledge none` 下准备任务照跑（旧流程的「整个研究阶段不跑」这一行已不存在）：它只整理素材与用户信息，
`sources` 留空。统一知识更新与本表正交：只看 `--knowledge update`，在任务结束后按 chunk 跑。

执行策略按任务解析（§10.5，`task_graph.resolve_execution_strategy`）：**agentic** 要两个条件同时成立——任务的链全是
`local_agent`，**且这个任务的 agent 调用真的会拿到 harness 工具**（`capabilities.chain_serves_tool_session`）。满足时准备任务与窗口
都不再由 harness 代取，prompt 改为「用授权的工具（`web_search`/`web_fetch`、`kb_*`）自己取、一次回复直接交付」。任一不成立则为
**batched**（上面的请求回合）。⚠ 第二个条件是必要的：传输由**会话档位、是否带媒体、driver 认不认 MCP** 三者派生
（[Agent 工具协议](llm_agent_tool_protocol.md) [§1 档位与形态](llm_agent_tool_protocol.md#1-档位与形态)），三条里任一不成立就走 capsule、那里没有工具——只看 backend 就选 agentic，
会让 harness 收起自己的请求回合、再叫一个没有工具的模型自己去搜，`retrieval=local` 就此静默变成 `none`。所以：
`correction_media != text` 的窗口任务永远是 batched；`api`/`resume` 两个档位永远是 batched；`per-window` 还要该 target 的 CLI
真的接受 per-invocation MCP server（`supports_mcp_config`，探一次 `--help`，按 target 记忆化，探不到就当作不支持）。
链上**每个**可能应答的 agent 都要满足——回退到链尾的那个也会读到同一句「你自己取」。
`FINESUB_TASK_STRATEGY=batched|agentic` 可钉死（开发用，优先于下面两条），`--test-profile` 为 batched。策略、来源
（`correction_chain` / `test_profile` / `env_pin` / `override`）与任务列表记在 `task_graph` artifact。

## 快速模式（--fast auto|on|off，默认 auto）

短输入把整段作为**单个窗口任务**处理：不跑准备任务（没有 context pack），首条消息就是纠错契约加请求通道
（`request_channel` 为真时：local，或知识开启且索引可读），按 `correction_media` 附加无媒体、音频或音视频整段剪辑；知识开启且索引可读时请求通道里可请求词条；
`--extra-info` 作为 `general_context.user_notes` 直接进窗口 prompt（`stages/plan.fast_context_pack`，没有别的环节消化它）。
`retrieval=native|none` 的 fast run 同样是单窗任务；没有可读知识库时没有请求通道。

auto 判定三个条件都过才启用（结果与数值写入 `fast_decision` artifact）：

- 输出：`k × c × 全量 csv_tokens ≤ 0.8 × 65536 − 10000 = 42428`；
- 输入：首条消息（countTokens，按请求通道形态组装）+ 剪辑媒体 token ≤ `prompt_input_limit − 56000`
  （`FAST_FILL_INPUT_RESERVE_TOKENS`，为回填轮的搜索结果与词条注入留余量；包络由绑定组的 catalog 行算出，免费 Gemini 上仍是 194000）；
- 质量护栏：整段 `<asr_result>` ≤ `max_window_subtitle_tokens`（默认 10,000，见「窗口拆分」）。
  快速窗口按定义就是全片，是最容易撞上这条的路径，所以它与两个预算条件同级参与判定。

⚠ 输入条件一律按 batched 的请求通道形态组装首条消息。agentic 执行者（全 agent 链）的整段窗实际发的是「自己取」那段
说明、且不带知识库索引，所以它的首条消息被**多算**：实测多 956 字符（空库）、约 2.8k（50 条索引）、约 6.7k（200 条），
对 138k 预算约 2%。多算的方向是安全的（退回常规多窗流程），要算准就得把路由与 driver 探测拉进 planner，所以不改
（2026-09-10 量过，owner 2026-09-30 结案）。只在全 agent + 纯文本 + `--fast on`、输入正好卡在线上几 k 之内时会报错。

`--fast on` 不满足即报错退出；`--fast off` 强制常规多窗流程。快速模式不写 `*-research-context.json`（没有准备任务）；
`--context-file` 传入的现成上下文照常注入。

## 输入与输出

输入以 `*-stable.json` 为字幕源。纠错翻译阶段会保留下列中间 artifact：

- `*-raw.srt`：从 `stable.json` 渲染出的完整 ASR 原文 SRT，便于人工排查；若请求的最终 postprocess profile 包含时间轴步骤（当前为 0/1），仅同步执行重叠修复 + 末端延长/闪轴闭合（profile `4 → 1`），文字保持原样。
- `*-corrected.srt`：从模型输出的 `corrected_text` 列渲染出的纠错后原文 SRT，便于分析 ASR 修正和误听模式。
- `*-translated.srt`：模型输出的中文字幕直出版本，尚未做最终 SRT 后处理。
- `*.srt`：最终中文字幕 SRT；默认由 `*-translated.srt` 经过 profile 0 后处理得到。
- `*-research-context.json`：准备任务的产物（context pack + 请求/交付两条回复的输出（含 analysis_notes）+ planning 元数据 + token 报告），写在 task artifact 目录下（如 `input.llm-artifacts/input-research-context.json`）；旧 run 若仍在 SRT 同级，首次触达时会迁入 artifact 目录。

纠错窗口输入格式是：

```text
global_id|local_start|duration|gap|text
```

时间单位为秒，展示到 0.1；`local_start` 以本窗口**剪辑音频的 0 秒**为基准，每条 CSV 记录占一个物理行。传给多模态纠错模型的音频是原始音频（不是人声分离后的 `*-vocal.ogg`；旧产物可能为 `*-vocal.flac` 兜底格式）按窗口裁剪出的 mono-16k AAC 片段（ffmpeg：`-c:a aac -ac 1 -ar 16000 -b:a 32k`）：范围是窗口首条字幕头到末条字幕尾，两端各加 5s padding；含全局第一条字幕的窗口头部 padding 为 60s（clamp 到 0），含全局最后一条字幕的窗口尾部 padding 为 60s（clamp 到音频末尾）。剪辑写到任务 artifact 目录下的 `clips/<chunk_id>.aac`（每次覆盖；随 artifact 目录一并清理），每个执行窗口（含 `-a`/`-b` 半窗）单独上传一次并在请求回合/交付轮/同窗重试间复用。窗口规划完成后即在后台线程预切首窗；进入窗口 *i* 时预切 *i+1*，切完即上传 Gemini，主循环仅在取当前窗 file ref 时等待就绪。`*-translated.srt` 时间轴由本地按源序号从原始 `stable.json` 回填，因此保留原始高精度。最终中文字幕 SRT 使用 `translation` 列再做后处理；`-corrected.srt` 使用 `corrected_text` 列。

默认命令只写出计划和 prompt（准备任务的首条消息 + 每窗口纠错 prompt）：

```powershell
python -m finesub.llm.correction_translation out/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts
```

执行 API 时默认串联：准备任务 → 纠错窗口循环（`--execute` 必须提供 `--audio`；旧的 `--audio-file-id` / `--upload-audio` 已随按窗剪辑上传移除）：

```powershell
python -m finesub.llm.correction_translation out/input-stable.json --audio data/input.wav -o out/input.srt --execute
```

这个 CLI 与管线的 `translated-srt` / `final-srt` 阶段、reference ingest 走**同一个任务**
（`run_correction_task`）：同样按存在性跳过已有的 `-translated.srt` / 最终 SRT（欠着的后处理与
`--knowledge update` 照做），同样把 `llm_harness` 计时写进 run metadata、在会话关闭后写任务报告。
要重跑就删产物，与管线一致。

相关 CLI 参数：

- `--media {text,audio,video}` / `--retrieval {none,local,native}` / `--difficulty {quality,intermediate,efficiency}`：开关轴（两个入口没写时取激活预设的默认值，出厂 `default` 下是 `audio/local/quality`；管线 CLI 的 `--llm-media video` 在纯音频输入上自动降到 audio 并提示）。`difficulty=efficiency` 钉死 `media=text` + `retrieval=none`（命令行没写的直接取钉死值，写了冲突值报错）；**`--knowledge` 与 `--style-mode` 在该档封顶在 `read`**：未写时照常读（知识库与风格都注入），显式 `update` 报错，预设或配置给的 `update` 降为 `read`（2026-08-12 整体禁用，2026-10-01 改为默认只读：粗纠最需要专名对；该档思考旋钮全为 low，不该往知识库里写）。**difficulty 只做两件事**（model-routing v2）：选任务组格子的 prompt 变体（出厂：纠错 quality→capableC、intermediate/efficiency→basicB）和该格的思考旋钮；它不再从应答 endpoint 读 capability tier（该列已删），也不再影响窗口几何。预设可以**按档位绑不同模型组**，出厂纠错就是这样（quality=capable 组、intermediate=两个 lite），所以「切到 intermediate 继续」是用户显式发起的降档，而非链内静默降质。（生产 CLI **没有** `--variant`：变体由任务组格子决定；按名覆盖只存在于 `tools/session_replay`，见 docs/prompt-iterate.md。）
- `--parallel-windows N`：`continuity=parallel` 的最大并投窗口数；没写时取激活预设的 `parallel_windows`，否则 1（2026-08-30 起；任务级并行是首选的并行面）；串行模式忽略。
- `--output-scale K`：输出估算系数 k（默认 1.0）；调大切出更小窗口。
- `--fast {auto,on,off}`：快速模式开关（默认 auto）。
- `--video PATH`：源视频文件；`--media video` 时 `--execute` 必填（见「视频」）。
- `--extra-info` / `--extra-info-file`：用户提供的额外信息（来源 URL、内容说明、额外要求），注入准备任务；fast 模式下作为窗口的 `user_notes` 直接注入。
- `--context-file <research-context.json>`：复用已有准备结果，跳过准备任务。
- `--research-only`：只跑准备任务并写出 `research-context.json`，不进入纠错。快速模式没有准备任务，所以这时 `--fast auto` 落成 `off`，`--fast on` 报冲突（`fast_mode_for_research_only`）。
- `--extra-style`：注入纠错 system prompt 的特殊翻译风格段落（默认为空）。

## 纠错输入的段序列

`chunking.load_segments_from_stable_json` 是纯加载：拿到的 stable JSON 已经是最终段序列，
源序号按位置顺次编号。曾经在这之前还有一道确定性预合并（`src/premerge.py` / stabilize
profile 3），2026-07-29 随 `segment_split` 迁到全局 DP 一并删除——分句器自己决定 ASR 段接缝
的去留，词中切断的碎片不再产生，预合并因此无对象可并。

**为什么这么定**（M.1–M.10 十条决策，含被证伪的假设与教训）见
[`llm_design_notes.md`](llm_design_notes.md)「已退役的 capability tier 与确定性预合并决策记录」
——决策台账只在那一处，这里不复述。

## 窗口拆分

Harness 采用"先估算窗口数、再均匀放置分割点"的规划方式：

1. 对全量 CSV 做一次 countTokens，按每行字符占比折算每段文本 token；每段媒体 token 按 `媒体速率 × 到下一段开始的时间跨度` 折算（`media=audio|video` 含音频 32 tok/s，`media=video` 再加画面 17.75 tok/s；`media=text` 为 0），得到每段的规划质量（mass）与前缀和。
2. 由输出约束（每窗字幕 token ≤ `窗口输出预算 / (k × c)`，其中窗口输出预算 = `0.9 × 65536 − 5000 = 53982`，k 为 `--output-scale`、c 为开关合成的输出系数；默认 `audio/local/quality` 算出约 11,996，实际由下面第 4 步的质量护栏 10,000 定）与输入约束（字幕+媒体+上下文 ≤ `prompt_input_limit`，免费 Gemini 上是 `194000`；含重叠与 padding 的固定加成）估算窗口数。规划时固定预留 `72000` tokens 的上下文额度（`WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS`），覆盖纠错调用中窗口 CSV/媒体之外的全部内容：静态 system prompt（实测 ~4k）+ user 脚手架 + 调查 context pack + serial advice 台账（≤8k）+ 请求回合 notes + 搜索结果块（≤20k）+ 知识库词条块（≤28k），最坏合计约 69k；窗口实际由输出公式或质量护栏限死，加大 reserve 几乎不改变窗口数。
3. 在均匀 mass 目标点附近（半径约 `0.4·n/k`，由近及远）snap 到合适边界；每个规划窗口再用真实 countTokens 预算校验，任一窗口超限则 `k+1` 全局重排（保持均匀），上限 `k0+16` 后报错；单段放不下直接报错。发生过重排（输入超预算导致窗口缩小）时写入 `window_plan_report` artifact（`estimated_windows`/`planned_windows`/`replan_attempts`/最后一次超限错误，分 research/correction 两个 phase），task report 渲染为独立 "Window Planning" 小节。
4. **质量护栏 `max_window_subtitle_tokens`**（`ModelLimits` 默认 10,000；config.toml `[chunking]` 可覆盖，`0` 关闭，非法/负值硬报错）：单窗 `<asr_result>` CSV（正文 + 重叠行）的 token 上限，独立于输出系数——窗口太长时翻译质量会掉，哪怕输出装得下。两处生效：第 2 步的窗口数估算取它与输出约束的较小者，第 3 步的真实 countTokens 校验后再硬查一次，超限走上面同一条 `k+1` 重排路径。
5. research 与 correction 可以采用不同窗口几何；research 笔记按 source-id 区间重映射。
   `research-context.json` 仍带完整 `planning` 元数据供审计，几何不再参与复用比较。

每个窗口包含：

- 本窗口的输入源序号范围与重叠源序号（物理上重复包含上一窗口尾部）。
- 重叠条数 = `边界前 30s 内开始的字幕条数`（v13 起纯内容驱动、无下限），并 clamp 到上一窗口长度以内。边界落在 >30s 空档处时重叠为 0——这是正确行为：拼接不该跨大空白，连续性由只读前文块负责。
- 只读前文 `preceding_segments`：窗口开始前最多 10 条 raw ASR 字幕（`PRECEDING_CONTEXT_MAX_SEGMENTS`，固定条数、**不设 gap-stop**——大空档后恰是冷启动风险最高处，负时间戳让模型自行判断新旧）。第一个窗口为空；-a 半窗继承父窗口的、-b 半窗回看父窗口自身尾部。纯输入、不参与翻译、不影响窗口位置与 clip 范围。
- 音频剪辑区间 `clip_start`/`clip_end`（见上文 padding 规则）。
- 预算估算：字幕文本 token、剪辑音频 token（含 padding）、预计输出 token、总量和计数来源。

边界选择规则：

- 优先在句号、问号、感叹号、省略号等自然句末截断，其次是较长静音（`even_sentence_or_gap_boundary`）。
- snap 半径内没有合适边界时在均匀目标点强切，标记 `forced_even_boundary`。

预算规则：

- 输入 prompt 上限：`prompt_input_limit`——由绑定组的 catalog 行算出
  （`min(max_input_tokens, context_window − 预留)` 取组内最小；预留自 2026-09-04 起是**预计输出**
  `output_scale × 输出系数 × 每窗字幕上限`，不再是声明的输出上限），免费 Gemini 上是 `194000`。
- **不再有单独的「模型上下文规划上限」**（2026-09-03 删）：包络已经从上下文里扣掉了输出，
  `输入 + 预期输出 ≤ context_window` 按构造成立，那道 `256000` 的检查永远不会响。
- API 输出上限：`output_limit`——组内最小的 `max_output_tokens`，免费 Gemini 上是 `65536`。
- 文本 token 计数按 **本地 tokenizer 二进制 → `countTokens` API → 启发式** 三级 fallback（`default_token_counter()`，逐 sha 缓存）：
  - 首选本地 `tokcount`（Go/`google.golang.org/genai/tokenizer`，源码在 `tools/tokcount/`，预编译产物 `bin/windows-amd64/tokcount.exe`，不列入 pyproject 依赖）。Python 进程内所有 counter 实例按 binary/model 共享一个 lazy 启动的 stdio server；默认空闲 300 秒自动退出，下次精确计数透明重启，避免逐次初始化 tokenizer。离线、免配额；它用 `gemini-2.5-flash` 词表，实测与 3.1-flash-lite 的 `countTokens` 相差**恒定 +1 token**（API 的 `contents` 外壳），Harness 已加回该 offset 使二者逐字一致。
  - 本地 binary 可执行时，截断/注入预算跳过启发式预检，直接使用常驻 server 的精确结果。本地 binary 不可用时才启用 heuristic fast path：明显低于上限则直接返回估算，接近或超过上限时进入 `countTokens` API 精确计数；API 再失败才回落启发式 counter（`HeuristicTokenCounter`，按字符类别加权求和：数字/拉丁/CJK/谚文/全角标点/其他文字/空格/ASCII 符号/其他，权重经实测拟合为**上界**——对每个测试类别 heuristic ≥ real，对实际喂给模型的字幕 CSV 最紧约 +1~8%）。旧版启发式因对 CJK 混合文本低估 25-40% 被弃用。
  - `execution_policy=agent-only` 时，counter chain 明确移除 `countTokens` HTTP backend，只允许本地 binary → 启发式；“不调用 provider API”覆盖预算估算辅助路径。调用方注入 client 时以该 client 的实际 execution settings 为准，不读取全局 policy 覆盖它。
  - `countTokens` 端点**完全免费**：不消耗任何生成配额、不计费、无实际速率约束，`.env` key 只用于鉴权。因此即便回落到 API 也不烧 quota。
  - 本地二进制在位时**默认 dry-run 无需联网/无需 key**；缺二进制才回落到 countTokens 端点。
  - **二进制从哪来**：源码 checkout 跑 `bin/` 里那份（不被跟踪：新 clone 没有它，回落到 `countTokens`，直到在 `tools/tokcount/` 编译或解开 Release 资产）；发布版 CLI 不带 `bin/`，改由 `runtime-manifest.json` 的 `tokcount` 资源下载到 `runtime/tokcount/<版本>/`，前端用 `GEMINI_TOKEN_COUNTER_EXE` 指名注入（`finesub_bootstrap.environment.token_counter_overrides`，解析顺序：用户显式设的环境变量 → 系统已有的 → 托管的）。它是**唯一一个装不上也照跑**的托管工具，不拿它当门槛：CLI 在 LLM 阶段任务前尽力拉一次、失败只警告。发布与版本规则见 `tools/tokcount/README.md`。
- 每轮候选规划需要 `k` 次 token 计数校验，通常一轮即收敛。
- 基于 token 上限的文本截断走 `finesub/llm/token_truncate.py::truncate_to_token_window`（插值+二分搜索最接近上限的安全切片，按切片长度缓存计数，只需个位数次 counter 调用；`keep="head"` 保留前缀/截尾部（默认），`keep="tail"` 保留后缀/截前缀；可选回退到自然句末边界）。两个默认开启的快速开关：本地 binary 不可用时，`lazy` 先用启发式 upper-bound 预检，`估算 × 1.02`（`lazy_safety_factor`，额外保险）≤ 上限则原样返回、零 API 计数；本地 binary 可用时直接精确计数；`quick` 把截断搜索的命中窗口放宽到 0.95/50（更少计数次数）。通用 `cap_tokens`、注入预算和累计 advice ledger 使用同一分流；显式传 `gold_ratio`/`abs_slack` 时 `quick` 不覆盖。
- 音频 token 本地按 Gemini 官方口径 `32 tok/s` 乘以**剪辑时长（含 padding）**估算；`media=video` 另加 `71 tok/frame × 0.25 fps = 17.75 tok/s`。由于每次调用只附本窗剪辑，估算口径与 provider 实际计费一致。
- 纠错输出估算：`k × c × csv_asr_result_tokens`（k = `--output-scale`，c = 3.5 + 1.0(retrieval≠none)，media、difficulty、continuity 都不改变 c；替代旧的 `csv × 5 + 10000` 启发式）。
- 窗口规划要求该估算 ≤ 窗口输出预算 `0.9 × 65536 − 5000 = 53982`（快速模式收紧为 `0.8 × 65536 − 10000 = 42428`）；真实 API 请求仍使用 `65536`。

## 纠错窗口调用形态（由 retrieval 与知识库决定）

- **有请求通道（`request_channel`）**：一个窗口是**一条对话**（`stages/correction/request_turn.py` 开场，`attempts.py` 收尾）。首条消息 =
  纠错契约 + 「获取缺失输入」一节：模型读完本窗输入后要么直接交付终稿（作为 attempt 0 验收，不再调用），要么只输出
  `<reasoning>`、`<window_notes>`（≤800 token）、可选 `<requested_entries>`/`<keep_entries>`（新请求 ≤8、与透传合计 ≤12）
  与 `<search_queries>`（≤8 条，可为空块）。harness 查库 + 一轮本地搜索后把词条全文与 `<search_results>` 作为下一条 user
  消息回填（模型的请求回复作 assistant 消息放回历史，剪辑跟在最后一条 user 消息上），交付轮与修复轮都在这条对话里；回填只有
  一次，再请求按格式错误重试。请求回合 best-effort：调用异常按空产物继续——**要选风格时除外**（见下一条）。产物 `RequestTurnProduct`（搜索结果 + window_notes +
  请求/保留的词条 key + 请求回复原文）按 base 窗口 id 缓存：同窗口的 validation 重试和 `-a`/`-b` 拆分半窗复用第一次的结果，
  不重复调用或重搜。`continuity=parallel` 下请求回合在前置 pass 并发跑（`previous_advice` 为空），屏障后再并投交付轮。
  `retrieval=none|native` 时（2026-09-30 起，有可读知识库才有）同一条路只请求词条：通道里没有搜索规则，请求回复以
  `<requested_entries>` 或 `<keep_entries>` 为准（local 仍以 `<search_queries>` 为准），回填只有词条；native 下通道的措辞改为
  「联网搜索用你自带的工具」，请求回合与交付轮都开原生搜索。
  否决过的替代：harness 按原文精确匹配、把命中的词条逐窗预注入（不加回合、更便宜）。精确匹配恰好漏掉最需要纠正的误听专名，
  而且是 harness 替模型选，与 agent 路「模型自己取」相反；匹配面能可靠覆盖误听（音近匹配）时可重评（2026-09-30）。
- **fast 窗口要选风格（`style_pick`，`--style auto`）**：选择是交付契约的一部分（[知识库契约](knowledge.md)「style 类别」）。
  batched：首条回复必须是请求回复并带一个合法的 `<style_selection>`——直接交付、只请求不选、选了不在候选里的名字都按格式错误
  重试，`REQUEST_TURN_FORMAT_RETRIES` 用完仍不合就抛错（任务失败），调用异常也直接抛；带搜索授权时只选不搜的回复合格。
  回填带上选中那套的全文。没有搜索、也没有可读知识库时，请求回合**只为选风格而开**：开场是「选择翻译风格」一节，回复里出现
  `<search_queries>` / `<requested_entries>` / `<keep_entries>` 算格式错误，回填只有风格、不要求 `<keep_entries>`
  （没有词条通道的回填都不要求它，只有搜索的那一格也一样，v91 起）。
  agentic：没有请求回合，交付里附选择，并须先用 `kb_read` 完整读过选中那套；没开知识库读取时窗口拿窄授权 `kb_tools=style`
  （只有 `kb_read`、只解析 `style/<名字>`），说明里只讲选风格，不再列不存在的检索工具。
- **`retrieval=native`、没有可读知识库**：没有请求通道（要选风格时例外，见上一条）；纠错调用自身设置 `native_search=True`，在组内过滤出能联网的成员并在模型内完成查证。
- **`retrieval=none`、没有可读知识库**：没有请求通道（要选风格时例外，见上），纠错调用也不开工具。
- **agentic** 的窗不走请求回合：说明里列出本窗真有的工具——联网检索只在 `local` 下有，知识库工具在索引可读时有。

交付轮任务组 `correction-mm`/`correction-text`（3.7 优先组）；`retrieval=native` 不换组，只是在组内过滤出能联网的成员（出厂为付费 3.7）。
回填消息带来 `<search_results>` 与 `<entry_details>`（请求的词条全文；准备任务的种子优先）；请求回合的 `<window_notes>`
留在被回放的 assistant 消息里，交付轮不再单独注入（首条消息的 `<pre_round_notes>` 槽在任务式流程里恒为空，空时整段不渲染）；交付轮不启用工具、不能再发起搜索。

### 视频（--video；`--media video`）

- `video` 档的会话媒体是**低清视频+音轨的 `.mp4` 剪辑**（同剪辑区间与 padding 规则；`finesub.media.ffmpeg.extract_video_clip`：decode 先 `-hwaccel auto` 失败退 CPU，编码 libx264 + AAC）。只切 `correction_media` 要的那种剪辑（`audio` → `.aac`，`video` → `.mp4`）；请求回合与交付轮共用同一份剪辑与上传。
- API 侧：mp4 以 `detail=low` + `video_metadata.fps=0.25` 经 REST 直传（→ Gemini 每 part `mediaResolution: {level: MEDIA_RESOLUTION_LOW}` 与 `videoMetadata.fps`），计费口径与规划一致：`32 tok/s（音轨）+ 71 tok/frame × 0.25 fps（画面）`。
- 快速模式的单窗同样按 `correction_media` 上传整段剪辑，请求回合与交付轮复用。
- **video→audio 一级安全网（client 层）**：带视频剪辑的调用落到「能听不能看」的候选 target 时，不再整个跳过，而是换发同窗音频剪辑（按需现切）、按 target 去重告警一次、并在 route decision trace 上记 `media_downgrade: "video->audio"`。这是安全网不是机制——正常配置应让接视频的格子里都是有视频能力的模型；audio 不降 text，native search 不降级。

### LLM session 级 resume（默认开启）

> **一句话口径**：**一个 harness LLM session 跑完、校验通过，就是一个 checkpoint。** 重跑时它
> 直接复用，不再调模型。下面的分层与失效键回答的是另一个问题——**改了什么之后这个 checkpoint
> 就不算数了**——不是"复用单位是什么"。
>
> 两个词的关系也在这里说清，别的地方不再重复：**窗口**是把字幕切成的一段，纠错链上**一段就是
> 一次 session**，所以「窗口」和「session」基本同义；**stage** 是流水线的一大步（调查 / 纠错
> / …），一个 stage 里装着好几个 session（比如准备阶段 = 请求回复 + 搜索 + 交付回复），它们各自存档，合起来
> 产出这个 stage 的产物。

传入 task artifact 目录且 `resume=True` 时，生产 harness 会把以下经当前 parser/contract 验证成功的原始响应追加到 `<artifact_dir>/session-checkpoints.jsonl`：准备任务的请求回复与交付回复（session `prepare-t1` / `prepare-t2`，超长素材按块 key），以及纠错窗口的请求回合（session `request`，key 为 chunk id；直接交付的回复同样入账，回放时作为 attempt 0 再过 validator）。ledger 为 append-only JSONL；每条 committed 记录包含 `schema_version`、稳定 session/key、`input_hash`、`content_hash`、原始 `content` 和模型 metadata。截断行、坏 JSON、未知 schema/status 或 content hash 不符的记录加载时直接忽略。

**一条例外（准入门 D 的 C 案，2026-08-19，docs/llm_local_agent.md §7）**：依赖隐含 provider
历史的调用——`agent_session_mode=resume` 下跨窗继承了 conversation handle 的那次 agent 调用——
被打上 `LLMCallResult.resumable=False`，**不写进本账本**：它的 hash 不覆盖那段只有厂商知道的
历史，两次不同输入会被认成同一次。代价只有「中断后那一次调用重发」，已提交的窗口/stage 不受
影响。

`input_hash` 覆盖精确组装后的 messages、`PROMPT_VERSION`、角色/输出上限/thinking 等调用配置，以及 messages 外的调用状态。**它一个字段都不放松**：docs/llm_local_agent.md §4 让它做 `submit()` 的 compare-and-swap 令牌，§12 让它成为未来统一 task runtime 的唯一提交口径。它只服务于“外层 stage/window 尚未提交”时的单次调用恢复：重启从确定性本地代码重建状态，到同一边界且 hash 精确命中才取旧响应，并再次走**当前** parser/contract；复验失败即 live 重打。这里的精确性是为了证明中断点能有效重建，不会向上推翻已经提交的 research context 或纠错窗口，也不要求整条任务保持同一模型/prompt。请求回合命中只省模型调用，关联搜索仍会重新执行。

因此细粒度可恢复边界是：准备任务的请求回复后与交付后；每窗请求回合后。纠错交付轮使用下述 `correction-windows.jsonl` 整窗提交缓存；知识更新另有 `knowledge-update-chunks.jsonl`。`--no-resume`（`resume=False`）使 session 与 correction-window 两种 ledger 都不读也不写，但不删除旧文件，也不关闭 pipeline 对完整 `*-research-context.json`/SRT 文件的 stage 级存在性复用。

### 纠错窗口中途 resume（默认开启）

成功且未截断的执行叶会把原始响应追加到
`<artifact_dir>/correction-windows.jsonl`。记录包含 chunk/source id、clip、continuity、
`task_fingerprint`（现为 stable 源指纹）、完整 `input_hash`、不含词条签名的 `input_hash_core`、实际注入词条 keys、
响应、实际 `variant` / `difficulty` 与兼容用 capability tier 元数据；拆分父记录另写
`split_into`，恢复时先重建同一棵 `-a/-b` 叶，因而
首次拆出的前半窗也可复用。任何缓存响应都要再过当前 variant-aware validator。

- `task_fingerprint` 是一份**显式 include 列表** `WINDOW_INVALIDATION_INPUTS`：`prompt_version`、`test_profile`、源 `*-stable.json` 解析后的 id/时间/文本序列（排版或无关 JSON 字段变化不算源变化）、源媒体的 path+size（**不含 mtime**：重下载同一份音频不是内容变化）。
  ⚠ **翻译风格不在里面**（2026-09-02）：`extra_style`（自由文本）与 `--style`（具名条目）中途改了都不作废已完成的窗。后果是产物前半段一个口吻、后半段另一个——owner 判定可接受，不值得为它重跑整批。这条推翻的是 `commit.py` 注释里曾明写的失败模式，理由在影响面：风格条目有界、形状固定、只碰风格，与知识库词条正文同一档。没有被分类的字段进不了这个列表，`_task_fingerprint` 会断言 payload 与列表一致——漏维护的失败方向是「多重跑一个窗口」，不是「错误复用」。
- 因此**不在**失效键里的有：execution identity（模型、预设、模型组、thinking、execution policy、agent driver/effort/超时；docs/llm_local_agent.md §11）、difficulty 与 variant（逐窗记录实际值，按记录里的 variant 回放校验）、知识库词条与常见错误正文、`task_update_feedback`、context pack，以及几何类旋钮（`correction_media`、`retrieval`、`--output-scale`、窗口上限、组包络）。
- 回放要求 `input_hash_core`（当前窗口 + 只读前文的 id/时间/文本）匹配，并把原始响应按记录中的实际 variant 再过当前 validator。完整 `input_hash`、注入词条 keys 与 `knowledge_version`（知识库嵌套 git HEAD，docs/llm_local_agent.md §8）留作审计。
- 只缓存"成功且非 output_limited"的窗口（与既有提交门槛一致）；`-a`/`-b` 半窗各自按 chunk id 缓存，拆分父记录保存 `split_into`，恢复时会重建并复用完整拆分树。
- **窗口计划持久化**：非 fast 且 resume 开启时，规划结果写入
  `<artifact_dir>/correction-window-plan.json`（含 `source_fingerprint` 与审计用 `geometry`）。复用门只看
  L2 `task_fingerprint`；计划的语义只是 chunk id 与 source-id 边界，文件里**只有**
  `chunk_id`/`segment_ids`/`overlap_ids`/`boundary_reason`（schema v2）。clip 区间、token 预算和
  只读前文都是派生态，恢复时按当前 profile、组包络、媒体时长与当前段列表重算——落盘的旧包络
  数字没有机会混进新一轮。
- 复用计划先展开既有 `split_into` 树、验证可回放叶，再对 pending 叶检查输入包络、输出 CSV
  上限和质量护栏；不合身则在发车前递归拆半，并先写同一套 `split_into` 记录。发生预拆时打印
  stderr 告警并写 `window_refit_report`；单段仍放不下会以包含 source ids、估算值、限额和失败面
  的明确错误终止。拆分只会让计划更细；若换大模型后希望合并成更少窗口，需显式删除
  `correction-window-plan.json`（这也放弃原 chunk id 对缓存的寻址）。
  ⚠ **预拆和重试拆分花的是同一份预算**：refit 也要过 `WindowGeometry.may_split`，深度到顶
  就以同一种错误终止，不会再拆。（它以前是**不问的**，所以复用计划每续跑一次就能再对半一层、
  一路越过上限——2026-09-03 修。）
  ⚠ **上限管的是「能不能新拆」，不是「能不能回放」**：上限从 2 降到 1 之前记下的 `0001-a-a`
  仍会被 `expand_cached_splits` 完整重建并回放——已完成的产物不因今天换了个参数而作废
  （`README_DEV.md`「复用的依据是任务身份」）。被拒的只是**从它再拆一次**。
- **serial**：接受同源且 core hash 匹配的记录，并按窗口顺序回放；
  回放重新执行 `merge → advice/transfer commit`，所以后续 live 窗看到的台账与首次运行一致。
- **parallel**：缓存可以是任意窗口子集。先回放可用叶，再对 pending 叶并投请求回合；屏障
  使用首次运行持久化在同一 JSONL、同一源指纹下的 `parallel_entry_set`，不会用
  “剩余窗口的请求”重新缩小会话固定集。key 固定，正文每次从当前 KB 重渲染。第二阶段只
  投 pending 纠错窗；所有在途窗排空、成功窗写缓存后才抛首个错误，最终结果按窗口顺序合并。
  同批累计 3 个失败会阻止尚未开始的窗口继续发车。
- **模式切换**：serial / parallel 双向都复用已完成窗口，结果可以合法混合两种执行方式。
  `parallel -> serial` 会**打印告警**：parallel 窗从不写 advice，所以恢复后的 serial 台账比首次
  运行短一截，后续 live 窗看到的东西因此不同——允许，但不静默。
- **research context（L3）**：planning metadata 全部落盘作审计；`research_reuse_key` 只比较
  stable 源、`prompt_version`、`extra_info_hash`、`test_profile` 与
  `research_semantics_id`（当前为 retrieval）。媒体/窗口几何、模型、知识版本、difficulty、continuity
  均不作废已提交调查。`planning_metadata` 的每个字段都必须显式归入 gate 或 audit-only 两组之一
  （`RESEARCH_REUSE_GATES` / `RESEARCH_REUSE_AUDIT_ONLY`，测试断言两组之并等于全部字段）——
  这是 L2 `_task_fingerprint` 那条断言在 L3 的对应物：漏分类会在测试里炸，而不是在字幕里。
  损坏/不兼容文件在覆盖前复制为同目录 `*.invalid[.N]`，备份失败则拒绝覆盖。fast context 规则不变。

## Prompt 信息

纠错窗口 prompt 会给模型这些信息：

- `<asr_result>`：本窗口需要处理的直接类 CSV 文本块（时间以剪辑 0 秒为基准），header 为 `local_id|start|duration|gap|text`。每个执行窗口（含 fast 单窗）都把目标行重编号为 `1..N`。
- `<preceding_context>`（v13）：窗口前最多 10 条 **raw ASR** 只读前文，与 `<asr_result>` 并列、使用同一时间基准；按时间顺序编号为 `1-M..0`，最近前文恒为 0。0/负数不属于输出范围，误引用会命中未知序号校验整窗重试。目标语侧连续性由 advice 台账与 context_pack 承载；advice 禁止携带无窗口命名空间的局部序号。请求回合与交付轮是同一条对话，前文块在首条消息里。
- 通用背景（`general_context` JSON）与按 source-id 覆盖关系选出的本窗口背景，来自背景调查，放在直接 ASR 块之前以提高 input cache 复用效率。旧 payload 中的 `audio_file`、`chunk_id`、`segment_range`、`boundary_reason`、`overlap_source_ids` 和 token budget 均不再发给模型。
- `<previous_advice>`：仅 `continuity=serial`；此前所有成功窗口 `<next_advice>` 的累积台账，
  请求回合和交付轮都能看到，整体≤8000 token，超限从最旧条目截断。parallel 不注入该块。
- `<entry_details>`：仅知识开启；来自准备任务的种子、local 请求回合的请求，或 parallel 屏障固定集的
  本地词条全文，可为空。
- `<search_results>`：仅 `retrieval=local` 的请求回合产生（fast 单窗同样），由本地搜索代理渲染、随回填消息注入。
  请求回合的 `<window_notes>` 留在被回放的 assistant 消息里；首条消息的 `<pre_round_notes>` 槽恒为空、整段不渲染（只有 replay 工具重打交付轮时还填它，才出现）。
- 原生搜索：`retrieval=native` 不注入本地 `<search_results>`，而由纠错调用自身启用工具。
- **任务 recap**：自 v9 起，每种 session 的 user prompt 末尾（payload 之后）都有一段 2-3 行的静态"最后提醒"，重申任务目标与输出格式关键约束（Gemini 长上下文最佳实践：指令重申放在大段 context 之后）。纠错与准备任务的 user 模板都有；回填消息末尾另有一句交付提醒。

模型随后输出 translated 终稿；basicA 还会先逐源完成 singles。tag parser 只认第一级同名块。输出结构按 variant 区分：capableB/C 使用下述九列 CSV；BasicA/B 使用带 header 的十列 CSV：

生产由任务组×difficulty 格子显式给出 variant（出厂纠错：quality→capableC，
intermediate/efficiency→basicB）；provider fallback 不再改变 prompt。纠错 live、窗口 resume cache
与 session replay 共用 variant-aware validator：capableC 不要求 `<singles>`，basicB 继承 capableB
合并并带 `start`（basicA 仍要求完整 `<singles>`），并绑定各自的 CSV 列布局。旧记录没有 variant
时才用 capability tier 作兼容回落；新记录逐窗保存实际 variant 与 difficulty。

```text
type|position|duration|gap|corrected_text|translation|conf|char_count|note
type|position|start|duration|gap|corrected_text|translation|conf|char_count|note
# 单源本身越过硬门槛，仍须如实输出
sub|1|2.5|4.6|5.6|...|...|high|13|
```

- `type`：留空或 `sub` 表示默认行为（拼合/纠错/翻译，`position` 填源序号）。~~`insert` 插轴已于 v63 全面废弃~~（旧模板已移至 `legacy/`，gitignored 不随仓库分发）。多个源片段合并时源序号用英文逗号连接，例如 `sub|3,4|1.9|0.2|good morning|你好|high|2|ASR 错分`。
- `start`：BasicA/B 以 CSV 列携带；单源抄输入 start，合并行抄首源 start。解析只校验存在和数值类型，最终时间轴仍按映射后的稳定源序号回填；抄值准确率仅作能力观测。`duration`/`gap` 同样是引导字段而非可信时间源。
- capableC 的局部推理使用目标行正上方的 `#` 注释；普通单源在界内前不输出。validator 只计数，不进入 SRT。
- `gap`（v37）：**本条结束后到下一条开始**的间隔秒数（与输入 ASR CSV 的 gap 同义），绝不是本条到前一句的距离；判断是否与前一句合并时须读取前一行 gap。引导用列，解析后丢弃。
- `conf`（v39）：`high`（very certain）/`median`（likely correct）/`low`（better to manually check）三档自评信心；旧缓存中的 1–9 数字仍会兼容映射为三档。`char_count`：独立加权译文字数列，位于 note 左侧；本地按“拉丁/数字/标点/空格=0.5，其余可见字符=1”复算并规范化，模型值不一致时把 warning 写入窗口 artifact。**逐行不一致之上还有一条窗口级 warning**（2026-09-04）：当**不符行占该窗全部行 ≥1/3****且**不符行的 `computed/reported` **中位数 >1**（即模型系统性地**少报**）时，另写一行说明。两个条件缺一不可，因为它们答的是不同的问题——占比说“系统性”，方向说“不只是数不准”。阈值在实现之前按 116 份历史 exchange 的离线基线预注册：基线里单窗最多 2.4% 的行不符，且 **38/38** 的比值都 <1——模型只会**多报**自己的长度；而唯一一次传输故障是 100% 的行、比值约 2.8，两个维度都在另一侧且各有约一个数量级的余量。⚠ 它**只加告警、不改归一**：字数照旧被替换成本地复算值。把它升成 error 会让一个没人标定过的阈值挡在每个只是数不准的模型前面；它要修的是更窄也更真实的一件事——证据此前在同一步里被覆盖掉、然后丢弃。统一公式由 `finesub.subtitles.metrics.weighted_char_count` 定义，并同时用于 pacing、annotated CSV 与通用 SRT 行宽汇总；它只衡量字幕显示长度，与 token 预算及 ASR 异常检测用的 `finesub.text.count_word_units` 相互独立。`note`：自由注记，是最后一列；prompt 要求文本中的 `|` 写成全角 `｜`，解析器仍宽容旧输出在末列使用半角分隔符。
- 统一入口是 `output_protocol.validate_correction_window_output`：它先按 variant 校验窗口局部 CSV，再把有效 `position` 与 discard 序号映射回稳定源序号。parser 对 type/note 宽松，`conf` 非法只告警不失败（仅供参考，从不单独判行失败）；`char_count` 格式会校验——**漂移行正是被它拦住的**（多一列会把非数字挤进 char_count）。结构性错误（列数不符；未知/乱序/重复源序号，含 discard 与普通行之间的冲突；意外 start 列；insert 行；空文本；缺时长列）判失败触发重试。⚠ **不要再加「译文行数＝输入条数」断言**：把多个源合并到一行是合法输出，而行数对得上也抓不到语义错位（反馈包 #22 就是结构全对、内容串行）——这里保的是**结构**（每个源序号都被交代），内容对不对要靠别的信号。
- **保留数下限 `KEPT_FLOOR`（owner 2026-09-30）**：一个窗口里没被 `discard` 的源序号数 ≤ `min(10, 总数×0.5−10)` 判失败，进修复轮。它不是新规矩，是把「全部 discard → 无有效行」这条既有判据从 100% 边界挪开——**coverage 只问源序号有没有被交代，不问窗口有没有产出字幕**。起因是 2026-08-22 canary：一行 `sub` + 其余全 `discard` 通过了全部结构校验，成品只剩一条。
  读法：40 条以上的窗口是「最多只留 10 条就拒」；不到 40 条时门槛随总数降，20 条及以下不再触发（只剩「全部丢弃」那条）。
  它取代了 2026-09-03 的 50% 丢弃比例上限，那条有两个问题：没有小样本保护（一段大半是歌的 1 分钟短片，6 句里丢 4 句是对的），也用不到拆分后的半窗上（归档里最坏的正确半窗丢 43.8%，见 `bench-baselines.md` [二十五 纠错窗口的丢弃比例](bench-baselines.md#二十五纠错窗口的丢弃比例05-的标定以及为什么它不能用在拆分叶子上)）。绝对下限没有这两个问题，所以**整窗和半窗一律适用**（owner 2026-09-30 确认）。正确输出里丢得最多的整窗也只丢 21.9%，离这条线很远，所以它是错误探测器而不是质量旋钮。
  ⚠ 仍是一处**写明的缺口**：没读正文、但留下超过下限条数的回复照样通过；补它要的是**另一种信号**（模型到底有没有收到窗口正文），已记在 `llm_followups.md`。
  这一条同样管着 `agent-task lint`（同一个 validator，同样的范围），所以看不到正文的 agent 在提交前就现形。
- **行尾 `<void>` 自弃标记（v12 起）**：模型写完一行才发现不对（时长失控、分组/取舍错误）时，可在行尾追加 `<void>` 废弃整行并另起重写。解析时带标记的行在一切结构检查**之前**剥离（内容再破也不报错），其源序号可被后续行重新使用；数量计入 `CsvValidationResult.voided_rows` 并写进 `correction_window_response` artifact（用于观测模型是否真的使用该通道）。全部行都自弃且无其他有效行时按"无有效行"判失败重试。
- 只有 `translation`（纠错 SRT 另用 `corrected_text`）进入 SRT；`type`/`duration`/`gap`/`conf`/`char_count`/`note` 留存在 `<stem>-annotated.csv`（9 列）。行时间轴按源序号从 `*-stable.json` 回填。知识更新阶段另会 overlay 最终 SRT 时间轴，生成含 start/end 的 10 列 `<final_csv>`。
  insert 的发射/去重/合并代码已于 2026-08-07 删除（两个生产调用点早已 `allow_insert=False`，整条路径不可达）；`KIND_INSERT` 常量保留，因为知识素材仍要从旧 `annotated.csv` 里读回该类型。

如果某段高度疑似 ASR 幻觉（含套话式幻觉）、无意义重复或非主播有效内容：basicA 仍须在 `<singles>` 输出并标注；所有变体都须在 `<translated>` 以 `discard|<局部序号>` 显式丢弃，不能静默漏掉（否则 coverage validation 失败）。v12 起 prompt 侧取向为**拿不准时保留并在 note 标记「疑似幻觉」**（人工删一条错留的幻觉比重听补一句被误删的台词便宜）。v16 起幻觉判定/保守保留/丢弃与插轴取舍收拢为独立的「幻觉与丢弃」fragment（`$hallucination_block`，取舍子句仅音频路线注入），套话幻觉另有独立反例（输入/输出块形态）；处置措辞仍按模态参数化（`$hallucination_handling`/`$noisy_span_handling`：音频路线以重听裁决，纯文本路线默认保留、禁止凭空"还原"台词），按类别特征描述（套话语域＋上下文脱节＋无对话区间）而非具体短语黑名单。

每行首先是字幕显示单元/时间单元，不是逐词语义对齐单元。`translation` 可以为了中文语序和阅读节奏，在同一连续语义单元内相对 `corrected_text` 前后错位；多数源保持独立，勿为对齐强行并成长字幕。少数三源例外仅限同一句连续三切，禁止四源以上。

`continuity=serial` 时，`</translated>` 后要求一个可为空的
`<next_advice>...</next_advice>`：只写本窗新增/修正的术语、说话人状态、未决指代等，
每窗≤800 token。Harness 按窗口 id 入账并注入后续窗口，台账整体≤8000 token；漏块按空
建议处理。`continuity=parallel` 时 system/user 都不提该块，也不解析或累积 advice。

`retrieval=local` 的纠错调用不开工具，查证由同一对话里的请求回合 + 本地搜索代理完成；
`retrieval=native` 的纠错调用自身启用原生搜索；`none` 两者都没有。所有本地 query、provider
和结果元数据都会写入 task artifact。

当且仅当采集反馈时（`--knowledge update`，或参考资料导入），纠错 prompt 会额外要求最后输出
`<task_update_feedback>`（v3：`knowledge_hints` + `asr_corrections` + `uncertainties`）。
它前面的尾块按实况变化：serial+local 可有 `<next_advice>`、`<keep_entries>`；serial 非 local
只有 advice；parallel 两者都没有。feedback 的 `source_ids` 使用本窗口正局部序号，落盘前
映射回稳定源序号；解析失败只告警不重试，也不进入最终 SRT。准备任务的交付同样采集
`research_task_feedback`。该档位纳入纠错 resume 的 task fingerprint。

## 注入上限

下列上限由 Harness 在运行时强制（常量定义见 `src/finesub/llm/routing/config.py` 与 `src/finesub/llm/web_search.py`）；全部以 **token** 计（用任务的 token counter 计数，截断走 `token_truncate.cap_tokens`）。超出 prompt 侧上限时截断或丢弃；超出输入硬上限时报错。

统一注入预算公式（搜索结果、深度提取、知识库词条共用，`injection_block_token_limit`）：**整块 ≤ 该轮单位上限 × 2000 + 4000 token**；块尾追加"注入预算说明"列出受影响条目，`RenderedBlock.report()` 全量记入 artifact。块内怎么分，两类不同：

- **搜索 / 提取结果**：单 section（单 query 结果 / 单 URL）≤ 4000 token，按优先序装填，装不下的截断或连同其后全部丢弃（先到先得）。搜索循环拿截断 / 丢弃名单决定事实优先级，所以这一类不改。
- **知识库词条**（`render_knowledge_entries_block`，2026-10-02）：每条先按所属类别 preset 的 `max_entry_tokens`（三个类别都是 6000）封顶；知识更新预取（上限 60k）里有两类条目先分、每条要么整条进要么不进、合计最多半块——本次更新的 style 条目，以及超过软阈值（5000）、被要求压缩的条目（这类不受 6000 封顶：看不全就压不了）；装不下的退回共享分配、照常截，也不挂压缩提示（`docs/knowledge.md`「条目的长度」）；其余条目「最长的先让」——求一个水位，短于它的整条进，长于它的截到它，所以块满时只削最长的几条。截断**按行**：条目头（H1、简介、`档案`）与元数据节总在，其余各节按顺序逐行装，装不下的那一行及其后的都不进；条目头之和都装不下时按优先序从尾部整条丢弃。块末说明写到「哪个条目的哪一节少了几行」，report 另记每条被削的原因（`cap` 超过单条上限 / `block` 块满被分）。**不变式：块永远不超过整块上限**。被截掉的行在知识更新里不分句柄（`retain_injected_handles`），曝光只记真正注入的行（`signals.injected_node_ids`）。纠错 system prompt 里的 style 块按同一上限逐套截断。

| 项目 | 上限 | 说明 |
| --- | --- | --- |
| 窗口规划 reserve | 72000 tokens | 规划每窗输入预算时扣除，覆盖窗口 CSV/媒体之外的全部内容（静态 prompt、context pack、建议台账、请求回合 notes、搜索块、词条块），最坏合计约 69k。research 与 correction 两侧必须同值（窗口 id 一致性）。 |
| 准备任务搜索 query | 8..16 条 | 本地执行硬上限；`min(16, 8 + sqrt(原始段数)//10)`（`research_search_query_limit`）。模型多出的 query 丢弃。 |
| 纠错窗口搜索 query | 8 条 | 每窗请求回合 `<search_queries>` 最多执行这么多条（Exa → Gemma4 → Tavily），再回填给交付轮。 |
| 搜索/提取结果渲染 | 单 section 4000 token；整块 `该轮query上限×2000+4000` token | 统一预算公式。软上限：单条 snippet/answer 600 token、单 URL 提取内容 1800 token（section 内部的排版控制）。 |
| 知识库词条注入 | 单词条按类别 `max_entry_tokens`（6000）；整块 `条数上限×2000+4000` token，块内最长的先让 | 准备任务的 request≤8、keep≤8、keep-first 合计≤12（整块≤28k；下游无请求能力的档位 request 直接 12）；请求回合新请求与透传同样合计≤12；本地预注入按自己的上限。 |
| 本地关键词预注入 | 8 条 | 用户备注与 index key/alias 的 casefold 子串匹配，按频次排序；仅知识开启时注入准备任务的首条消息，直接交付时也留在种子里。旧的 text 直注纠错窗路径已删除。 |
| `--extra-info` URL 预提取 | 8 个 URL | 从 `--extra-info` 去重后的 HTTP(S) 链接；`retrieval=local` 时在准备任务首条消息前 deep extract，注入 `<note_url_extracts>`（`<search_results>` 文本），块预算 `8×2000+4000`。 |
| `analysis_notes`（准备任务的请求回复） | 1500 token | 解析后 Harness 截断；只是回填后交付的起点，不注入纠错窗。 |
| `window_notes`（纠错请求回合） | 800 token | 请求回合的预搜索分析；截断后进产物记录，交付轮通过被回放的请求回复看到它。 |
| `next_advice` | 800 token/窗；台账整体 8000 token | 仅 `continuity=serial`；按窗口 id 累积并注入后续窗口（含拆分叶）。parallel 完全撤除。 |
| Prompt 输入硬上限 | `prompt_input_limit`（免费 Gemini 上 194000） | ✱ **不是常量**（2026-09-03 起）：由绑定组的 catalog 行算出，`min(max_input_tokens, context_window − 预留)`（预留=预计输出，2026-09-04）。准备任务与纠错窗调用 API 前走 countTokens；超出即硬错误（无 map/reduce）。 |
| 快速模式回填 reserve | 56000 tokens | 快速窗首条消息的输入门槛 = `prompt_input_limit` − 56000（`FAST_FILL_INPUT_RESERVE_TOKENS`），为回填轮的搜索结果（≤20k）+ 词条（≤28k）留余量。这几个上限是绝对值、不随组包络收缩，包络放不下时闸门按名拒绝；不为小上下文模型缩它们（owner 2026-09-30：这类模型已基本没有）。 |
| 纠错窗输出上限 | `output_limit`（免费 Gemini 上 65536） | 请求回合与交付轮共用（Gemini 3.x 上 thinking 与可见输出竞争同一预算）。 |
| 准备任务输出 | **无 harness 上限**（请求打满答题模型自己的 `max_output_tokens`） | 请求回复与交付回复都是。`SESSION_OUTPUT_MAX_TOKENS`（32000，× `--output-scale`）自 2026-09-04 起只是**上下文预留**，不再当上限发出去。 |

## 说话人裁判（--diarize judge-text / judge-audio，默认关）

⚠ 仍在开发中；实验与验收的结论在 `docs/diarization-findings.md`「裁判放在哪」「生产裁判的验收」。用户向说明在 [`docs/manual/diarization.md`](manual/diarization.md)「LLM 说话人裁判」。

- **开关与模态**：pipeline 的 `--diarize` 四档（`stages.DIARIZE_LEVELS`），`judge-text` / `judge-audio` 让 LLM 阶段的请求带 `speaker_judge="text"` / `"audio"`；单独跑纠错模块用它自己的 `--speaker-judge off|text|audio`。
  模态只看这一个值，不看纠错的 `correction_media`：`audio` 时裁判调用走 `correction-mm` 组（`run.judge_group`），纠错照自己的组。开跑前用 `usable_endpoints` 判一次这组里有没有能听音频的模型——选路时没有能听的候选只会以「没有路由」失败、到不了降级——没有就整次按纯文本判、告警一次（`task.judge_media`）。
  目标阶段低于 `translated-srt`、或说话人分离没跑成时，pipeline 往 stderr 打一行（`_warn_if_the_judge_cannot_run`）。
- **位置**：`correction_translation._run_task` 里，准备任务之后、纠错窗口之前（`stages/speaker_judge/task.judge_task`）。路线一里两边的判决互不读取，只共用窗口边界：
  纠错窗口计划由 `resolve_correction_plan` 一次运行解析一次（读计划文件，指纹对得上就沿用；读不到就新算并写回；fast 模式用 `FastDecision.window`），交给裁判与纠错。
  ⚠ 别在同一次运行里再解析一次：第二次会读到第一次刚写的计划文件、当成复用，纠错账本就会拿已丢失的计划续跑。`mode == "research"` 不跑裁判。
- **窗口**：一个纠错窗口的核心行（重叠区之后的行）就是一个裁判窗（owner 2026-10-09）；窗 id 就是纠错窗口的 `chunk_id`。前后带不判的上下文，长度同纠错跨切点带的重叠（owner 2026-10-09）：切点前 30 秒内开始的行（`chunking.overlap_count_for_boundary`，长停顿之后为 0），窗后镜像：本窗最后一行之后 30 秒内开始的行（长停顿之前为 0）。纠错的 prompt 不带后文块，是因为窗尾那段会被下一窗当重叠区重做、拼接时取新的那份，切点附近的行的最终版本因此看过后文；裁判一行只判一次、不重做，就把这段后文显式带上，效果一致。窗前只取重叠，不取纠错另带的最多 10 行只读前文（`preceding_context_for_boundary`，长停顿处也不停），所以长停顿之后开始的裁判窗没有前文（owner 2026-10-09：上下文同纠错的重叠那一部分）。行数由行的时间算，sidecar 不记。
  第一次切好的窗列表写进 `<stem>-speaker-judge.json`，此后以它为准：纠错之后重新规划不重切、不重判；只有 sidecar 不在、版本不是 2、或行的 id / 时间 / 文字变了（重跑过 ASR）才按当前计划整份重切。
- **放不下时对半拆一次**（owner 2026-10-09：跟着纠错窗的分割，正常不特判；兜底照纠错的拆窗，见[重试与拼接](#重试与拼接)）。纠错规划已按纠错的输出公式卡过窗的大小，裁判每行的输出比纠错少（GPT-6.1 Sol 约 35 token / 行含思考，纠错约 90 / 源段），所以裁判不另估输出。三种情形对半拆：
  - API 用量贴着输出上限（`is_likely_output_limited`，主信号）：不再重试，立刻拆。agent 的用量是整个会话累计的，不拿来比。
  - 重试用完、最后一次回答 `<judge>` 开了没闭合（`answers.cut_off`，兜底，同纠错只看最后一次；agent 通道只有它）。
  - 开跑前按裁判自己那组的限额（`planning_limits_for`）估输入超了（主要防 `judge-audio`：整窗录音与样本）。
  切点取离中间最近的合理边界（`windows.halves`，规则同 `chunking.split_window_in_half`，不带重叠：半窗前后的上下文按同一规则取）。被拆的窗在自己那一项里记 `parts`（窗列表从不增项，并行写入靠下标），半窗 id 是 `窗id-a` / `窗id-b`，各有完整的重试次数。**只拆一层**：半窗再撞上限、输入还超、或重试用完时回答仍开着，就记为失败、退回声学标签（纠错在这里停掉整个任务；裁判是可选的，不拖垮纠错）。不记行数上限，下一个窗照常整窗发。
- **人物候选**：开了裁判时准备任务多交一个 `general_context.speakers`（`fragment_prepare_speakers_{duty,field}_v1.md`，要求轻：只凭标题、简介与台词直接看得出的，不另外检索；关着时 prompt 逐字不变）。它不进纠错窗口（`ContextPack.general_prompt_text` 去掉，分块调查的每块里也去掉）。候选顺序：`speakers` → 知识库主播（`roster.roster_entries`）→ `entities` 里的人物，按名字原文去重（`speakers` 里的人在 `entities` 里也有时，prompt 要它照抄 `entities` 的 `name`，同一人才不会列两遍），最多 30 个；整份候选直接进裁判 prompt，裁判可以写名单外认得出的人（owner 2026-10-09）。读取兼容分块调查合并后的 `{块 id: 块}` 形状。准备任务的复用不看开没开裁判：已有的调查结果没有 `speakers` 就只用另外两类候选。
- **每窗一次调用**（owner 2026-10-09；准备任务已经列出全片的人，不另挑人）：纠错同一个思考档，组按裁判的模态。回答不合法重试，每单元（窗或半窗）最多 4 次，用完记为失败、退回声学标签，续跑再判。
  额度类错误（按路由的 `classify_failure` 判为配额或限速：429、402 等；或来自至少两个单元、连续 3 次不花输出 token 的空回答）停掉整个裁判，没判的窗记为待判，把额度留给纠错；其他错误只让那一窗失败。窗口并行度跟纠错窗口同一个设置。
- **跨单元的新建轨**（owner 2026-10-09）：`continuity == serial` 时按顺序判，上一个单元（窗，或被拆开的窗的前一半）新建或接着用的轨，连同裁判给的名字、目前为止的例句与声音样本，进下一个单元的画像；裁判答这条轨号就算并进去，sidecar 记成 `carried: {window: 单元 id, label, of}`（不记号码：号码由读的一侧按单元顺序编；`of` 是源单元当时那份判决的摘要 `judgments.verdict_of`，源单元不再成立、或被重判成另一份判决时这条结果作废，也不再传给下一个单元；作废的那些行回到声学标签，接力的单元本身不因此重判）。`parallel` 时各单元各开新轨，不合并；同一个 `@甲` 在两个半窗里也是两条轨。
- **续跑**：每判完一个单元立即写 sidecar。重判条件只有失败、待判、身份不对（单元内行、它们的整个 `speaker` 对象、画像用到的轨）、结果读不了（缺 `results` 或形状不对：`judgments.holds`）。裁判自己的 prompt 版本（`run.PROMPT_VERSION`，独立于纠错的 `PROMPT_VERSION`）随每个判决记下，升级时不重判仍对得上的单元（owner 2026-10-09：可复用的窗不因 prompt 升级丢掉）；换模态续跑时已判的单元也照旧沿用、只告警。要按新 prompt 重判就删 sidecar。
  **已完成的任务**（成品在）：sidecar 已覆盖全部窗口（不看版本）时照旧走 `reused` 捷径；没覆盖时走完整的任务边界，复用准备任务的上下文，只补裁判（窗取存下的纠错计划的边界，不看它的指纹——那是纠错实际跑过的窗；计划不在就新算一份、不写回；排窗失败只告警），
  再按现有 `-annotated.csv` 的 `position` 列重渲 `spk` 列（`refresh_annotated_speakers`），不重跑纠错、不碰 SRT；CSV 不在时只告警。
- **dry-run**：`--prompt-dir` 多出 `speaker-judge-0001.txt`（候选只有知识库的，准备任务还没跑），`plan.json` 的 `speaker_judge` 给窗数与输入 token；用 dry-run 自己算的计划，不写计划文件。
- **读回**：`speech/diarization/judgments.read_judgments` 是唯一的读取函数（`spk` 列、分轨导出共用），读的一侧同样逐层校验，见其模块 docstring。分轨导出在 raw-srt 阶段与 LLM 阶段之后各跑一次，每次先清掉上次导出的文件。
- **开发开关**：`FINESUB_SPEAKER_JUDGE_MARKS=0` 让裁判看不到置信标（「裁判放在哪」那组实验的输入），默认看得到（写成 `s3（低）`、`s3（低，备选 s5）`、`s4（补）`，与量过置信标对裁判有没有帮助的那组实验相同）。

## 任务 Artifact 记录

显式指定 `--task-artifact-dir` 时，`task-artifacts.jsonl`、`exchanges/`、
`task-report.md` 和 pipeline 的 LLM round 汇总全部使用该目录；不会再回退扫描或写入
默认 `<stem>.llm-artifacts`。其中 `task-artifacts.jsonl` 会记录：

- `fast_decision`：快速模式判定（mode/enabled/reason 与输出、输入两侧的估算值和预算）。
- `task_graph`：紧随 `fast_decision` 的一条**目标任务图**记录（2026-09-09 起）：执行策略（`batched`/`agentic`）与其来源（`correction_chain`／`test_profile`／`override`）、能力授权（search＝`--retrieval`、knowledge、style）以及任务列表（普通＝`prepare`+`translate`，fast＝单个 `whole` 的 `translate`，`--knowledge update` 追加 `knowledge_update`）。它是审计面，不是 resume 依据。⚠ 记的那个 `strategy` 是**纠错链**的：策略按任务解析，准备任务读的是研究链，两条链不同时（例如研究钉 agent、纠错混 API）这一条不代表准备任务实际走了哪条——准备任务自己的那次解析记在 `research-context.json` 的 `rounds.task_flow`。
- `prepare_task_request` / `prepare_task_response`：准备任务的请求回复与交付回复（模型、usage、解析错误，以及 `emits_queries`/`emits_entries`/`native_search`/`agentic` 等形态字段；token 报告计入 `phase: research`）；直接交付时只有后者。
- `research_search_results` / `correction_search_results`：本地搜索的 query 列表、每条 query 的 provider/条数/URL/错误，以及注入文本长度。
- `correction_query_response`：纠错窗请求回合的回复（模型、usage、解析错误、提取出的 query 与 window_notes；`delivered` 标记直接交付、`task_flow="batched"`）。⚠ **`delivered=true` 的那条不是一次独立调用**：同一个回复接着就是这一窗的 attempt 0，由下面的 `correction_window_response` 记录，所以它不进会话统计、不出 token 行、也不再写一份 exchange（记录本身留着，它是「请求通道开着但没被用」的审计面）；`correction_query_call_error`：请求回合调用异常（best-effort，不中断任务）；`correction_query_format_retry`：请求回合的格式重试。三个名字沿用旧查询轮的产物契约（bench 打分与已有 run 的读法不分叉）。
- `window`：窗口 id、源序号范围、起止时间、重叠源序号、只读前文序号（`preceding_source_ids`，v13）、音频剪辑区间（`clip_start`/`clip_end`）、边界原因和规划预算。
- `request`：请求输出上限、消息文本字符数和请求文本哈希（fingerprint），以及是否附加 Gemini 文件。真实 token 数以 provider usage 和 token 分布报告为准，request 侧不再做本地估算。
- `provider`：从 raw response 中提取的 `usageMetadata` / `usage`、模型版本、response id 和 prompt feedback 等 provider 元数据；如果 provider 没返回 usage，该字段为空。
- `response`：响应文本长度和哈希，以及提取出的 `next_advice`。完整模型文本仍保留在 `response_content`，供失败分析和任务中知识更新使用。
- `correction_window_retry` 会额外记录失败窗口、重试窗口和（拆分时的）后半窗口。
- `exchanges/` 子目录：每次 LLM API 交互一个可读 markdown 文件。串行阶段（准备任务的各条回复、知识更新）为 `NNN-<call>.md` 按调用顺序编号；纠错窗（请求回合 + 各次交付 attempt）为 `NNN-MM-<call>.md`——`NNN` 是**调度时**按窗口顺序预领的 block 号、`MM` 是该窗内的调用序号，因此并行模式下两次相同的 run 产出完全相同的文件名（完成顺序不影响编号；缓存回放的窗口不占号）。文件顶部先渲染 API Calls 表，逐行列出本 logical attempt 内所有底层 `completion()` 调用（provider tier、model、api key name、同 key+model 的 call #、return code、发起/返回时间、耗时、超出软超时的秒数——只有超过软超时的那次才有值）；模型 fallback 或同 key retry 的失败尝试也会保留，最后一行通常是成功调用。随后是去重后的元数据头（attempt、finish/校验状态、`input_tokens`、`output_tokens_breakdown`、逐输入块 token 等；provider/model/key 只在 API Calls 表出现）、逐 message 的请求全文（音频附件以 `[附件文件: ...]` 标注），以及完整模型响应全文（含 `<reasoning>` 等标签，不再单独摘录）。覆盖准备任务、请求回合、交付轮（含重试/拆分）和统一知识更新调用（逐 chunk 一个文件）；供知识库更新、prompt 迭代等后续任务直接提取信息。
  元数据头包含**`prompt_version`**（当前 `PROMPT_VERSION`，便于对照契约）、**输入分块 token 统计**（`exchange_metadata.py` 的 `*_input_components`，用任务 counter 对各标签块计数，0 值省略）和合并 token 行（`uncached / cached / total` input，`visible / thinking / total` output）：准备任务 `transcript/extra_info/note_url_extract/<每个可匹配类别>_index/preinjected_entry/knowledge_injection/search_injection`；请求回合与交付轮 `csv/media/knowledge(context pack)/entry_details/advice_ledger/pre_round_notes/preceding_context/各类别 index/search_injection/expected_output`；⚠ **index 那几个键是按可匹配类别派生的**（`exchange_metadata.index_block_tokens`，2026-09-07 前写死为 `streamer_index_tokens` / `common_index_tokens` 两个），所以键的**数量随类别集合变**，读日志时别假设正好两个；知识更新 `window_packs/kb_entries/prompt_tokens_estimate`。纠错 validation failed 时，header 会保留错误理由并从错误文本中摘出 row/source id 位置摘要。
- 检索/词条注入类 artifact 均带 `render_report`（included/truncated/dropped 与块 token 数）；请求回合响应另记 `requested_entries/injected_entries/missing_entries`；本地预注入记 `knowledge_preinjection`（匹配词、频次、来源 phase）。
- `content_filter_ladder` / `content_filter_blacklist`：Gemini PROHIBITED_CONTENT 阶梯恢复——某次调用被拦后按「URL leave-one-out → 丢全部 URL → 丢全部检索注入」重建 prompt（不占 validation/parse 重试预算）；定位到的毒块按 `content_hash` 写入黑名单，同任务后续窗口/轮次 render 前预剔除；resume 启动时从 artifact 加载。知识更新与请求回合无检索注入可丢时只做原样重试一次。
- `window_plan_report`：仅当规划因输入超预算发生 `k+1` 重排时写入（research/correction 各自 phase），task report 渲染成 "Window Planning" 小节。
- `token_distribution_report`：每阶段一条（`phase: research` 在准备任务后写入并同时并入 `research-context.json` 的 `token_report`；`phase: correction` 在最终 SRT 写出后写入）。`rows` 逐调用记录 `call/chunk_id/attempt/model/finish_reason` 和 token 分布（`prompt_text_tokens` / `prompt_audio_tokens` / `thinking_tokens` / `output_tokens` / `total_tokens`，来自 Gemini REST `usageMetadata` 的 `promptTokensDetails` 模态拆分与 `thoughtsTokenCount`）；`totals` 为各项求和加 `call_count`，供调参参考。
- `api_call`：本地非 LLM API 调用计数（如 `gemini_file_upload`、`web_extract`），供 `task-report.md` 汇总。
- `../<stem>-metadata.json`：与主产物同级的 pipeline metadata；只记录下载、人声分离、VAD-ASR、LLM harness 和 pipeline 总耗时、相关 worker，以及从本次 resolved artifact 目录汇总的 LLM logical-round 耗时。同一 batch logical run 的后续 pass 保留前一 pass 已执行的 stage；stage snapshot 整条替换，不会出现 `reused` 携带旧执行耗时。单轮跨度覆盖该轮全部失败 attempt、endpoint fallback 与 validation/format retry；底层 API 行仍以 `exchanges/` 为明细来源。轻量的 stabilize/SRT 导出/后处理不单列。
- `task-report.md`：任务完成后给用户阅读的**运行时摘要**——输出路径、上述核心阶段/总耗时和 worker、LLM logical-round 耗时/API attempt/retry、API 调用计数、分阶段/会话 token 用量、**按 (provider tier, model) 汇总的 token 成本视图**（cached 与 uncached input 分列，因为二者计价不同；tier 从 `route_decision` 里答题的那个候选反查，因为同一个模型在免费档和付费档是两笔账，本地 agent 档则按订阅计费、其行只用于缓存与上下文核算；**既反查不出答题者、表里计的 token 又全为零的调用不进该表**（判的是数值不是键在不在：response 记录的 `usage` 恒为补零的全键 dict）——它不是一行账，只会渲染成 `unknown / unknown` 加一次调用和一串零，读起来像「认不出的 provider」，实为「失败在任何 provider 上报之前」，会话表里那一行照留）、fallback 与疑似 IP/代理风控 warning、**所有失败调用的逐条列表**（任何 `*_call_error`，不只下面两类各有专节的——否则 agent 未 submit 这类失败会让本节只剩一句「No fallback was recorded」）、Gemini File 403 提示、重试/拆窗、SRT 后处理与知识库更新摘要。注入上限与 thinking effort 的静态说明见上文「注入上限」「LLM thinking effort」，不在此文件重复。

## 最终 SRT 后处理

当前支持 profile `-1`、`0`、`1`、`2`、`3`、`4`，其他值会直接报错。默认 profile `0` 按顺序执行 profile `3 → 4 → 1 → 2`，从 `*-translated.srt` 生成最终 `*.srt`：

- profile `1`（时长）：每条字幕末端先固定后延 `0.3s`（不得越过下一条开始）；随后把剩余小于 `0.3s` 的短闪轴空隙闭合到下一条开始。
- profile `2`（标点）：中文逗号、中文句号和中文全角空格替换为英文空格；每行首尾 whitespace trim；不修改时间轴。
- profile `3`（繁简）：整篇 opencc t2s 试转，字符差异率超过阈值才判定为繁体并整体转简。
- profile `4`（重叠）：检测相邻字幕重叠（前一条结束晚于后一条开始），把前一条的结束提前到后一条开始，并向 stderr 打一条 `Warning:` 报告条数与首个实例。重叠是上游时间轴缺陷，不是渲染选择，所以必须可见。乱序输入（后一条开始早于前一条开始）会把该条压成零时长而不是让结束早于开始。
- profile `0`（默认）：`3`（繁简）→ `4`（重叠）→ `1`（时长）→ `2`（标点）。`4` 必须排在 `1` 前面：`1` 的「不得越过下一条开始」会把重叠的字幕**截短**并计进 `duration_extended`，先解重叠才能让那个 cap 退化成 no-op。

`--postprocess-profile -1` 不做时间轴或文本清理；实现仍会解析并重新渲染 SRT，因此字幕时间与文本语义保持不变，但不保证字节级原样复制。

`*-raw.srt` 跑同一套时间轴策略 `4 → 1`（见「产物」一节），文字保持 ASR 原样。顺序常量
`TIMELINE_POSTPROCESS_PROFILES` 是唯一来源，profile `0` 与 raw 导出共用，不会各写各的。

`final_srt` artifact 与 `task-report.md` 会记录请求的 `profile`、实际 `applied_profiles`、重叠修复条数 `overlaps_fixed`、末端延长条数 `duration_extended`、闪轴闭合条数 `flash_extended`、标点替换数与 trim 行数。

## 重试与拼接

每个纠错窗口的目标是一次 API 交互完整成功。

**API 调用的两段超时**（`llm_runtime.api_timeouts`，只管 REST 生成调用；本地 agent 用
`[llm] local_agent_timeout_seconds`，上传/countTokens/检索各有自己的）：

- 非流式调用在答案生成完之前一个字节都不回，所以等待时间就是整段生成时间，时限必须随**这次
  允许的输出量**变：软超时 = 180 s + `maxOutputTokens` ÷ 25 tok/s，硬超时 = 软超时 × 2；
  没有给上限的调用按 65536 算。64k 输出的软超时约 46 分钟、8k 约 8 分钟。三个数可在
  `config.toml` 的 `[llm]` 下覆盖：`api_timeout_overhead_seconds`（≥0）、
  `api_timeout_floor_tokens_per_second`（≥1）、`api_hard_timeout_factor`（≥1）。
- 用的是**实际发出的 `maxOutputTokens`**（已按上下文剩余截过，思考 token 也算在里面），**不是**
  规划用的输出估计（`output_reserve`）——答案本来就允许超过估计，按估计定时限会把这些答案在
  生成途中切断。
- **软超时不断连接**：到点发 `llm-api-soft-timeout` warning，继续等；在软、硬之间返回的结果照常
  使用，attempt 记 `soft_timeout_sec` / `soft_timeout_overrun_sec`（exchange 文件的 API Calls
  表多一列），并发 `llm-api-late-response`。只有**硬超时**才断开并记 `NO_RESPONSE_TIMEOUT`——断开
  之后结果不可能再到达本进程，所以「超时后返回」只能靠不断开来观察。等待几乎不耗资源，而重试要
  从零重新生成。
- ⚠ **硬超时是整次调用的截止，不是 httpx 的 timeout。** 后者只限单次读写的等待，服务端隔一会儿
  发一个字节（分块传输的 body 慢慢往外滴）就永远触发不了，调用会一直占着窗口槽位、连续超时退出
  也不会发生。所以 `http.post_within_deadline` 自己守截止，三件事：
  1. 请求在工作线程里跑，调用方只等到截止——哪怕工作线程卡在谁也打断不了的 DNS 解析里，调用方
     也准点返回。等待按 0.5 s 分片（Windows 上带超时的 `join` 打断不了），所以 Ctrl+C 立即生效，
     并且同样拆掉在途请求；
  2. 到点**拆掉**请求而不是丢下：对 socket 做 `shutdown`（能唤醒另一个线程里阻塞的读，Linux 上
     光 `close` 不行），再关 client。socket 通过 httpcore 的 `trace` 扩展在每一步产出时拿到——
     先是 TCP 的，再是替换它的 TLS 的（`wrap_socket` 会让原 socket 失效，只 shutdown 它等于
     什么都没做），所以还没收到响应时也拆得掉。⚠ TLS 握手进行中那一小段没有可拆的 socket，
     由 connect 超时（≤45 s）兜住，调用方不受影响；
  3. 同一个 `trace` 在「开始连接 / 连接完成 / 开始 TLS / 开始发请求头」这几步检查截止，**过了截止
     就拒绝往下走**——卡在 DNS 里的工作线程回来后也发不出请求，不会在没人等的时候触发一次远端生成。

  是否超期一律按**单调时钟**判断，不看计时器有没有跑过：逐次读超时与截止共用同一个预算，谁先触发
  是调度噪声（Windows 上线程唤醒约晚 15 ms），异常类型不能取决于它。只要截止决定了结局，调用方
  拿到的就是 `DeadlineExceeded`——`httpx.TimeoutException` 的子类、文本带 "timed out"，重试与
  分类规则照旧把它当超时；截止之前的真失败（例如 45 s 的连接超时）原样抛出。httpx 的 timeout
  保留，只作停滞检测。证明在 `test/test_llm_http_deadline.py`（真 socket 的几条要
  `--run-network-mock`）。
- connect 阶段单独 45 s，不随输出量放大。
- 连续两次硬超时仍按原规则立即抛出（`CONSECUTIVE_TIMEOUT_ABORT_COUNT`），所以一个窗口最坏占用
  约两个硬超时。

**媒体上传有自己的重试预算，与模型调用的分开。** `RoleClient(max_retries=...)` 从拿到 media
ref 之后才开始计，所以上传本身的一次网络中断（2026-08-20 实例：finalize 步
`httpx.ReadError [WinError 10054]`，经 `http_proxy`）曾直接终止整个 LLM 阶段。现在
`_upload_gemini_file_rest` 内部：

- 最多 3 次尝试，退避 1s / 3s 加 ≤0.5s jitter；`429`/`503` 等带 `Retry-After` 时用它——只认
  秒数形式且 ≤30s，HTTP-date 形式按固定退避。**每次尝试新建 resumable session 整体重传**——中断后的旧 session 状态不明，也没有
  offset 查询；不做断点续传。文件内容只读一次。
- 可重试判定**按 httpx 异常类型**（`ConnectError`/`ReadError`/`WriteError`/
  `RemoteProtocolError`/`TimeoutException`）与状态码（408/425/429/500/502/503/504），不看
  错误文本——模型调用那套 `is_retryable_provider_error` 靠 `timeout`/`unavailable` 之类的
  词，10054 一个都不含。其余 4xx、缺 `x-goog-upload-url`、本地文件错误立即失败。
- finalize 成功之后只重试轮询：state GET 与 `countTokens` 各自按单次请求计 3 次，正常的
  `PROCESSING` 等待不消耗它；不会重新上传造成远端重复文件。`countTokens` 的 400 是「尚未采样」
  的就绪信号，照旧等待；其余非 200 走同一套分类（429/5xx 重试，401/403 立即失败，不会被当成
  没就绪等满 5 分钟）。
- 超时从单一 600s 拆成 `httpx.Timeout(connect=45, read=600, write=600, pool=45)`，是每个网络
  操作的上限，不是整次上传的总时限；connect 给到 45s 是照顾代理 CONNECT + TLS。
- 上传跑在 `WindowClipPrefetcher` 线程里。prefetcher 关闭时先 set 一个 `threading.Event`，
  上传在每次尝试之间、退避等待与轮询间隔里都检查它，收到即抛 `UploadCancelled`；**已经在线上的那个请求
  不能被打断**，仍可能跑满自己的超时——取消只保证不进入下一次尝试。
- 最终失败抛出的是最后一次的原始异常（不包一层），日志 `gemini-upload-failed` 带尝试次数。

- 不做多轮续写。
- 输出上限判定用两个信号：usage 计数（输出+thinking token >= `65536 - 100`，主信号）与 `<translated>` 开标签无闭标签（兜底）；finish reason（`MAX_TOKENS` 等）只记录、不参与判断，原因见下一条。
- **agent 传输的 usage 不参与这个判定**：CLI `result` 事件里的 usage 是跨工具轮累计的整个会话，不是单轮输出，和单轮上限比会把完整的结果误判成截断、白拆一次窗口。产物记 `basis: agent_session_cumulative_usage_not_comparable`；这条路径上的截断只认显式信号（错误结果、输出校验失败）。
- 拆分判据（2026-08-08 修正）：**主信号是 usage 计数**（`output_tokens_plus_thinking_tokens`）。`finish_reason` 记录进产物但**不参与判断**——`46206b1` 有意降级它，因为 flash 会在输出完整时误报 `length`，害得生产窗口 0001 把一个通过校验的完整结果白拆一次。`<translated>` 开标签无闭标签这条内容启发式作为**兜底**：仅当 usage 缺失或低报（导致主信号为假）**且窗口重试已用尽**时才触发，避免「只是需要再试一次」的窗口被提前拆开。此前它完全没有接线，于是截断的回复表现为普通校验失败：同一个超长窗口再发 5 次，然后以 `RuntimeError: Window NNNN failed validation` 杀掉整个任务。满足以上任一即把当前窗口**对半拆分**重试：在窗口中间附近选择合适边界（句末 > 长静音 > 片段边界）拆成两半，两半之间保留与正常窗口同规则的动态重叠（切点前 30s 内条数，纯内容驱动，稀疏处可为 0），先处理前半，再处理后半；每个半窗有自己的音频剪辑与上传，-a 继承父窗口的只读前文、-b 回看父窗口尾部。
- 子窗口 chunk id 是 `父id-a` / `父id-b`；两个子窗口按窗口 id 继承父窗口的 window context。serial 下前半的 `<next_advice>` 传给后半；parallel 没有 advice，叶窗口可独立执行。总调用次数仍受两档重试预算约束（见下）；单片段窗口无法拆分时同窗口重试。⚠ **只拆一层**（`WindowGeometry.MAX_SPLITS = 1`；owner 2026-09-02 定的 2，**2026-09-03 改为 1**）：半窗还不成就不是尺寸问题了，再拆只是拿配额换同一个失败——此时抛 `RuntimeError` 停掉该 task。所以生产能产生的 chunk id 只有 `父id-a` / `父id-b` 两层，**不会出现 `0001-a-a`**。深度直接从 chunk id 数 `-` 得出，不另记计数器（id 就是血缘，计数器会与它漂移）——读法与上限是两件事，`split_depth` 对一个 `0001-a-a` 仍答 2，只是没人再造得出它。
- CSV 格式错误、未知源序号、重复源序号或源序号乱序默认同窗口重试，重试后仍失败则报错。
- **同窗口重试是修复轮，不是盲重掷**（2026-08-15）：`reason=validation_same_window`
  的下一次 attempt 会带上**上一轮的输出**和**校验器给出的每一条错误**。表示形态由
  transport 决定——无状态端点收到追加的 `assistant`（上轮输出）+ `user`（错误清单 +
  「重新输出完整结果，不要 diff」，模板 `fragment_repair_round_v1.md`）两轮；本地 agent
  收到的是 capsule 输入（`input/previous-output.txt`、`input/validation-errors.txt`）。
  修复轮**不额外占预算**：它就是原来那次重试，只是 prompt 里多了东西，所以
  `--max-retries-per-window`、checkpoint 身份与 artifact schema 都不变。
  三种情况下不带修复上下文，退回原来的盲重试：
  - **窗口变了**——拆半窗或截断重试，上一轮的输出讲的是别的行；
  - **装不下**——附加后超出该候选的 `max_input_tokens` 时逐候选丢弃（决策记
    `repair_context: dropped_input_limit`），因为丢掉辅助也比丢掉这次尝试好；
  - **agy 且没有会话复用**——见下。
  产物里 `correction_window_response.repair_round` 记录这次 attempt 有没有带，
  `correction_window_retry.repair_context` 记录下一次会不会带。**exchange 渲染的 message
  列表是基础 prompt**——修复轮是在 client 里按后端追加的，不经过调用方组装的那份，所以
  不在下方 message 里；exchange 头因此只在带了时写一行 `repair_round`，指向上一 attempt
  的 exchange（那里有上一轮输出全文与 `## Validation` 里的错误原文）。
  ⚠ `repair_round` 只说明调用方发了修复上下文，不说明 driver 用上了：被拒收时它仍记 true，
  审计要同时看 `execution_attempts[0].repair_context`（`declined_by_driver` 即实为盲重掷）。
  **效果（2026-09-30 结案）**：当时归档里 10 次修复轮全部一次通过；其中 6 次是在上一轮真实内容上修的
  （另 4 次上一轮只有一个时间戳或被 driver 拒收，等同盲重试），这 6 次保住了上一轮 99.5%–100% 的字幕行，
  没有一次整窗重写，唯一的回归是一次在 `corrected_text` 里各丢了一个假名、译文未变。所以按「保住已经对的
  内容」判，修复轮优于整窗重来；收益在内容保持而不在省 token——agent 路修复回复照样重发整窗。
  盲重试的通过率没有对照数据，「通过率更高」这一点没有立论。
- **agy 只在写出这份答案的那个会话里修**（owner 决定 2026-08-15）。新开会话要把整窗连同
  上一轮输出重发一遍，而那第二份对模型不是同一个东西：在自己的会话里它是「刚写的」，
  换个会话就成了「别人给的一段文本」，代价还落在 agy 最贵的媒体窗上。所以
  `session_scope=assignment` 且有 conversation handle 才接修复上下文，否则该次重试保持
  盲掷（执行记录写 `repair_context: declined_by_driver`）。复用会话时上一轮输出已在
  上下文里，prompt 因此只指向 `validation-errors.txt`，不为第二份副本付钱。
- **会话内修复（2026-08-17）**：一个窗口的整条重试链共用一个
  `repair_session_key`（`correction-<chunk_id>`，由 `attempts.py` 传入）。第 0 次尝试开
  会话并记下 handle，之后每次修复带着它以 `session_scope=assignment` 回到同一会话，
  于是上面那条对 agy 的限制自然满足，其他 agent 也省掉整窗重发。**只在窗口内**——新
  窗口一定开新会话，跨窗复用仍然关闭（那是另一回事：跨窗复用在 agy 上实测净亏，见
  [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) [§3.1 agy 上的 A/B](llm_local_agent_experiments.md#31-agy-上的-ab复用是净亏2026-08-14)；重测协议在
  [`llm_followups.md`](llm_followups.md)）。
  `supports_session_reuse=false` 的 driver 拿到 `assignment` 是**发车前硬失败**而不是
  降级，所以先探能力，探不到就全重放；会话变冷（TTL / compact / CLI 被杀）则回落一次
  全重放，即本条改动前的行为。**额度耗尽与策略拒绝不走这条回落**——重建会话补不回
  额度，重试只会白发一次车，还会盖掉配额账本用来判冻结的两次连续失败之一。会话缓存
  按 `(provider_tier, model, chain)` 索引，与 driver 缓存同口径：session id 属于签发它
  的那个 CLI，而后续 attempt 是独立路由的，可能落到另一家或同模型的另一个档。这**不是**长驻 worker 的接线；`agent_session_mode`
  旋钮自 2026-08-19 起由这条路读取（四档：`api`/`per-window` 默认/`resume` 实验开关/
  `pseudo-conversational` 拒绝），本条描述的正是默认档 `per-window`，见
  [`llm_local_agent.md`](llm_local_agent.md) [§12.1.1 档位是唯一配置面](llm_local_agent.md#1211-档位是唯一配置面传输由它派生2026-08-22-起-per-window-默认即工具会话)。
- **重试预算是两档**（2026-08-19）：
  `--max-retries-per-window`（默认 5）是**一条会话链内**的修复次数；链用尽后
  `--max-replacements-per-window`（默认 1）次把窗口交给**全新会话**——修复上下文丢弃，
  agent 开新会话、无状态端点盲重掷，每次替换照常重新路由。一个窗口的调用总数是两个
  (n+1) 的乘积（出厂默认 6×2=12 次上界；典型路径「一两次修复就过」不变）。
  `correction_window_retry.replacement` 记录下一次是否为替换。**两个独立旋钮而不是一个
  总预算**（owner 2026-08-19）：两档的单价差一个量级——第一档省的是不重开会话与可能的
  前缀缓存命中（不是「只发 delta」，assignment 每轮仍重发 protocol + run_context + 整份
  manifest），第二档是完整重放加冷缓存——所以「最坏 (档1+1)×(档2+1) 次」这个数严重高估
  真实成本，第二档保持「最大替换次数」的字面意思作为一等旋钮。原始论证在维护者的本地归档里。
  两档都登记在 `finesub.settings`（`[llm] max_retries_per_window` / `max_replacements_per_window`），命令行值优先。
  ⚠ 将来别的前端要带这两档，**两档必须同批**，不能只加一个。
- **并行熔断按「会话链耗尽」计数，不按窗口**（2026-08-19）：两档之后一个窗口要烧满
  12 次调用才算失败，按窗口计会让熔断的价码随第二档翻倍（实测 4 lane 24→48 次）。
  按链计与改动前同价——链耗尽正是熔断一直在计的东西，两档之前「一个窗口失败」就等于
  一条链耗尽。同一个计数器也收「窗口彻底失败」，因为最后一条链走的是抛错路径、到不了
  链边界的钩子；熔断跳闸后在飞的窗在下一次 attempt 边界停手。
- **熔断不保证批次待在日额度以内，owner 2026-08-19 判定不必修**。实测（4 lane，
  第一档 5）：齐步 26 次调用、错峰 34–36 次；关掉第二档则两种调度都是 24 次。
  地板 24 = `lane 数 × 链长`，**不可约**——每条 lane 的首链都在任何证据出现之前就花掉了；
  多出来的部分是「信号还没到阈值时合法启动的替换链」，那些调用在发出的当下都有依据
  （彼时只有一个窗口失败，而第二档的定义就是「这个窗口的会话退化了」）。
  **超额不是失败模式，是一次路由事件**：多出来的调用换到 PerDay 429，而 429 要连续
  `DAILY_STRIKE_COUNT` 次才锁档（并发同时撞上的多个 429 由 `departed_at` 闸门判为**一次**
  观测），锁上之后该候选按 `daily_exhausted` 跳过，落到下一个免费 flash、直至付费尾。
  净代价是「在注定失败的批次上多几次失败尝试」加「该模型当日免费额度提前用完」。
  key 的用法与 serial 一致（2026-08-19 改判，并行不再额外钉 key，见
  [`llm_harness_routing.md`](llm_harness_routing.md)）：配额错误先在同一把 key 上原地
  sticky 重试，预算花光才轮到下一把，所以「免费档耗尽」不等于「这条链到此为止」。
  带媒体的调用仍钉一把（文件属于上传那把 key 的 project）。**被否的更紧方案**：替换链发起前预留配额能把错峰压到 ~20，但代价是
  「两个窗口各自只需要一次替换」就熔断掉一个本来能跑完的批次——拿真失败换一次配额事件，
  方向反了。真要卡死上限，杠杆是 `--parallel-windows`（2 lane 实测 24 次）或调小第一档。
- **只有输出上限（usage 主信号）、或重试用尽后 `<translated>` 明显截断（兜底）才对半拆分**；
  格式错误一律同窗口重试。Gemini `503 high demand` 只做同请求退避重试，失败后记
  `correction_window_call_error` 并停止——**不拆窗口**（那不是窗口太大）。重试预算内触发的
  拆分：前半窗继续用剩余预算、后半窗排队为新单元；在最后一次重试上触发的拆分则两个半窗
  都作为全新单元重新入队（各自完整重试预算，串行模式下重走 resume 检查），而不是以
  "failed validation" 终止。
- **采样参数**：生成调用默认显式传 `temperature=1.0`；validation/parse retry 的第 N 次
  logical attempt 用 `temperature=max(0, 1.0 - 0.01×N)`，并在末尾 user message 追加
  `(seed=N)` 文本提示（Gemini REST 没有原生 seed 参数）。成功后的下一个独立窗口/轮次从
  attempt 0 恢复。`top_p` / `top_k` 不显式设置，保留 provider 默认。
- **Content filter 阶梯**（独立于上述 validation 重试）：`finish_reason=content_filter` 且空输出时，调用侧按注入 unit（URL 提取 / query 结果）逐级丢弃重建 prompt 再调；同任务黑名单跨窗口生效。全部丢弃后仍被拦则报错（疑似源文本触发）。准备任务（首条消息的 URL 预提取、回填消息的搜索块）、纠错请求回合与交付轮、知识更新均接入（知识更新与无检索注入可丢的请求回合仅原样重试）。

当前本地校验包括：

- 必须有且仅有一个 `<translated>...</translated>` 块。
- 列**按位置严格解析**：9 列（`type|position|duration|gap|corrected_text|translation|conf|char_count|note`），basic 档在 `position` 后多一列 `start` 共 10 列。`duration`/`gap`/`char_count` 须为数字。列数不符即结构性错误，**不做启发式重排**——只有末列 `note` 允许含半角 `|`（prompt 要求正文用全角 `｜`），解析按 `maxsplit` 保住它。
  旧的 3/5/7 列容忍已删除（2026-08-07）。此前解析器靠单元格内容猜布局，多一列的行会被整体左移一格，把**未翻译的原文**写进 `translation` 且不报错。唯二的消费者都不涉及 `data/` 里不可再生的人工标注：`annotated.csv` 由纠错运行自己产出、可重跑，`tools/session_replay` 按约定为按需维护。
- 本地校验暂不限制 `<translated>` 每条 `sub` 引用的源序号数量；仍校验源序号存在、唯一且顺序正确。prompt 要求通常使用单源或两源，只在少数同一句连续三切时允许三源，并禁止四源以上；不向模型暴露这一临时校验放宽。
- 源序号必须属于当前窗口，且每个源序号最多出现一次。
- 同一行和不同行的源序号都必须保持源时间顺序。
- 空 `<translated></translated>` 合法，表示当前窗口全部丢弃。

拼接规则（重叠区倾向采用最新窗口译文，且对跨界 merge 安全）：

- 所有窗口输出先解析为带源序号的字幕片段。
- 旧结果中**完全落在**当前窗口源序号集内的行被替换为新窗口输出（newest wins）。
- 旧结果中**跨越重叠边界的 merge 行**（含当前窗口之外的源序号，如 `79,80,81` 只有 81 在重叠区）整行保留，并"认领"其落在当前窗口内的序号；与认领序号冲突的新行被丢弃，其因此丢失的其他序号再从被替换的旧行回填，保证不产生覆盖空洞。
- 新窗口有意丢弃的序号保持丢弃（回填只针对"因冲突丢失"的序号）；已知边界情形：双跨界 merge 行回填时可能连带恢复个别被有意丢弃的序号。
- 加入新窗口字幕片段后按原始时间轴排序并重新编号。

## 知识库更新行为

知识库结构、任务反馈采集与统一知识更新的完整行为见 [`knowledge.md`](knowledge.md)；本文其余章节仅涉及采集开关对纠错 prompt/resume fingerprint 的影响。

## 任务级并行与 agent 槽位预算（2026-08-30 落地）

多个纠错 run 可同进程并行（设计与验收记录在维护者的本地归档里）。
现行行为：

**三个旋钮，语义分开。** `[llm] local_agent_max_parallel`（默认 4）是**物理上限**——本机 +
订阅同时活跃的 agent CLI 进程数，**每个 vendor 一份**进程级预算（`AgentSlotBudget`，按
driver_id 共享——同订阅的不同模型共用一池；agy 的 tool-slot project 按 domain root 共享，
两个并发 run 拿到不同的 `.finesub-tool-<slot>`）。`finesub.pipeline --max-parallel-tasks`
（默认 2）是**准入上限**——同时活跃的任务数，纯 API 任务同样受它约束。
`--llm-parallel-windows`（默认 1）是**单任务的意愿上限**。三者都是上限不是配额：约束是
瞬时槽位占用，三者都由用户覆盖。**它们不是彼此独立的**：保底格在**任务起步**就预留，所以
`local_agent_max_parallel` 小于 `--max-parallel-tasks` 时，第二个任务会卡在起步的 `reserve`
上等到第一个任务整体结束——连它本可以先跑的纯 API 阶段也一起等。要并行多个任务，物理上限
必须不低于准入上限。

**保底 lane（不变式 I1）。** 路由链可达 agent 后端的任务起步时在**每个可达 vendor 的预算**里
各 `reserve` 一格（`TaskSlotAccount` + `TaskSlotClaims`，按 catalog 过近似判定需求；调用落到
哪个池按次决定，只保首池会让路由到第二家 vendor 的调用没有保底——reviewer 2026-08-30 P1；
纯 API/测试档不占）。必得 lane 的每次调用在**本池的** claim 上锁内一步兑现（reserved→held）、
调用结束摆回（held→reserved），**任何时刻都不会被别的任务的可选扇出饿死**——可选调用只吃
`free = limit - held - reserved`。pseudo-conversational 的长驻 host 在创建线程捕获 claim 集、
由 supervisor 一次 enter 消费保底，CLI 退出才摆回；task 结束时保底仍被 host 占着的话由那次
enter 的退出直接释放回 free（不炸收尾、不漏 reserved）。兑现从不等待：等待意味着账面与池
脱节，直接报错。

**动态扇出。** 并行纠错的每个阶段取 `want = min(1 + ⌊free/A⌋, parallel_windows, 待跑数)`
（`A` = 进程内有 agent 需求的活跃任务数）。批繁忙时 `claim_cap→1`，每个任务自动退化为
单 lane 串行——语义仍是 `continuity=parallel`（advice 链已撤），只是顺序执行；`continuity`
永远不由分配器改写。

**lane = run 发放的逻辑 ordinal**（`run_context.LaneOrdinalPool`）。lane 是对话身份：`resume`
会话与 pseudo host 都按它键。ordinal 属于 run 而非线程/池——查询与纠错两个先后线程池领到
同一组 1..N、命中同一批会话；串行段（research/知识更新）的 lane 在阶段期间借给池、之后取回。
pin（知识库快照）、session registry、reporter、lane 同走线程池 initializer
（`run_context.bind_llm_worker`）进 worker——ContextVar 不会自己进新线程。

**conversational 强制 serial。** 纠错格绑定 conversational 后端时 `continuity=parallel` 被
强制改写为 serial 并告警（`conversational-forced-serial`）：一个人的 agent 是一条队列，扇出
只会剥掉 advice 台账而不省墙钟。它不占 driver 槽（不进预算）。**同一条闸门也管研究分块**：
超长素材的分块调查在扇出前按**研究格自己的**路由问一次（纠错格的答案说明不了这一阶段），
命中就整段串行并发同一条告警——纠错阶段的改写发生在研究之后，覆盖不到它（2026-08-30 复审）。

**知识库侧**：并发 run 各读各的 pin、写路径进 per-root 单写者队列、真并发冲突按类型化解，
见 [`knowledge.md`](knowledge.md)。**batch 侧**：组串行 / LPT / 调度序记录，见
[`batch-scheduler.md`](batch-scheduler.md)「llm bin 的并发语义」。**上限怎么定**：owner
2026-08-30 裁定不做标定实验——并发上限是用户偏好（各家 agent 对并行流量宽容；代价是 5h
额度被快速耗尽导致 session 中途白费，以及墙钟收益边际递减、token 效率与上下文损失边际
递增），选值指引在 [`manual/agent.md`](manual/agent.md)；出厂默认 windows 1 / tasks 2。
