# 本地检索代理与背景调查

> 从 [`llm_harness_behavior.md`](llm_harness_behavior.md) 拆出（2026-08-16，原文 930 行）。
> 那份仍是 harness 运行时行为的总入口；本文是它的一块，单独成篇是因为读者不同。

## 本地检索代理

gemini-3.x 免费层级不开放 `google_search` grounding（实测立即 429），所以纠错/调查模型不直接启用联网工具；检索由 `finesub/llm/web_search.py` 在本地执行，含 **search**（网页搜索）与 **extract**（单 URL 深度整页提取）两类。Gemma4 fallback 使用 `gemma-4-31b-it` + 通用 Search grounding 免费配额（约 1500 RPD），区别于 Gemini 3 免费层级专用 grounding 0 RPD：

- search 顺序：Exa（`type:"deep"`，`contents.highlights.query` + summary，`x-api-key`）→ Gemma4 grounded（`GEMINI_FREE`，默认 `<|think|>` + 中等深度；若接地 metadata 为空，会用同一请求去掉可见 thinking token 自动重试一次）→ Tavily（`auto_parameters` + `include_answer=advanced` + `max_results=10`，Bearer）。**没有免 key 的本地兜底**：全部失败则返回错误结果。
- extract 顺序：Exa `/contents`（summary + highlights）→ Gemma4 grounded（prompt 中先把 URL 百分号转义还原为标准字符）→ Tavily `/extract`（`chunks_per_source=5`）。同样无本地兜底（全部失败则返回错误结果）。
  search 曾有一层免 key 的 DuckDuckGo HTML 抓取兜底，**0.5.0 移除**：页面一改它就静默解析出空结果，
  而「零条结果的成功」读起来正是「网上查不到这个」——那是检索最不能编造的一个答案。
  本地 search/extract 之后可接开源本地检索 MCP。
- key pool：`.env` 的 `GEMINI_FREE` / `GEMINI_PAID` / `EXA_KEYS` /
  `TAVILY_KEYS` 只保存 `{name:key,...}`；根目录 `config.toml` 的 `[pools]` 按名字
  筛选和重排，[`config.example.toml`](../config.example.toml) 为模板。空/缺失 pool
  默认取 Gemini Free 前 2 把、Exa/Tavily 前 3 把，Gemini Paid 默认全取且无推荐
  上限；显式 pool 超过推荐数只告警、不截断。`[providers]` 可关闭 Exa、Gemma4
  grounded 或 Tavily；Gemma4 复用选定的 Gemini Free pool。未配置 key
  或被关闭的 provider 静默跳过（不产生 fallback 事件）。
- key 不可用判定：Exa 遇 401/402/403/429、Gemma4 遇 401/403/429、Tavily 遇 401/403/429/432/433 视为该 key 不可用，在 runtime state 目录中按 provider 锁定 24h 并选 pool 内下一 key 重试；源码 checkout 默认 `<root>/.state`，可用 `FINESUB_STATE_DIR` 覆盖，wheel 无 checkout 时使用用户 state 目录。某 provider 全 key 锁定/失败即回退下一 provider。Gemma4 如果没有返回 usable `groundingMetadata.groundingChunks`，也视为 provider 失败并继续 fallback。全部失败则该 query/URL 记为"搜索失败/提取失败"，流程继续，不中断任务。
- **引导语（guided query）**：search query 与 extract URL 均可带一句话引导语——对 search 映射到 Exa 的 `highlights.query` 与 Gemma4 的 `search_goal`（Tavily search 无对应即忽略），对 extract 映射到 Exa `highlights.query` / Gemma4 `extract_goal` / Tavily extract `query`；只影响网页重点提取方向，不改变搜索关键词。
- `search_many`/`extract_many` 按 `(query|url, 引导语)` 去重、按上限截断（准备任务 `min(16, 8 + sqrt(raw字幕片段数)//10)` 条 / 每纠错窗口 8 条；extract 只由 harness 对 `--extra-info` 里用户给的 URL 发起，模型不再指定提取目标，见下），并做 1.5s 限速（Exa 的 10 qps 限制远宽于此）。
- Gemma4 search 单次 pass 最多接收 8 条 query；若 pending query 更多，会按 8 条一批自动分批调用。Gemma4 grounded REST 调用单独使用 1200s timeout；多次真实测试中，触达最大输出时耗时可到约 900s。Exa/Tavily 共用 `DEFAULT_SEARCH_TIMEOUT_SECONDS = 60s`（2026-09-16 由 30s 上调，未进 `config.toml`）：30s 落在 provider 自身的耗时分布之内而非之外——一次调查密集的任务里 15 条 Exa 查询有 6 条读超时，且全部在重试时以 24.5–30.0s 成功，即天花板切掉的是正要返回的健康调用，每次代价是等两遍。⚠ 代价在另一头：读超时走通用重试（`max_retries = 7`，退避合计约 63.5s），provider **卡死不响应**时一条查询换到下一个 provider 前的最坏等待从约 304s（8×30s）变成约 544s（8×60s），且查询串行。今天按这个代价接受；若实测出现卡死的 provider，先改的是读超时的重试次数，不是把超时调回去。
- Gemma4 search 单 query 注入量级（2026-07-10 真实 8-query smoke）：按 `render_search_results([result])` 计，观测范围约 283–1079 tokens，中位数约 1062 tokens。Google grounding 的 redirect URL 较长，title+URL token 往往占主要部分；预算估算可先按 300–1200 tokens/query 记，source-heavy 查询留到 1500 tokens/query 更稳。
- 结果按 query 分组渲染（provider、标题、URL、摘要，含长度截断）后注入 `<search_results>` 块；query section 用 `--- query: ... ---`，深度提取 URL section 用 `--- 深度提取 url: ... ---`，避免 Markdown `###` 标题与正文错位。图片及其 URL 在清洗阶段丢弃。
- **模型不指定提取目标**（2026-09-09 起）。搜索 loop 删除后，`extract` 只剩 harness 对 `--extra-info` 里用户给的 URL
  做的预提取（≤8 个，`retrieval=local`，准备任务首条消息前）；准备任务与纠错窗的请求通道里模型能发的只有 search query（可带引导语）与词条请求。
  旧 loop 的「extract 只能选已展示过的 URL」白名单（`normalize_url_key` 与 `rejected_extract_urls`）随之删除——它守的通道
  已不存在。search query 从不设此限：它只发往 Exa/Gemma4/Tavily，注入攻击者没有接收端。

### 已验证的 API 响应结构（Exa/Tavily 2026-07-04，Gemma4 2026-07-10 实测）

四个接口均用真实 key 打通（HTTP 200），响应字段与 `web_search.py` 的解析一致；清洗后只保留文本，`image`/`images` 一律丢弃。

| 接口 | 顶层键 | `results[0]` 字段 | 解析取用 |
| --- | --- | --- | --- |
| Exa `/search`（`type:"deep"`） | `requestId, resolvedSearchType, results, searchTime, costDollars` | `id, title, url, highlights[], summary, image` | `title` / `url` / `summary`+`highlights`（`_exa_snippet`） |
| Exa `/contents`（extract） | `requestId, results, statuses, costDollars, searchTime` | `id, title, url, author, highlights[], summary, image` | `title` / `summary`+`highlights` |
| Tavily `/search`（`include_answer=advanced`） | `query, follow_up_questions, answer, images, results, auto_parameters, response_time, request_id` | `title, url, content, score, raw_content` | `title` / `url` / `content`，顶层 `answer` | 
| Gemma4 grounded `generateContent` | `candidates, usageMetadata` | `groundingMetadata.groundingChunks[]`, `groundingMetadata.groundingSupports[]`, content JSON block | `groundingChunks.web.title/uri` 作来源；`groundingSupports.segment.text` + `groundingChunkIndices` 作支持片段；content JSON 仅用于按 query/URL 分组和摘要 |
| Tavily `/extract`（`chunks_per_source=5`） | `results, failed_results, response_time, request_id` | `url, title, raw_content, images` | `raw_content`（回退 `content`）；`failed_results` 用于报错 |

要点：`type:"deep"` 被 Exa 接受（返回 `resolvedSearchType`）；`include_answer=advanced` 实测返回非空 `answer`；guided query 经 `highlights.query` 生效（Exa summary/highlights 会围绕引导语聚焦）。Gemma4 返回必须含 grounding chunks，否则不把未接地模型正文注入。Exa 接口返回 `costDollars`，属计费/额度调用。

## 准备任务（一条对话，按轴收放请求通道）

准备任务（`stages/prepare_task.py`，2026-09-09 起取代旧的 r1/r2 两轮 + 多轮搜索 loop）在 `research` 任务组上跑，角色标签
`general_capable`。输入是带窗口标记的紧凑字幕文本（`源序号|文本`，边界处插入 `--- window N ---`）、用户额外信息、
`retrieval=local` 下由 harness 预提取的 `--extra-info` URL（≤8 个，`<note_url_extracts>`），以及知识开启时的各类别 index
与按备注命中预注入的词条（≤8 条）。哪些请求块出现由 `retrieval` 与知识索引实况决定、与 `media` 无关——逐档表见
[`llm_harness_behavior.md`](llm_harness_behavior.md)「准备任务与请求通道」；这里只讲这条对话本身。

- **首条消息**：完整任务 + 交付契约 + 「获取缺失输入」一节（三种形态：batched 请求通道 / agentic 自取 / 无外部证据，
  `fragment_prepare_{request_channel,agentic_acquire,no_evidence}_v1`）。模型要么直接交付 `<context_pack>`，要么只输出
  `<reasoning>`、`<analysis_notes>`（≤1500 token）与请求块：`<requested_entries>` / `<keep_entries>`（有索引时；
  request/keep canonicalize 后各≤8、合计≤12，keep-first，超限从 request 尾部丢弃；下游没有请求能力的档位 request 直接 12）、
  `<search_queries>`（仅 `local`；上限 `min(16, 8 + sqrt(原始段数)//10)`，可带 ` >> 引导语`）。
- **回填**：harness 从本地 KB 加载词条全文、执行一轮本地搜索，把两者作为下一条 user 消息追加（`prepare_task_results_v1`；
  模型的请求回复原样作为 assistant 消息放回历史，REST 适配器因此仍只发一段合法历史）。回填只有一次：第二条回复必须交付，
  再提请求按解析失败处理、走同一重试预算，不会有第二次回填。
- **交付**：`<context_pack>` JSON——`general_context` 注入所有窗口；harness 把模型按窗口 id 给出的 `window_contexts`
  绑定为 `{window_id, first_source_id, last_source_id, context}` 区间后落盘。`retrieval=native` 的交付调用自身开原生搜索
  工具；`none` 且无索引时 prompt 要求 `sources` 留空、推断进 `uncertainties`。采集反馈时（`--knowledge update`，或参考资料导入）交付还带
  `<task_update_feedback>`；`--style auto` 时带 `<style_selection>`（见 [`knowledge.md`](knowledge.md)）。分块调查时每块
  都看到风格目录、各交一票，汇总按票数（同一处「style 类别」）。
- **执行策略 agentic**（任务链全是 `local_agent`）：请求通道关闭，prompt 改为「用授权的工具自己取、一次回复直接交付」。

纠错时按正文 source-id 覆盖匹配窗口笔记：优先拼接完全落入当前窗的旧笔记，否则取包含当前窗的
最小旧笔记；横切旧区间则不注入。拼接上限 8,000 token，截断写
`window_context_truncated` artifact（上限常量与其他注入上限一起在 `config.py`）。
笔记没有区间的 context（模型刚吐出、尚未绑定的中间态，或落盘早于区间寻址的旧产物）
**不可注入**，因此读取时直接判为不兼容：走 `*.invalid[.N]` 备份并重跑准备任务，而不是
只拿 `general_context` 交差——那等于把用户已经付过配额的逐窗笔记静默丢掉。
绑定时窗口 id 不在计划里的笔记无处安放、只能丢弃，此时落 `window_context_bind_report`
并向 stderr 告警：模型把 id 系统性写错会丢光逐窗笔记，而产物看上去完全正常（已绑定、
0 条），上面那道「未绑定即不兼容」的闸恰恰看不见这种情况。

因此 `media=text` 也可以有完整的准备任务；旧的"文本路线不调查、关键词条直塞纠错窗"
路径已经删除。

预算与失败处理：

- 每条消息必须满足 `prompt_input_limit`；超限直接报错，提示先切分音频。不做 map/reduce。
  ✱ 它**不是常量**（2026-09-03 起）：由绑定组的 catalog 行算出，免费 Gemini 上仍是 194000。
- 模型输出标签块/JSON 解析失败时同请求最多再试 5 次（`max_parse_retries=5`，共 6 次调用），仍失败则任务失败（`<context_pack>` 缺失时会尝试直接解析裸 JSON 兜底）。
- 每条回复、usage token 计数和解析错误写入 task artifact（`prepare_task_request` / `prepare_task_response`）；builder 的完整入参落
  `research-prepare-input.json`（`session_replay` 的 `prepare-task` 读它）。
- 首条消息的 URL 预提取块与回填消息的搜索块都接入 content-filter 阶梯（`docs/llm_harness_behavior.md`「重试与拼接」）。
- 请求回复与交付回复各自是一个 session checkpoint（`prepare-t1` / `prepare-t2`，key 为块 id 或 `main`），中断后按精确输入哈希复用。

**旧形态（2026-09-09 删除）**：r1 提需求 → 本地搜索 loop（Research Contract / Progress / Evidence Pack，最多 3 轮，
`search_judge` 任务组的 judge 逐轮筛证据）→ r2 消化；fast 另有一个把 r1 与每窗查询轮合并的融合轮。三者都是独立 API 调用，
靠 harness 在轮间搬运状态。要回溯它们的正文，看 `f5ba89df` 的父提交（`research.py` / `search_loop.py` / `stages/fast_session.py`
及 `research_round{1,2}_v2` / `search_loop_v1` / `fast_round1_v1` 模板）；取舍依据在
维护者的本地计划里（不随公开快照发布）。

## 超长素材分块调查（2026-08-30 落地，plan W7 / P8）

「超限直接报错」在闸门之上不再成立：调查 transcript 超过
`[chunking] research_transcript_max_tokens`（默认 **100,000**，`0` 关闭）时，研究阶段按
**整数个调查窗口**切块（`plan_research_chunks`，贪心、确定性、绝不切开窗口；单窗超限自成
一块），每块独立跑完整的准备任务（同一条对话形态）。阈值以下沿用单次路径，行为不变；
`retrieval=native` 不进入分块；`retrieval=none` 也按块跑准备任务（只挑词条与整理素材），拼合是确定性的。
194k 的每轮输入硬查仍然在——它现在约束的是单块。

- **Resume**：每块的产物落在 `research-context-cNN.json`（按块输入哈希重放，重跑只补没跑完的
  块）；准备任务的 session checkpoint（key 为块 id）与 exchange 前缀都带块 id，互不覆盖。
  所有块完成后才拼最终 context pack，纠错不会对着半份上下文开跑。
- **拼合是纯函数**：块 id（c01、c02…按素材先后）、顺序、合并说明文字全部固定。逐窗笔记按
  互不重叠的 source-id 区间直接串联；各块 `general_context` 合并为
  `{合并说明, c01: …, c02: …}` 一份、替换所有窗口看到的全局部分——合并说明告诉纠错模型：
  不同块对同一专名分歧时**采信证据更充分的一段**并在 note 标注。不加额外 LLM 归并轮。
- **合并全局块有独立上限**（`RESEARCH_MERGED_GENERAL_MAX_TOKENS = 20k`，与知识词条的
  `injection_block_token_limit` 无关）：超限从**尾部整块**丢弃并告警，绝不截断 JSON 中段。
- `<keep_entries>` 按块序并集去重后取前 `KB_TRANSFER_MAX_ENTRIES`（local；none 用窗口总额）。
- **块间在 `continuity=parallel` 下并行执行**（serial 下保持串行）：一块从头到尾骑一条
  lane（请求回复→回填→交付共享该 lane 的会话与前缀缓存），块散到不同 worker，扇出 =
  `min(块数, --llm-parallel-windows, claim_cap)`（与其它任务内并行面同受 W4 分配约束）。
  定序靠**共享单实例**：块间共用一个 ExchangeLogger / SessionCheckpointStore /
  WebSearchClient（实例锁串行编号与追加）；exchange 文件编号因此按完成序、
  两次运行可不同（与并行纠错窗同款 A/B 噪声），拼合结果不受影响（按块 id 定序）。
