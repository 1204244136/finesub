# Agent 长驻会话：约束放松准则与实测

> 本文从 [`llm_local_agent.md`](llm_local_agent.md) 拆出（2026-08-16，原文 2,198 行），
> **章节号已按本文重编、从 1 起**——每份文档独立编号，才能在中间插入新节。跨文档引用
> 一律写成 `` `文件名` §N `` 的形式，由 `test_doc_links.py` 守着：指不到的章节号会红。

本文 2026-10-10 之前的 CodeBuddy 测量均来自 WorkBuddy 附带 CLI；当前已改用独立 CLI 2.164.0。
新版的短调用与 MCP 领取／提交任务已验证，见 [CodeBuddy driver 的当前接线](llm_local_agent.md#1215-codebuddy第五家-driverclaude-code-的分支2026-09-04-接线)；
本文的长窗口、续会话质量与成本结论尚未在新版复测。

## 多供应商 driver：通用接口、约束放松准则与长驻会话

> 基线 checkpoint：**`769331f`**（"make one retrieval request one request all the way down"）。
> 它之前的 13 个提交是已审的修复批次，是本节全部改造的回退点；agent 多供应商工作从
> `e9cc6b7` 起，全部可以在不碰 `769331f` 的前提下推翻重做。

本节的实施过程（改造前的 Codex 专用形状、G-A 逐条改动）已移到维护者的本地归档里，下面直接是当前契约。

### 1 约束放松准则（owner 定，2026-08-13）

现有约束偏紧。**满足以下任一条即可放松**：

1. 约束影响了生成质量；
2. 约束带来不可忽略的调用成本；
3. 部分 agent 无法满足约束里过于吹毛求疵的部分。

两条护栏（放松不等于静默降级）：

- **必须具名**：放松后实际拿到的保证要写进 `execution_attempt.isolation`，产物能回答"这一次
  到底跑在什么隔离下"。
- **不得跨越安全底线**：写/执行/MCP/computer-use 工具的禁用、env secret 剔除、进程树回收、
  输出上限、**失败现场证据**这五项不在可放松范围内——它们既不影响质量也几乎不产生成本。
  （注意是**失败**现场：G-D 之后成功 episode 会被删除，成功路径的证据由 `exchanges/` 承担。）
- **不得在用户磁盘上拉屎**（owner 要求，2026-08-13）：agent 运行时留下的任何东西要么自动清，
  要么用户一条命令能清干净。详见 15.6。

### 2 逐条约束重估表

| 约束 | 现状 | 适用准则 | 建议 |
| --- | --- | --- | --- |
| `--no-session-persistence` / `--ephemeral` | 每次调用一个全新会话 | 2（成本）+ 挡住 [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做) 长驻目标 | **放松**，见 15.5 |
| 每调用一个新 capsule | 每次重建目录与 manifest | 2（小） | **由 [§2 逐条约束重估表](#2-逐条约束重估表) 取代**：改成一次性 episode，transport 成功即删、失败留存。"每 task 一个"是本表写在 G-D 之前的旧提法，不要按它实现 |
| `--safe-mode` 而非 `--bare` | 已放松 | 3（`--bare` 与 OAuth 互斥） | 保持，已具名记录 |
| 工具穷举拒绝 + `tool_use` 审计 | Claude Code 专有 | — | 保留（安全底线）。init 工具集不匹配已从违规降为告警，理由见 `llm_local_agent.md` [§7 Prompt、重复上下文与 compact](llm_local_agent.md#7-prompt重复上下文与-compact) |
| native 轮必须出现 `web_search` 事件 | **已放松**：无 completed search 记录 `native_search_not_used` note | 1（模型判断无需检索时被误判失败） | 保持当前语义；真正的不足留给业务校验。**note 只说观察到什么，不说原因**——搜索被 project hook 拒掉时同样为空，2026-08-30 曾据旧措辞误判成「模型没搜」（[`llm_local_agent_agy.md`](llm_local_agent_agy.md) [§6.1 tool-session 也要一对 proj…](llm_local_agent_agy.md#61-tool-session-也要一对-project2026-08-30-补齐)） |
| 恰好一个终态事件 | 多/少都判违规 | 3 | 保留，但允许各家声明自己的终态事件名 |
| 结果字节上限 1 MiB | 全局 | — | 保留 |
| 单请求 900s 超时 | 全局 | 3（多模态长片段可能不够） | 改为按 driver/媒体档可配 |

### 3 单 agent session 连续处理多个 harness session（G-B，会话复用已实施，A/B 未做）

**这原是 owner 目标里没做到的部分。** `llm_local_agent.md` [§3 两种正式模式](llm_local_agent.md#3-两种正式模式)/[§4 统一任务协议与状态机](llm_local_agent.md#4-统一任务协议与状态机)/[§10 多 Agent、lease、TTL 与调度](llm_local_agent.md#10-多-agentleasettl-与调度) 描述的是长驻 worker 连续领 task；改造前的实现是
它的**反面**——每个 harness session 起一个进程、一个会话，答完即弃。

已观测到的代价（2026-08-13 smoke，**n=1/臂、极小 prompt，只能当信号不能当标定**）：
一句 "reply OK" 的调用，Claude Code 侧 `input_tokens` 3,589（冷）或 `cache_read` 11k–17k +
`cache_creation` 2.4k–10.7k。即每次调用都要重新铺一遍 CLI 自己的 system prompt 与我们的框架
文本，而会话随即丢弃。真实纠错窗的正文占比更高，比例会稀释，但**绝对量按调用次数线性叠加**。

落地顺序（不改变 `llm_local_agent.md` [§12 模块边界与分阶段实施](llm_local_agent.md#12-模块边界与分阶段实施) 的六阶段，只是把第 1、3 步的前置条件补上）：

1. **先测再改（agy 一支已测，2026-08-14；Codex/Claude 未测）**：见下方 [§3.1](#31-agy-上的-ab复用是净亏2026-08-14)–[§3.4 Claude Code 的 resume 确…](#34-claude-code-的-resume-确实吃到缓存n1-信号2026-08-14)。
   agy 上小任务复用净亏 46%，默认因此保持 `session_scope=task`；但那批测量整体落在**缓存不
   生效的小前缀区间**（[§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14)），生产尺寸窗口必须重测才能下结论。
2. **会话身份进 checkpoint（已实施的是 B；准入门 D 已于 2026-08-19 改定 C，迁移未做）**：assignment scope 纳入 epoch/digest/lineage，
   task scope 保留全重放基线。
3. **transport 增能力位（已实施）**：`supports_session_reuse` 由各家 driver 的 probe 实测填充——
   Claude Code 认 `--resume` + `--session-id`，Codex 认 `codex exec resume <SESSION_ID>`（探针跑
   `exec resume --help` 并要求 `SESSION_ID` 与 `--json` 同时出现），agy 认 `--conversation`。
   不支持的家继续走 one-shot，不阻塞。
4. **lease/TTL 与 `max_parallel`（已实施）**：runtime 按 worker 分桶发 lease，driver 自带
   `max_parallel` 信号量与 `conversation_ttl_seconds` 退役窗口，见 `llm_local_agent.md` [§10 多 Agent、lease、TTL 与调度](llm_local_agent.md#10-多-agentleasettl-与调度)。
5. **无轮询等待（已实施）**：ready task 由 `submit` 夹带；API 依赖未完成时按 `llm_local_agent.md` [§4.2 等待、API 依赖与唤醒](llm_local_agent.md#42-等待api-依赖与唤醒) 由 runtime 挂起
   waiter，headless 由 `run_until_complete` 挂 watcher（`still_waiting` 只是 watcher 的活性
   边界，不返回给调用方）；conversational 在当前 turn 启动标准 watcher，进程/宿主异常时才显式
   manual resume。

**实施形态。** `AssignmentHeadlessWorker` 与 task-scope 基线共用同一个 `HeadlessTaskWorker`：
首轮走全重放（此时它的语义身份就是 `session_scope=task` 的那一份，因为逻辑内容确实是全重放），
之后每轮仍带自己的 protocol / context / manifest，provider 历史只当加速缓存。driver 侧
`session_scope=assignment` 时 cwd 换成协调域内 `.conversations/<key digest>` 的常驻目录（capsule
仍是一次性的），并强制一轮之内不得出现第二个 conversation handle。

**handle 丢了不会卡死。** provider handle 是可丢缓存，因此 runtime 有
`reset_conversation`：resume 轮的调用抛任何 `LocalAgentError`（会话被供应商清掉、resume 落到别的
会话、超时）时，worker 记一次 epoch 递增并**在同一 task 内立刻全重放到新会话**，每轮最多重建
一次——首轮失败没有可归咎的缓存，重建后再失败就是 driver 自己的问题。重建原因与被退役的 handle
按 epoch 记在会话行的 `resets` 里，不因后续成功而被覆盖。

**供应商侧残留是这项复用的已知代价：** assignment scope 下 Codex 不能再传 `--ephemeral`（否则
无从 resume），会话因此落在 `~/.codex/sessions`，在 FineSub 的清理域之外——与 agy 的 project
注册同类。`python -m finesub.llm.agent.agent_cleanup` 只删自己的协调域（含 `.conversations`），不去改供应商的用户级
记录。

#### 3.1 agy 上的 A/B：复用是净亏（2026-08-14）

**只测了 agy 这一支。** owner 明确：Gemini 的缓存机制与另两家有区别，不要泛化；Codex 与
Claude Code 彼此接近，但**都未测**。跑另外两家时注意 Codex 的 **cache write 显示恒为 0 是显示
缺陷**（实际在正常缓存）——但 **cache read 那一列是准的**，对照要看 read。agy 这边 owner 判定
usage 统计本身可信，所以下面的 0 是真的没命中，不是没报。

测具：`tmp/agy_reuse_ab.py`。同一个 runtime、同一个 driver、同一批 4 个纠错 task，唯一变量是
`session_scope`；agy 1.1.13 / `gemini-3.7-flash` / effort=low。臂间 n=5，另有 5 次同配置复跑作
噪声基线。

> **口径陷阱，先读这条。** agy `result` 事件里的 `input_tokens` 是**整个 conversation 的累计
> 值**，不是本轮用量。每次调用各开一个会话时它恰好等于单次用量，一旦一个会话跨多轮就会把前面
> 每一轮再数一遍。本节第一版正是拿累计数（复用臂）比单次数（非复用臂），得出过一个假的
> "3.3× 且超线性"。真实数字改由 **agy 自己的逐次 generation 账本**给出：
> `~/.gemini/antigravity-cli/conversations/<id>.db` 的 `gen_metadata`，一行一次真实 LLM 请求
> （`tmp/agy_gen_metadata.py` / `tmp/agy_ab_recount.py`）。两臂用同一账本重算，比值才可信。
> 代码侧已把这条钉进 `_agy_usage` 的 `conversation_cumulative` 标志。

| 臂 | 真实 prompt tokens（中位，来自 `gen_metadata`） | 逐 turn（rep0） | 墙钟（中位） |
| --- | --- | --- | --- |
| `task`：每 task 新会话，全重放 | **43,718** | 各轮约 10.9k | 55.7s |
| `assignment`：复用一个会话 | **63,638** | 10.8k / 14.2k / 17.6k / 21.0k | 53.4s |
| 噪声基线（`task` 同配置复跑） | 43,611 | — | 59.4s |

- **复用贵 46%（1.46×），增长是线性的**：每轮多带约 3.4k，就是上一轮的问答留在上下文里。
  噪声基线（43,611）与 `task` 臂（43,718）几乎重合，说明这 46% 远在噪声之外；但它**不是**
  第一版说的 3.3× 超线性——那个数字是口径错误的产物，已作废。
- 每次调用内部其实是 **2 次 LLM 请求**（先决策，再在 `view_file` 读到任务正文后作答），
  两臂都一样，所以不影响比较。
- **墙钟没有优势**：三组中位数落在 53–59s，而同配置的两组（`task` 与噪声基线）之间就差 3.7s，
  说明这个量级的差异都在噪声里。墙钟也**没有做负载控制**（其中一次与本地测试套件并行跑过），
  不要基于它下任何结论。
- **几乎完全没有命中缓存**：按 `gen_metadata` 逐条核对，复用臂 40 次 generation 里只有 1 次
  拿到 8,278 的 cache read，其余全 0；非复用臂同样全 0。**这一条不受上面的口径错误影响**
  （账本里 cache read 字段无命中时直接缺省），追查见 [§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14)。
- **输出长度这次测不出结论**：两臂逐轮输出的组内跨度都比组间差距大（复用臂 turn3 中位 2,559、
  跨度 696–3,217；非复用臂 turn3 中位 2,109、跨度 1,044–2,624），每 assignment 的输出总量在
  复用臂是 1.7k–9.0k。想说"复用会让模型变简/串味"就得专门设计质量对照，本次**没有**评估——
  [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做) 开头那条风险仍然悬着。

**结论：agy 路径上默认保持 `session_scope=task`。** 代码路径保留——它是 Codex/Claude 未来那组
A/B 的前提，也是 `llm_local_agent.md` [§13 长期基线与明确暂缓的能力](llm_local_agent.md#13-长期基线与明确暂缓的能力) 里"单任务模式必须长期保留"的对照面。要动另外两家的默认值，必须先各自跑
同一套 A/B。

> **但这组数字只对小任务成立，不要拿去决策生产。** 本节两臂都跑在**不缓存区间**（[§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14) 的
> 约 16k 门槛之下），而生产尺寸的纠错窗前缀 19.6k、确实命中 16.3k（[§3.3](#33-生产尺寸纠错窗实测门槛确实被越过2026-08-14)）。复用臂在生产
> 尺寸下会开始吃缓存，46% 这个数必须重测才能引用。

#### 3.2 缓存为什么没接上：前缀太小，不是路不通（2026-08-14）

> **先读结论（本节结论已于当日改写两次，以这一条为准）**：agy 这条路**能**缓存，包括在我们
> 自己的受控 project + 自定义 agent + `--sandbox` + `--print` headless 形态下。它需要**约 1.6 万
> token 量级的共享前缀**；在那之下**完全不缓存**。我们此前所有测量的前缀都在 4.9k–10.7k，
> 全部落在不缓存区间——所以"复用拿不到缓存"是**我们把任务做得太小**，不是供应商的限制。
>
> 实测（`tmp/agy_pseudo_turns_probe.py`，单次 headless 调用内 4 次 generation）：
>
> | 共享前缀 | gen#2 / gen#3 的 `cache_read` |
> | --- | --- |
> | ≤ 10.7k（小文件） | **0 / 0** |
> | ~16.8k（大文件先读） | **16,327 / 16,328** |
>
> 两次大前缀实验读回的都是同一个 **16,328** 的块——缓存按固定块存，增量部分不缓存。
> 本机其他会话的命中也吻合：它们起手就是 15.5k 前缀，在 gen#2 读回 16.3k。
>
> **因此下面这两小节里"我方无可修""重启条件是等供应商"的说法作废**，它们的实验没有错，
> 但整批都跑在不缓存区间里，本来就检测不到任何差异。
>
> **agy 缓存行为的当前全貌（2026-08-14 收口，逐条都有实测）**：
>
> | 事实 | 出处 |
> | --- | --- |
> | 写入门槛约 **1.6 万 token**：某次请求的前缀过了才写，之下连写都不发生 | 本节 + [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做) 末 |
> | 写入后**隔 1–2 次请求**才可读；"没命中"要先看有没有过门槛，再谈滞后 | [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做) 末 |
> | **会话内 resume 继承缓存**（跨进程也继承） | 本节「resume 本身没问题」 |
> | **跨 session 不继承**：同输入全新会话仍要自己重写一遍 | [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做) |
> | **生产尺寸纠错窗（前缀 19.6k）确实过门槛并命中 16.3k** | [§3.3 生产尺寸纠错窗实测](#33-生产尺寸纠错窗实测门槛确实被越过2026-08-14) |
> | 过门槛**不保证**命中：mapped 两次里一次读回 16,344、一次为 0 | [§3.3 生产尺寸纠错窗实测](#33-生产尺寸纠错窗实测门槛确实被越过2026-08-14) |
> | 同一轮次内媒体**不卸载、不打断缓存**；读第二个视频也一样 | [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做) |
> | 被逐一排除且**不要重做**：resume 机制、`view_file` 通道、capsule 路径、cwd、TTL、进程边界、`sessionID`、会话长度 | 本节 |
> | 无法做到：让**首轮**请求本身过门槛（argv 受 32k 字符限制；`AGENTS.md` 不进首轮前缀） | [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做) |



上一节留下的问题是"没命中是我们的调用方式，还是这条路本身"。测具 `tmp/agy_cache_probe.py`
用一个对照把两者分开：**把同一段 13.5k token 的 prompt 逐字节原样连发三次，每次各起新会话**。
如果连这个都拿不到缓存，那复用实现就不可能是原因。

| 臂 | input_tokens | `cached_input_tokens` |
| --- | --- | --- |
| 相同 prompt，每次新会话 ×3 | 13,498 / 13,500 / 13,492 | **0 / 0 / 0** |
| 相同 prompt，同一会话 resume ×3 | 13,668 / 37,220 / 69,192 | 0 / 0 / 0 |

~~结论：在我们这种 headless 调用形态下隐式缓存没有生效~~ —— **这条当时的结论是错的，连同它的
理由**："远超任何合理的最小前缀阈值"正好说反了：13.5k 恰恰**在**门槛之下（见上方方框）。
这批实验的真实价值只有一条：**它们整批跑在不缓存区间，因此只能证伪、不能证实**。
（表里的 `input_tokens` 仍是 `result` 事件的累计口径，见 [§3.1 agy 上的 A/B](#31-agy-上的-ab复用是净亏2026-08-14) 的口径陷阱；cache 那一列是
0/缺省，不受影响。）

**顺带证伪了两个看起来很像答案的假设，第二个是第一个的补课。**

我方每次调用会变的东西只有两样，两样都试过：

| 假设 | 测具 | 结果 |
| --- | --- | --- |
| prompt 里写着 `<capsule.messages_path>`，而 capsule 每次新 id → **第一条用户消息就唯一** | `agy_cache_probe2.py`：任务文件复制到固定名字，prompt 写固定名字 | 13,339 / 13,332 / 13,413，cached **仍全 0** |
| task scope 的 **cwd = `capsule.root`，每次都是新目录**（coding agent 的前言通常带工作区路径） | `agy_cache_probe3.py`：路径与 cwd 同时钉死（assignment scope 的常驻会话目录），但每次仍开新会话 | 13,341 / 13,339 / 13,320，cached **仍全 0** |

**probe2 单独看是不成立的**——它只固定了 prompt 文本里的路径，进程仍在每次不同的目录里启动，
分不清"路径不是原因"和"前面还有别的东西也在变"。probe3 才是干净对照：prompt、任务文件路径、
cwd、project id、custom agent、model、effort、环境变量全部一致，会话全新，**依然 0**。
这两个实验都做过了，**不要重做**。

顺带一个观察：即使我方全部钉死，三次的 input 总量仍在 13,320–13,341 之间小幅浮动，其中一次
`reasoning_output_tokens=0` 而另两次有——说明 agent 内部的执行轨迹本身每次就不同，我们连
"请求流完全一致"都保证不了，更谈不上稳定前缀。

**供应商侧的机制说明（`../common/session_cache_analysis.md`，owner 提供）** 把剩下的空白补上了。
那份报告用同一个 `gen_metadata` 账本量了交互式会话，结论与我们这边并不矛盾：

- agy 的隐式前缀缓存**在连续会话里工作得很好**，命中率稳定在 85%–98%；所以"这条路不支持缓存"
  是错的说法；
- **服务端闲置 TTL 约 180–300s**：超过就整体归零。我们的 resume 间隔只有 11–19s，所以**这不是
  我们的原因**；
- **`view_file` 载入的大载荷在进入下一个用户轮次时会被剥离**（只留文本层执行摘要），这会改写
  历史消息的 token 序列，使插入点之后的前缀哈希失效——报告里对应的命中率就掉到 15%–27%，
  下一次调用再回弹到 97%；
- 换模型会造成缓存隔离，冷启动。

**这解释了我们看到的全部现象**：FineSub 给 agy 的任务正文**每一轮都是经 `view_file` 进来的**，
所以每个 resume 轮的开头都恰好落在"上一轮载荷被剥离、前缀失效"这个点上；而我们每次调用只有
一个用户轮次（一个 `--print` 进程 = 一轮），**永远拿不到报告里那个"下一次调用回弹 97%"**。
那唯一一次 8,278 的命中（占 13%）正好落在报告给出的 15%–27% 区间下沿，与"局部失效"吻合。

因此：

- [§3.1 agy 上的 A/B](#31-agy-上的-ab复用是净亏2026-08-14) 的 +46% 仍是那批测量的真实结果，但它是**在"完全不缓存"这个区间里测出来的**——任务
  太小，两臂都够不着门槛。**不能当作生产尺寸窗口的结论**：真实纠错窗带上 protocol、context
  pack 与知识之后很可能越过门槛，而 resume 轮已证实**能**吃到缓存，那时复用臂的成本结构完全
  不同，46% 很可能不成立，必须重测；
- **我方有可做的事**，不是等供应商：让稳定材料（protocol / context / 知识）真正落在前缀里并
  足够大，这是我们自己的 prompt 组装问题；
- 门槛只框到区间（≤10.7k 不缓存、~16.8k 缓存），**没有精确定位**——当时以为 `view_file` 返回约
  12k token 就封顶、用它调不细前缀（2026-08-22 核实：上限其实是 ≈46k 字节/次且可按 `ContentOffset`
  续读，见 `llm_local_agent_agy.md` [§5 MCP 工具结果的内联上限](llm_local_agent_agy.md#5-mcp-工具结果的内联上限约-4k-字节超过即外置成文件无预览2026-08-22)；那次没细调前缀的原因不成立，但结论不变）。要定位得换一条能自由控制前缀大小的注入路径。

**resume 本身没问题，已单独验证。** `tmp/agy_resume_threshold_probe.py`：与生产完全一样的
resume 形态（每轮新起 `--print` 进程、带 `--conversation <id>`），只是 turn 0 先读一个大文件
把前缀撑过门槛：

| gen | 属于 | `uncached` | `cache_read` |
| --- | --- | --- | --- |
| #0 | turn0 冷启动 | 4,692 | 0 |
| #1 | turn0 读完大文件（前缀写入） | 16,218 | 0 |
| #2 | **turn1（跨进程 resume）** | 16,458 | 0 |
| #3 | **turn2（跨进程 resume）** | 4,478 | **12,229** |

跨进程 resume **确实继承缓存**。滞后现象（#1 写、#2 仍冷、#3 才读到）本身也有解释，见 [§3 单 agent session 连续处理多个…](#3-单-agent-session-连续处理多个-harness-sessiong-b会话复用已实施ab-未做)
末尾：**写入本身有门槛**，过了门槛的那一次才写，之后 1–2 次请求内可读；没过门槛则连写都不发生。

**被逐一排除的解释（都已实测，不要重做）**：resume 机制本身、`view_file` 载荷跨轮剥离（内联
正文后仍 0）、prompt 里的 capsule 路径、cwd、TTL（命中会话的 gen 间隔同样是 1–2s）、进程边界
（单进程内 4 次 generation 也全 0）、`sessionID`（命中与不命中的会话完全相同）、会话长度
（命中的会话只有 3 个 gen）。唯一的判别量是**共享前缀大小**，外加"写入后隔一次请求才可读"。

注意这仍然只说 agy。Claude Code 的初步信号见 [§3.4 Claude Code 的 resume 确…](#34-claude-code-的-resume-确实吃到缓存n1-信号2026-08-14)：它在 6.6k 前缀上就能缓存，门槛明显低得
多——**各家门槛不同，这是又一个不能跨供应商外推的量**。

#### 3.3 生产尺寸纠错窗实测：门槛确实被越过（2026-08-14）

前面几节全部跑在玩具任务上，所以"agy 不缓存"其实只是"任务太小"。这次拿**真实生产窗**验证：
素材 `BV1ojjc6MEAs`（33.6 分钟），先按原流程用 API 跑完 `difficulty=quality`
（`media=audio, retrieval=local, continuity=serial`——当时的 profile 串还带 `planning_media=audio`，那条轴 2026-09-09 已删——2 窗），
再把窗口 0001 的**冻结 prompt**（system 12,185 + user 9,282 tokens）连同同一段 17 分钟音频
发给 agy，连发两次（`tmp/agy_correction_live.py`）。

| | call 0（89.7s） | call 1（122.9s） |
| --- | --- | --- |
| gen#0 | uncached 4,820 / read 0 | 4,819 / 0 |
| gen#1（prompt+音频落地） | uncached **19,579** / read 0 | 19,573 / 0 |
| gen#2 | uncached 13,718 / **read 16,347** | 11,649 / **read 16,344** |

- **真实窗口的前缀是 19.6k，稳稳越过约 16k 的写入门槛**，缓存建立并在同一次调用内被读回
  （16.3k）。[§3.1 agy 上的 A/B](#31-agy-上的-ab复用是净亏2026-08-14) 的 A/B 之所以一次都没命中，就是任务太小——**那份 −46% 到此正式不适用于
  生产尺寸**；
- 命中发生在**调用内**（agy 自己的三次 generation）。两次调用各是新会话，task scope 下不继承——
  这正是 `session_scope=assignment` 要拿走的那块 16.3k，现在它是有真实价值的；
- **音频费率的一次实测：Gemini REST 侧 25.82 tok/s。** 同一窗的 API 返回带 modality 明细
  （`prompt_text_tokens` / `prompt_audio_tokens`），窗口 0001 是 text 23,906 + audio
  **25,476** = 49,382，clip 实测 **986.589s**（窗口 0002 为 25,817，同量级）。
  **估算仍按 32 不变**（owner 定）：32 是上限，实际随抖动/压缩偏低，包络必须用上限。
  这条只用于理解差距，不用于改公式。
- **音频确实不进 agy 的 prompt 计数器，而且这次不依赖 32 tok/s 那个假设。** 初稿用"按 Gemini 侧
  32 tok/s 该 33k"来反推少记，是循环论证（那是 REST 音频通道的口径，agy 走单帧 MP4 +
  `view_file`；而 agy 账本本就对媒体记账不稳）。改用**自量的文本**作判据：本次发出的
  system+user 经本地计数器是 **21,467 tokens**，而账本 gen#0+gen#1 合计 **24,728**——
  差额约 3.2k 要同时装下工具声明、agy 自己的前言**和 17 分钟音频**。任何合理费率下音频都塞不
  进 3.2k。`input_preparation` 同时确认音频**确实送进去了**：
  `mode=audio_single_frame_mp4`、4,112,227 → 3,934,030 bytes、`visual_frames_for_audio=1`。

**产物质量（同一窗，303 条源）**：

| | 输出行 | discard | 合并行 | 覆盖 | thinking tokens |
| --- | --- | --- | --- | --- | --- |
| API（quality 档） | 264 | 0 | 37 | 303/303 | — |
| agy call 0 | 222 | 0 | **77** | 303/303 | 884 |
| agy call 1 | 281 | **7** | 22 | 303/303 | 17,042 |

三份都结构合法、覆盖完整。但**agy 两次之间的差异比它与 API 的差异更大**（合并 77 vs 22、
discard 7 vs 0、thinking 884 vs 17,042）。听写层面三份也会分歧（同一句 `美しいスタッカート`
vs `スカラシュ`，API 与 call 0 一致、call 1 不同）——单句孰对需要人听音频裁决，这里不下结论。

> **上面这两行 agy 是跛的，别引用。** 配置里带了 `local_agent_reasoning_effort = "low"`，
> 而该项**覆盖**任务格子映射的思考档——correction-mm/quality 映射的是
> `thinking_level=medium`（budget 26,214），API 那一臂正是按它跑的。

**去掉该覆盖后重跑（对等对照）**：

| | 输出行 | discard | 合并行 | 覆盖 | thinking |
| --- | --- | --- | --- | --- | --- |
| API（quality/medium） | 264 | 0 | 37 | 303/303 | — |
| agy mapped call 0 | 276 | 4 | 27 | 303/303 | 40,341 |
| agy mapped call 1 | 269 | 2 | 34 | 303/303 | 17,253 |

**结论反转**：按映射思考档跑，agy 两次彼此接近（276/269 行、27/34 合并）也接近 API
（264 行、37 合并）。此前"agy 合并/取舍很不稳定"的判断，主要来自实验者自己钉的
`effort=low`，不是 agy 的固有行为。**教训：对照 agent 与 API 时不要覆盖思考档**——
它是任务格子的一部分，覆盖了就不是同一个任务配置。

缓存在生产尺寸下**也不是每次都命中**：mapped 两次里 call 0 读回 16,344、call 1 为 0
（gen#2 uncached 28,136）。门槛是必要条件，不是充分条件。

**工具缺口**：`tools/session_replay` 无法把会话派发到 agent 链（agent-only 下报
`capability_mismatch=2`，尽管候选链第一位就是 agy 且能力/驱动就绪），且它的 `prepare_media`
会先把剪辑上传 Files API（与 agent-first 的惰性上传相悖）。本次因此改用它 dry-run 冻结下来的
prompt 直接发车——上游注入一致，但**不是 replay 路径**。修它是另一件事。

#### 3.4 Claude Code 的 resume 确实吃到缓存（n=1 信号，2026-08-14）

**这是随手一测，不是标定。** 每臂 n=1、极小任务、只试了 Haiku 4.5，按仓库纪律
（臂间 n≥5 + 同配置噪声基线）**不足以支撑改默认值**；记下来是因为它与 agy 的结论方向相反，
足以推翻"复用没用"这个跨供应商的印象。测具 `tmp/claude_reuse_probe.py`，CLI 2.1.231。

| 臂 | turn | `cache_read` | `cache_creation` | 成本 |
| --- | --- | --- | --- | --- |
| resumed（同一 session 三轮） | 0 / 1 / 2 | 0 / **6,628** / **10,158** | 6,628 / 3,530 / 3,455 | $0.0189 / $0.0092 / $0.0091 |
| fresh（每轮各起 session） | 0 / 1 / 2 | 0 / 0 / 0 | 6,619 / 6,619 / 6,617 | $0.0200 / $0.0189 / $0.0194 |

- **resume 命中整段前缀**：turn2 的 10,158 恰好等于前两轮写入的 6,628+3,530。三轮合计
  **$0.0372 vs $0.0583，复用便宜 36%**。
- **口径已核**：Claude Code 自己的 transcript
  （`~/.claude/projects/<workspace>/<session>.jsonl`）逐条与 `result` 事件一致，**是逐轮值，
  不是 agy 那种会话累计**。被咬过一次之后，这一步不能省。
- **顺带一个对现行默认不利的观察**：fresh 臂每轮都写约 6.6k `cache_creation` 而永远无人读取。
  cache write 按 1.25× 计费，也就是说 `session_scope=task` 在这家上是在**白付缓存写入**。
- 为什么这里能命中而 agy 不能：Claude Code 的任务正文走 stdin/prompt 进入历史，不像 agy 那样
  经 `view_file` 载入、再在下一轮被剥离（[§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14)）。
- **闲置 TTL > 5 分钟**（`tmp/claude_ttl_probe.py`，n=1）：同一 session 两轮之间空闲 **400 秒**，
  resume 轮仍读回 **6,627**（正是 turn0 写入的那块），成本 $0.0182 → $0.0083。
  与 agy/Gemini 侧"闲置 180–300s 即释放"（[§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14)）明显不同，行为上像长 TTL 那一档。
  **只测了 400s 这一个点**，上界未定。

**要据此改 Claude Code 的默认 scope，必须先补齐正式 A/B**（n≥5、真实纠错窗、含噪声基线），
并单独测 Codex——它与 Claude Code 接近但未测，且读它的账本时注意 cache write 显示恒 0 是显示
缺陷、cache read 才准。

文件系统是这项设计的 durable memory：stable prompt/protocol、context pack、task manifest 与已验收
进度都按 digest 落盘，Agent 按需重读。provider conversation id 只是可丢的加速句柄；丢失或 compact
后新建 conversation，仍从同一 control namespace rehydrate。它解决“摘要忘了 prompt”的问题，
准入门 D 的现行答案是 **C**（2026-08-19，取代 B）：复用只是加速，依赖隐含历史的调用不产出可复用
checkpoint；拿不到稳定 lineage 的 driver 照旧自动使用 `session_scope=task` 的每 task 完整逻辑
重放。**runtime 侧代码实现的仍是 B**（记录 lineage 并纳入身份），那一半迁移挂在
[`llm_local_agent.md`](llm_local_agent.md) [§12 模块边界与分阶段实施](llm_local_agent.md#12-模块边界与分阶段实施) 第 3 步；生产窄路已是 C
（`LLMCallResult.resumable` + 三处 L1 提交点跳过入库）。

**风险**：跨 task 复用会话意味着上一窗的输出留在上下文里。对纠错任务这既可能提高一致性
（术语、风格），也可能造成串味与 compact 丢失。这属于准则 1，要用质量 A/B 判，不能凭直觉。

#### 3.5 生产尺寸复用 A/B 的协议（未跑，2026-08-19 补写）

> 按本协议做过的：3.8 试跑（n=1）与 [3.9 `resume` 对 `per-window`](#39-resume-对-per-window跨窗信息传递与单窗质量2026-10-02agy-每臂-n5codex-luna-每臂-n3)（正式两段）。标题里的「未跑」保留，免得断掉引用它的锚点。

[§3.1 agy 上的 A/B](#31-agy-上的-ab复用是净亏2026-08-14)–[§3.4 Claude Code 的 resume 确…](#34-claude-code-的-resume-确实吃到缓存n1-信号2026-08-14) 之后仍然没有一个能拍板的数：agy 的净亏只对小任务成立（[§3.3 生产尺寸纠错窗实测](#33-生产尺寸纠错窗实测门槛确实被越过2026-08-14) 已证明生产窗能过缓存
门槛），Claude Code 是 n=1，Codex 一次没测。而 `agent_session_mode` 的默认值、
[`llm_local_agent.md`](llm_local_agent.md) [§12.1.1 档位是唯一配置面](llm_local_agent.md#1211-档位是唯一配置面传输由它派生2026-08-22-起-per-window-默认即工具会话) 那根线接不接、以及
[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) 的收益估算，**三件事都挂在这次重测
上**。所以这里把协议写死，避免又跑出一批口径不一、不能合并的数。

**循环依赖已解（2026-08-19）**：此前跨窗复用在生产里是关的，「用生产尺寸窗口测复用」与
「接不接那根线」互为前提。出口就是当时设想的实验开关，**已接线**：`agent_session_mode` 由
`client.py` 窄路读取（四档，默认 `per-window` 即原行为，见
[`llm_local_agent.md`](llm_local_agent.md) [§12.1 会话形态](llm_local_agent.md#121-会话形态四档已决定什么还差什么接在哪)），跨窗复用只在显式设成 `resume` 时发生
（每 worker lane 一条会话）。测量因此走的是生产代码路径而不是另一套探针。注意 `resume` 下
继承了隐含历史的调用不写可复用 L1（准入门 D 的 C 案），逐窗 attempt 记账不受影响。
（[§3.1](#31-agy-上的-ab复用是净亏2026-08-14) 那次用的探针 `tmp/agy_reuse_ab.py` 是 scratch，已经不在了；不要指望复用它。）

**臂**：每家 driver（agy / Claude Code / Codex）各两臂——`agent_session_mode=per-window`
（**现行默认**）与 `agent_session_mode=resume`——外加**同配置复跑**作噪声基线。每臂 `n≥5`，
臂间同素材同顺序。

> **臂名按四档写，不要按 `session_scope` 写**（2026-08-19 复审更正：初稿写 `task` vs
> `assignment`）。四档之后 `assignment` 这个 scope 同时是 `per-window` 与 `resume` 两档的实现
> 结果，拿它当臂名已经不唯一了。更要紧的是**基线选谁**：基线取 `api` 会把「窗内修复是否续会话」
> 这第二个变量一起搬进对照，而本次要测的只有**跨窗**复用，所以基线必须是生产默认 `per-window`。

**素材**：真实纠错窗，不是探针任务。至少两份，一份专名密集、一份稀疏；窗口数 ≥ 5，
让第 2 轮之后的复用真的有机会发生（[§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14)：写后隔 1–2 次请求才可读）。

**固定量**：素材、知识库快照、`PROMPT_VERSION`、difficulty/retrieval/knowledge 三个开关、
`continuity`（先用 serial；parallel 是下面的第二阶段）、driver 版本与 CLI 版本。任一项变了
就是另一次实验。

**逐次记录**（缺一不可，前两项是被咬过的坑）：

1. **token 口径要先自证**：agy 的 `result` 事件是**会话累计**、Claude Code 是**逐轮**、Codex 的
   cache write 显示恒 0 只有 read 准。每家先跑一次两轮探针确认口径，再开始记账；
2. `cache_read` / `cache_creation` / uncached input / output，逐轮值；
3. 墙钟（逐窗与总计）、attempt 数、修复轮次数、validation 失败类型；
4. 会话事件：`reset_conversation` 触发次数与原因（handle 丢失 / 漂移 / TTL）；
5. 成品质量：与同素材精修字幕的对照，重点是**跨窗一致性**（专名、人称、语气）——它既是复用
   最可能的收益，也是「上一窗输出留在上下文里」最可能的害处（见本节前面那条**风险**）。

**判据（owner 定，2026-08-19，先写死避免事后挑）**：

- **门槛是质量：质量不变差就接线。** 成本与墙钟照记照报，但**不作否决项**。理由是当前的
  cache miss 大概率是 agent 侧的缺陷而不是架构的性质（[§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14) 已证明 agy 能缓存、只是门槛与继承
  规则古怪；[§3.4](#34-claude-code-的-resume-确实吃到缓存n1-信号2026-08-14) 的 Claude Code 方向相反），上游修掉之后成本账会翻过来——**不该因为今天的账
  不划算就永久否掉一个架构选择**；
- 反过来，**质量劣化是硬否决**：跨窗串味是复用固有的风险，不是可以指望别人修的东西；
- 逐家判定：`agent_session_mode` 是任务组级、按 cell 解析，本来就支持逐家不同，所以哪家过了
  改哪家，不要为一家的结果动全局默认；
- 成本数据仍要留档并写进本节——它是判断「上游修了没有」的基线，也是工具化协议估收益的输入。

**这次实验不覆盖**：`resume` 搬上工具会话之后的形态——那是另一套传输，届时要重跑，不能沿用
本次的数（共存已验证、接线待定，见 [`llm_followups.md`](llm_followups.md)「`resume` 档在工具会话下仍未定义」）。

**实施（owner 2026-09-30 定目标；只做试跑，已跑完，结果在 [3.8 会话档位与传输试跑](#38-会话档位与传输试跑q1q32026-09-30每臂-n1)）**。三个问题，一套跑法：

- **Q1 `resume` 对 `per-window`**（本节的接线判据，量质量）：agy 3.7 Flash、CodeBuddy DeepSeek-V4.1-Flash
  ——都是 medium 档模型；
- **Q2 `pseudo-conversational` 对 `resume`**（两种跨窗形态，主要量**花费**：token、缓存、墙钟，以及重置、
  提前退出、修复轮）：CodeBuddy hy3、agy 3.7 Flash、Codex GPT-6 Luna。质量预期相同，所以允许用低能力模型；
- **Q3 `per-window` 下 capsule 对工具会话**（同一档位、两种传输，量质量与花费）：agy 3.7 Flash、
  CodeBuddy DeepSeek-V4.1-Flash。capsule 臂用 dev-only 的 `FINESUB_AGENT_TRANSPORT=capsule` 钉住。

| 臂（`<目标>-<档位>[-capsule]`） | 目标 | 用在 |
| --- | --- | --- |
| `agy37f-per-window`（×2） | `local-agy-media-gemini-3_7-flash` | Q1、Q3 的基线；第二次是同配置复跑，当噪声基线 |
| `agy37f-resume` | 同上 | Q1、Q2 |
| `agy37f-per-window-capsule` | 同上 | Q3 |
| `agy37f-pseudo-conversational` | 同上 | Q2 |
| `wbdsv41f-per-window`（×2） / `-resume` / `-per-window-capsule` | `local-codebuddy-deepseek-v4_1-flash` | Q1、Q3 |
| `wbhy3-resume` / `-pseudo-conversational` | `local-codebuddy-hy3` | Q2 |
| `luna-resume` / `-pseudo-conversational` | `local-codex-completion-gpt-6-luna` | Q2 |

- **素材**：维护者本地精修对照组里三份单人素材（第三方真实素材，名字不进本文），各截前约 10 分钟、调窗口上限切成
  6 窗，每窗 prompt 约 5 万字符（固定脚手架占大头），在缓存门槛之上；三份的长度与上限见 [3.8 会话档位与传输试跑](#38-会话档位与传输试跑q1q32026-09-30每臂-n1)。
  片段长度按 owner 的约束定：单个 driver 每个实验不超过 30 分钟素材，够验证即可。⚠ 窗更小、片段更短，跨窗效应
  比整集弱；结论只对这个窗长成立。
- **固定**：`--media text`（工具协议纯文本；带媒体的调用任何档位都单独起会话，会抹掉要测的差别）、
  `--retrieval none --knowledge none --fast off --continuity serial --difficulty quality`；三个文本格
  （research / correction-text / knowledge）绑同一目标、同一档位，`execution_policy = "agent-only"`。
- **次数**：每臂 1 次，基线臂复跑 1 次。读数只当信号——下面的判据要 n≥5，试跑达不到。
- **自动指标**：`tools/bench/score_session_ab.py --clip-end 575.48`（复用 `score_flow_ab.py` 的对齐与
  相似度）：与精修的平均相似度、对上率、低相似行、首次校验通过率、重试与调用错误；跨窗效应看**窗 1 与
  窗 2 起**分开的相似度；成本是 uncached / cached input 与输出，含 pseudo 档记在
  `agent-session-usage.json` 的会话总账。精修 ASS 直接读：花括号里是被替换掉的机器原句，剥掉；`注释`
  样式不计。
- **质量判定（Q1、Q3）**：相似度只是粗信号，判定靠**盲评**：`tools/bench/judge_session_ab.py pack` 把两条
  run 的译文逐窗并排、随机标成 X / Y（密钥另存），附 ASR 原文与精修参照；外派两个不同模型各评一遍
  （逐窗 X / Y / 平，错误带时间点，专名跨窗是否一致），`collect` 按密钥还原；两份评审不一致的窗口由
  维护者逐条复核。同配置复跑的两条 run 也评一遍——它们之间的偏好就是这把尺子的噪声。
- **判据**：沿用上面 owner 定的「质量不变差就接线」；「不变差」沿用任务式流程验收的非劣口径
  （读数在 [`bench-baselines.md`](bench-baselines.md)「二十八、任务式流程的验收对照」），跑前写死：令
  δ = max(0.02, 基线臂相似度的标准差)，`resume` 相似度均值 ≥ `per-window` − δ、对上率 ≥ − 0.03、
  首次校验通过率 ≥ − 0.10；盲评里输给基线的窗口数不超过噪声对（同配置复跑）之间的分歧。逐家判定。
- **跑法**：`python -m tools.bench.run_session_ab --stable <clip-stable.json> --out out/session-ab/<素材>
  --n 1 --max-window-subtitle-tokens 1200 --arms <臂…>`（复跑加 `--start 2`），不同 CLI 的臂分开进程并行。

#### 3.6 pseudo-conversational 首次 canary：agy 37f、4 窗、一条会话（2026-08-22）

**配置**：`tmp/agent-canary-agy.toml`（`agent-only`，所有 cell 绑 `local-agy-media-gemini-3_7-flash`，
`agent_session` 全 pseudo，`[chunking] max_window_subtitle_tokens = 4000`），素材
`out/kaguya60/kaguya60-stable.json`（553 段 → 4 窗 69/80/91/88 段，每窗 prompt ≈44k 字符），
`--media text --retrieval none --knowledge none --fast off`，serial。块按文件交给 agy 的
`view_file` 读（`llm_local_agent_agy.md` [§5 MCP 工具结果的内联上限](llm_local_agent_agy.md#5-mcp-工具结果的内联上限约-4k-字节超过即外置成文件无预览2026-08-22)）。

**读数**：

| 项 | 值 |
| --- | --- |
| 墙钟 | 201s 整跑（逐窗 92s / 37s / 30s / 33s，第一窗含 CLI 启动与协议读取） |
| CLI 会话 | 1 条，0 次 premature stop，0 次 `still_waiting`（harness 在窗间的间隔短于长轮询） |
| MCP 调用 | 9：`next_task` ×5、`submit` ×4；**每窗一次 submit 即 accepted，零修复** |
| `view_file` | 6：协议文件**只在第一窗读一次**，后三窗只读各自 payload（长驻会话省下的正是这份） |
| 会话 usage（agy 累计口径） | uncached input 215,056；cached input 1,181,202；output 88,824（其中思考 63,740） |
| compaction | 仅会话第一步的 bootstrap CHECKPOINT；300k 级上下文没有触发中途 compact |
| 成品 | 505 条、2019 行；窗口边界（69/149/240）处连贯，专名「宫子」跨窗一致 |

**口径说明**：task-report 的 per-call usage 全为 0 是预期——pseudo 下 usage 按会话记账
（`usage_attribution="session"`）。总账写在产物目录的 `agent-session-usage.json`，并由任务报告
加进 Provider Token Totals 的 token 列（调用次数仍按窗口算）；会话出事故没被清理时，
`<root>/control/session-usage.json` 里还留一份。
没有 kaguya60 的精修参照，质量只能定性（成品可读、格式全对、无串窗）。

**同日此前的两次失败**（都不是会话机制的问题）：第一次 4 窗因 agy 把 >4k 字节的 MCP 回复外置成
文件、模型完全看不到 protocol/payload 而全部耗尽预算（两条会话 19 万 uncached + 81 万 cached）；
随后的 1 窗用 MCP 分页「跑通」但成品是一行 `test` + 全部 `discard`——**校验器放过了一份几乎全丢的
窗口**，这是一个独立的校验洞，记在 followups。

**缓存逐请求账本**（agy `conversations/<id>.db` 的 `gen_metadata`，17 次真实请求；f2 = uncached、
f5 = cached、f3 = 输出、f9 = 思考，合计与会话总账分毫不差）：

| 请求 # | 属于 | uncached | cached | 命中率 |
| ---: | --- | ---: | ---: | ---: |
| 0–2 | 窗 1 启动（bootstrap、next_task、读协议） | 16.1k / 16.3k / 17.6k | 0 | 0% |
| 3–5 | 窗 1 读 payload、生成、submit | 14.0k / 7.6k / 19.6k | 16.3k / 28.5k / 32.7k | 54% / 79% / 63% |
| 6–8 | 窗 2 | 26.3k / 3.1k / 10.1k | 49.0k / 73.5k / 73.6k | 65% / 96% / 88% |
| 9–11 | 窗 3 | 19.3k / 4.3k / 7.0k | 81.8k / 98.1k / 102.2k | 81% / 96% / 94% |
| 12–15 | 窗 4 | 18.2k / 3.1k / 10.0k / 17.3k | 106.3k / 122.6k / 122.6k / 130.8k | 85% / 98% / 92% / 88% |
| 16 | seal 后最后一轮 | 5.2k | 143.1k | 96% |

前 3 次冷启动零命中（前缀 ≈16k，正卡在 [§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14) 的写入门槛附近）；从第 4 次起每轮都命中，cached 即
「上一轮的全部上下文」、逐轮单调上涨，uncached 只剩本轮新进来的东西（一个窗的 payload ≈18k +
上一轮输出回显 ≈10k）。后三窗窗内命中 81–98%，整会话 85%。**这回答了 [§3.2 缓存为什么没接上](#32-缓存为什么没接上前缀太小不是路不通2026-08-14) 悬着的问题：生产尺寸
前缀下 agy 的缓存稳定工作**，此前「几乎零命中」只是小任务没过门槛。代价结构：每窗 uncached
≈30–35k、与窗数线性；上下文 4 窗到 145k（catalog 上 37f 是 1M；`gen_metadata` 里另有一个 256,000 的
上限字段，含义未证实，可能是 agy 自己的压缩阈值——什么时候撞 compaction 要另测）。

**本次 canary 回答了什么**：取活/交活/终止契约、一窗一 task、替换轮起新会话、seal 退出在真机上
都成立；agy 上一条会话服务整条 run 的成本结构（协议只读一次、缓存读占 85%）。**没回答**质量对照
（无参照）与 `continuity=parallel`；Claude Code / Codex 的 pseudo 没跑。

**`continuity=parallel` 是本协议的第二阶段，不是排除项。** assignment 形状已定为「一 assignment
× N worker」（`llm_local_agent.md` [§12.1.2 resume](llm_local_agent.md#1212-resumetransport-齐了缺的是谁来保证租约盖得住) 第 3 点），随之打开的调度策略有两条待验证假设，按同一
套口径测、**第一条优先**：

1. **连续窗口给同一 worker vs 任意分配**——相邻窗口讲的是相邻的话，同一条会话里连着做，跨窗
   一致性（专名、人称、语气）可能更好。这是**质量**假设，而质量正是上面的门槛，所以它排在
   成本问题前面；
2. **同类型 session 归同一 agent**（correction 一组、query/research 一组）——每个 agent 只面对
   一种任务形状是否更 focus，顺带前缀更稳定、更容易吃到缓存。

这两条的臂在 serial 上没有意义（只有一条 lane），必须在 parallel 上单独跑：固定量里的
`continuity` 换成 `parallel` 并记录 `parallel_windows`。


#### 3.7 复审修复后的验收：2 窗、agy 37f（2026-08-22）

同素材同配置（`tmp/agent-canary-agy-2w.toml`，`max_window_subtitle_tokens = 8000` → 2 窗），
验的是复审那三条修复在真机上成不成立：

| 验收项 | 结果 |
| --- | --- |
| 成功会话不留 assignment root | ✓ 本次没有新增 `assignments/session-*`（此前 4 个都是修复前的 run 留下的） |
| 审计包每 task 一份 | ✓ capsule 里 `audit-call-0001` / `audit-call-0002`，含 manifest、blocks 正文、outcome、artifact、mcp-frames |
| 会话 usage 进任务报告 | ✓ `agent-session-usage.json` + 报告一行 `LOCAL_AGY / gemini-3.7-flash / 2 calls / 159,885 input / 515,018 cached / 22,187 output / 51,779 thinking` |

**第一次跑没通过第三项，暴露了两个真问题**（都已修）：

1. **`correction_translation.main()` 是第二条 run 路径**：它自己跑 research、自己调
   `execute_correction_windows`、自己写报告，完全不经过 `run_full_correction`——run 作用域与
   usage 记账只接在后者上，所以模块 CLI 跑出来的报告仍是 0 token，且它的 research 阶段跑在
   scope 之外（拿私有注册表、会话活到进程退出）。现在所有入口都只造一个请求交给
   `run_correction_task`，run 路径只剩一条。
2. **同一 agent 在报告里裂成两行**：逐窗记录用 `LOCAL_AGY`、会话总账被折成小写，
   `provider_usage` 于是有两个键——一行有调用数没 token，一行有 token 没调用数。折算不再改
   大小写。

另一条**不是缺陷但值得记**：本次 agent 在两个 task 都 accepted 之后碰了一次原生工具、被 guard
拒绝，agy 把整个 result 翻成 `ERROR`（`tool call denied by pre-tool hook`）。
[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§4 完成与终止契约](llm_agent_tool_protocol.md#4-完成与终止契约) 的
「driver 在 accepted 之后出错先读 runtime」照常生效，两窗产物完好；terminal `result` 事件里的
usage 也照常带着（`conversation_cumulative`），没有丢账。

#### 3.8 会话档位与传输试跑：Q1–Q3（2026-09-30，每臂 n=1）

**结论**：三份素材、每臂各 1 次，**没有一档、一种传输在质量上稳定变差**：Q1 的 `resume`、Q3 的 capsule 与基线
的输赢，几乎都能追到某一次运行恰好把一个 ASR 错名改对或没改对，而同配置复跑两次之间也有同样大的输赢。
唯一不能这样解释的一处：agy 的 `resume` 在第三份素材上把主播自称混成三种写法（其余所有臂都整篇统一），
这正是 §3.5 判据里「跨窗串味」那一类，但只出现一次。花费上：**工具会话比 capsule 贵**（三份素材、两家同向）；
跨窗两档（`resume`、pseudo）的缓存读取涨得最多，pseudo 在 Codex 上最贵、最慢。⚠ n=1。按
[§3.5 生产尺寸复用 A/B 的协议](#35-生产尺寸复用-ab-的协议未跑2026-08-19-补写) 重做的 `resume` 对 `per-window` 见下一节 [3.9 `resume` 对 `per-window`](#39-resume-对-per-window跨窗信息传递与单窗质量2026-10-02agy-每臂-n5codex-luna-每臂-n3)。

**素材**（都是维护者本地精修对照组里的单人素材，名字不进本文）：甲 9.6 min / 192 段 / 窗口上限 1200 → 6 窗；
乙 9.9 min / 134 段 / 上限 800 → 6 窗；丙 9.1 min / 205 段 / 上限 1200 → 6 窗。乙、丙打开了
`local_agent_keep_episodes`。所有臂都是 `--media text --retrieval none --knowledge none --fast off --continuity serial`；现在知识库默认已是 `read`，开着知识库时工具会话与 capsule 的执行策略不同，Q3 的结论不能直接套到默认配置上。Q2 的 CodeBuddy 低档模型在甲上是 hy3，hy3 的 pseudo 跑不通后改 GLM-5.3-Flash 补测三份。

**质量**（盲评：两位评委各评一遍，6 窗；数字是「挑战者对基线 `per-window` 第 1 次」的胜 / 负 / 平，噪声行是
第 2 次对第 1 次；甲的评委是 Opus · DeepSeek，乙、丙是 Gemini 3.7 Flash · DeepSeek——Opus 当时额度用尽）

| 素材 | 对比 | agy 3.7 Flash | CodeBuddy DeepSeek-V4.1-Flash |
| --- | --- | --- | --- |
| 甲 | 噪声 | 1/1/4 · 2/0/4 | 0/5/1 · 0/5/1 |
| 甲 | Q1 `resume` | 2/0/4 · 2/1/3 | 1/3/2 · 1/4/1 |
| 甲 | Q3 capsule | 2/0/4 · 2/1/3 | 1/2/3 · 2/2/2 |
| 乙 | 噪声 | 0/2/4 · 0/4/2 | 2/2/2 · 2/0/4 |
| 乙 | Q1 `resume` | 0/2/4 · 0/4/2 | 2/1/3 · 1/0/5 |
| 乙 | Q3 capsule | 0/1/5 · 0/1/5 | 0/0/6 · 1/0/5 |
| 丙 | 噪声 | 2/0/4 · 5/0/1 | 0/1/5 · 0/1/5 |
| 丙 | Q1 `resume` | 2/1/3 · 0/3/3；对第 2 次 0/6/0 · 0/6/0 | 0/4/2 · 0/1/5 |
| 丙 | Q3 capsule | 3/0/3 · 5/0/1；对第 2 次 2/0/4 · 3/0/3 | 2/0/4 · 0/0/6 |

- **输赢多半是一个错名的抽样**：甲里 ASR 把「本渡楓」听成「本土楓」，乙里把「岡咲」听成「岡崎」，丙里把主播
  父母的昵称听成「パパ長／ママなが」。每次运行的准备任务（两档完全相同、都开新会话）这一次有没有认出它，决定了
  后面写对还是全程写错——甲：CodeBuddy 的 `resume` 与 `per-window` 第 2 次都写错；乙：12 个臂里只有 agy
  `per-window` 第 1 次与 capsule 写对；丙：agy 五次里 `per-window` 两次和 `resume` 写成「爸爸长」，capsule
  与 pseudo 没有——错与不错和档位、传输都对不上。所以「挑战者输给第 1 次」常常只是第 1 次恰好抽到对的那面，
  丙里第 1 次又恰好偏弱，挑战者对它全赢。**只对一次基线评，会把基线那一次的运气算成档位的差**；丙补评了「对
  第 2 次」才看清。
- **唯一的档位信号**：丙里 agy `resume` 对第 2 次 6 窗全输，两位评委理由一致——它把主播自称写成「诺娜／侬酱／诺诺」
  三种（15/7/2 次），其余每个臂都整篇只用一种（「诺娜」「小侬」或「乃乃」）。续会话本该更一致，这次反而漂了。
  甲、乙的 agy `resume` 没有这种情况；CodeBuddy 的 `resume` 在丙里输的是零散误译，评委明说称呼「各自内部一致」。
- 与精修字幕的平均相似度全部落在 0.54–0.63，同配置复跑差约 0.015——自动指标分不出档位。

**花费**（按各家原始用量重算：`resume` 下三家报的都是会话累计值，按会话取最大值再相加；「未缓存」对 CodeBuddy
是它报的缓存写入——它的 `input_tokens` = 缓存写入 + 缓存读取；agy 与 Codex 的输出含思考；单位万 token，
墙钟秒；三格依次是素材甲 / 乙 / 丙）

| 臂 | 未缓存输入 | 缓存读取 | 输出 | 墙钟 |
| --- | --- | --- | --- | --- |
| agy `per-window`（工具会话）第 1 次 | 59.0 / 44.5 / 55.3 | 72.2 / 64.1 / 58.8 | 17.3 / 14.8 / 18.9 | 690 / 548 / 654 |
| agy `per-window` 第 2 次 | 46.8 / 47.1 / 49.5 | 65.7 / 61.6 / 62.8 | 15.8 / 15.1 / 16.9 | 596 / 577 / 607 |
| agy `per-window-capsule` | 24.1 / 21.3 / 20.2 | 6.5 / 8.2 / 9.8 | 9.4 / 7.6 / 10.9 | 375 / 313 / 387 |
| agy `resume`（capsule） | 23.2 / 26.9 / 23.7 | 154.5 / 119.5 / 192.5 | 8.3 / 5.6 / 7.8 | 353 / 250 / 347 |
| agy `pseudo-conversational` | 26.0 / 29.8 / 35.4 | 243.2 / 336.8 / 202.3 | 11.7 / 9.6 / 9.4 | 384 / 346 / 320 |
| WB DeepSeek `per-window` 第 1 次 | 23.5 / 18.0 / 23.2 | 77.9 / 52.3 / 70.2 | 19.5 / 10.6 / 16.6 | 987 / 495 / 1014 |
| WB DeepSeek `per-window` 第 2 次 | 19.9 / 20.4 / 18.3 | 73.1 / 60.6 / 80.0 | 15.3 / 11.2 / 17.8 | 698 / 538 / 789 |
| WB DeepSeek `per-window-capsule` | 13.1 / 4.4 / 5.5 | 7.4 / 12.2 / 17.7 | 18.9 / 13.1 / 18.9 | 943 / 583 / 815 |
| WB DeepSeek `resume`（capsule） | 28.2 / 29.0 / 35.0 | 21.8 / 15.2 / 13.7 | 16.8 / 10.5 / 15.9 | 834 / 457 / 697 |
| WB GLM-5.3-Flash `resume` | 53.0 / 57.8 / 50.7 | 0 / 3.3 / 0 | 4.8 / 3.3 / 4.7 | 1194 / 836 / 1176 |
| WB GLM-5.3-Flash `pseudo` | ⚠ 漏记 / 19.5 / 20.3 | ⚠ / 238.9 / 160.6 | ⚠ / 3.6 / 8.1 | 1477 / 701 / 1321 |
| WB hy3 `resume`（只跑了甲） | 28.4 | 8.1 | 11.3 | 1118 |
| Codex Luna `resume`（修复后） | 15.3 / 13.4 / 16.8 | 41.9 / 38.0 / 54.1 | 3.8 / 2.5 / 3.9 | 712 / 471 / 749 |
| Codex Luna `pseudo` | 23.3 / 22.2 / 24.1 | 334.7 / 260.0 / 302.6 | 7.5 / 7.8 / 11.3 | 1504 / 1433 / 2536 |

- **Q3（同在 `per-window` 档，工具会话对 capsule）**：三份素材两家同向，工具会话贵——agy 未缓存 2.0–2.7 倍、缓存读取
  6–11 倍，墙钟慢约 0.6–0.9 倍；CodeBuddy 未缓存 1.5–4.7 倍、缓存读取 4–11 倍，墙钟相当。成因见下面 agy 逐请求账本。
  **accept 后立即结束 CLI 之后重测**（2026-10-01，乙，agy；工具会话 2 次、capsule 1 次，7 个任务全部 accept 即退出）：
  工具会话未缓存 33.7 / 31.4 万、缓存读取 39.1 / 42.0 万、输出 14.4 / 13.1 万、墙钟 561 / 493 s；与改动前同序号的两次逐对比，未缓存降 24% / 33%、缓存读取降 39% / 32%（前后 agy 从 1.2.13 升到
  1.2.14，capsule 前后几乎没变——未缓存 21.3 → 19.4 万——可作部分对照）；
  capsule 19.4 / 9.8 / 7.5 万、303 s。差距缩小但方向不变：未缓存 1.6–1.7 倍、缓存读取 4.0–4.3 倍、输出与墙钟仍约 1.6–1.9 倍。
  只测花费：accept 后才退出，不改答案。当时 CodeBuddy 那半未能完成——WorkBuddy 附带 CLI 2.156.0
  在桌面端之外读不到加密登录态。独立 CLI 2.164.0 已解决当前调用的认证问题；这组成本对照仍未重测。
- **Q1**：`resume` 恒走 capsule、基线 `per-window` 在这两家上走工具会话，花费差混着传输差；只看档位要对
  `per-window-capsule`：agy 未缓存相当（1.0–1.3 倍）、缓存读取 15–24 倍；CodeBuddy 未缓存 2.2–6.6 倍——续会话的
  历史越积越长。输出（含思考）反过来：`resume` 在 6 格里都比 `per-window-capsule` 少（agy 少 12–28%，CodeBuddy 少 11–20%；
  同配置复跑的输出差可达约 22%）；agy 上逐窗看更清楚，第 1 窗与 capsule 持平、第 2 窗起稳定下降。墙钟 `resume` 也
  6 格都快一些（6–22%）。方向一致、单格在噪声内，算「少重复推理」的弱信号。
- **Q2（pseudo 对 `resume`）**：agy 未缓存 1.1–1.5 倍、缓存读取 1.0–2.8 倍、墙钟相当；Codex 未缓存 1.4–1.7 倍、
  缓存读取 6–8 倍、墙钟 2–3.4 倍（成因见下）；CodeBuddy GLM 反过来——它的 `resume` 几乎吃不到缓存（缓存读取 0–3 万），
  pseudo 的未缓存只有 `resume` 的 0.34–0.4 倍，墙钟在乙上更快、在甲丙上慢一些。**CodeBuddy 的 `resume` 吃不到历史缓存
  不是 GLM 独有**：乙、丙逐窗看，DeepSeek 的 `resume` 每窗未缓存（CodeBuddy 报的缓存写入）随历史涨（乙 1.8 万→10 万），
  缓存读取多数窗约 2 万、止于第一个乱码处——历史每轮写进缓存、下一轮却读不到；agy 与 Codex 的 `resume` 历史大多命中。
  **根因在 CodeBuddy 自己的 `--resume`**（2026-10-01 按它的会话记录与请求追踪逐请求核实）：续会话时它从文件尾往前、
  按 64 KiB 一块读会话记录 jsonl，每块单独按 UTF-8 解码，块边界切在中日文多字节字符上就成了 U+FFFD；文件每窗变长、
  边界跟着移，乱码位置每窗不同，请求前缀从第一个乱码起就和上一窗对不上——缓存只命中到那里（乙、丙按边界公式预测的
  乱码位置与追踪逐字节吻合，第一处乱码越靠前缓存读取越少）。我们的调用每窗除 handle 外逐字相同，用量换算也对得上
  逐请求之和。同一进程内 DeepSeek 能缓存整段历史。2.156.0 的代码里这段读法还在。GLM 另多一层：新进程连固定前缀都
  读不到（同一进程的 pseudo 里能读到），原因没查。⚠ 甲的 GLM pseudo 第一条会话中途
  退出、换了新 CLI，会话账只记下约 2.5 万输入，那格不可用。GLM 的 pseudo 三次里有一次要换会话才交稿，hy3 的两次都没交稿。
  **2026-10-11 在独立 CLI 2.164.0 上复测，分块解码的乱码不再出现**（hy3、`--effort low`、`-p` 加 `--resume`，两组各一次）：
  第一轮发 12 万 / 24.5 万字节的中文测试文本（每行一个编号加 30 个汉字），第二轮续会话，让模型逐字抄回按边界公式
  （从文件尾往前每 64 KiB）落在多字节字符中间的那几段。三种读取时机（追加第二轮之前、追加用户消息之后、再追加快照
  之后）各至少有一处边界切在字中间，抄回的行与原文逐字相同，没有 U+FFFD。另有一段 24 行被模型整段编造，与乱码无关，
  是 hy3 在约 9.5 万 token 上下文里的召回失败。第二轮的缓存读取 52,352 / 95,744，等于第一轮的整段输入
  （51,952 / 95,339），不再止于前 2 万。2.164.0 的 trace 会截断载荷、看不到请求体，所以这是行为证据，不是逐字节核对；
  DeepSeek / GLM 上 `resume` 的缓存命中、GLM 新进程读不到固定前缀那一层都没有重测。
- **agy 工具会话贵在哪（逐请求账本，`conversations/<id>.db` 的 `gen_metadata`，同一窗口）**：工具会话一窗 5 次
  请求、未缓存 76,469 / 缓存读取 93,886，capsule 3 次请求、48,264 / 0。差额几乎全在**最后一次**：模型在第 4 次
  请求里把约 2.5 万 token 的答案写成 `submit` 的参数，收到「已接受」后还要再发一次请求收尾，这时刚写出的答案
  作为新内容进上下文、没有缓存可吃——这一次就是 28,145 未缓存。缓存门槛只影响第一次请求（约 1.6 万的冷启动，
  正卡在写缓存门槛附近）。工具会话这一窗的输出（含思考）也是 capsule 的 2 倍。⇒ accept 之后不再让 CLI 发
  收尾请求，按这一窗的账本每窗能省约 2.8 万未缓存——已据此在 accept 后立刻结束 CLI
  （`[llm] local_agent_exit_on_accept = auto`，Codex 除外；见 [`llm_local_agent_runtime.md`「目标形态」](llm_local_agent_runtime.md#目标形态)）。
  实施后重测，每个任务实际少 1.5–2.2 万未缓存（见上面 Q3 那条）。
- **Codex pseudo 为什么缓存读取是 `resume` 的 8 倍、墙钟 2 倍**（第二份素材上打开
  `local_agent_keep_episodes`、用 Codex 自己的 rollout 逐请求核实）：Codex 把 MCP 工具包在一个 JS 执行工具里调用，
  每窗固定是 `next_task` → `read_context`（第 1 窗两次、之后一次，读的正是 `next_task` 已内联的块）→ 把答案写进脚本
  调 `submit`；交稿后的那次请求同时是下一窗的 `next_task`。于是每窗 3–4 次请求，**每次都重读整条 run 累积的上下文**
  （1.6 万一路涨到 22 万，共 24 次请求）；`resume` 每窗 1 次请求、上下文涨到约 15 万。7 个 task 之间没有长轮询空等。
- **我们自己的账算错了两处**（重算时发现，已修：`a80e73fd`，按用量来源分家换算）：CodeBuddy（与
  Claude Code 同一条转换）的缓存读取与写入在转成统一用量时被丢掉，账上缓存恒为 0；agy 的 `input_tokens`
  本来就不含缓存，转换却当它含缓存再减一次，未缓存输入几乎全记成 0。任务报告的 token 合计受同样影响。上表是按各家原始用量重算的，不受这两处影响。


**两个真问题**

1. **Codex 的 `resume` 收不到任务**（已修，`d133b7b2`）：`codex exec resume` 只在 prompt 写成 `-` 时才读
   stdin，driver 照旧把 prompt 写在参数上，于是每个续上的窗口都回答「未收到 stdin JSON」，直到第二档换新会话
   才交出结果；26 分钟那次的第 9 窗换了新会话也没过，整条 run 失败。修后 6 窗同一会话、每窗第一次就过。
2. **CodeBuddy hy3 守不住工具协议**：`pseudo-conversational` 两次运行、四条会话，都是调了 `next_task`、写出
   一万多 token 的回答、却从不调 `submit` 就结束，准备任务即失败。同一个 hy3 在 `resume`（capsule，不需要
   自己调工具交稿）下能跑完。这是低档模型的能力边界，不是档位的问题；但它说明「把低能力模型放在工具会话
   形态上」要先验过。

#### 3.9 `resume` 对 `per-window`：跨窗信息传递与单窗质量（2026-10-02，agy 每臂 n=5、Codex Luna 每臂 n=3）

**结论**：`resume` 把前面窗口的信息带到了后面——在看结果前冻结的「必须靠前文才能译对」的点上（主分析：强、中两档），
五格里四格 `resume` 的每次运行都高于 `per-window` 的每次运行，剩下一格（agy 戊）范围重叠；专名写法也更统一。
单窗整体质量两家都只有弱信号、方向相反：agy 戊 `per-window` 稍好（后段窗口差得多），Luna 戊 `resume` 稍好，
两处置换检验 p 都在 0.1 上下，其余各格看不出差别。花费：未缓存输入两臂相当，缓存读取 `resume` 多 4–20 倍。
**两家都不接入、默认值不动**（owner 2026-10-02）：不是因为质量，而是这条线还没探索完——`resume` 进生产前的
两项前提（每窗只发增量、历史上限或轮换，见 [`llm_followups.md`](llm_followups.md)「未做：`resume` 档在工具会话下仍未定义」）
都没做；agy 后段的迹象若属实，也可能正是历史过长所致。所以 `resume` 留在实验档。⚠ Luna 只到 n=3（owner 定不补）。

**设计**（相对 [3.8 会话档位与传输试跑](#38-会话档位与传输试跑q1q32026-09-30每臂-n1) 的改动）：

- **素材**：维护者本地精修对照组里两份多人对谈节目与一份同系列节目，各剪前 25 分钟：丁（专名密）、戊（专名稀疏）、
  己（与丁同系列，一个人名被 ASR 写成七八种形式，只给 Luna 跑）。每窗字幕 token 上限 3600，切成 5–6 窗、每窗约
  5 分钟（生产默认不设上限时约 10 分钟一窗；试过 1200 上限切成 18 窗约 1 分钟一窗，太碎，会放大 `resume` 的优势，弃用）。
- **两臂**：capsule 版 `per-window` 与 `resume`，只差会话档位、传输相同。开关同 3.8：`--media text --retrieval none
  --knowledge none --fast off --continuity serial`。
- **共用准备任务**：每份素材只跑一次准备任务（agy），各臂各家都用它（`run_session_ab.py --prepare` / `--context-file`）。
  3.8 里质量的输赢多半来自「准备任务那一次有没有认出某个错名」，共用之后这一项噪声不复存在。
- **三把尺**（评委全部是 agy 3.7 Flash，盲评）：
  1. **逐窗排名**：同一窗口所有运行的译文一起排序（agy 10 份、Luna 6 份），可并列；两臂比平均名次。
  2. **逐点评审**：看结果之前由另一个 agent 只读原文与精修字幕挑定的点，冻结并记哈希，每份素材 10–16 个，按强 / 中 / 弱
     分档；冻结时定下主分析只用强、中两档，弱档单独报。每点每份判对 / 半对 / 错（1 / 0.5 / 0）。
  3. **专名一致性**：每个反复出现的专名与称呼，数每次运行偏离它自己主写法的次数（越少越统一）。

**读数**（范围是每臂各次运行；名次越小越好）

| 家 | 素材 | 臂 | n | 逐点得分（主分析） | 弱档 | 偏离主写法 | 平均名次 | 首次校验通过 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agy | 丁 | `per-window` | 5 | 0.857 ×5 | 0.67–0.83 | 21–30 | 5.22 | 24/25 |
| agy | 丁 | `resume` | 5 | **1.0 ×5** | 0.67–0.83 | 19–21 | 5.78 | 24/25 |
| agy | 戊 | `per-window` | 5 | 0.69–0.89 | 0.25–1.0 | 25–29 | 4.75 | 28/30 |
| agy | 戊 | `resume` | 5 | 0.77–0.89 | 0.75–1.0 | 22–25 | 6.25 | 30/30 |
| Luna | 丁 | `per-window` | 3 | 0.79–0.93 | 0.67 | 19–24 | 3.53 | 13/15 |
| Luna | 丁 | `resume` | 3 | **1.0 ×3** | 0.83 | 19–20 | 3.47 | 11/15 |
| Luna | 戊 | `per-window` | 3 | 0.42–0.58 | 0.5–1.0 | 25–31 | 4.11 | 18/18 |
| Luna | 戊 | `resume` | 3 | **0.69 ×3** | 0.75–1.0 | 28–29 | 2.89 | 18/18 |
| Luna | 己 | `per-window` | 3 | 0.864 ×3 | 1.0 | 17–18 | 3.60 | 13/15 |
| Luna | 己 | `resume` | 3 | **0.909 ×3** | 0.9–1.0 | 16–20 | 3.40 | 14/15 |

- **跨窗信息传递**：粗体那四格里，`resume` 的每次运行都高于 `per-window` 的每次运行——5 对 5 是能有的最强区分
  （单侧 p≈0.004），3 对 3 也是（p≈0.05）。己的差只在一个点上（0.909 对 0.864）：共用的准备结果里已有那个人名，
  `per-window` 多半也写对。agy 戊重叠。agy 丁 `resume` 第 1 次有一条判词评委写的不是对 / 半对 / 错，剔除不计。
- **单窗质量**：按「两臂无差别」的零假设，单个窗口上两臂平均名次之差的标准差约 1.9（agy，5 对 5）、1.5（Luna，3 对 3）；
  第 1 窗两臂条件完全相同（都是新会话），它的差只是这个噪声里的一个样本，不能当噪声的尺子。
  各窗平均名次（`per-window` / `resume`）：

  | | 第 1 窗 | 2 | 3 | 4 | 5 | 6 |
  | --- | --- | --- | --- | --- | --- | --- |
  | agy 戊 | 5.4 / 5.6 | 4.9 / 6.1 | 5.8 / 5.2 | 4.1 / 6.9 | 3.4 / 7.6 | 4.9 / 6.1 |
  | agy 丁 | 5.2 / 5.8 | 5.8 / 5.2 | 4.7 / 6.3 | 5.7 / 5.3 | 4.7 / 6.3 | — |
  | Luna 戊 | 4.2 / 2.8 | 3.3 / 3.7 | 4.0 / 3.0 | 4.0 / 3.0 | 4.2 / 2.8 | 5.0 / 2.0 |
  | Luna 己 | 4.5 / 2.5 | 4.2 / 2.8 | 2.3 / 4.7 | 4.5 / 2.5 | 2.5 / 4.5 | — |

  按运行做置换检验，只有两格到 0.1 上下、方向相反：agy 戊 `per-window` 更好（25 对里赢 20 对，p≈0.075），差集中在第 4、5 窗
  （约 1.5 和 2.2 个标准差，且是 6 窗里最大的两窗）；Luna 戊 `resume` 更好（9 对里赢 8 对，p≈0.10）。其余各格 p≥0.35。
  评委记下的错误同样对称：agy 戊 `resume` 误译 18 对 8、添加 10 对 3；Luna 戊反过来，`per-window` 误译 20 对 11。
  两处都只算迹象。
- **花费**（千 token；`resume` 按会话取最大值；不含共用的准备任务）：

  | 家 | 素材 | 臂 | 未缓存输入 | 缓存读取 | 输出（含思考） |
  | --- | --- | --- | --- | --- | --- |
  | agy | 丁 | `per-window` / `resume` | 175–226 / 225–252 | 65–155 / 1,292–1,752 | 101–122 / 87–113 |
  | agy | 戊 | `per-window` / `resume` | 206–286 / 237–334 | 82–200 / 1,635–1,936 | 117–148 / 65–121 |
  | Luna | 丁 | `per-window` / `resume` | 124–170 / 143–365 | 63–94 / 326–937 | 36–42 / 34–56 |
  | Luna | 戊 | `per-window` / `resume` | 137–155 / 138–172 | 65–86 / 461–492 | 39–40 / 37–46 |
  | Luna | 己 | `per-window` / `resume` | 125–158 / 143–209 | 63–103 / 334–698 | 36–41 / 36–47 |

  Luna 丁那次 365 / 937 有两部分：前 4 窗首次都没过校验（CSV 某几行字段不够），修复轮在同一会话里续、整份任务与上一版
  答案再发一遍，历史翻倍；第 5 窗一窗又多出约 11.4 万未缓存、缓存读取只多约 1.1 万，像是缓存在这一窗断了（原因没查），
  约占总未缓存的三成。同一种格式错在这条会话里连续出现，其余 `resume` 运行没有——是否「`resume` 把一次格式失误
  带到后面」只有这一例，未定。墙钟不可比：这一轮各次运行是同时并行跑的。

**局限**：
- 逐点评审的强点少：丁 1 个、戊 5 个、己 3 个，主分析里多数是中档。
- 评委只有一位，没有重评量评委自己的波动。
- 评委是能读本机文件的 agent，拿到的是评审包的路径；标签与运行的对照表放在评审包的同级目录，评委按任务要求只读评审包，
  但技术上读得到对照表。下次把评审包正文直接交给不能读文件的评委。
- 两臂都开着 `continuity serial`，`resume` 量的是「在已有的跨窗建议之上，隐式历史还能多带多少」。

**数据**：每次运行的产物在维护者本地（可按 `tools/bench/run_session_ab.py` 重跑，花 agent 额度）；冻结的评测清单、
计入结论的评审原文与对照表、逐次运行的汇总读数在维护者本地的数据目录里（不可再生的部分）。
打包与汇总用 `tools/bench/rank_session_ab.py`（逐窗排名）与 `tools/bench/probe_session_ab.py`（逐点评审、专名一致性；
`collect --probes <清单>` 按冻结的分档出主分析）。

#### 3.10 纠错 driver 对比：GPT-6.1 Sol 与 GPT-6 Luna（2026-10-09，hina-05 整份 3 窗，各 n=1）

一小时的日语直播（hina-05），纯文本、quality 档、同一份准备任务（Sol 跑的，Luna 复用），只换纠错窗口的 driver（都经 Codex）。

- 用量：Sol 未缓存输入 34.6 万、缓存读 593 万、输出 4.7 万；Luna 26.7 万、303 万、5.6 万。
- 质量：两版按时间对齐后有差异的组 758 个（三窗 275、249、234）。匿名为 A / B 交给 GPT-6 Luna 盲审（一窗一次，只给文字与背景，不给音频）：Sol 实质更好 38 组（三窗 12、15、11），Luna 实质更好 28 组（15、9、4），其余 692 组记为平局或仅凭文字无法判定。
- 审查列出的实质差异：Sol 多在日文听错与专名的修正（「小橋さん」→「大橋さん」、「雷の龍」→「ライノ竜」、「北条の国」→「豊穣の邦」、「謎くらい」→「ナド・クライ」）；Luna 多在中文保留「听说 / 似乎」等转述与推测语气。Sol 把素材里的「真央」写成「M・A・O」；Luna 有两处在日文行里写了中文地名（「沃陸の邦」「沃陆之邦の選手」）。没有对应字幕的组：Sol 6 个，Luna 1 个。审查引用的 9 组已对照原文核实。
- 限定：一份素材、一个 pass；审查者与被审的一方是同一个模型（盲审）；对齐与分组用的是一次性脚本，没进仓库。
