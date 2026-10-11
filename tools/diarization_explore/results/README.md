# 冻结的评测记录

这个目录只放**被 tracked 文档引用的、已冻结那一轮的**逐文件读数，一轮一个文件，
文件名带日期。它们随仓库公开发布（`tools/` 不在 `$PrivatePaths` 里），所以：

- **别往这里堆中间产物。** 一轮跑出来的原始逐文件 dump、时间线、概率矩阵留在 `tmp/`；
  只有当某份 tracked 文档要引用某个数，那一轮的汇总 JSON 才进来。
- **进来之后不再改。** 复算要对得上文档里的数——改了就等于悄悄改了已发表的结论。
  发现算错了，新开一个文件并在文档里写明取代关系。
- **一轮一个文件。** 当前共 2.8 MB：

| 文件 | 那一轮是什么 | 谁在引用 |
| --- | --- | --- |
| `accuracy-20260916.json` | 74 份合格素材的基线评测（收口 W4 的文件清单也取自这里的 `inventory.valid`） | `docs/diarization-findings.md` 2.4 与 7、`w4_segmented.py` |
| `accuracy-v2-20260916.json` | 七份开发集调参 + 九份姐姐咚咚冻结确认（base / high / right80 / community / DiariZen） | findings 2.4、本地归档的第二轮协议 |
| `diarizen-v3-20260917.json` | DiariZen 第三轮（双向身份合并、同家族三视图、原生边界、三模型局部投票） | 本地归档的第三轮协议与报告 |
| `backend-cost-w2-20260920.json` | **W2 重算**：姐姐咚咚 15 集两后端，用修好的编辑代价尺子（无标签行不再免费）+ 新增的 `leading ≥10 行` | 本地归档的收口计划（W2 表） |
| `backend-cost-w2b-hina-20260920.json` | **W2b 重算**：hina 19 份，同上 | 本地归档的收口计划（W2b 表） |
| `w2-bias-20260920.json` | **A3 重算**：两个语料的样本内 / crossfit / 轨数零假设，修了两处（无标签行计价、crossfit 改帧级留出） | 本地归档的收口计划（「下一批 ①」） |
| `c1-base-refresh-20260920.json` | **C1**：`base + refresh` 对 `base`，姐姐咚咚 9 份留出 + hina 7 份开发集；两组都没过预注册门槛 | 本地归档的收口计划（C1 那节） |
| `dz-track-drift-20260920.json` | 轨内漂移信号（聚类侧 `foreign_rate`）对轨不纯度，51 条轨，r=+0.075，**不成立** | 本地归档的说话人身份计划（B.1） |
| `dz-real-single-speaker-20260920.json` | **Q13 第一次读数**：真·单人 `kaguya60` 走完整生产路径，对 39 份多人素材的参照带（缓存 embedding 重聚类） | 本地归档的说话人身份计划（A.5） |
| `dz-identity-meancentroid-20260920.json` | **身份分离重算**：簇心改用加权**均值**（原来混着轨长），姐姐咚咚同集 4/53 + 跨集 374/745 | `docs/diarization-findings.md` 第 8 节 |
| `nemotron3-20260924.json` | **Nemotron-3-Diarization 一轮**：公开真值集 DER 与重叠精确率 / 召回（21 份，含前几轮缓存臂）、W2 / W2b 预注册编辑代价（对 DiariZen）、输入扰动、三后端单遍耗时、CPU 单份粗测、平滑阈值扫描、显存随时长，以及超过 8 人的方法台（`over8.py`：VoxConverse ≥9 人 61 份 / 魔裁 4 小时 / AliMeeting 5 份 / hina 19 / 姐姐咚咚 15）。由 `nemotron3_freeze.py` 从当时 worktree `tmp/` 的逐文件输出重算（那些输出已丢弃，重生成须先重跑推理）。⚠ 记录里的 `scripts` 摘要是对 LF 换行的文件算的，Windows 检出（autocrlf）要先把 CRLF 换回 LF 再比；`over8.py` 冻结后只加过一行注释（`bc52cfce`），所以它的摘要对不上；`recurvbx` / `enrollmerge` / `vbx-ahc0.45` 在 VoxConverse 上只有筛选子集或部分文件，看各自的 n | `docs/diarization-findings.md` 2.6 与 7、维护者本地的换默认后端计划 |
| `nemotron3-port-20260925.jsonl` | **生产移植对 NeMo eager**（`nemotron3_port.py`，预注册）：4 份（hina-04 / hina-10 / 魔裁 01、05），每份 σ=1e-7 噪声臂与移植臂的帧级 / 链级差异，外加一份 CPU 臂；修了 bf16 前端之后那一轮（修之前 hina-10 帧级 15%，只记在 findings） | `docs/diarization-findings.md` 2.6「生产移植」 |
| `dz-max-speakers-rerun-20260920.json` | **Q11 臂 1 的 24 份跑**（中位 +530、区间 [−37, +1353]、23/24 变差）——取代 `dz-max-speakers-20260920.json` 那份 10 份的 | 本地归档的收口计划、`docs/manual/tuning.md`、`cluster()` docstring |

⚠ **W2 / W2b 原来只写进 `tmp/`**，而它们的数被 tracked 文档引用着——正是本目录 README 这条规则要防的情况（2026-09-20 审计 §5.3）。`backend_cost.py --write` 现在写到这里。

⚠ `dz-max-speakers-20260920.json` 里是**被取代的 10 份跑**：臂 1 后来扩到 24 份（23/24 变差，中位 +530），那一轮的记录当时没有落盘。上表引用它的地方要改引新记录，重跑之前**不要**把文档里的 23/24 改回 10/10 —— 10/10 是旧的那一轮。
