# `diarizen_export/` —— 把 DiariZen 导成生产能用的产物

生产端**不得 import `diarizen`，也不得 import `pyannote`**（研究路径靠 vendored 的 pyannote 分支跑）。
和 Sortformer 的 D1 决定一样：生产只需要前向，那就只导出前向。

```powershell
python tools/diarizen_export/export_diarizen.py --device cuda --out tmp/diarizen-export
python tools/diarizen_export/export_diarizen.py --device cpu  --out tmp/diarizen-export-cpu
```

⚠ **每个设备一份导出。** trace 把设备记进了图里，两个方向都不通：CUDA 上 trace 的件在 CPU 上
从 TorchScript 内部报错，CPU 上 trace 的件在 CUDA 上也报（2026-09-21 实测）。两份的前向各自与 eager 逐位相同。

**用户不跑这一步**（owner 2026-09-21）：`--diarize` 按设备下载我们托管的两份
（[`spurtcarl/finesub-diarizen-cuda`](https://huggingface.co/spurtcarl/finesub-diarizen-cuda)、
[`-cpu`](https://huggingface.co/spurtcarl/finesub-diarizen-cpu)，manifest 里的 `diarizen-ts-cuda` / `-cpu`，
钉 commit、按 sha256 校验）。重新导出之后用 `release.py` 发布：它按 `release_card.md.tmpl` 写模型卡
（⚠ CC-BY-NC 4.0，逐个组件写来源与许可），`--upload` 需要有写权限的 HF 登录，并打印要贴回 manifest 的条目。
默认后端 nemotron3 只用这里导出的**声纹**那一件加 PLDA，而这两样本身是 CC BY 4.0：`release_embedding.py` 把两个设备的
`diarizen_embedding.ts` 与 PLDA 发到单独的 [`spurtcarl/finesub-speaker-embedding`](https://huggingface.co/spurtcarl/finesub-speaker-embedding)
（manifest 的 `speaker-embedding`，模型卡是 `embedding_card.md.tmpl`；PLDA 先核对与 pyannote community-1 逐字节相同才发）。
`FINESUB_DIARIZATION_ARTIFACTS` 指到本地导出目录时优先用本地的，不下载——⚠ 它必须是**本机设备**那一份：生产在加载前从图本身读出它是在哪个设备上 trace 的（CUDA 版的代码里带 `cuda:0` 常量），对不上就提示并跳过。

产物三件（本地导出与托管的都是这三件）：

| 件 | 大小 | 是什么 |
| --- | ---: | --- |
| `diarizen_segmentation.ts` | ≈279 MB | WavLM-large + Conformer，一次前向 `(batch, channel, samples) → (batch, frames, powerset)` |
| `diarizen_embedding.ts` | ≈26.8 MB | WeSpeaker 的 **resnet 部分**，`(fbank, weights) → embedding` |
| `plda/` | 小 | `plda.npz` + `xvec_transform.npz`，聚类要的两张表 |

⚠ **`plda/` 不是导出来的，是从 checkpoint 里拷过来的**，脚本顺手做了。
它放在这里而不是留给 `hf_weights.prepare()`，是因为**一次运行永远不碰 Hub 权重**——
manifest 里 `diarizen` / `wespeaker` 那两条存在的理由是让这个脚本取得到 checkpoint，不是让跑的时候取。

⚠ **WeSpeaker 只导了一半。** 它的前端是 `torch.vmap` 包 `torchaudio.compliance.kaldi.fbank`，
tracer 进去即死；而 `forward` 恰好就是 `compute_fbank` → `resnet(fbank, weights)[1]` 两步。
所以 fbank 在生产侧用 torchaudio 的公开 API 重算（`diarizen_features.py`，本来就是声明依赖）。
拆成两半正是「bug 藏在没人验的那半」的形状 ⇒ **两半都验**，`_check_production_fbank` 比对生产 fbank 与
`model.compute_fbank`，两种长度上逐位相同。

## 动它之前必须知道的三条

1. ⚠ **不要 `torch.jit.freeze`。** 它把 BatchNorm 折进前面的卷积，答案移动 **4.5e-4 ~ 7.3e-4**
   （三个批次实测，而 eager 与自身逐位相同）。不 freeze 的 trace 在每个批次上都**逐位相同**。
   本仓早就知道这个坑的形状——`tools/sortformer_export/README.md` 对 NeMo 的 `model.export()`
   写着同一句话，那个也是 freeze。知道了还是走了进去。
2. **要先清掉 `@torch.jit.export` 标记**（脚本里做了）。WavLM 那层的 `extract_features_mc` 带着它，
   一个标记就会让 `torch.jit.trace` 去 script 整个类，连我们从不走的死分支
   （`normalize_waveform` 带显式 lengths 的那条）也要编，而那条编不过。
   ⚠ 这只改「哪些方法进产物」，不改任何一个方法算什么——**逐位校验是那句话的证据，不是注释**。
3. **要把 `feature_grad_mult` 设回 1.0**（脚本里做了）。`GradMultiply` 只缩放梯度、前向是恒等，
   推理时纯属空转，而 TorchScript 序列化不了 Python 的 `autograd.Function`。

（第四条，不在导出里但同源：Lightning 的 `trainer` 属性**抛 `RuntimeError` 而非 `AttributeError`**，
`hasattr` 扫描直接炸，包 wrapper 没用——扫描会递归到子模块。）

## 验收：**报，不判**

⚠ **不相等不拒绝写盘**，README 早先写的相反，那是错的。换产物形态（eager → TorchScript）是
`README_DEV.md` 开发原则的 **A 类**改动，逐位相等是**免费的**无回归证明而不是门槛；做不到时该做的是**刻画抖动**。
逐位相等只有在**声称语义不变**时才是必须的，而那个「声称」当初是我自己加的。

所以 `verify()` 报 `max abs` / `max rel`，四个 `check_*.py` 报差异率，都不用退出码下判断：

| 脚本 | 比什么 | 实测 |
| --- | --- | --- |
| `export_diarizen.py` 的 `verify()` | traced vs eager，**同一形状重复 4 次** | 见下 |
| `check_s1.py` | 局部分割 vs 冻结缓存 | 0.0214% 的帧 |
| `check_s2.py` | 声纹 vs 冻结缓存 | 余弦 0.999999464，max 1.42e-3 |
| `check_s4.py` | 聚类+组装 vs 冻结缓存（喂**缓存的**分割） | 2.3781% 的帧，说话人数相同 |
| `check_port.py` | 端到端**逐字幕**标签改变 | 6 份：中位 0.669%、最大 1.151%，说话人数 6/6 全同 |
| `noise_arm.py` | 这条链对 σ=1e-7 的自身敏感度 | 上面那个数要对着它读 |

⚠ **`verify()` 必须在同一形状上重复调用。** 第一版跑 batch 1、2、3 各一次就通过了，
而 profiling executor **按形状特化**、**首次运行之后**才重编译并融合算子——
那三次全是「该形状的第一次调用」，差异恰恰从第二次才出现。**校验通过，bug 完好无损地留在产物里。**
⇒ 一般规律：**「跑一次就过」的等价性校验几乎必然是假的**。

⚠ 校验还要覆盖**存盘再读回**：序列化是独立一步，它刚刚才拒绝写出一个 Python 函数，
显然不是恒等变换。

⚠ 校验要在**与生产相同的开关下**跑：`_jit_fusion_disabled()` + `_tf32_disabled()`，batch 固定 2。
这条链上没有「小到无所谓」的数值差——硬 powerset argmax 把末位浮点差放大成离散标签翻转，
四个开关都实测改过答案。机制与三条纪律在 `docs/diarization-findings.md` 5.2。

⚠ 权重是 **CC-BY-NC 4.0**，我们托管的转换版同样是。manifest 里 `diarizen` / `wespeaker` 两条是给导出脚本取
checkpoint 的，`diarizen-ts-cuda` / `-cpu` 两条是运行时下载的。用户向的许可说明在 `docs/manual/diarization.md`。
