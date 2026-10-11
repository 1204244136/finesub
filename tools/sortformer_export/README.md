# `sortformer_export/` —— 把 Sortformer 导成生产能跑的三个件

生产**不吃 `nemo_toolkit`**（计划 §8 的 D1）：它是 36 个直接依赖的训练面机器
（lightning / hydra / wandb / datasets / lhotse…），会进 `pylock.win-py312.toml`、被每个托管运行时
装上，只为跑一个 117M 模型的前向。这个目录是**唯一**需要 nemo 的地方——由我们跑一次，
产出三个 TorchScript 件，管线用已经钉死的 torch 2.11 加载（含 GPU）。

```powershell
conda run -n asr python tools/sortformer_export/export_sortformer.py `
    <diar_streaming_sortformer_4spk-v2.1.nemo 的路径> --out-dir <目标目录>
```

产出（切分沿用 NVIDIA 自己的导出边界 `forward_for_export`）：

| 件 | 大小 | 内容 |
| --- | ---: | --- |
| `sortformer_preprocessor.ts` | ~145 KB | 128-mel 前端 |
| `sortformer_core.ts` | ~469 MB | pre_encode + 17 层 Conformer + 18 层 Transformer + 4 路 sigmoid |
| `sortformer_cache_update.ts` | ~567 KB | 同步 `streaming_update`（静音档案、封顶判断、压缩） |

## ⚠ 不要改回 `model.export()`

核是**我们自己 trace 的**，不走 NeMo 的 `model.export()`——那个函数在 trace 之后调
`torch.jit.freeze()`（`nemo/core/classes/exportable.py`），freeze 把 BatchNorm 折进卷积：
数学等价，数值差约 1e-4。

**这个架构吃不下这 1e-4**：说话人缓存按**离散 top-k** 保留帧，一次选择翻转，两边缓存就永久
分叉。61.6 分钟实测：0.35% 的帧翻转、段数差 1–2%、一处边界差 **156 秒**。不 freeze 之后，
循环真正用到的三种形状（首块无左上下文、中间块、末块短窗）**逐个对 eager 逐位相等**。

trace 的形状取循环里占 99% 调用的那一种（满窗 + 满缓存 + 空 fifo）。⚠ 补零到满宽**反而**
不相等（实测 1.09e-4），所以末块保持原样、别去补齐。

## 两份逐字拷贝，以及为什么它们只住在这里

TorchScript 拒绝 NeMo 的两个方法**照原样**编译，所以脚本里各抄了一份并做最小改动。
⚠ **两份拷贝都只存在于导出器里，管线永远看不到它们**——这是这个目录存在的理由。

| 拷贝 | 挡路的是什么 |
| --- | --- |
| `FilterbankFeatures.forward` | 实例级 `torch.no_grad` 重绑、模块级全局 `CONSTANT`（sqrt 的 guard）、`log_zero_guard_value_fn` 的 isinstance 分支、`pad_value` 是 int 而 `pad()` 要 `Optional[float]` |
| `_compress_spkcache` | 一句 `scale_factor=2` 把 int 字面量传给 float 参数 |

⚠ **拷贝烤进去的每条假设都在导出前被断言**（`stft_pad_amount is None`、`use_grads False`、
`log` 的 add-guard、`frame_splicing 1`、`normalize NA`、`nb_augmentation_prob 0`、
`async_streaming False`）。换 checkpoint 若不满足，脚本**报错退出**而不是产出一个静默偏移的件。

⚠ 还有一条假设落在**驱动**侧而不是这里：mel 拷贝去掉了 NeMo 那两个 `autocast(enabled=False)`
（TorchScript 要求常量 device），所以**调用方不得在 autocast 下跑它**。

## 验收

这个脚本**不做**一致性检查。导出完必须跑计划 §5a 的 G1–G4（帧级概率偏差、阈值化帧一致率、
段数与边界、轨迹占比），素材与门槛都写在那里，**门槛先于测量写死**。

许可：NeMo 是 Apache-2.0，两份拷贝保留 NVIDIA 的版权。模型权重用 **v2.1**
（NVIDIA Open Model License）；**v1 是 CC-BY-NC，不进产品**。
