# 安装 patched CTranslate2(ASR 必需)

ASR 阶段需要一份打过补丁的 CTranslate2。PyPI 上的原版无法运行本项目的 ASR，必须额外安装补丁版本。

下面的命令适用于 **Windows + Python 3.12 + NVIDIA 显卡**。其它平台用同一个 Release 里对应的 wheel，
覆盖方式相同，逐平台说明见 [仓库安装](repo-install.md)「其他平台」。文件名末段与平台的对应是
**按平台 × 显卡**，不是只看显卡：

| 平台 + 显卡 | 末段 |
| --- | --- |
| Windows / Linux + NVIDIA | `cu128` |
| **Windows + AMD** | `cu128`（识别在 CPU 上跑，见 [仓库安装](repo-install.md)「Windows 上的 A 卡」） |
| Linux + AMD | `rocm7.2` |
| macOS | `cpu` |

⚠ 除 Windows 外的 wheel 由 CI 构建，**发布前 Release 里还没有它们**；Windows + AMD 不受影响，
它用的就是已经发布的那份 `cu128`。构建与发布的规矩见 [`../ct2-distribution.md`](../ct2-distribution.md)。

## 装

先按[仓库安装](repo-install.md)（uv 或 pip 流程）装好项目本体。那一步会装入 **stock** 版 CTranslate2，随后**必须**用补丁 wheel 覆盖它（仓库安装手册已包含该步骤；本页供故障排查时参考）:

```powershell
# uv:
uv pip install --reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.4.0/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl"

# pip:
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.4.0/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl"
```

## 验证安装

```powershell
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

- 打印 `4.8.1+finesub0.4.0.cu128` —— 正确。
- 输出为 `4.8.1` 则说明仍是原版，请重新执行上面的覆盖安装命令。

## 故障排查

**运行 ASR 时报 `CTranslate2 was not built with the WT refine trace extension`**
当前安装的是原版，请重新执行覆盖安装命令。

**报 `CTranslate2 was not built with WT disfluency weight support`**
装的是早期的补丁版，缺语气词检测需要的那部分，请用本页的 wheel 重新覆盖安装。这项检查在加载模型时进行，
不会跑完一趟再交出空字幕。

**AMD 卡上识别回退到 CPU，警告说 `an AMD card needs the ROCm/HIP build`**
装的是 CUDA 版或 CPU 版，换成文件名带 `rocm` 的 wheel。⚠ 这条只对 Linux 成立：Windows 的 ROCm 版
还没发布，那里的识别就留在 CPU（[`repo-install.md`](repo-install.md)「Windows 上的 A 卡」）。

**重装或升级项目后再次出现该错误**
重装项目会按 `==4.8.1` 装回原版，请再次执行覆盖安装命令。

**报 `FlashAttention is built for ... only`，或在 RTX 30 / 40 系上识别结果异常**
RTX 30 系及更新的 N 卡上，补丁版默认让 Whisper 的 encoder 用 flash attention（fp16）。自己转换的、head dim 不是 64 的模型
装不进这些 kernel，会在第一次识别时报这个错；RTX 30 / 40 系走的那套预编译 kernel 也还没在实卡上验证过。两种情况都先设环境变量
`FINESUB_CT2_FLASH_ATTENTION=off` 关掉它，换回标准 attention 再跑（PowerShell：`$env:FINESUB_CT2_FLASH_ATTENTION = "off"`）。
