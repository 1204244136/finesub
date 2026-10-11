"""W4's other half: what does "always segment" cost in wall clock?

W4 settled that segmenting every file is a noise-level change to the ANSWER. It never measured the
price, because `w4_segmented.py` recorded one `elapsed` covering all three arms -- so the ratio was
not recoverable from what was written down, and this is the rerun that gets it.

The cost is not obviously one-directional. Segmenting moves each piece to the device separately and
brings features back to host memory, so short files now pay a round trip they used not to; long
files stop paying 2.50 GiB per hour of peak and may come out ahead. That is why this reads several
lengths rather than an average over everything.

Through `tools/bench/discipline.py`, so the six conditions hold by construction rather than by my
remembering them: first runs discarded and printed, absolute ms beside every RTF, the two arms
alternated AB/BA with the order effect cancelled by geometric mean rather than averaged, a scale
check, empty output refused, and all eight config fields declared.

⚠ Run it on an idle machine and read the printed GPU baseline before trusting the number.

    python tools/diarization_explore/w4_timing.py
    python tools/diarization_explore/w4_timing.py --ids hina-07 hina-01 --rounds 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # `tools/`, for `bench.discipline`
import accuracy_lab as lab  # noqa: E402
from bench.discipline import (  # noqa: E402
    RunConfig,
    environment,
    paired_ratio,
    refuse_empty,
    scale_check,
)

OUT = lab.WT / "tmp" / "diarization-w4-timing"


def gpu_baseline(release: bool = False) -> dict:
    """What else is on the card. A timing taken beside someone else's job is not a timing.

    ⚠ `release=True` for the reading taken AFTER the run: without emptying the cache first this
    probe reports our own allocator, not a neighbour. The first run of this script read 1935 MiB
    before and 9124 MiB after and briefly looked contaminated -- all of it was the whole-file arm's
    own 3.3 GiB of features on a 79-minute file, and the card was back to 1904 MiB the moment the
    process exited.
    """

    if release:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        used, total, utilisation = (int(part) for part in raw.split(", "))
        return {"vram_used_mib": used, "vram_total_mib": total, "utilisation_pct": utilisation}
    except Exception as error:  # a missing probe is a missing note, not a failed benchmark
        return {"error": str(error)}


def pick(ids: list[str] | None) -> list[str]:
    """Shortest, median and longest of the qualified set -- the ratio is expected to vary with length."""
    if ids:
        return ids
    data = json.loads((HERE / "results" / "accuracy-20260916.json").read_text(encoding="utf-8"))
    lengths = []
    for ident in data["inventory"]["valid"]:
        path = lab.source(ident)
        if path.exists():
            lengths.append((path.stat().st_size, ident))
    lengths.sort()
    return [lengths[0][1], lengths[len(lengths) // 2][1], lengths[-1][1]]


def arms(runner, waveform, samples):
    """`(whole, seg)`: identical audio, identical model, only the tiling differs."""
    import torch

    from finesub.speech.diarization import runner as module
    from finesub.speech.diarization.chunking import FeatureSegment

    tensor = torch.from_numpy(waveform).unsqueeze(0)

    def one_segment(count, **kwargs):
        return [FeatureSegment(0, count, 0, 0, module.feature_frame_counts(count)[1])]

    def seg():
        return runner.diarize(tensor, samples)

    def whole():
        original = module.feature_segments
        module.feature_segments = one_segment
        try:
            return runner.diarize(tensor, samples)
        finally:
            module.feature_segments = original

    return whole, seg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--rounds", type=int, default=4, help="must be even")
    args = parser.parse_args()

    import soundfile as sf
    import torch

    from finesub.speech.diarization.runner import SortformerRunner
    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        raise RuntimeError("CUDA unavailable; this GPU experiment does not silently fall back")
    torch.set_num_threads(4)

    before = gpu_baseline()
    print(f"GPU before: {before}", flush=True)

    # ⚠ `lab.ARTIFACTS`, not a path built from `lab.WT`: the exported TorchScript lives in the
    # MAIN checkout and every worktree reads it from there. Spelling it out again here is how the
    # first run of this script died.
    runner = SortformerRunner(lab.ARTIFACTS, device="cuda")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ident in pick(args.ids):
        wave, rate = sf.read(lab.source(ident), dtype="float32", always_2d=True)
        assert rate == 16000, rate
        waveform = wave.mean(axis=1)
        seconds = len(waveform) / 16000
        whole, seg = arms(runner, waveform, len(waveform))

        # Condition 5, and condition 1's discarded first run in the same call: an arm that
        # produced nothing is a failure rather than a fast record, and these two calls are the
        # warm-up whose cost `thermal_state="warm"` declares the timings exclude.
        refuse_empty(np.asarray(whole().cpu()), what=f"{ident} whole-file timeline")
        refuse_empty(np.asarray(seg().cpu()), what=f"{ident} segmented timeline")

        print(f"\n{ident}: {seconds / 60:.1f} min, {args.rounds} rounds", flush=True)
        result = paired_ratio(whole, seg, rounds=args.rounds, labels=("whole", "seg"))
        config = RunConfig(
            backend="sortformer-v2.1 TorchScript", quantization="fp32", device="cuda",
            threads=4, thermal_state="warm", load_excluded=True, clip_seconds=seconds,
            aggregate="median",
            notes=f"W4 timing; GPU before {before}",
        )
        rows.append({"ident": ident, "seconds": seconds, "config": config.__dict__,
                     "result": result})
        # Condition 2: absolute milliseconds beside every ratio, never a ratio alone.
        whole_ms, seg_ms = np.median(result["whole_ms"]), np.median(result["seg_ms"])
        print(f"  whole {whole_ms:9.1f} ms (RTF {whole_ms / 1000 / seconds:.4f})", flush=True)
        print(f"  seg   {seg_ms:9.1f} ms (RTF {seg_ms / 1000 / seconds:.4f})", flush=True)
        print(f"  ratio {result['ratio']:.3f}×  次序效应 k={result['order_effect']:.3f}", flush=True)

    after = gpu_baseline(release=True)
    print(f"\nGPU after: {after}")

    # Condition 4: a longer file must cost proportionally more. What this catches is not a slow
    # arm but a fake-fast one -- a long file that came back as quickly as a short one did not run.
    scale: dict[str, object] = {}
    if len(rows) >= 2:
        for label in ("whole", "seg"):
            points = sorted((row["seconds"], float(np.median(row["result"][f"{label}_ms"])))
                            for row in rows)
            checked = scale_check(points, quantity="ms")
            scale[label] = checked
            print(f"  [{'PASS' if checked['passed'] else 'FAIL'}] scale check {label}:"
                  f" 最差偏离 {checked['worst_deviation']:.2f}")

    ratios = [row["result"].get("ratio") for row in rows if row["result"].get("ratio")]
    if ratios:
        geometric = float(np.exp(np.mean(np.log(ratios))))
        print(f"\n整片 / 分段 的耗时比，几何平均 {geometric:.3f}×"
              f"（>1 = 分段更快）  逐份 {[round(r, 3) for r in ratios]}")
    (OUT / "w4-timing.json").write_text(json.dumps(
        {"rows": rows, "gpu_before": before, "gpu_after": after, "scale_check": scale,
         "environment": environment(), "script": lab.digest(__file__)},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'w4-timing.json'}")


if __name__ == "__main__":
    main()
