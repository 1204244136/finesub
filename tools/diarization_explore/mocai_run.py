"""Diarize one whole 魔裁 part, un-separated, under the 8 GiB VRAM guard. One file per process.

The owner's 2026-09-16 decision after the 4-hour check (separation bought nothing for the timeline, mean
-0.7 pp, matching 希娜's -0.73): work on the un-separated audio. `--no-separate` produces a `-vocal.ogg` at
the same 16 kHz mono spec and path separation would, so this is the production shape minus separation.

One file per process on purpose: PyTorch's caching allocator keeps the high-water mark for the life of the
process, and five files in one process pushed the guard's reading to 6.73 GiB where one file needs ~4.3.

    python tools/diarization_explore/mocai_run.py 1        # input1 -> out/mocai-in1/mocai-in1-diar-nosep.json
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WT / "src"))

import vram_guard  # noqa: E402  -- before anything touches CUDA

SOURCE = Path("D:/ClipTranslArchive/yuka")


def main() -> None:
    part = int(sys.argv[1])
    stem = f"mocai-in{part}"
    out_dir = WT / "out" / stem
    vocal = out_dir / f"{stem}-vocal.ogg"
    timeline = out_dir / f"{stem}-diar-nosep.json"
    if timeline.exists():
        print(f"{stem}: timeline already there")
        return
    if not vocal.exists():
        subprocess.run(
            [sys.executable, "-m", "finesub.pipeline", str(SOURCE / f"input{part}.mp4"),
             "--stage", "vocal", "--no-separate", "--language", "ja", "-o", str(out_dir / f"{stem}.srt")],
            check=True, cwd=WT, stdout=subprocess.DEVNULL,
        )

    vram_guard.install()
    import torch
    from finesub.speech.diarization import stage as D

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = D.run_diarization(vocal, device="cuda")
    D.write_timeline(result, timeline)
    hours = max(t.end for t in result.tracks) / 3600
    print(f"{stem}: {hours:.2f} h, {len(result.tracks)} segments, "
          f"allocator peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB, "
          f"reserved peak {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB, "
          f"watchdog peak {vram_guard.peak_gib():.2f} GiB, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
