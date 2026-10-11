"""W9: does DiariZen survive a long recording at all, and what does it cost?

Sortformer is streaming -- its memory is flat in the length of the file, and mocai's eleven hours
cost it 1.69 GiB and a few minutes. DiariZen is offline: local segmentation over the whole file,
then WeSpeaker embeddings, then PLDA + VBx global clustering. The clustering step is the one nobody
has watched on anything this long, and its cost is not linear in the audio -- it is a function of
how many embeddings come out of it.

⚠ **This is a feasibility reading, not a quality one.** mocai has no ground truth, so there is
nothing here to score against; what the run answers is whether the thing completes, what it costs,
and how many speakers it decides there are. "It does not finish" is equally an answer (plan W9).

The window: three hours starting at hour four of `mocai-in2` (owner 2026-09-19 said a 2-4 hour
sample is enough). Deep into the recording rather than at its start -- by hour four Sortformer's
arrival-order cache has already turned its slots over several times (findings 4.3), which is the
regime the question is about. The same window is sliced out of Sortformer's existing whole-part
timeline for free, so the two are read over identical audio.

    python tools/diarization_explore/w9_long.py            # 3 h from hour 4
    python tools/diarization_explore/w9_long.py --start-hour 0 --hours 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402

PART = "mocai-in2"
OUT = lab.WT / "tmp" / "diarization-w9"
FRAME = lab.FRAME


def source() -> Path:
    """The 16 kHz mono delivery of the un-separated part -- the same file findings 4.3 was read from."""
    path = lab.WT / "out" / PART / f"{PART}-vocal.ogg"
    if not path.exists():
        raise SystemExit(f"{path} is missing; it is produced by the --no-separate pipeline run")
    return path


def sortformer_window(start_seconds: float, seconds: float) -> dict:
    """The existing whole-part Sortformer timeline, cut to the same window."""
    path = lab.WT / "out" / PART / f"{PART}-diar-nosep.json"
    if not path.exists():
        return {"available": False}
    # `-diar.json` holds one flat row per span: {"track", "start", "end"}.
    spans = json.loads(path.read_text(encoding="utf-8"))["tracks"]
    end = start_seconds + seconds
    active: dict[int, float] = {}
    for span in spans:
        low, high = max(span["start"], start_seconds), min(span["end"], end)
        if high > low:
            active[span["track"]] = active.get(span["track"], 0.0) + (high - low)
    return {
        "available": True,
        "tracks_active": len(active),
        "speech_seconds": sum(active.values()),
        "speech_seconds_sorted": sorted(active.values(), reverse=True),
    }


def peak_host_gib() -> float:
    """Windows peak working set for this process; 0.0 where the API is not there."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        # ⚠ Without this the pseudo-handle (-1) comes back as a Python int that ctypes then
        # refuses to pass as a handle -- the call raises and a bare `except` reports 0.0 GiB.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return 0.0
        return counters.PeakWorkingSetSize / 1024**3
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-hour", type=float, default=4.0)
    parser.add_argument("--hours", type=float, default=3.0)
    args = parser.parse_args()

    import vram_guard

    vram_guard.install()
    import soundfile as sf
    import torch

    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        raise RuntimeError("CUDA unavailable; this GPU experiment does not silently fall back")
    torch.set_num_threads(4)

    start_seconds, seconds = args.start_hour * 3600.0, args.hours * 3600.0
    path = source()
    began = time.perf_counter()
    waveform, rate = sf.read(path, dtype="float32", always_2d=True,
                             start=int(start_seconds * 16000), frames=int(seconds * 16000))
    assert rate == 16000, rate
    waveform = waveform.mean(axis=1)
    read_seconds = time.perf_counter() - began
    print(f"read {len(waveform) / 16000 / 3600:.2f} h in {read_seconds:.1f}s", flush=True)

    import accuracy_diarizen as dz

    OUT.mkdir(parents=True, exist_ok=True)
    loading = time.perf_counter()
    pipeline = dz.load_pipeline()
    load_seconds = time.perf_counter() - loading
    print(f"pipeline loaded in {load_seconds:.1f}s", flush=True)

    running = time.perf_counter()
    annotation = dz.infer(pipeline, waveform)
    infer_seconds = time.perf_counter() - running
    probs = dz.raster(annotation, len(waveform))

    speakers = annotation.labels()
    per_speaker = {}
    for turn, _, who in annotation.itertracks(yield_label=True):
        per_speaker[who] = per_speaker.get(who, 0.0) + (turn.end - turn.start)
    ordered = sorted(per_speaker.values(), reverse=True)

    payload = {
        "part": PART,
        "window": {"start_seconds": start_seconds, "seconds": len(waveform) / 16000},
        "elapsed": {"read": read_seconds, "load": load_seconds, "infer": infer_seconds},
        "rtf": infer_seconds / max(len(waveform) / 16000, 1e-9),
        "peak_vram_gib": vram_guard.peak_gib(),
        "peak_host_ram_gib": peak_host_gib(),
        "speakers": len(speakers),
        # How lopsided the result is: a clustering that shatters one person into many leaves a long
        # tail of tiny speakers, which is exactly the over-split this backend is known for.
        "speech_seconds_sorted": ordered,
        "tail_under_30s": sum(1 for value in ordered if value < 30.0),
        "sortformer_same_window": sortformer_window(start_seconds, len(waveform) / 16000),
        "audio": lab.digest(path),
        "script": lab.digest(__file__),
        "model": dz.provenance(),
    }
    stem = f"{PART}-h{args.start_hour:g}+{args.hours:g}"
    np.savez_compressed(OUT / f"{stem}.npz", probs=probs)
    (OUT / f"{stem}.rttm").write_text(annotation.to_rttm(), encoding="utf-8")
    (OUT / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("speech_seconds_sorted", "model")}, indent=2), flush=True)
    print(f"\nwrote {OUT / f'{stem}.json'}", flush=True)


if __name__ == "__main__":
    main()
