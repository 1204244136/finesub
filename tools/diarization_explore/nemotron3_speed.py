"""Wall time of one diarization pass: Nemotron-3 (NeMo eager) vs the two production backends.

One arm, one file, one process. Load is timed apart and excluded; then two passes over the same
waveform (the first is cold -- CUDA/cuDNN warm-up inside the process -- and is reported, not
dropped). Production arms go through `Backend.ensure` / `Backend.activity`, i.e. exactly what
`run_diarization` calls, so DiariZen's time includes the embeddings it computes for evidence.

Declared configuration: device cuda (RTX 5070 Ti), fp32 for all three (Nemotron eager, no
compile; production backends as shipped), batch 1, 16 kHz mono float32 already in memory
(decode excluded), threads = torch default, aggregate = none (each pass printed).

    python tools/diarization_explore/nemotron3_speed.py nemotron3 onetonton-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import nemotron3_cost as nc
import nemotron3_probe as n3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", choices=("nemotron3", "diarizen", "sortformer"))
    p.add_argument("ident")
    args = p.parse_args()

    import vram_guard
    vram_guard.install()
    import soundfile as sf
    import torch

    wave, sr = sf.read(nc.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000, sr
    wave = wave.mean(axis=1)
    duration = len(wave) / sr

    t0 = time.perf_counter()
    if args.arm == "nemotron3":
        model = n3.load_model()
        run = lambda: n3.infer(model, wave, sr)[1]
    else:
        from finesub.speech.diarization.stage import resolve_backend
        backend = resolve_backend(args.arm)
        handle = backend.ensure(None, "cuda")
        tensor = torch.from_numpy(wave).unsqueeze(0)
        run = lambda: backend.activity(tensor, handle=handle, device="cuda", max_speakers=None).activity
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    passes = []
    for _ in range(2):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = run()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        if out is None or len(out) == 0:
            sys.exit(f"{args.arm}: empty output -- a failure, not a timing")
        passes.append(dict(seconds=elapsed, rtfx=duration / elapsed,
                           peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                           frames=int(len(out))))
    record = dict(arm=args.arm, ident=args.ident, audio_seconds=duration, load_s=load_s, passes=passes,
                  watchdog_peak_gib=vram_guard.peak_gib())
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
