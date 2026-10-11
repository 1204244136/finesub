"""Nemotron-3 as the speaker reference (owner 2026-09-24), both forms `diarization-findings.md` 2.6 measured:

  n3      Nemotron-3-Diarization alone (8 slots, 10 ms), smoothed by production `segments_from_activity`
  n3vbx   Nemotron-3 as DiariZen's local segmentation + WeSpeaker/PLDA/VBx (`over8.method_vbx`), with the
          production embedding configuration (`prod=1`: batch 2, TF32 off) -- the exploration path of the
          default backend since 2026-09-26 (`nemotron3`; the production port matches it, findings 2.6)

Inference is `nemotron3_probe.infer` unchanged; only the paths point at the main checkout's `tmp/`, where
the exploration environment lives (weights, NVIDIA-NeMo/Speech@cf724ac, lhotse). Output per item, same
shape as a production `-diar.json` track list: `data/ts-diar/episodes/<id>/{n3,n3vbx}-diar.json` (main checkout).

    PYTHONPATH="<main>/tmp/nemo-main:<main>/tmp/n3-deps:<worktree>/src" \
        C:/Users/Carl/miniconda3/envs/asr/python.exe -m tools.ts_diar_explore.n3_baseline hina-72 ...
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from .common import EPISODES, WT, vocal_track

sys.path.insert(0, str(WT / "tools" / "diarization_explore"))
import nemotron3_probe as n3  # noqa: E402
import over8  # noqa: E402



def tracks(activity, frame: float) -> list[dict]:
    from finesub.speech.diarization.segments import segments_from_activity
    return [{"track": t.track, "start": t.start, "end": t.end}
            for t in segments_from_activity(np.asarray(activity).tolist(), frame_seconds=frame)]


def write(item: str, arm: str, rows: list[dict], frame: float, elapsed: float) -> None:
    out = EPISODES / item / f"{arm}-diar.json"
    out.write_text(json.dumps({"metadata": {"model": arm, "frame_seconds": frame, "elapsed": round(elapsed, 1),
                                            "track_count": len({r["track"] for r in rows})},
                               "tracks": rows}, indent=1), encoding="utf-8")
    print(f"{item} {arm}: {len({r['track'] for r in rows})} tracks, {elapsed:.1f}s -> {out}", flush=True)


def main() -> None:
    import soundfile as sf
    import torch

    model = None
    for item in sys.argv[1:]:
        if (EPISODES / item / "n3vbx-diar.json").exists():
            continue
        wave, sr = sf.read(str(vocal_track(item)), dtype="float32", always_2d=True)
        assert sr == 16000, sr
        wave = wave.mean(axis=1)
        if model is None:
            model = n3.load_model()
        t0 = time.perf_counter()
        _, probs = n3.infer(model, wave, sr)
        frame = round(len(wave) / sr / probs.shape[0], 3)
        write(item, "n3", tracks(probs, frame), frame, time.perf_counter() - t0)
        torch.cuda.empty_cache()
        t1 = time.perf_counter()
        activity, frame20 = over8.method_vbx(item, probs, wave, prod=1.0)
        write(item, "n3vbx", tracks(activity, frame20), frame20, time.perf_counter() - t1)


if __name__ == "__main__":
    main()
