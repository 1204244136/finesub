"""DiariZen local powerset activity + community-1 embedding/VBx/reconstruction.

Uses saved local neural output, never reference identities. This isolates whether
the local overlap model or global assignment explains the difference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score
from accuracy_v2 import OUT


def infer(waveform, local_path):
    import torch
    from scipy.ndimage import median_filter
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    sys.modules["torchcodec"] = None
    sys.modules["speechbrain"] = None
    sys.modules["nemo"] = None
    from pyannote.audio import Pipeline
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    saved = np.load(local_path)
    window = SlidingWindow(start=saved["window"][0], duration=saved["window"][1], step=saved["window"][2])
    rf = SlidingWindow(start=saved["receptive_field"][0], duration=saved["receptive_field"][1], step=saved["receptive_field"][2])
    local = SlidingWindowFeature(median_filter(saved["data"], size=(1, 11, 1), mode="reflect"), window)
    pipeline = Pipeline.from_pretrained(lab.WT / "tmp/diarization-models/community-1")
    pipeline.segmentation_batch_size = pipeline.embedding_batch_size = 8
    pipeline.to(torch.device("cuda"))
    # No segmentation forward is called. Replace inference with a metadata carrier
    # as well, so community's 10 s / 17 ms geometry cannot contaminate reconstruction.
    pipeline._segmentation.model = SimpleNamespace(specifications=SimpleNamespace(powerset=True), receptive_field=rf)
    pipeline.get_segmentations = lambda file, hook=None: local
    last = [0.]

    def hook(step, artifact, file=None, total=None, completed=None):
        now = time.perf_counter()
        if now-last[0] > 30 or completed == total:
            print(f"recluster {step} {completed}/{total}", flush=True)
            last[0] = now
    return pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": 16000}, hook=hook).speaker_diarization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ident")
    args = p.parse_args()
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from accuracy_diarizen import raster
    torch.set_num_threads(4)
    wave, sr = sf.read(lab.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000
    directory = OUT / args.ident
    start = time.perf_counter()
    annotation = infer(wave.mean(axis=1), directory/"diarizen-local.npz")
    probs = raster(annotation, len(wave))
    stable, _, grid, people, *_ = lab.rm.episode(args.ident, lab.rm.sources()[args.ident])
    payload = dict(arm="dz_community", metrics=lab.score(probs, stable, grid, people),
        bcubed=identity_score(probs, stable, grid), elapsed=time.perf_counter()-start,
        identity=dict(audio=lab.digest(lab.source(args.ident)), script=lab.digest(__file__),
                      local=lab.digest(directory/"diarizen-local.npz")), watchdog_peak_gib=vram_guard.peak_gib())
    np.savez_compressed(directory/"dz_community.npz", probs=probs)
    (directory/"dz_community.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (directory/"dz_community.rttm").write_text(annotation.to_rttm(), encoding="utf-8")
    print(args.ident, payload["metrics"]["fixed"], payload["bcubed"], flush=True)


if __name__ == "__main__":
    main()
