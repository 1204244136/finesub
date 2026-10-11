"""Explicit frozen candidates for arbitrary audio and existing stable subtitles.

Community requires the exploration environment with pyannote.audio 4.0.3 installed.
No reference labels, paid API, ASR decoding, or modification of the source run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]

#: ⚠ Frozen meanings. `base` is the geometry that shipped until 2026-09-20 -- the one the recorded
#: nepolabo reading (9/20) used -- and `high` is built on it; both equal `accuracy_lab.BASE` /
#: `HIGH` (`test_accuracy_lab.py` pins that; this script runs in the research env and does not
#: import the lab). Whatever ships TODAY is `sortformer-default`, read from the dataclass so it
#: cannot fall behind. An arm that followed the dataclass under the old name changed meaning the
#: day right80 shipped (Codex audit 2026-09-21).
SORTFORMER_ARMS = {
    "sortformer-base": dict(chunk_len=188, chunk_right_context=1, fifo_len=0, spkcache_len=188,
                            spkcache_update_period=188),
}
SORTFORMER_ARMS["sortformer-high"] = SORTFORMER_ARMS["sortformer-base"] | dict(
    chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300)


def sortformer_parameters(backend):
    """The geometry a Sortformer arm runs with -- the ONE place both inference and the record read.

    The record used to be a hard-coded copy of `high`, so `base` and `default` ran right=1 / 80 and
    wrote right=40 into their metadata (Codex re-review 2026-09-21).
    """
    from dataclasses import asdict
    from finesub.speech.diarization.chunking import SortformerGeometry
    if backend == "sortformer-default":
        return asdict(SortformerGeometry())
    return asdict(SortformerGeometry(**SORTFORMER_ARMS[backend]))


def infer(audio, model_dir, backend, hook=None):
    import torch
    import soundfile as sf
    from finesub.speech.preprocessing.audio import resample_if_needed
    from finesub.speech.runtime.device import cuda_usable
    import numpy as np
    from finesub.speech.diarization.segments import segments_from_activity
    if not cuda_usable():
        raise RuntimeError("candidate_run requires a working CUDA device")
    signal, rate = sf.read(audio, dtype="float32", always_2d=True)
    waveform, _ = resample_if_needed(torch.from_numpy(signal.mean(axis=1)).unsqueeze(0), rate, 16000)
    if backend in ("sortformer-high", "sortformer-base", "sortformer-default"):
        from finesub.speech.diarization.runner import SortformerRunner
        from finesub.speech.diarization.chunking import SortformerGeometry
        geometry = SortformerGeometry(**sortformer_parameters(backend))
        runner = SortformerRunner(model_dir, geometry=geometry)
        probabilities = runner.diarize(waveform.squeeze(0)).cpu().numpy()
        return segments_from_activity(probabilities.tolist())
    # This script owns its process. Disable optional integrations unused by this
    # checkpoint, avoiding installed torchcodec DLL / SpeechBrain API conflicts.
    for optional in ("torchcodec", "speechbrain", "nemo"):
        sys.modules[optional] = None
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(model_dir)
    pipeline.segmentation_batch_size = 8
    pipeline.embedding_batch_size = 8
    pipeline.to(torch.device("cuda"))
    output = pipeline({"waveform": waveform, "sample_rate": 16000}, hook=hook)
    speakers = output.speaker_diarization.labels()
    probabilities = np.zeros((int(np.ceil(waveform.shape[-1]/1280)), max(1, len(speakers))), dtype=np.float32)
    for turn, _, who in output.speaker_diarization.itertracks(yield_label=True):
        probabilities[max(0, int(turn.start/0.08)):int(np.ceil(turn.end/0.08)), speakers.index(who)] = 1
    # Use exactly the same 80 ms grid and smoothing as the evaluated lab arm.
    return segments_from_activity(probabilities.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--stable", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--backend", choices=["community", "sortformer-high", "sortformer-base", "sortformer-default"], required=True)
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    args = p.parse_args()
    if args.out.exists():
        p.error("--out must be a new directory; original run artifacts are never overwritten")
    if not args.audio.is_file() or not args.stable.is_file() or not args.model_dir.is_dir():
        p.error("audio, stable subtitles, and a local model directory are required")
    if args.backend == "community" and not (args.model_dir / "config.yaml").is_file():
        p.error("community requires a local model directory with config.yaml")
    payload = json.loads(args.stable.read_text(encoding="utf-8"))
    if not isinstance(payload.get("segments"), list):
        p.error("stable JSON must contain a segments list")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    import vram_guard
    vram_guard.install()
    import torch
    torch.set_num_threads(4)
    tracks = infer(args.audio, args.model_dir, args.backend)
    from finesub.speech.diarization.annotate import speaker_metadata
    from accuracy_export import write_preview
    from accuracy_lab import digest
    model = ("pyannote/speaker-diarization-community-1" if args.backend == "community" else
             "nvidia/diar_streaming_sortformer_4spk-v2.1")
    parameters = ({"segmentation_batch_size": 8, "embedding_batch_size": 8, "exclusive": False}
                  if args.backend == "community" else
                  sortformer_parameters(args.backend))
    metadata = speaker_metadata(model=model, artifacts_revision="",
        track_count=len({t.track for t in tracks}), parameters={"source_audio_sha256": digest(args.audio),
        "model_files": {str(f.relative_to(args.model_dir)): digest(f) for f in args.model_dir.rglob("*")
                        if f.is_file() and f.suffix in (".ts", ".bin", ".safetensors", ".npz", ".yaml")},
        "backend": args.backend, "tool_sha256": digest(__file__), **parameters})
    stem = args.stable.stem.removesuffix("-stable")
    write_preview(payload, tracks, metadata, args.out, stem)
    print(args.out)


if __name__ == "__main__":
    main()
