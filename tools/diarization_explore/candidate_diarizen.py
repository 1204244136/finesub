"""Opt-in DiariZen research preview for arbitrary audio and stable subtitles.

Requires the pinned local source/dependencies/weights described in the README.
Weights are CC-BY-NC 4.0. Writes derived copies into a new directory only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy_lab as lab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    p.add_argument("--stable", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-seconds", type=int, choices=[8, 16], default=16,
                   help="local window; 8 s is an experimental configuration with domain-dependent results")
    p.add_argument("--native-boundaries", action="store_true", help="preserve native intervals instead of the evaluated 80 ms raster/smoothing path")
    args = p.parse_args()
    if args.out.exists():
        p.error("--out must be a new directory")
    if not args.audio.is_file() or not args.stable.is_file():
        p.error("audio and stable subtitle files are required")
    payload = json.loads(args.stable.read_text(encoding="utf-8"))
    if not isinstance(payload.get("segments"), list):
        p.error("stable JSON must contain a segments list")
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    from finesub.speech.preprocessing.audio import resample_if_needed
    from finesub.speech.diarization.segments import segments_from_activity
    from finesub.speech.diarization.annotate import speaker_metadata
    from accuracy_diarizen import load_pipeline, infer, raster, provenance
    from accuracy_export import write_preview
    if not cuda_usable():
        raise RuntimeError("DiariZen preview requires a working CUDA device")
    torch.set_num_threads(4)
    wave, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    waveform, _ = resample_if_needed(torch.from_numpy(wave.mean(axis=1)).unsqueeze(0), sr, 16000)
    signal = waveform.squeeze(0).numpy()
    annotation = infer(load_pipeline(inference_overrides={"seg_duration": args.window_seconds}), signal)
    if args.native_boundaries:
        from finesub.speech.diarization.segments import TrackSegment
        speakers = sorted(annotation.labels(), key=int)
        tracks = [TrackSegment(speakers.index(who), max(0., turn.start), min(len(signal)/16000, turn.end))
                  for turn, _, who in annotation.itertracks(yield_label=True)
                  if min(len(signal)/16000, turn.end) > max(0., turn.start)]
        tracks.sort(key=lambda t: (t.start, t.track))
    else:
        tracks = segments_from_activity(raster(annotation, len(signal)).tolist())
    metadata = speaker_metadata(model="BUT-FIT/diarizen-wavlm-large-s80-md-v2", artifacts_revision="",
        track_count=len({t.track for t in tracks}), parameters=dict(
            backend="diarizen-research", source_audio_sha256=lab.digest(args.audio),
            window_seconds=args.window_seconds,
            tool_sha256=lab.digest(__file__), implementation=provenance(),
            weights_license="CC-BY-NC-4.0", boundary_mode="native" if args.native_boundaries else "raster-smoothed",
            frame_seconds=None if args.native_boundaries else lab.FRAME))
    stem = args.stable.stem.removesuffix("-stable")
    write_preview(payload, tracks, metadata, args.out, stem)
    print(args.out)


if __name__ == "__main__":
    main()
