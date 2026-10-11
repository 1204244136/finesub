"""Plan 5o.11 probe 3: the per-frame speaker probabilities behind the cached timeline.

`run_diarization` throws the sigmoids away and returns only segments, so this calls the runner
directly and saves the `(frames, 4)` array. ⚠ It **annotates, never replaces**: `-diar.json` is the
one artifact that is not reproducible (same audio on different hardware moves ~7% of subtitles to
another speaker), and every S4 arm is built on the cached one. The array is used only to put a
confidence on segments that already exist.

  DIAR_S4_CORPUS=ja python s4_diar_probs.py nepolabo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import s4_projection as proj


def probabilities(name):
    """`(frames, tracks)` sigmoids for one session, cached beside the timeline."""
    out = proj.ALI / name / f"{name}-diar-probs.npz"
    if not out.exists():
        import soundfile
        import torch

        from finesub.speech.diarization.runner import SortformerRunner
        from finesub.speech.preprocessing.audio import TARGET_SR, resample_if_needed

        signal, rate = soundfile.read(str(proj.ALI / name / f"{name}-vocal.ogg"), dtype="float32", always_2d=True)
        waveform, _ = resample_if_needed(torch.from_numpy(signal.mean(axis=1)).unsqueeze(0), rate, TARGET_SR)
        activity = SortformerRunner(proj.WORK / "export_v2", device="cuda").diarize(waveform)
        np.savez_compressed(out, probs=activity.float().cpu().numpy())
    return np.load(out)["probs"]


def segment_confidence(name):
    """Cached segment -> mean sigmoid of its own track over its frames."""
    probs = probabilities(name)
    tracks = json.loads((proj.ALI / name / f"{name}-diar.json").read_text(encoding="utf-8"))["tracks"]
    out = []
    for seg in tracks:
        a = int(seg["start"] / proj.FRAME)
        b = max(a + 1, int(np.ceil(seg["end"] / proj.FRAME)))
        out.append(float(probs[a:b, seg["track"]].mean()) if a < len(probs) else float("nan"))
    return tracks, np.array(out)


def stable_confidence(name):
    """The **pipeline's own** segments with their diarization labels and each label's confidence.

    Probe 3 used the raw Sortformer runs -- hundreds of sub-second fragments. These are the units
    production actually emits (VAD + ASR + stabilization, then `labels_for`), so filling them in is
    the same shape as producing a better A0. One line per segment, `tracks` is what production
    labelled it with (often two).
    """
    from finesub.speech.diarization.stage import labels_for, read_timeline

    probs = probabilities(name)
    payload = json.loads((proj.ALI / name / f"{name}-stable.json").read_text(encoding="utf-8"))
    segments = payload["segments"]
    labels = [tuple(s.extras.get("tracks", ())) for s in labels_for(segments, read_timeline(proj.ALI / name / f"{name}-diar.json"))]
    out = []
    for segment, tracks in zip(segments, labels):
        a = int(segment["start"] / proj.FRAME)
        b = max(a + 1, int(np.ceil(segment["end"] / proj.FRAME)))
        window = probs[a:b] if a < len(probs) else None
        out.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "tracks": list(tracks),
            "confidence": [float(window[:, k].mean()) if window is not None else float("nan") for k in tracks],
        })
    return out


def main():
    name = sys.argv[1]
    tracks, conf = segment_confidence(name)
    print(f"{name}: {len(conf)} 段，本轨概率均值 中位 {np.nanmedian(conf):.3f}，"
          f">0.8 {np.mean(conf > 0.8):.1%}，>0.5 {np.mean(conf > 0.5):.1%}，<0.3 {np.mean(conf < 0.3):.1%}")
    short = np.array([t["end"] - t["start"] for t in tracks]) < 0.5
    if short.any():
        print(f"   其中 < 0.5 s 的 {short.sum()} 段：中位 {np.nanmedian(conf[short]):.3f}")


if __name__ == "__main__":
    main()
