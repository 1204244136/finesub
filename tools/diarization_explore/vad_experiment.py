"""Experiment: Sortformer as a multi-speaker-aware VAD + overlap detector.

Runs diarize(include_tensor_outputs=True), builds frame-level speech-activity
(sum over speakers), compares against the project's VAD-produced segments
(stable.json), and reports overlap ratio (frames with >1 active speaker).

Usage:
    python vad_experiment.py <audio.wav> <stable.json> [--out out.json]
"""
import argparse
import json
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import torch
from nemo.collections.asr.models import SortformerEncLabelModel

FRAME = 0.08  # 80ms per Sortformer frame


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("stable_json", help="project stable.json (segments with start/end)")
    p.add_argument("--out", default=None)
    p.add_argument("--thr", type=float, default=0.5, help="activity threshold")
    return p.parse_args()


def load_stable_segments(path):
    d = json.load(open(path, encoding="utf-8"))
    segs = d if isinstance(d, list) else d.get("segments", d.get("items", []))
    out = []
    for s in segs:
        st = s.get("start", s.get("start_time"))
        en = s.get("end", s.get("end_time"))
        if st is None or en is None:
            continue
        out.append((float(st), float(en)))
    return out


def main():
    args = parse_args()
    t0 = time.time()
    model = SortformerEncLabelModel.restore_from(
        restore_path="models/diar_streaming_sortformer_4spk-v2.1.nemo",
        map_location="cuda", strict=False)
    model.eval().cuda()
    try:
        model.sortformer_modules.chunk_len = 340
        model.sortformer_modules.chunk_right_context = 40
        model.sortformer_modules.fifo_len = 40
        model.sortformer_modules.spkcache_update_period = 300
    except Exception:
        pass
    print(f"[load] {time.time()-t0:.1f}s")

    t1 = time.time()
    segs, tensors = model.diarize(audio=[args.audio], batch_size=1, include_tensor_outputs=True)
    print(f"[diarize+tensor] {time.time()-t1:.1f}s")
    probs = tensors[0][0].cpu().numpy()  # (T, S) — drop batch dim
    print(f"tensor shape: {probs.shape}  (T*{FRAME}s = {probs.shape[0]*FRAME:.1f}s audio)")

    act = probs.sum(axis=1)  # (T,)
    speech_frames = act >= args.thr
    overlap_frames = (probs >= args.thr).sum(axis=1) >= 2

    # -> VAD segments
    vad_segs = []
    in_speech = False
    for i, sp in enumerate(speech_frames):
        if sp and not in_speech:
            start = i * FRAME
            in_speech = True
        elif not sp and in_speech:
            vad_segs.append((round(start, 3), round(i * FRAME, 3)))
            in_speech = False
    if in_speech:
        vad_segs.append((round(start, 3), round(len(speech_frames) * FRAME, 3)))
    vad_speech = sum(e - s for s, e in vad_segs)
    total_sec = len(speech_frames) * FRAME
    print(f"\n=== Sortformer VAD ===")
    print(f"speech coverage: {vad_speech:.1f}s / {total_sec:.1f}s = {vad_speech/total_sec*100:.1f}%")
    print(f"overlap frames (>=2 spk): {overlap_frames.mean()*100:.1f}% of time, {overlap_frames.sum()*FRAME:.1f}s")
    per_spk = {}
    for s in range(probs.shape[1]):
        per_spk[s] = (probs[:, s] >= args.thr).sum() * FRAME
    print("per-speaker active time:", {f"spk{k}": round(v, 1) for k, v in sorted(per_spk.items())})

    # compare vs project VAD segments
    stable = load_stable_segments(args.stable_json)
    stable_speech = sum(e - s for s, e in stable)
    print(f"\n=== project stable.json ===")
    print(f"segments={len(stable)} speech={stable_speech:.1f}s")

    # overlap between the two VAD masks (in time)
    def segs_to_mask(segs_, n_frames):
        mask = np.zeros(n_frames, dtype=bool)
        for s, e in segs_:
            a = max(0, int(s / FRAME)); b = min(n_frames, int(np.ceil(e / FRAME)))
            mask[a:b] = True
        return mask

    n_f = len(speech_frames)
    m_sf = speech_frames
    m_st = segs_to_mask(stable, n_f)
    inter = (m_sf & m_st).sum() * FRAME
    union = (m_sf | m_st).sum() * FRAME
    print(f"\n=== VAD agreement ===")
    print(f"intersection={inter:.1f}s union={union:.1f}s Jaccard={inter/union*100:.1f}%")
    print(f"Sortformer-only (project VAD missed): {(m_sf & ~m_st).sum()*FRAME:.1f}s")
    print(f"project-only (Sortformer missed): {(m_st & ~m_sf).sum()*FRAME:.1f}s")

    out = args.out or args.audio.replace(".wav", ".vad.json")
    with open(out, "w") as f:
        json.dump({
            "vad_segments": vad_segs, "overlap_ratio": float(overlap_frames.mean()),
            "overlap_sec": float(overlap_frames.sum() * FRAME),
            "per_spk_sec": {str(k): round(v, 1) for k, v in per_spk.items()},
            "stable_segments": len(stable), "stable_speech_sec": round(stable_speech, 1),
            "jaccard": round(inter / union, 4),
            "sf_only_sec": round((m_sf & ~m_st).sum() * FRAME, 1),
            "stable_only_sec": round((m_st & ~m_sf).sum() * FRAME, 1),
        }, f, ensure_ascii=False, indent=1)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
