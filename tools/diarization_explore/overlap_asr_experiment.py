"""Experiment: speaker-prompted ASR on overlapping speech, aided by Sortformer.

Select overlapping windows from the Sortformer segments. For each window:
  - baseline: transcribe the mixed audio directly (whisper-timestamped)
  - prompted: build a per-speaker activity mask from Sortformer segments,
    apply it to the audio ("attend to this speaker only"), transcribe each
    speaker's version separately.

Usage:
    python overlap_asr_experiment.py <sortformer.json> <audio.wav> [--windows 3] [--model large-v3-turbo]
"""
import argparse
import json
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sortformer_json")
    p.add_argument("audio")
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--model", default="large-v3-turbo")
    p.add_argument("--language", default="ja")
    return p.parse_args()


def find_overlap_windows(segs, n=3, min_ov=1.0, max_win=14.0):
    segs_sorted = sorted(segs, key=lambda s: s["start"])
    cands = []
    for i in range(len(segs_sorted) - 1):
        a = segs_sorted[i]
        for j in range(i + 1, min(i + 6, len(segs_sorted))):
            b = segs_sorted[j]
            ov_st = max(a["start"], b["start"])
            ov_en = min(a["end"], b["end"])
            if ov_en - ov_st >= min_ov and a["label"] != b["label"]:
                cands.append((ov_st, ov_en, a["label"], b["label"]))
    cands.sort(key=lambda c: -(c[1] - c[0]))
    picked = []
    for c in cands:
        ov_len = c[1] - c[0]
        win_len = ov_len + 1.0
        if win_len > max_win:
            continue
        st = max(0, c[0] - 0.5)
        en = c[1] + 0.5
        if any(not (en <= p[0] - 2 or st >= p[1] + 2) for p in picked):
            continue
        picked.append((round(st, 2), round(en, 2), c[2], c[3], round(ov_len, 2)))
        if len(picked) >= n:
            break
    return picked


def build_speaker_mask(segs, win_st, win_en, speaker, sr, n_samples):
    """Binary mask (window-local): 1 where Sortformer has `speaker` segments inside window."""
    mask = np.zeros(n_samples, dtype=np.float32)
    for s in segs:
        if s["label"] != speaker:
            continue
        a = max(win_st, s["start"]); b = min(win_en, s["end"])
        if b <= a:
            continue
        ia = int((a - win_st) * sr); ib = int((b - win_st) * sr)
        ia = max(0, ia); ib = min(n_samples, ib)
        mask[ia:ib] = 1.0
    return mask


def main():
    args = parse_args()
    segs = json.load(open(args.sortformer_json, encoding="utf-8"))["segments"]
    sig, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    sig = sig[:, 0] if sig.shape[1] > 1 else sig[:, 0]

    windows = find_overlap_windows(segs, args.windows)
    if not windows:
        print("no overlap windows found"); return
    print(f"selected {len(windows)} overlap windows:")
    for w in windows:
        print(f"  [{w[0]}-{w[1]}]s overlap={w[4]}s  {w[2]} <-> {w[3]}")

    import whisper_timestamped as whisper
    t0 = time.time()
    model = whisper.load_model(args.model, device="cuda")
    print(f"whisper load: {time.time()-t0:.1f}s")

    results = []
    for wi, (st, en, spk_a, spk_b, ov) in enumerate(windows):
        ia, ib = int(st * sr), int(en * sr)
        wav = sig[ia:ib]
        win = {"window": [st, en], "overlap_s": ov, "speakers": [spk_a, spk_b]}

        # baseline
        r = whisper.transcribe(model, wav, language=args.language, vad=False)
        baseline_text = " ".join(seg_["text"].strip() for seg_ in r["segments"]).strip()
        win["baseline"] = baseline_text
        print(f"\n--- window {wi} [{st}-{en}]s ov={ov}s ---")
        print(f"  [baseline mixed ] {baseline_text}")

        # speaker-prompted per speaker
        win["prompted"] = {}
        for spk in [spk_a, spk_b]:
            mask = build_speaker_mask(segs, st, en, spk, sr, len(wav))
            masked = wav * mask
            # pad silence where mask is 0 to keep alignment readable
            r2 = whisper.transcribe(model, masked, language=args.language, vad=False)
            text = " ".join(seg_["text"].strip() for seg_ in r2["segments"]).strip()
            win["prompted"][spk] = {"mask_ratio": float(mask.mean()), "text": text}
            print(f"  [prompted {spk}   ] {text}")
        results.append(win)

    out = args.sortformer_json.replace(".json", ".overlap-asr.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
