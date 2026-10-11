"""Verify with ASR whether the VAD difference regions are real speech.

- sf_only: Sortformer VAD segments NOT covered by project stable.json (the ~413s
  Sortformer "recovered").
- stable_only: project stable segments NOT covered by Sortformer VAD (the ~52s
  Sortformer "missed").

Each region (merged, >= min_len) is transcribed with whisper large-v3-turbo;
a region counts as real speech if it yields non-empty text with average word
confidence >= CONF_THR.

Usage:
    python vad_diff_asr.py
"""
import json
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import whisper_timestamped as whisper

SF_JSON = "nepolabo.vad.json"          # vad_segments
STABLE = "C:/Users/Carl/Documents/Carl/projects/asr-playground/out/nepolabo/nepolabo-stable.json"
AUDIO = "nepolabo-vocal-16k.wav"
LANG = "ja"
CONF_THR = 0.5
MIN_LEN = 0.8
MERGE_GAP = 0.6


def diff(segs, refs):
    """segments of `segs` not covered by `refs` (interval subtraction)."""
    refs = sorted(refs)
    out = []
    for s in sorted(segs):
        cur = list(s)
        for r in refs:
            if r[1] <= cur[0]:
                continue
            if r[0] >= cur[1]:
                break
            if r[0] > cur[0]:
                out.append((cur[0], r[0]))
            cur[0] = max(cur[0], r[1])
            if cur[1] <= cur[0]:
                break
        if cur[1] > cur[0]:
            out.append(tuple(cur))
    return out


def merge(segs, gap=MERGE_GAP, min_len=MIN_LEN):
    segs = sorted(segs)
    out = []
    for s in segs:
        if out and s[0] - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], s[1]))
        else:
            out.append(s)
    return [s for s in out if s[1] - s[0] >= min_len]


def main():
    vad = json.load(open(SF_JSON, encoding="utf-8"))["vad_segments"]
    d = json.load(open(STABLE, encoding="utf-8"))
    stable = sorted((float(s["start"]), float(s["end"])) for s in d["segments"])

    sf_only = merge(diff(vad, stable))
    st_only = merge(diff(stable, vad))
    print(f"sf_only (Sortformer 补回): {sum(e-s for s,e in sf_only):.1f}s, {len(sf_only)} 段")
    print(f"stable_only (Sortformer 漏掉): {sum(e-s for s,e in st_only):.1f}s, {len(st_only)} 段")

    t0 = time.time()
    model = whisper.load_model("large-v3-turbo", device="cuda")
    print(f"whisper load: {time.time()-t0:.1f}s")

    import soundfile as sf
    sig, sr = sf.read(AUDIO, dtype="float32")
    if sig.ndim > 1:
        sig = sig[:, 0]

    def classify_cropped(regions, name):
        total = sum(e - s for s, e in regions)
        speech_s = 0.0; non_s = 0.0; empty_n = 0; lowconf_n = 0
        samples = []
        for si, (st, en) in enumerate(regions):
            ia, ib = int(st * sr), int(en * sr)
            clip = sig[ia:ib]
            r = whisper.transcribe(model, clip, language=LANG, vad=False)
            confs = []
            text_parts = []
            for seg_ in r["segments"]:
                for w in seg_.get("words", []):
                    c = w.get("confidence", 0.0)
                    confs.append(c)
                    text_parts.append(w.get("word", ""))
            text = "".join(text_parts).strip()
            avg_c = float(np.mean(confs)) if confs else 0.0
            dur = en - st
            is_speech = bool(text) and avg_c >= CONF_THR
            if is_speech:
                speech_s += dur
            else:
                non_s += dur
                if not text:
                    empty_n += 1
                else:
                    lowconf_n += 1
            if len(samples) < 8:
                samples.append((st, en, dur, text[:60], avg_c, "语音" if is_speech else "非语音"))
        print(f"\n=== {name}: {len(regions)} 段, 总 {total:.1f}s ===")
        print(f"判定为真语音: {speech_s:.1f}s ({speech_s/total*100:.1f}%)")
        print(f"判定为非语音/噪声: {non_s:.1f}s ({non_s/total*100:.1f}%)   [空转写 {empty_n}, 低置信 {lowconf_n}]")
        for st, en, dur, tx, c, tag in samples:
            print(f"  [{st:7.1f}-{en:7.1f}] {dur:4.1f}s conf={c:.2f} {tag} | {tx}")
        return speech_s, non_s

    classify_cropped(sf_only, "A. Sortformer 补回的 (sf_only)")
    classify_cropped(st_only, "B. Sortformer 漏掉的 (stable_only)")


if __name__ == "__main__":
    main()
