"""Plan 5r B: re-decode each speaker's own stretch of an overlap window (with a far anchor) as correction candidates.

For a pipeline output dir (`<stem>-vocal.ogg`, `<stem>-diar.json`, `<stem>-stable.json`):

  --plan     overlap windows = runs where >= 2 tracks sound, padded W_PAD and merged; per window and per
             track sounding >= MIN_TRACK in it: that track's frames (T1 cut) behind its own solo speech
             >= 30 s away (T2 anchor). Same cut, anchor and join rules as `s4_timecut.plan` (5h.17, 5o.7).
  --decode   run the jobs through production's QwenReferee (qwen-asr env, GPU) and write
             <stem>-overlap-candidates.json: [{start, end, track, text, anchored}]; the anchor's transcript
             is stripped from the front. A track with no far anchor is decoded without one.
  --insert   B1: write the candidates into <stem>-stable.json as extra segments (`candidate: true`,
             labelled with their track, overlapping the production segments in time; no words). The
             original is kept as <stem>-stable.pre-candidates.json.

  python redecode_candidates.py OUT_DIR --plan|--decode|--insert
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

SR, FRAME = 16000, 0.08
W_PAD, JOIN, MIN_TRACK, ANCHOR_FAR, ANCHOR_TOTAL, ANCHOR_MIN, GAP = 1.0, 0.2, 0.3, 30.0, 4.0, 1.0, 0.6


def runs(mask):
    out, start = [], None
    for i, flag in enumerate(np.append(mask, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    return out


def cut(audio, spans):
    gap = np.zeros(int(JOIN * SR), dtype=np.float32)
    pieces = []
    for s, e in spans:
        pieces += [audio[int(s * SR):int(e * SR)], gap]
    return np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)


def load(out_dir):
    import soundfile

    stem = out_dir.name
    audio, rate = soundfile.read(str(out_dir / f"{stem}-vocal.ogg"), dtype="float32", always_2d=True)
    if rate != SR:
        raise SystemExit(f"expected {SR} Hz, got {rate}")
    audio = audio.mean(axis=1)
    tracks = json.loads((out_dir / f"{stem}-diar.json").read_text(encoding="utf-8"))["tracks"]
    frames = int(len(audio) / SR / FRAME) + 2
    active = np.zeros((frames, 1 + max(t["track"] for t in tracks)), dtype=bool)
    for t in tracks:
        active[int(t["start"] / FRAME):int(np.ceil(t["end"] / FRAME)), t["track"]] = True
    return stem, audio, active


def far_anchor(audio, active, k, w0, w1):
    alone = active[:, k] & (active.sum(axis=1) == 1)
    candidates = [(a * FRAME, b * FRAME) for a, b in runs(alone) if (b - a) * FRAME >= ANCHOR_MIN]
    candidates = [r for r in candidates if r[1] < w0 - ANCHOR_FAR or r[0] > w1 + ANCHOR_FAR]
    candidates.sort(key=lambda r: min(abs(r[0] - w1), abs(r[1] - w0)))
    pieces, total = [], 0.0
    for s, e in candidates:
        take = min(e - s, ANCHOR_TOTAL - total)
        if take <= 0:
            break
        pieces.append(audio[int(s * SR):int((s + take) * SR)])
        total += take
    return np.concatenate(pieces) if pieces else None


def plan(out_dir):
    stem, audio, active = load(out_dir)
    busy = active.sum(axis=1) >= 2
    windows = []
    for a, b in runs(busy):
        w0, w1 = max(0.0, a * FRAME - W_PAD), b * FRAME + W_PAD
        if windows and w0 <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], w1)
        else:
            windows.append([w0, w1])
    jobs = []
    for w0, w1 in windows:
        f0, f1 = int(w0 / FRAME), int(np.ceil(w1 / FRAME))
        sub = active[f0:f1]
        for k in range(sub.shape[1]):
            if sub[:, k].sum() * FRAME < MIN_TRACK:
                continue
            spans = [(f0 * FRAME + a * FRAME, f0 * FRAME + b * FRAME) for a, b in runs(sub[:, k])]
            body = cut(audio, spans)
            anchor = far_anchor(audio, active, k, w0, w1)
            job = {"id": f"{w0:.2f}-{w1:.2f}/{k}", "start": spans[0][0], "end": spans[-1][1], "track": k,
                   "window": [round(w0, 2), round(w1, 2)], "anchored": anchor is not None}
            jobs.append((job, body, anchor))
    print(f"{stem}: {len(windows)} overlap windows, {sum(w1 - w0 for w0, w1 in windows):.0f} s; "
          f"{len(jobs)} track jobs, {sum(j['anchored'] for j, _, _ in jobs)} anchored")
    return stem, jobs


def decode(out_dir, batch=16):
    from finesub.speech.verification.qwen_referee import QwenReferee
    import difflib

    stem, jobs = plan(out_dir)
    referee = QwenReferee(device="cuda")
    try:
        audio_in, anchors = [], []
        for job, body, anchor in jobs:
            if anchor is not None:
                audio_in.append(np.concatenate([anchor, np.zeros(int(GAP * SR), np.float32), body]))
                anchors.append(anchor)
            else:
                audio_in.append(body)
                anchors.append(None)
        texts = []
        for i in range(0, len(audio_in), batch):
            texts += [t for t, _ in referee.transcribe_batch(audio_in[i:i + batch])]
        anchor_texts = []
        present = [a for a in anchors if a is not None]
        for i in range(0, len(present), batch):
            anchor_texts += [t for t, _ in referee.transcribe_batch(present[i:i + batch])]
    finally:
        referee.close()
    it = iter(anchor_texts)
    out = []
    for (job, _, anchor), text in zip(jobs, texts):
        if anchor is not None:
            a = next(it)
            best_k, best = 0, 0.0
            for k in range(0, min(len(text), int(len(a) * 1.5) + 2) + 1):
                score = difflib.SequenceMatcher(None, text[:k], a).ratio()
                if score > best:
                    best_k, best = k, score
            text = text[best_k:] if best >= 0.5 else text
        text = text.strip()
        if text:
            out.append({**job, "start": round(job["start"], 3), "end": round(job["end"], 3), "text": text})
    target = out_dir / f"{stem}-overlap-candidates.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{target}: {len(out)} candidates")


def insert(out_dir):
    stem = out_dir.name
    stable = out_dir / f"{stem}-stable.json"
    backup = out_dir / f"{stem}-stable.pre-candidates.json"
    if not backup.exists():
        shutil.copyfile(stable, backup)
    payload = json.loads(backup.read_text(encoding="utf-8"))
    candidates = json.loads((out_dir / f"{stem}-overlap-candidates.json").read_text(encoding="utf-8"))
    extra = [{"start": c["start"], "end": c["end"], "text": c["text"], "words": [], "lang": "ja", "candidate": True,
              "speaker": {"labels": [], "block": 0, "track": c["track"], "tracks": [c["track"]],
                          "coverage": {str(c["track"]): 1.0}}} for c in candidates]
    payload["segments"] = sorted(payload["segments"] + extra, key=lambda s: (s["start"], s["end"]))
    payload.setdefault("metadata", {})["overlap_candidates"] = {"inserted": len(extra), "source": backup.name}
    stable.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{stable}: {len(extra)} candidate segments inserted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--decode", action="store_true")
    group.add_argument("--insert", action="store_true")
    arguments = parser.parse_args()
    out_dir = Path(arguments.out_dir)
    if arguments.plan:
        plan(out_dir)
    elif arguments.decode:
        decode(out_dir)
    else:
        insert(out_dir)


if __name__ == "__main__":
    main()
