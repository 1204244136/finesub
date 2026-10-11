"""Plan 5h.2: does re-decoding overlap regions recover the drowned speaker?

Arms: baseline (stable words), sham (same clip, no mask), masked (per active
track, others attenuated to 0.1 with 50 ms ramps). Regions come from our own
Sortformer timeline -- production signals only, never the reference.
Run with --decode (GPU) once, then --score as often as needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

import s4_headroom as ref  # sibling module; PYTHONPATH=<repo>/src;<this directory>

# Paths: DATA = <repo>/data/diarization (hand-kept, main checkout), WORK =
# <repo>/tmp/diarization_explore (scratch; the three .ts go in WORK/export_v2).
import os as _os
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(_os.environ.get("DIAR_EXPLORE_DATA", ROOT / "data" / "diarization"))
WORK = Path(_os.environ.get("DIAR_EXPLORE_WORK", ROOT / "tmp" / "diarization_explore"))
ALI = WORK / ref.WORK_NAME
FRAME, MERGE_GAP, MIN_REGION, CONTEXT, PER_SESSION = 0.08, 0.3, 0.5, 3.0, 150
ATTENUATION, RAMP, SR = 0.1, 0.05, 16000


def timeline(session):
    path = ALI / session / f"{session}-diar.json"
    if not path.exists():
        from finesub.speech.diarization.stage import run_diarization, write_timeline

        write_timeline(run_diarization(ALI / session / f"{session}-vocal.ogg", artifacts=WORK / "export_v2"), path)
    return json.loads(path.read_text(encoding="utf-8"))["tracks"]


def activity(tracks, frames):
    active = np.zeros((frames, 1 + max(t["track"] for t in tracks)), dtype=bool)
    for t in tracks:
        active[int(t["start"] / FRAME) : int(np.ceil(t["end"] / FRAME)), t["track"]] = True
    return active


def regions(active):
    both = active.sum(axis=1) >= 2
    spans, start = [], None
    for index, flag in enumerate(np.append(both, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append([start * FRAME, index * FRAME])
            start = None
    merged = []
    for span in spans:
        if merged and span[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    kept = [span for span in merged if span[1] - span[0] >= MIN_REGION]
    if len(kept) > PER_SESSION:
        kept = [kept[round(i * (len(kept) - 1) / (PER_SESSION - 1))] for i in range(PER_SESSION)]
    return kept


def gain_for(active, track, begin, end):
    samples = end - begin
    gain = np.ones(samples, dtype=np.float32)
    first, last = begin / SR / FRAME, end / SR / FRAME
    for frame in range(int(first), min(int(np.ceil(last)), len(active))):
        row = active[frame]
        if not row[track] and row.any():
            lo = max(0, int(frame * FRAME * SR) - begin)
            hi = min(samples, int((frame + 1) * FRAME * SR) - begin)
            gain[lo:hi] = ATTENUATION
    width = int(RAMP * SR)
    return np.convolve(gain, np.ones(width, dtype=np.float32) / width, mode="same")


def decode(model, clip, offset):
    segments, _ = model.transcribe(
        clip, language=ref.LANGUAGE, word_timestamps=True,
        condition_on_previous_text=False, vad_filter=False,
    )
    return [
        {"start": offset + w.start, "end": offset + w.end, "word": w.word}
        for segment in segments
        for w in (segment.words or ())
    ]


def run_decode(sessions):
    import soundfile

    from finesub.speech.recognition.fw_refine_backend import RefinedWhisperModel
    from finesub.speech.recognition.transcribe import REFINE_SEC

    model = RefinedWhisperModel("large-v3-turbo", device="cuda", compute_type="float16", refine_sec=REFINE_SEC)
    for session in sessions:
        out = ALI / session / "projection.json"
        if out.exists():
            continue
        tracks = timeline(session)
        audio, rate = soundfile.read(str(ALI / session / f"{session}-vocal.ogg"), dtype="float32", always_2d=True)
        assert rate == SR
        audio = audio.mean(axis=1)
        active = activity(tracks, int(len(audio) / SR / FRAME) + 2)
        records = []
        for start, end in regions(active):
            begin, finish = max(0, int((start - CONTEXT) * SR)), min(len(audio), int((end + CONTEXT) * SR))
            clip, offset = audio[begin:finish], begin / SR
            first, last = int(start / FRAME), int(np.ceil(end / FRAME))
            live = [k for k in range(active.shape[1]) if active[first:last, k].any()]
            records.append(
                {
                    "start": start,
                    "end": end,
                    "sham": decode(model, clip, offset),
                    "masked": {str(k): decode(model, clip * gain_for(active, k, begin, finish), offset) for k in live},
                }
            )
        out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(session, "regions", len(records), "tracks", len(set(t["track"] for t in tracks)))


def region_text(words, start, end):
    return "".join(ref.normalize(w["word"]) for w in words if start - ref.PAD <= (w["start"] + w["end"]) / 2 <= end + ref.PAD)


def spurious(text, reference_bigrams):
    grams = ref.bigrams(text)
    total = sum(grams.values())
    return None if not total else sum(n for g, n in grams.items() if g not in reference_bigrams) / total


def score(sessions):
    pooled = {"worse": {"base": [], "sham": [], "masked": []}, "better": {"base": [], "sham": [], "masked": []},
              "spur": {"base": [], "sham": [], "masked": []}}
    for session in sessions:
        records = json.loads((ALI / session / "projection.json").read_text(encoding="utf-8"))
        rows = ref.score(session)
        all_ref = [r for r in ref.reference(session)]
        words = ref.hypothesis(session)
        base_words = [{"start": m, "end": m, "word": t} for m, t in words]
        for record in records:
            record["sham_text"] = region_text(record["sham"], record["start"], record["end"])
            record["masked_text"] = "".join(
                region_text(w, record["start"], record["end"]) for w in record["masked"].values()
            )
            nearby = set()
            for r in all_ref:
                if r["start"] < record["end"] + 1 and r["end"] > record["start"] - 1:
                    nearby |= set(ref.bigrams(r["text"]))
            for arm, text in (("base", region_text(base_words, record["start"], record["end"])),
                              ("sham", record["sham_text"]), ("masked", record["masked_text"])):
                value = spurious(text, nearby)
                if value is not None:
                    pooled["spur"][arm].append(value)
        local = {"worse": {"base": [], "sham": [], "masked": []}, "better": {"base": [], "sham": [], "masked": []}}
        for a, b in ref.pairs(rows):
            middle = (max(a["start"], b["start"]) + min(a["end"], b["end"])) / 2
            if not any(r["start"] <= middle <= r["end"] for r in records):
                continue
            for member in (a, b):
                touching = [r for r in records if r["start"] < member["end"] + ref.PAD and r["end"] > member["start"] - ref.PAD]
                base = ref.window_text(words, member["start"], member["end"])
                member["arms"] = {
                    "base": member["recall"],
                    "sham": ref.recall(member["text"], base + "".join(r["sham_text"] for r in touching)),
                    "masked": ref.recall(member["text"], base + "".join(r["masked_text"] for r in touching)),
                }
            worse, better = sorted((a, b), key=lambda m: m["recall"])
            for arm in ("base", "sham", "masked"):
                local["worse"][arm].append(worse["arms"][arm])
                local["better"][arm].append(better["arms"][arm])
        for side in ("worse", "better"):
            for arm in local[side]:
                pooled[side][arm].extend(local[side][arm])
        n = len(local["worse"]["base"])
        if n:
            w = {arm: mean(v) for arm, v in local["worse"].items()}
            print(f"{session}: pairs {n}  worse base {w['base']:.3f} sham {w['sham']:.3f} masked {w['masked']:.3f}")
    n = len(pooled["worse"]["base"])
    w = {arm: mean(v) for arm, v in pooled["worse"].items()}
    b = {arm: mean(v) for arm, v in pooled["better"].items()}
    s = {arm: mean(v) for arm, v in pooled["spur"].items()}
    print(f"\nPOOLED pairs {n}")
    print(f"  worse  base {w['base']:.3f}  sham {w['sham']:.3f} ({100 * (w['sham'] - w['base']):+.1f} pp)  "
          f"masked {w['masked']:.3f} ({100 * (w['masked'] - w['base']):+.1f} pp)")
    print(f"  better base {b['base']:.3f}  sham {b['sham']:.3f}  masked {b['masked']:.3f}")
    print(f"  spurious base {s['base']:.3f}  sham {s['sham']:.3f}  masked {s['masked']:.3f}  (regions {len(pooled['spur']['base'])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--score", action="store_true")
    arguments = parser.parse_args()
    if arguments.decode:
        run_decode(ref.SESSIONS)
    if arguments.score:
        score(ref.SESSIONS)


if __name__ == "__main__":
    main()
