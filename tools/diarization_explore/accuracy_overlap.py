"""Continuous-time, overlap-inclusive zero-collar DER and overlap attribution.

Full audio UEM; reference RTTM/TextGrid speaking turns, not subtitle styles.
The Hungarian mapping is scoring-only and never feeds inference.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import accuracy_lab as lab
from accuracy_v2 import OUT

PUBLIC = lab.rm.MAIN / "data/diarization/public"
CALLS = ["2188", "2235", "2225", "2218", "1967", "2223", "2096", "1642", "2210", "1928"]
ALI = ["R8001_M8004", "R8009_M8018", "R8007_M8010"]


def timeline(rows, duration):
    """Union duplicate per-speaker intervals before counting active speakers."""
    changes = {0.: [], float(duration): []}
    for a, b, speaker in rows:
        a, b = max(0., a), min(duration, b)
        if b > a:
            changes.setdefault(a, []).append((speaker, 1))
            changes.setdefault(b, []).append((speaker, -1))
    counts, spans = {}, []
    points = sorted(changes)
    for index, a in enumerate(points[:-1]):
        for speaker, change in changes[a]:
            counts[speaker] = counts.get(speaker, 0) + change
        spans.append((a, points[index+1], {s for s, n in counts.items() if n > 0}))
    return spans


def score_intervals(reference, hypothesis, duration):
    refs, hyps = timeline(reference, duration), timeline(hypothesis, duration)
    ref_ids = sorted({s for _, _, s in reference})
    hyp_ids = sorted({s for _, _, s in hypothesis})
    ri, hi = {s: i for i, s in enumerate(ref_ids)}, {s: i for i, s in enumerate(hyp_ids)}
    matrix = np.zeros((len(ref_ids), len(hyp_ids)))
    events, i, j = [], 0, 0
    while i < len(refs) and j < len(hyps):
        a, b, r = refs[i]
        c, d, h = hyps[j]
        width = min(b, d)-max(a, c)
        if width > 0:
            events.append((width, r, h))
            for x in r:
                for y in h:
                    matrix[ri[x], hi[y]] += width
        if b <= d:
            i += 1
        if d <= b:
            j += 1
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {hyp_ids[y]: ref_ids[x] for x, y in zip(rows, cols)}
    total = miss = fa = confusion = overlap = predicted = tp = all_correct = 0.
    three = three_correct = four = four_correct = 0.
    for width, r, h in events:
        correct = len(r.intersection(mapping[s] for s in h if s in mapping))
        total += width*len(r)
        miss += width*max(0, len(r)-len(h))
        fa += width*max(0, len(h)-len(r))
        confusion += width*(min(len(r), len(h))-correct)
        overlap += width*(len(r) >= 2)
        predicted += width*(len(h) >= 2)
        tp += width*(len(r) >= 2 and len(h) >= 2)
        all_correct += width*(len(r) >= 2 and correct == len(r))
        three += width*(len(r) >= 3)
        three_correct += width*(len(r) >= 3 and correct == len(r))
        four += width*(len(r) >= 4)
        four_correct += width*(len(r) >= 4 and correct == len(r))
    return dict(der=(miss+fa+confusion)/max(total, 1e-12),
                miss=miss/max(total, 1e-12), false_alarm=fa/max(total, 1e-12),
                confusion=confusion/max(total, 1e-12), reference_speaker_seconds=total,
                overlap_seconds=overlap, overlap_precision=tp/max(predicted, 1e-12),
                overlap_recall=tp/max(overlap, 1e-12),
                overlap_all_speakers_recall=all_correct/max(overlap, 1e-12),
                three_plus_seconds=three,
                three_plus_all_recall=three_correct/three if three else None,
                four_plus_seconds=four,
                four_plus_all_recall=four_correct/four if four else None)


def dataset(ident):
    corpus = "alimeeting" if ident.startswith("R") else "callhome-jpn"
    root = PUBLIC / corpus
    audio = root / "audio" / f"{ident}.wav"
    rttm = root / "rttm" / f"{ident}.rttm"
    if rttm.exists():
        rows = []
        for line in rttm.read_text().splitlines():
            f = line.split()
            if f and f[0] == "SPEAKER":
                rows.append((float(f[3]), float(f[3])+float(f[4]), f[7]))
        truth = rttm
    else:
        truth = root / "textgrid" / f"{ident}.TextGrid"
        rows = []
        for block in re.split(r"item \[\d+\]:", truth.read_text(encoding="utf-8"))[1:]:
            speaker = re.search(r'name = "(.*?)"', block).group(1)
            for a, b, text in re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', block, re.S):
                if text.strip():
                    rows.append((float(a), float(b), speaker))
    return audio, truth, rows


def annotation_rows(annotation):
    return [(turn.start, turn.end, str(who)) for turn, _, who in annotation.itertracks(yield_label=True)]


def worker(ident, arm):
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    if not cuda_usable():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    audio, truth, reference = dataset(ident)
    waveform, sr = sf.read(audio, dtype="float32", always_2d=True)
    assert sr == 16000
    waveform = waveform.mean(axis=1)
    directory = OUT / "overlap" / ident
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if arm == "diarizen":
        from accuracy_diarizen import load_pipeline, infer, provenance
        annotation = infer(load_pipeline(), waveform, cache_dir=directory)
        hypothesis = annotation_rows(annotation)
    elif arm == "community":
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        sys.modules["torchcodec"] = None
        sys.modules["speechbrain"] = None
        sys.modules["nemo"] = None
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(lab.WT / "tmp/diarization-models/community-1")
        pipeline.segmentation_batch_size = pipeline.embedding_batch_size = 8
        pipeline.to(torch.device("cuda"))
        output = pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr})
        hypothesis = annotation_rows(output.speaker_diarization)
    else:
        from finesub.speech.diarization.segments import segments_from_activity
        probs = lab.run_arm(waveform, arm)
        np.savez_compressed(directory/f"{arm}.npz", probs=probs)
        hypothesis = [(t.start, t.end, str(t.track)) for t in segments_from_activity(probs.tolist())]
    result = dict(arm=arm, metrics=score_intervals(reference, hypothesis, len(waveform)/sr),
        reference=reference, hypothesis=hypothesis, duration=len(waveform)/sr,
        identity=dict(audio=lab.digest(audio), reference=lab.digest(truth), script=lab.digest(__file__)),
        elapsed=time.perf_counter()-start, watchdog_peak_gib=vram_guard.peak_gib())
    if arm in ("community", "diarizen"):
        from finesub.speech.diarization.segments import segments_from_activity
        speakers = sorted({s for _, _, s in hypothesis})
        probs = np.zeros((int(np.ceil(len(waveform)/1280)), max(1, len(speakers))), dtype=np.float32)
        for a, b, speaker in hypothesis:
            probs[max(0, int(a/lab.FRAME)):int(np.ceil(b/lab.FRAME)), speakers.index(speaker)] = 1
        np.savez_compressed(directory/f"{arm}.npz", probs=probs)
        deployed = [(t.start, t.end, str(t.track)) for t in segments_from_activity(probs.tolist())]
        result["production_metrics"] = score_intervals(reference, deployed, len(waveform)/sr)
    if arm == "diarizen":
        result["identity"]["model"] = provenance()
    elif arm == "community":
        root = lab.WT / "tmp/diarization-models/community-1"
        result["identity"]["model_files"] = {
            p.relative_to(root).as_posix(): lab.digest(p) for p in root.rglob("*")
            if p.is_file() and p.suffix in (".bin", ".yaml", ".npz") and ".cache" not in p.parts}
    else:
        result["identity"]["model_files"] = {p.name: lab.digest(p) for p in lab.ARTIFACTS.glob("*.ts")}
        result["identity"]["inference_script"] = lab.digest(lab.__file__)
    (directory/f"{arm}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(ident, arm, result["metrics"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ident")
    p.add_argument("arm", choices=["base", "high", "community", "diarizen"])
    args = p.parse_args()
    worker(args.ident, args.arm)


if __name__ == "__main__":
    main()
