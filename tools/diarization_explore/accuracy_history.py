"""Revisit the historical nepolabo failure on identical audio and owner spot labels."""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import accuracy_lab as lab
from accuracy_v2 import OUT


def truth_rows():
    path = lab.rm.MAIN / "out/nepolabo/nepolabo-spotcheck.md"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        found = re.match(r"\| (\d+):(\d+\.\d+) \| ([\d.]+)s .*\| ([1-4])(?:\(mix\))?$", line)
        if found:
            minute, second, duration, who = found.groups()
            rows.append(dict(start=int(minute)*60+float(second), end=int(minute)*60+float(second)+float(duration), who=int(who)))
    return path, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # ⚠ `base` added 2026-09-20 (C2): every historical reading here ran `high` and never the
    # geometry production actually ships, so "high wins on nepolabo" had no control to win against.
    p.add_argument("model", choices=["base", "high", "community", "diarizen"])
    p.add_argument("--single-speaker-control", action="store_true")
    args = p.parse_args()
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    if not cuda_usable():
        raise RuntimeError("Historical GPU evaluation requires a working CUDA device")
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity, track_seconds
    torch.set_num_threads(4)
    ident = "kaguya60" if args.single_speaker_control else "nepolabo"
    audio = lab.rm.MAIN/f".worktrees/speaker-clustering/tmp/sortformer/{ident}-vocal-16k.wav"
    waveform, sr = sf.read(audio, dtype="float32", always_2d=True)
    assert sr == 16000
    start = time.perf_counter()
    directory = OUT/"historical"/ident
    directory.mkdir(parents=True, exist_ok=True)
    if args.model == "diarizen":
        from accuracy_diarizen import load_pipeline, infer, raster
        annotation = infer(load_pipeline(), waveform.mean(axis=1), cache_dir=directory)
        tracks = segments_from_activity(raster(annotation, len(waveform)).tolist())
    else:
        from candidate_run import infer
        backend = {"high": "sortformer-high", "base": "sortformer-base"}.get(args.model, "community")
        model_dir = (lab.WT/"tmp/diarization-models/community-1" if args.model == "community"
                     else lab.ARTIFACTS)
        tracks = infer(audio, model_dir, backend)
    truth, rows = (None, []) if args.single_speaker_control else truth_rows()
    if not args.single_speaker_control and len(rows) != 20:
        raise ValueError("The frozen nepolabo reference must contain 20 labelled windows")
    assignments = assign_all(rows, tracks)
    ids = sorted({label.track for label in assignments if label.track is not None})
    matrix = np.zeros((len(ids), 4), dtype=int)
    for row, label in zip(rows, assignments):
        if label.track is not None:
            matrix[ids.index(label.track), row["who"]-1] += 1
    a, b = linear_sum_assignment(-matrix)
    mapping = {ids[x]: int(y+1) for x, y in zip(a, b)}
    result = dict(model=args.model, labelled_points=len(rows),
        point_accuracy=float(matrix[a, b].sum()/max(len(rows), 1)),
        tracks=track_seconds(tracks), mapping=mapping,
        predictions=[dict(**r, track=l.track, predicted=mapping.get(l.track)) for r, l in zip(rows, assignments)],
        identity=dict(audio=lab.digest(audio), labels=lab.digest(truth) if truth else None, script=lab.digest(__file__)),
        elapsed=time.perf_counter()-start, watchdog_peak_gib=vram_guard.peak_gib(),
        limitation="20 manually labelled subtitle windows, optimal mapping on those points; not full-file DER")
    if args.single_speaker_control:
        seconds = list(result["tracks"].values())
        result.pop("point_accuracy")
        result["track_count"] = len(seconds)
        result["secondary_track_share"] = 1-max(seconds)/sum(seconds) if seconds else None
        result["limitation"] = "Known single-speaker control; track fragmentation diagnostic, no full speech activity ground truth or DER."
    if args.model == "diarizen":
        from accuracy_diarizen import provenance
        result["identity"]["model"] = provenance()
        result["identity"]["inference_script"] = lab.digest(lab.HERE/"accuracy_diarizen.py")
    else:
        result["identity"]["model_files"] = {
            p.relative_to(model_dir).as_posix(): lab.digest(p) for p in model_dir.rglob("*")
            if p.is_file() and p.suffix in (".ts", ".bin", ".yaml", ".npz") and ".cache" not in p.parts}
        result["identity"]["inference_script"] = lab.digest(lab.HERE/"candidate_run.py")
    (directory/f"{args.model}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result, flush=True)


if __name__ == "__main__":
    main()
