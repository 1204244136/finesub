"""Plan 5j-A: under the high-quality streaming geometry, does our export still sit
inside NeMo's own perturbation spread? (The 5a.4b criterion, re-run on a new geometry.)

Both sides get the same geometry; ours is the production `SortformerRunner`.
Writes a record carrying the git HEAD and start time (audit 2.1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import combinations
from pathlib import Path

import soundfile
import torch

from finesub.speech.diarization.chunking import SortformerGeometry
from finesub.speech.diarization.runner import SortformerRunner

HIGH_QUALITY = {"chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40, "spkcache_update_period": 300}


def per_subtitle_labels(activity, segments):
    labels = []
    for segment in segments:
        first = int(segment["start"] / 0.08)
        last = min(int(segment["end"] / 0.08) + 1, activity.shape[0])
        if last <= first:
            labels.append(None)
            continue
        totals = (activity[first:last] > 0.5).double().sum(dim=0)
        best = int(totals.argmax())
        labels.append(best if totals[best] > 0 else None)
    return labels


def agreement(first, second):
    decided = [(a, b) for a, b in zip(first, second) if not (a is None and b is None)]
    return 100.0 if not decided else sum(a == b for a, b in decided) / len(decided) * 100


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("nemo")
    parser.add_argument("audio")
    parser.add_argument("stable")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--geometry", choices=("default", "high"), default="high")
    parser.add_argument("--record", default="")
    args = parser.parse_args()
    started = time.time()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    device = torch.device("cuda")

    signal, rate = soundfile.read(args.audio, dtype="float32", always_2d=True)
    assert rate == 16000, rate
    waveform = torch.from_numpy(signal.mean(axis=1)).unsqueeze(0).to(device)
    payload = json.loads(Path(args.stable).read_text(encoding="utf-8"))
    segments = payload["segments"] if isinstance(payload, dict) else payload

    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(args.nemo, map_location="cuda")
    model.eval()
    overrides = HIGH_QUALITY if args.geometry == "high" else {}
    for key, value in overrides.items():
        setattr(model.sortformer_modules, key, value)
    print("nemo geometry:", {k: getattr(model.sortformer_modules, k) for k in (
        "chunk_len", "chunk_left_context", "chunk_right_context", "fifo_len", "spkcache_len", "spkcache_update_period")})

    def nemo_run(x):
        with torch.inference_mode():
            features, lengths = model.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]], device=device))
            return model.forward_streaming(features, lengths).squeeze(0).float().cpu()

    generator = torch.Generator(device="cpu").manual_seed(31337)
    tiny = (torch.rand(waveform.shape, generator=generator) - 0.5).to(device) * 1e-9
    labels = {}
    for name, audio in {
        "nemo/plain": waveform,
        "nemo/x(1+1e-7)": waveform * (1.0 + 1e-7),
        "nemo/x(1-1e-7)": waveform * (1.0 - 1e-7),
        "nemo/+1e-9 dither": waveform + tiny,
    }.items():
        labels[name] = per_subtitle_labels(nemo_run(audio), segments)

    geometry = SortformerGeometry(**overrides)
    runner = SortformerRunner(args.artifacts, device="cuda", geometry=geometry)
    t0 = time.time()
    ours_activity = runner.diarize(waveform).float().cpu()
    ours_seconds = time.time() - t0
    labels["ours/export"] = per_subtitle_labels(ours_activity, segments)

    nemo_names = [n for n in labels if n.startswith("nemo/")]
    spread = [agreement(labels[a], labels[b]) for a, b in combinations(nemo_names, 2)]
    ours = [agreement(labels["ours/export"], labels[n]) for n in nemo_names]
    inside = min(ours) >= min(spread)
    duration = waveform.shape[1] / rate
    result = {
        "head": head, "started": started, "audio": Path(args.audio).name, "geometry": args.geometry,
        "subtitles": len(segments), "nemo_spread": [round(min(spread), 3), round(max(spread), 3)],
        "ours_vs_nemo": [round(min(ours), 3), round(max(ours), 3)], "inside": inside,
        "ours_rtf": round(ours_seconds / duration, 4),
    }
    print(json.dumps(result))
    if args.record:
        with open(args.record, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
