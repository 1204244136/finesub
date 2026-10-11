"""Plan 5i: more than four speakers -- onepass vs sliding windows + cross-window linking.

VoxConverse dev (RTTM truth). Both paths run the production `SortformerRunner`.
Metric: attribution accuracy (AA) over frames where exactly one reference
speaker talks and at least one output identity is active -- the loudest output
identity, one-to-one Hungarian-mapped to reference speakers. Relative only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import parse_rttm  # noqa: E402

ROOT = HERE.parents[1]
DATA = Path(os.environ.get("DIAR_EXPLORE_DATA", ROOT / "data" / "diarization"))
WORK = Path(os.environ.get("DIAR_EXPLORE_WORK", ROOT / "tmp" / "diarization_explore"))
VOX = DATA / "public" / "voxconverse"

#: Plan 5j: which streaming geometry every script here runs. `default` is what
#: all numbers before 5j were measured on; `high` is NeMo's high-quality preset.
GEOMETRY = os.environ.get("DIAR_GEOMETRY", "default")
GEOMETRIES = {
    "default": {},
    "high": {"chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40, "spkcache_update_period": 300},
}


def make_runner():
    from finesub.speech.diarization.chunking import SortformerGeometry
    from finesub.speech.diarization.runner import SortformerRunner

    return SortformerRunner(WORK / "export_v2", device="cuda", geometry=SortformerGeometry(**GEOMETRIES[GEOMETRY]))
FRAME, THRESHOLD = 0.08, 0.5
WINDOW, HOP, LINK_IOU = 60.0, 30.0, 0.3
MANY, FEW = 12, 8


def select():
    rows = []
    for rttm in sorted((VOX / "voxconverse-master" / "dev").glob("*.rttm")):
        reference = parse_rttm(rttm)
        if reference:
            rows.append((rttm.stem, len({s.speaker for s in reference}), max(s.end for s in reference) / 60))

    def even(items, count):
        return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]

    many = even([r for r in rows if r[1] >= 5 and r[2] <= 12], MANY)
    few = even([r for r in rows if 2 <= r[1] <= 4 and r[2] <= 12], FEW)
    return many, few


def audio(name):
    path = WORK / "vox" / f"{name}.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DATA / "public" / "voxconverse_dev_wav.zip") as archive:
            path.write_bytes(archive.read(f"audio/{name}.wav"))
    return path


def truth(name, frames):
    reference = parse_rttm(VOX / "voxconverse-master" / "dev" / f"{name}.rttm")
    speakers = sorted({s.speaker for s in reference})
    grid = np.zeros((frames, len(speakers)), dtype=bool)
    for s in reference:
        grid[int(s.start / FRAME) : int(np.ceil(s.end / FRAME)), speakers.index(s.speaker)] = True
    return grid


def run(runner, waveform):
    import torch

    probabilities = runner.diarize(torch.from_numpy(waveform).unsqueeze(0)).float().cpu().numpy()
    return probabilities.reshape(-1, probabilities.shape[-1])


def windowed(runner, waveform, rate):
    from scipy.optimize import linear_sum_assignment

    total = len(waveform) / rate
    windows, start = [], 0.0
    while True:
        end = min(total, start + WINDOW)
        probs = run(runner, waveform[int(start * rate) : int(end * rate)])
        windows.append({"start": start, "end": end, "probs": probs, "gid": {}})
        if end >= total:
            break
        start += HOP
    next_gid = 0
    for index, window in enumerate(windows):
        live = [k for k in range(window["probs"].shape[1]) if (window["probs"][:, k] > THRESHOLD).any()]
        if index:
            previous = windows[index - 1]
            lo, hi = window["start"], previous["end"]
            a = previous["probs"][int((lo - previous["start"]) / FRAME) : int((hi - previous["start"]) / FRAME)] > THRESHOLD
            b = window["probs"][: a.shape[0]] > THRESHOLD
            a = a[: b.shape[0]]
            before = list(previous["gid"])
            if before and live:
                iou = np.zeros((len(before), len(live)))
                for i, p in enumerate(before):
                    for j, q in enumerate(live):
                        union = (a[:, p] | b[:, q]).sum()
                        iou[i, j] = (a[:, p] & b[:, q]).sum() / union if union else 0.0
                for i, j in zip(*linear_sum_assignment(-iou)):
                    if iou[i, j] >= LINK_IOU:
                        window["gid"][live[j]] = previous["gid"][before[i]]
        for k in live:
            if k not in window["gid"]:
                window["gid"][k] = next_gid
                next_gid += 1
    frames = int(total / FRAME) + 1
    out = np.zeros((frames, max(next_gid, 1)), dtype=np.float32)
    centers = np.array([(w["start"] + w["end"]) / 2 for w in windows])
    for t in range(frames):
        owner = windows[int(np.argmin(np.abs(centers - t * FRAME)))]
        local = t - int(round(owner["start"] / FRAME))
        if 0 <= local < owner["probs"].shape[0]:
            for k, g in owner["gid"].items():
                out[t, g] = max(out[t, g], owner["probs"][local, k])
    return out


def attribution(probs, grid):
    from scipy.optimize import linear_sum_assignment

    frames = min(len(probs), len(grid))
    probs, grid = probs[:frames], grid[:frames]
    use = (grid.sum(axis=1) == 1) & ((probs > THRESHOLD).any(axis=1))
    out = probs[use].argmax(axis=1)
    ref = grid[use].argmax(axis=1)
    table = np.zeros((probs.shape[1], grid.shape[1]))
    np.add.at(table, (out, ref), 1)
    rows, cols = linear_sum_assignment(-table)
    matched = table[rows, cols].sum()
    claimed = set(cols.tolist())
    orphan = sum(table[:, s].sum() for s in range(grid.shape[1]) if s not in claimed)
    identities = int(((probs > THRESHOLD).sum(axis=0) * FRAME >= 2.0).sum())
    return {"frames": int(use.sum()), "matched": float(matched), "orphan": float(orphan), "identities": identities}


def main():
    import soundfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args()
    many, few = select()
    if arguments.list:
        for label, group in (("many", many), ("few", few)):
            print(label, [(n, s, round(d, 1)) for n, s, d in group])
        return
    runner = make_runner()
    results = {}
    for label, group in (("many", many), ("few", few)):
        for name, speakers, _ in group:
            waveform, rate = soundfile.read(str(audio(name)), dtype="float32", always_2d=True)
            waveform = waveform.mean(axis=1)
            assert rate == 16000, rate
            grid = truth(name, int(len(waveform) / rate / FRAME) + 1)
            one = attribution(run(runner, waveform), grid)
            win = attribution(windowed(runner, waveform, rate), grid)
            results[name] = {"group": label, "speakers": speakers, "onepass": one, "windowed": win}
            print(
                f"{label:<4} {name} ref {speakers:2d} | onepass AA {one['matched'] / max(one['frames'], 1):.3f} "
                f"ids {one['identities']} orphan {one['orphan'] / max(one['frames'], 1):.1%} | windowed AA "
                f"{win['matched'] / max(win['frames'], 1):.3f} ids {win['identities']}",
                flush=True,
            )
    (WORK / "over4.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for label in ("many", "few"):
        rows = [r for r in results.values() if r["group"] == label]
        aa = {arm: sum(r[arm]["matched"] for r in rows) / sum(r[arm]["frames"] for r in rows) for arm in ("onepass", "windowed")}
        orphan = sum(r["onepass"]["orphan"] for r in rows) / sum(r["onepass"]["frames"] for r in rows)
        print(f"POOLED {label}: onepass AA {aa['onepass']:.3f} (orphan speech {orphan:.1%}) | windowed AA {aa['windowed']:.3f} | "
              f"delta {100 * (aa['windowed'] - aa['onepass']):+.1f} pp")


if __name__ == "__main__":
    main()
