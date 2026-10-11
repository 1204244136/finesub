"""Controlled Sortformer experiments; see docs/archive/diarization-accuracy-lab-20260916.md.

Each batch worker is a fresh process with the 8 GiB guard. Inference never sees gold.
Artifacts are scratch, separate from production. Resume checks audio and script identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(WT / "src")]
import numpy as np
import refined_multi as rm

OUT = WT / "tmp" / "diarization-accuracy-lab"
ARTIFACTS = rm.MAIN / "tmp" / "diarization_explore" / "export_v2"
DEV = ["hina-01", "hina-02", "hina-03", "hina-04", "hina-07", "hina-11", "hina-12", "nagi-bd"]
FRAME = 0.08

#: ⚠ **Every arm here is built on the geometry production shipped UNTIL 2026-09-20**, spelled out
#: rather than read from `SortformerGeometry()`. The owner then moved `chunk_right_context` 1 -> 80;
#: an arm that inherited the dataclass defaults would have followed it silently, and a re-run of
#: `base` (or `cache2`, `ema`, `refresh`, which only override other fields) would compare against a
#: configuration none of the recorded readings used while keeping the same name (Codex audit
#: 2026-09-21). Today's default is a different arm, not a new meaning for this one.
BASE = dict(chunk_len=188, chunk_right_context=1, fifo_len=0, spkcache_len=188,
            spkcache_update_period=188)
HIGH = BASE | dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source(ident):
    return WT / "out" / "refined-2609" / ident / f"{ident}-vocal.ogg"


def score(probs, stable, grid, people, frame_seconds=FRAME):
    """Production smoothing/assignment, plus fixed denominator and cross-fitted owners.

    `frame_seconds` is the grid `probs` is ON -- not the truth grid, which stays at `FRAME`.
    Everything after segmenting is in seconds, so a finer activity map can be scored against the
    same 80 ms truth; that is what A3 (native 20 ms boundaries) needs.
    """
    from finesub.speech.diarization.segments import segments_from_activity
    from finesub.speech.diarization.assign import assign_all

    tracks = segments_from_activity(probs.tolist(), frame_seconds=frame_seconds)
    labels = [label.to_dict() for label in assign_all(stable, tracks)]
    active = np.zeros((len(grid), probs.shape[1]), dtype=bool)
    for t in tracks:
        active[int(t.start / FRAME):int(np.ceil(t.end / FRAME)), t.track] = True
    folds = (np.arange(len(grid)) // 750) % 2
    owners = []
    for fold in (None, 0, 1):
        mask = np.ones(len(grid), dtype=bool) if fold is None else folds != fold
        table = active[mask].T.astype(np.float64) @ grid[mask].astype(np.float64)
        owners.append(np.where(table.sum(axis=1) > 0, table.argmax(axis=1), -1))
    total = covered = correct = lenient = count = cross = 0.0
    row_correct, weights, predictions = [], [], []
    for s, label in zip(stable, labels):
        lo, hi = int(s["start"] / FRAME), max(int(s["start"] / FRAME) + 1, int(np.ceil(s["end"] / FRAME)))
        ref = grid[lo:hi].sum(axis=0)
        if not ref.any():
            continue
        who, weight = int(ref.argmax()), s["end"] - s["start"]
        ids = label.get("tracks", [])
        total += weight
        right = bool(ids and owners[0][ids[0]] == who)
        row_correct.append(right)
        weights.append(weight)
        predictions.append(int(owners[0][ids[0]]) if ids else -1)
        if ids:
            covered += weight
            correct += weight * right
            lenient += weight * (who in owners[0][ids])
            count += weight * len(ids)
            fold = int((s["start"] + s["end"]) / 2 // 60) % 2
            cross += weight * (owners[fold + 1][ids[0]] == who)
    return {
        "fixed": correct / max(total, 1e-12), "crossfit": cross / max(total, 1e-12),
        "strict": correct / max(covered, 1e-12), "lenient": lenient / max(covered, 1e-12),
        "coverage": covered / max(total, 1e-12), "labels": count / max(covered, 1e-12),
        "tracks": len({t.track for t in tracks}), "people": len(people), "seconds": total,
        # W1: how many tracks somebody has to decide about. `tracks` alone counts ones that never
        # reach the spk column, so it answers neither "how much work" nor "how fragmented".
        "naming": __import__("cost_metrics").naming_cost(probs, stable, grid, frame_seconds=frame_seconds),
        "row_correct": row_correct, "weights": weights, "predictions": predictions,
    }


def align(base, other):
    """One-to-one track alignment without reference labels; preserve all four slots."""
    from scipy.optimize import linear_sum_assignment
    n = min(len(base), len(other))
    a, b = base[:n].astype(np.float64), other[:n].astype(np.float64)
    overlap = a.T @ b
    union = a.sum(axis=0)[:, None] + b.sum(axis=0)[None, :] - overlap
    rows, cols = linear_sum_assignment(-overlap / np.maximum(union, 1e-9))
    result = np.zeros_like(base)
    result[:n, rows] = other[:n, cols]
    return result


class CacheUpdate:
    """Use the exported NVIDIA selection with a controlled prediction refresh."""
    def __init__(self, original, refresh):
        self.original, self.refresh = original, refresh

    def __call__(self, cache, history, has, silence, n_silence, embs, predictions, current):
        if has:
            history = (1 - self.refresh) * history + self.refresh * current
        return self.original(cache, history, has, silence, n_silence, embs, predictions, current)


def arm_geometry(arm):
    """The `SortformerGeometry` keyword arguments an arm runs with -- always complete, never inherited."""
    if arm == "high":
        return dict(HIGH)
    if arm == "cache2":
        return BASE | dict(spkcache_len=376)
    return dict(BASE)


def run_arm(waveform, arm, base=None):
    import torch
    from finesub.speech.diarization.runner import SortformerRunner
    from finesub.speech.diarization.chunking import SortformerGeometry
    runner = SortformerRunner(ARTIFACTS, geometry=SortformerGeometry(**arm_geometry(arm)))
    if arm == "cache2":
        runner._cache_update.m.spkcache_len = 376
    if arm in ("refresh", "ema"):
        runner._cache_update = CacheUpdate(runner._cache_update, 1.0 if arm == "refresh" else 0.25)
    if arm == "no_latest":
        runner._cache_update.m.scores_boost_latest = 0.0
    if arm == "split":
        import over4
        import over4_v2 as split
        import over4_anchor
        # No foreign historical cache: this worker saves complete arm output itself.
        original = over4_anchor.cached
        over4_anchor.cached = lambda runner, key, name, compute: compute()
        try:
            return split.split(runner, "lab", waveform, base, split.CONFIGS["split-trig5"])
        finally:
            over4_anchor.cached = original
    signal = waveform
    offset = 0
    if arm.startswith("shift"):
        offset = int(arm.removeprefix("shift"))
        signal = np.pad(waveform, (offset * 1280, 0))
    elif arm == "micro":
        signal = waveform * np.float32(1.0000001)
    result = runner.diarize(torch.from_numpy(signal)).cpu().numpy()
    return result[offset:offset + len(base)] if offset and base is not None else result


def worker(args):
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    if not cuda_usable():
        raise RuntimeError("CUDA unavailable; this GPU experiment does not silently fall back")
    torch.set_num_threads(4)
    path = source(args.ident)
    signal, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 16000
    waveform = signal.mean(axis=1)
    data = rm.episode(args.ident, rm.sources()[args.ident])
    stable, _, grid, people, *_ = data
    directory = OUT / args.ident
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"audio": digest(path), "script": digest(__file__), "frames": len(waveform),
                "base": BASE, "high": HIGH,
                "artifacts": {p.name: digest(p) for p in ARTIFACTS.glob("*.ts")}}
    base = None
    for arm in dict.fromkeys(["base", *args.arms]):
        destination = directory / f"{arm}.npz"
        metadata = directory / f"{arm}.json"
        if destination.exists() and metadata.exists() and json.loads(metadata.read_text())["identity"] == identity:
            probs = np.load(destination)["probs"]
            print(f"{args.ident} {arm} cached", flush=True)
        else:
            begin = time.perf_counter()
            print(f"{args.ident} {arm} start ({len(waveform)/16000/60:.1f} min)", flush=True)
            probs = run_arm(waveform, arm, base)
            elapsed = time.perf_counter() - begin
            result = score(probs, stable, grid, people)
            np.savez_compressed(destination, probs=probs)
            metadata.write_text(json.dumps({"identity": identity, "arm": arm, "metrics": result,
                "elapsed": elapsed, "allocated_peak_gib": torch.cuda.max_memory_allocated()/1024**3,
                "reserved_peak_gib": torch.cuda.max_memory_reserved()/1024**3,
                "watchdog_peak_gib": vram_guard.peak_gib()}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{args.ident} {arm} fixed={result['fixed']:.4f} crossfit={result['crossfit']:.4f} "
                  f"coverage={result['coverage']:.4f} tracks={result['tracks']} seconds={elapsed:.1f}", flush=True)
            torch.cuda.empty_cache()
        if arm == "base":
            base = probs


def batch(args):
    ids = args.ids or DEV
    if ids == ["all"]:
        # Known EX#2 mismatch: reject mechanically using audio/axis durations as well.
        import soundfile as sf
        ids = [i for i, ass in rm.sources().items() if source(i).exists() and
               sf.info(source(i)).duration >= max(b for _, b, _ in rm.refined_lines(ass)) * 0.9]
    OUT.mkdir(parents=True, exist_ok=True)
    for ident in ids:
        command = [sys.executable, "-u", __file__, "worker", ident, "--arms", *args.arms]
        with (OUT / f"{ident}.log").open("a", encoding="utf-8") as log:
            print(f"START {ident}", flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"END {ident} exit={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{ident} failed; inspect {OUT / (ident + '.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("worker")
    p.add_argument("ident")
    p.add_argument("--arms", nargs="+", default=["split"])
    p = sub.add_parser("batch")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--arms", nargs="+", default=["split"])
    args = parser.parse_args()
    (worker if args.command == "worker" else batch)(args)


if __name__ == "__main__":
    main()
