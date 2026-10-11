"""W4: is "always featurize in segments" a noise-level change, or does it move the answer?

Sortformer USED to featurise a whole file in one call under 90 minutes and in segments above it,
because the single call costs 2.50 GiB per hour of audio. This experiment asked whether closing
that fork -- always segmenting -- is a noise-level change. It is (see the plan's W4 entry), so the
fork was removed on 2026-09-19 and this script now reconstructs the whole-file arm itself.

Criteria frozen in `docs/archive/diarization-closeout-plan.md` sections 2 and 2c BEFORE this ran:

(a) DID IT REORGANISE -- measured by the per-subtitle label change rate, NOT by Delta-fixed.
    `fixed` names every track after its own best match, so it is blind to a permutation, which is
    the typical shape of a re-drawn attractor (section 2c).
      a1  share of files that "jumped" (> 0.5 pp of rows change): segmented minus perturbation,
          paired bootstrap 95% upper bound at most +15 pp
      a2  among files where either arm jumped, segmented P90 at most 1.5x the perturbation P90
(b) DID QUALITY MOVE -- Delta-fixed paired mean, 95% interval not wholly below -1.0 pp.

The perturbation arm is the null: the same whole-file path on a waveform nudged by Gaussian noise
of sigma 1e-7. It answers "how much does this chain move when nothing meaningful changed", and the
probe showed that answer is a per-file coin flip (probe 8.6), so the band comes from the spread
across 74 files -- one draw each, exactly like the segmented arm.

    python tools/diarization_explore/w4_segmented.py run
    python tools/diarization_explore/w4_segmented.py score --write
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402
from accuracy_report import bootstrap_mean_ci  # noqa: E402

OUT = lab.WT / "tmp" / "diarization-w4"
NOISE_SEED = 20260919
JUMP_PP = 0.5
RATE_LIMIT_PP = 15.0
P90_FACTOR = 1.5


def qualified_ids():
    """The 74 files the accuracy lab qualified (audio at least 90% of the reference span)."""
    data = json.loads((HERE / "results" / "accuracy-20260916.json").read_text(encoding="utf-8"))
    return list(data["inventory"]["valid"])


def arms_of(waveform):
    rng = np.random.default_rng(NOISE_SEED)
    return {
        "whole": waveform,
        "seg": waveform,
        "noise": (waveform + rng.normal(0.0, 1e-7, len(waveform))).astype(np.float32),
    }


def worker(ident: str) -> None:
    import vram_guard

    vram_guard.install()
    import soundfile as sf
    import torch
    from finesub.speech.diarization import runner as dr
    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        raise RuntimeError("CUDA unavailable; this GPU experiment does not silently fall back")
    torch.set_num_threads(4)
    directory = OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    wave, sr = sf.read(lab.source(ident), dtype="float32", always_2d=True)
    assert sr == 16000
    waveform = wave.mean(axis=1)
    started = time.perf_counter()
    for name, signal in arms_of(waveform).items():
        path = directory / f"{name}.npz"
        if path.exists():
            continue
        # The only difference between `whole` and `seg` is how the SAME audio is handed to the
        # front end, so the tiling is replaced rather than the input. ⚠ The 90-minute fork this
        # originally toggled was REMOVED once these numbers came in (plan closeout W4), so the
        # whole-file arm now has to be reconstructed here: one segment covering everything is
        # exactly the single call the old path made.
        original = dr.feature_segments
        if name == "whole":
            def one_segment(samples, **kwargs):
                from finesub.speech.diarization.chunking import FeatureSegment

                return [FeatureSegment(0, samples, 0, 0, dr.feature_frame_counts(samples)[1])]

            dr.feature_segments = one_segment
        try:
            probs = lab.run_arm(signal, "base")
        finally:
            dr.feature_segments = original
        np.savez_compressed(path, probs=probs)
    (directory / "run.json").write_text(json.dumps(dict(
        ident=ident, seed=NOISE_SEED, seconds=len(waveform) / 16000,
        elapsed=time.perf_counter() - started, audio=lab.digest(lab.source(ident)),
        watchdog_peak_gib=vram_guard.peak_gib()), indent=2), encoding="utf-8")
    print(f"{ident} done in {time.perf_counter() - started:.0f}s", flush=True)


def label_change(base, other, stable, grid):
    """Share of reference-covered rows whose leading track moves, after a one-to-one alignment."""
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    width = max(base.shape[1], other.shape[1])

    def pad(p):
        return p if p.shape[1] >= width else np.hstack(
            [p, np.zeros((len(p), width - p.shape[1]), p.dtype)])

    a, b = pad(base), pad(other)[: len(base)]
    if len(b) < len(a):
        b = np.vstack([b, np.zeros((len(a) - len(b), width), b.dtype)])
    b = lab.align(a, b)
    first = [s.track for s in assign_all(stable, segments_from_activity(a.tolist()))]
    second = [s.track for s in assign_all(stable, segments_from_activity(b.tolist()))]
    changed = total = 0
    for row, x, y in zip(stable, first, second):
        lo = int(row["start"] / lab.FRAME)
        hi = max(lo + 1, int(np.ceil(row["end"] / lab.FRAME)))
        if not grid[lo:hi].any():
            continue
        total += 1
        changed += x != y
    return 100.0 * changed / max(total, 1)


def score_one(ident, rm):
    directory = OUT / ident
    probs = {}
    for name in ("whole", "seg", "noise"):
        path = directory / f"{name}.npz"
        if not path.exists():
            return None
        probs[name] = np.load(path)["probs"]
    data = rm.episode(ident, rm.sources()[ident])
    if data is None:
        return None
    stable, _, grid, people, *_ = data
    row = {"ident": ident}
    base = probs["whole"][: len(grid)]
    for name in ("seg", "noise"):
        other = probs[name][: len(grid)]
        row[name] = {
            "label_change_pp": label_change(base, other, stable, grid),
            "fixed_delta_pp": 100.0 * (lab.score(other, stable, grid, people)["fixed"]
                                       - lab.score(base, stable, grid, people)["fixed"]),
        }
    return row


def report(rows, write=False):
    seg = np.array([r["seg"]["label_change_pp"] for r in rows])
    noise = np.array([r["noise"]["label_change_pp"] for r in rows])
    jump_seg, jump_noise = seg > JUMP_PP, noise > JUMP_PP
    diffs = (jump_seg.astype(float) - jump_noise.astype(float)) * 100
    rate_ci = bootstrap_mean_ci(list(diffs))
    either = jump_seg | jump_noise
    p90_seg = float(np.percentile(seg[either], 90)) if either.any() else 0.0
    p90_noise = float(np.percentile(noise[either], 90)) if either.any() else 0.0
    fixed = [r["seg"]["fixed_delta_pp"] for r in rows]
    fixed_ci = bootstrap_mean_ci(fixed)

    print(f"\nW4, {len(rows)} files — segmented vs whole-file, with a noise arm as the null\n")
    print(f"{'':22}{'segmented':>12}{'noise (null)':>14}")
    print(f"{'files that jumped':22}{jump_seg.sum():>9}/{len(rows)}{jump_noise.sum():>11}/{len(rows)}")
    print(f"{'median label change':22}{np.median(seg):>11.2f}%{np.median(noise):>13.2f}%")
    print(f"{'P90 (among jumpers)':22}{p90_seg:>11.2f}%{p90_noise:>13.2f}%")
    print(f"{'max label change':22}{seg.max():>11.2f}%{noise.max():>13.2f}%")
    band = f"[{rate_ci[0]:+.2f}, {rate_ci[1]:+.2f}]" if rate_ci else "(n < 5)"
    print(f"\n  a1 jump-rate difference {diffs.mean():+.2f} pp {band}"
          f"  -> upper bound {'<=' if rate_ci and rate_ci[1] <= RATE_LIMIT_PP else '>'} +{RATE_LIMIT_PP:.0f}")
    print(f"  a2 P90 ratio {p90_seg / p90_noise if p90_noise else float('inf'):.2f}x"
          f"  -> {'<=' if p90_noise and p90_seg <= P90_FACTOR * p90_noise else '>'} {P90_FACTOR}x")
    fband = f"[{fixed_ci[0]:+.3f}, {fixed_ci[1]:+.3f}]" if fixed_ci else "(n < 5)"
    print(f"  b  fixed delta {np.mean(fixed):+.3f} pp {fband}")
    gates = {
        "a1 jump rate": bool(rate_ci and rate_ci[1] <= RATE_LIMIT_PP),
        "a2 jump size": bool(p90_noise and p90_seg <= P90_FACTOR * p90_noise),
        "b  no systematic shift": bool(fixed_ci and not fixed_ci[1] < -1.0),
    }
    print()
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  => {'噪声级，可开' if all(gates.values()) else '不是噪声级，按 §2 第 2 行复查'}")
    if write:
        (OUT / "w4-results.json").write_text(json.dumps(
            {"rows": rows, "gates": gates, "script": lab.digest(__file__)}, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT / 'w4-results.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "score"))
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    ids = args.ids or qualified_ids()
    if args.worker:
        worker(ids[0])
        return
    if args.command == "run":
        OUT.mkdir(parents=True, exist_ok=True)
        for ident in ids:
            if (OUT / ident / "run.json").exists():
                print("CACHED", ident, flush=True)
                continue
            print("START", ident, flush=True)
            with (OUT / f"{ident}.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", __file__, "run", "--worker", "--ids", ident],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print("END", ident, flush=True)
        return
    import refined_multi as rm

    rows = [r for r in (score_one(i, rm) for i in ids) if r]
    print(f"scored {len(rows)} of {len(ids)} files")
    report(rows, write=args.write)


if __name__ == "__main__":
    main()
