"""Second-round geometry/cache factorial; frozen first-round workers stay untouched."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score

OUT = lab.WT / "tmp" / "diarization-accuracy-v2"
HIGH = lab.HIGH
#: ⚠ **The geometry production shipped UNTIL 2026-09-20, not today's default.** It used to be
#: `SortformerGeometry`'s defaults spelled out; then the owner moved `chunk_right_context` to 80
#: (see that field's comment for the reading that decided the annotation). The literal values stay
#: frozen here on purpose: every `base*` arm below and every reading already recorded against them
#: was produced with right=1, and silently tracking the dataclass would repoint those comparisons
#: at a different configuration while keeping their names.
BASE = lab.BASE
ARMS = {
    "high_ema": (HIGH, .25),
    "high_refresh": (HIGH, 1.),
    "long680": (HIGH | dict(chunk_len=680, spkcache_update_period=640), 0.),
    "long1360": (HIGH | dict(chunk_len=1360, spkcache_update_period=1320), 0.),
    "long680_ema": (HIGH | dict(chunk_len=680, spkcache_update_period=640), .25),
    "high_cache376": (HIGH | dict(spkcache_len=376), 0.),
    "high_right80": (HIGH | dict(chunk_right_context=80), 0.),
    # C1 (2026-09-20): the one cell nobody had measured. EVERY arm above is built on HIGH, and
    # production runs BASE -- so `refresh`, the only knob whose development reading was a large
    # F1 gain at almost no `fixed` cost, had never been tried on the geometry that ships.
    # ⚠ `base_geom`, not "base": summary mode reads an arm called `base` out of the first-round
    # lab directory, and two different things under one name is how a comparison silently pairs
    # the wrong runs.
    "base_geom": (BASE, 0.),
    "base_refresh": (BASE, 1.),
    # ⚠ The arm the owner's 2026-09-20 decision actually ships. `high_right80` above is
    # HIGH|right=80 -- chunk_len 340, fifo 40, cache period 300 -- and production's
    # `SortformerGeometry` is chunk_len 188, right 1, fifo 0. "Turn right80 on by default"
    # therefore names a geometry NOBODY HAS RUN, and the C2 reading of the same day says HIGH
    # itself has no advantage over BASE, so the measured arm cannot stand in for this one.
    "base_right80": (BASE | dict(chunk_right_context=80), 0.),
}
DEV = [i for i in lab.DEV if i != "hina-12"]


def worker(ident, arm):
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    from finesub.speech.diarization.runner import SortformerRunner
    from finesub.speech.diarization.chunking import SortformerGeometry
    if not cuda_usable():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    path = lab.source(ident)
    params, alpha = ARMS[arm]
    identity = dict(audio=lab.digest(path), script=lab.digest(__file__),
                    parent=lab.digest(lab.__file__), geometry=params, alpha=alpha,
                    artifacts={p.name: lab.digest(p) for p in lab.ARTIFACTS.glob("*.ts")})
    directory = OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / f"{arm}.json"
    output = record.with_suffix(".npz")
    if record.exists() and output.exists() and json.loads(record.read_text())["identity"] == identity:
        print(f"{ident} {arm} cached", flush=True)
        return
    signal, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 16000
    start = time.perf_counter()
    runner = SortformerRunner(lab.ARTIFACTS, geometry=SortformerGeometry(**params))
    if "spkcache_len" in params:
        runner._cache_update.m.spkcache_len = params["spkcache_len"]
    if alpha:
        runner._cache_update = lab.CacheUpdate(runner._cache_update, alpha)
    print(f"{ident} {arm} inference start {len(signal)/sr:.1f}s", flush=True)
    probs = runner.diarize(torch.from_numpy(signal.mean(axis=1))).cpu().numpy()
    elapsed = time.perf_counter() - start
    stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
    metrics = lab.score(probs, stable, grid, people)
    bcubed = identity_score(probs, stable, grid)
    np.savez_compressed(output, probs=probs)
    record.write_text(json.dumps(dict(identity=identity, arm=arm, metrics=metrics, bcubed=bcubed,
        elapsed=elapsed, allocated_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        watchdog_peak_gib=vram_guard.peak_gib()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{ident} {arm} fixed={metrics['fixed']:.4f} F1={bcubed['f1']:.4f} elapsed={elapsed:.1f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["worker", "batch", "summary"])
    p.add_argument("--ids", nargs="+", default=DEV)
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    # ⚠ The onetonton corpus is reached by rebinding `lab` (`accuracy_cross.configure`), not by
    # naming different idents -- its sources, output root and reference loader all differ. The
    # flag has to reach the WORKER subprocess too, which is why batch mode forwards it.
    p.add_argument("--corpus", choices=("hina", "onetonton"), default="hina")
    args = p.parse_args()
    if args.corpus == "onetonton":
        from accuracy_cross import configure
        configure()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "worker":
        assert len(args.ids) == len(args.arms) == 1
        worker(args.ids[0], args.arms[0])
    elif args.mode == "batch":
        for ident in args.ids:
            for arm in args.arms:
                print(f"START {ident} {arm}", flush=True)
                with (OUT / f"{ident}-{arm}.log").open("a", encoding="utf-8") as log:
                    subprocess.run([sys.executable, "-u", __file__, "worker", "--ids", ident,
                                    "--arms", arm, "--corpus", args.corpus],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"END {ident} {arm}", flush=True)
    else:
        for arm in ["base", "high", "community", *args.arms]:
            root = lab.OUT if arm in ("base", "high", "community") else OUT
            values = [json.loads((root/i/f"{arm}.json").read_text(encoding="utf-8"))
                      for i in args.ids if (root/i/f"{arm}.json").exists()]
            if values:
                print(arm, len(values), "fixed", np.mean([v["metrics"]["fixed"] for v in values]),
                      "F1", np.mean([v["bcubed"]["f1"] for v in values]))


if __name__ == "__main__":
    main()
