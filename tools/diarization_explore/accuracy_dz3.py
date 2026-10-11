"""DiariZen round three: geometry and identity assignment, inference without gold.

All arms share one local prediction/embedding cache per geometry. Official default
is replayed as a numerical control before any engineering variant is evaluated.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time

import numpy as np
import accuracy_lab as lab
from accuracy_v2 import DEV, OUT as PREVIOUS

OUT = lab.WT / "tmp/diarization-dz3"
SCRIPT_DIGEST = lab.digest(__file__)
VARIANTS = {
    "default": {},
    "ahc50": {"ahc_threshold": .5},
    "ahc70": {"ahc_threshold": .7},
    "fa035": {"Fa": .035},
    "fa14": {"Fa": .14},
    "overlap_constraint": {},
    "active_constraint": {},
    "unconstrained": {},
}


def active_assignment(scores, activity):
    """Hungarian assignment excluding silent slots from centroid competition."""
    from scipy.optimize import linear_sum_assignment
    scores = np.nan_to_num(scores, nan=float(np.nanmin(scores)) if np.isfinite(scores).any() else 0.)
    result = np.full(scores.shape[:2], -2, dtype=int)
    for c, values in enumerate(scores):
        active = np.flatnonzero(activity[c].sum(axis=0) > 0)
        rows, cols = linear_sum_assignment(values[active], maximize=True)
        result[c, active[rows]] = cols
    return result


def overlap_assignment(scores, activity):
    """Exact best assignment with cannot-link only for coactive local speakers.

    Four local slots need at most their four best candidate identities: at most
    three other slots can forbid candidates. Inactive slots impose no constraint.
    """
    scores = np.nan_to_num(scores, nan=float(np.nanmin(scores)) if np.isfinite(scores).any() else 0.)
    result = np.argmax(scores, axis=2)
    for c, values in enumerate(scores):
        active = np.flatnonzero(activity[c].sum(axis=0) > 0)
        conflicts = (activity[c].T @ activity[c]) > 0
        pairs = [(i, j) for i in active for j in active if i < j and conflicts[i, j]]
        if not any(result[c, i] == result[c, j] for i, j in pairs):
            continue
        choices = [np.argsort(-values[i], kind="stable")[:len(active)] for i in active]
        best, best_score = None, -np.inf
        for candidate in itertools.product(*choices):
            assignment = dict(zip(active, candidate))
            if any(assignment[i] == assignment[j] for i, j in pairs):
                continue
            score = sum(values[i, k] for i, k in assignment.items())
            if score > best_score:
                best, best_score = assignment, score
        # A model with fewer global clusters than concurrent locals cannot satisfy
        # every constraint. Preserve argmax rather than inventing an identity.
        if best is not None:
            for i, k in best.items():
                result[c, i] = k
    return result


def save_annotation(ident, name, annotation, samples, provenance):
    from accuracy_diarizen import raster
    from accuracy_identity import identity_score
    from accuracy_native import score_timeline
    from finesub.speech.diarization.segments import TrackSegment
    directory = OUT / ident
    probs = raster(annotation, samples)
    np.savez_compressed(directory / f"{name}.npz", probs=probs)
    (directory / f"{name}.rttm").write_text(annotation.to_rttm(), encoding="utf-8")
    stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
    labels = annotation.labels()
    tracks = [TrackSegment(labels.index(s), t.start, t.end)
              for t, _, s in annotation.itertracks(yield_label=True)]
    native, native_bc = score_timeline(tracks, max(1, len(labels)), stable, grid, people)
    result = dict(arm=name, identity=provenance,
                  metrics=lab.score(probs, stable, grid, people),
                  bcubed=identity_score(probs, stable, grid),
                  native=dict(metrics=native, bcubed=native_bc))
    (directory / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(ident, name, "fixed", result["metrics"]["fixed"], "F1", result["bcubed"]["f1"], flush=True)


def worker(ident, duration, variants, cpu_replay=False):
    import vram_guard
    if not cpu_replay:
        vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    from accuracy_diarizen import load_pipeline, infer, provenance
    if not cpu_replay and not cuda_usable():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    start = time.perf_counter()
    directory = OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"w{duration}"
    cache = directory / prefix
    cache.mkdir(exist_ok=True)
    if cpu_replay and not (cache / "embeddings.npy").exists():
        raise ValueError("CPU replay requires existing embeddings")
    waveform, sr = sf.read(lab.source(ident), dtype="float32", always_2d=True)
    assert sr == 16000
    waveform = waveform.mean(axis=1)
    pipeline = load_pipeline(cpu=cpu_replay, inference_overrides={"seg_duration": duration})
    # Official segmentation Inference.__call__ disables TF32 globally. Replaying
    # its cache must preserve that side effect before extracting voice embeddings.
    from pyannote.audio.utils.reproducibility import fix_reproducibility
    fix_reproducibility(pipeline.device)
    identity = dict(audio=lab.digest(lab.source(ident)), model=provenance(),
                    duration=duration, precision="official-fix-reproducibility")
    manifest = cache / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError(f"Inference cache identity mismatch: {cache}")
    if not manifest.exists() and (cache / "embeddings.npy").exists():
        raise ValueError(f"Unverified embedding cache: {cache}")
    manifest.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    from pyannote.audio.utils.signal import Binarize
    existing = cache / "diarizen-local.npz"
    if not existing.exists() and duration == 16:
        existing = PREVIOUS / ident / "diarizen-local.npz"
        if existing.exists():
            prior = json.loads((existing.parent / "diarizen.json").read_text())["identity"]
            if prior["audio"] != identity["audio"] or prior["model"] != identity["model"]:
                raise ValueError("Previous local cache identity mismatch")
    if existing.exists():
        saved = np.load(existing)
        sw = SlidingWindow(start=saved["window"][0], duration=saved["window"][1], step=saved["window"][2])
        local = saved["data"].copy()
        pipeline.get_segmentations = lambda *a, **kw: SlidingWindowFeature(local.copy(), sw)
    collected = {}
    original = pipeline.get_embeddings

    def embeddings(file, segmentations, **kwargs):
        collected["segmentations"] = segmentations
        path = cache / "embeddings.npy"
        if path.exists():
            values = np.load(path)
        else:
            values = original(file, segmentations, **kwargs)
            np.save(path, values)
        collected["embeddings"] = values
        return values

    pipeline.get_embeddings = embeddings
    annotation = infer(pipeline, waveform, cache_dir=cache)
    if cpu_replay:
        from accuracy_diarizen import raster
        previous = np.load(directory / f"{prefix}_default.npz")["probs"]
        if not np.array_equal(raster(annotation, len(waveform)), previous):
            raise ValueError("CPU default replay differs from GPU original")
    provenance = dict(**identity, script=SCRIPT_DIGEST,
                      adapter=lab.digest(lab.HERE / "accuracy_diarizen.py"),
                      local=lab.digest(cache / "diarizen-local.npz"),
                      embeddings=lab.digest(cache / "embeddings.npy"))
    # Gold is loaded only by the scorer after the default inference has finished.
    save_annotation(ident, prefix + "_default", annotation, len(waveform), provenance)
    seg = collected["segmentations"]
    count = pipeline.speaker_count(seg, pipeline._segmentation.model._receptive_field, warm_up=(0., 0.))
    count.data = np.minimum(count.data, pipeline.max_speakers).astype(np.int8)
    original_assignment = pipeline.clustering.constrained_argmax
    for variant in variants:
        if variant == "default":
            continue
        params = dict(pipeline.PIPELINE_PARAMS["clustering"]) | VARIANTS[variant]
        pipeline.clustering.instantiate(params)
        pipeline.clustering.constrained_assignment = variant != "unconstrained"
        pipeline.clustering.constrained_argmax = (
            (lambda scores, const_location=None: overlap_assignment(scores, seg.data))
            if variant == "overlap_constraint" else
            (lambda scores, const_location=None: active_assignment(scores, seg.data))
            if variant == "active_constraint" else original_assignment)
        clusters, _, _ = pipeline.clustering(collected["embeddings"], segmentations=seg,
                                             min_clusters=1, max_clusters=20)
        clusters[seg.data.sum(axis=1) == 0] = -2
        discrete, _ = pipeline.reconstruct(seg, clusters, count)
        changed = Binarize(onset=.5, offset=.5, min_duration_on=0., min_duration_off=0.)(discrete)
        save_annotation(ident, prefix + "_" + variant, changed, len(waveform), provenance | {"clustering": params})
    resource_file = "resources-cpu-replay.json" if cpu_replay else "resources.json"
    (cache / resource_file).write_text(json.dumps(dict(elapsed=time.perf_counter()-start,
        allocated_peak_gib=0. if cpu_replay else torch.cuda.max_memory_allocated()/1024**3,
        watchdog_peak_gib=vram_guard.peak_gib()), indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=DEV)
    p.add_argument("--durations", nargs="+", type=int, default=[16])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--worker", action="store_true")
    p.add_argument("--cpu-replay", action="store_true")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.worker:
        worker(args.ids[0], args.durations[0], args.variants, cpu_replay=args.cpu_replay)
        return
    for ident in args.ids:
        for duration in args.durations:
            print("START", ident, duration, flush=True)
            with (OUT / f"{ident}-w{duration}.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", __file__, "--worker", "--ids", ident,
                    "--durations", str(duration), "--variants", *args.variants,
                    *(["--cpu-replay"] if args.cpu_replay else [])],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            print("END", ident, duration, flush=True)


if __name__ == "__main__":
    main()
