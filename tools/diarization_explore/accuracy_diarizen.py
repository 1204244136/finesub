"""Isolated official DiariZen pipeline; no global package replacement.

Clone BUTSpeechFIT/DiariZen into tmp/diarizen-source and download official
BUT-FIT/diarizen-wavlm-large-s80-md-v2 and pyannote/WeSpeaker weights first.
Only namespace/version metadata and decoded-audio I/O are adapted here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import types
import hashlib

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score
from accuracy_v2 import OUT


def provenance():
    source = lab.WT / "tmp/diarizen-source"
    files = sorted(p for p in source.rglob("*.py") if ".git" not in p.parts)
    code = hashlib.sha256()
    for p in files:
        code.update(p.relative_to(source).as_posix().encode())
        code.update(bytes.fromhex(lab.digest(p)))
    models = lab.WT / "tmp/diarization-models"
    return dict(source_sha256=code.hexdigest(),
                model_files={p.relative_to(models).as_posix(): lab.digest(p)
                    for p in (models/"diarizen-v2").rglob("*")
                    if p.is_file() and ".cache" not in p.parts},
                embedding=lab.digest(models/"wespeaker/pytorch_model.bin"), batch_size=2)


def load_pipeline(cpu=False, *, inference_overrides=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    # The explicitly downloaded official WeSpeaker checkpoint contains Lightning metadata.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import pyannote
    source = lab.WT / "tmp/diarizen-source"
    pyannote.__path__ = [str(source / "pyannote-audio/pyannote"), *pyannote.__path__]
    sys.path[:0] = [str(source), str(lab.WT / "tmp/diarizen-deps")]
    version = types.ModuleType("pyannote.audio.version")
    version.__version__, version.git_version = "3.1.0", "DiariZen-vendored"
    sys.modules[version.__name__] = version
    sys.modules["speechbrain"] = None
    sys.modules["nemo"] = None
    import torch
    import tomllib
    # The package __init__ eagerly imports unused training/VAD tasks tied to
    # torchaudio's removed AudioMetaData. Load only the requested inference class.
    import pyannote.audio
    package = types.ModuleType("pyannote.audio.pipelines")
    package.__path__ = [str(source / "pyannote-audio/pyannote/audio/pipelines")]
    sys.modules[package.__name__] = package
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    package.SpeakerDiarization = SpeakerDiarization
    from diarizen.pipelines.inference import DiariZenPipeline
    models = lab.WT / "tmp/diarization-models"
    embedding = models / "pyannote-wespeaker/pytorch_model.bin"
    embedding.parent.mkdir(exist_ok=True)
    if not embedding.exists():
        shutil.copyfile(models / "wespeaker/pytorch_model.bin", embedding)
    config = tomllib.loads((models / "diarizen-v2/config.toml").read_text())
    config["inference"]["args"]["batch_size"] = 2
    config["inference"]["args"].update(inference_overrides or {})
    available = torch.cuda.is_available
    try:
        if cpu:
            torch.cuda.is_available = lambda: False
        pipeline = DiariZenPipeline(models / "diarizen-v2", str(embedding), config_parse=config)
    finally:
        torch.cuda.is_available = available
    progress = {}
    for name in ("get_segmentations", "get_embeddings"):
        original = getattr(pipeline, name)

        def monitored(*args, _original=original, _name=name, **kwargs):
            def hook(*unused, total=None, completed=None, **ignored):
                now = time.perf_counter()
                if now - progress.get(_name, 0) > 30 or completed == total:
                    print(f"DiariZen {_name} {completed}/{total}", flush=True)
                    progress[_name] = now
            return _original(*args, **kwargs, hook=hook)
        setattr(pipeline, name, monitored)
    return pipeline


def infer(pipeline, waveform, cache_dir=None):
    # Official __call__ uses torchaudio.load and channel 0. Supply the same already
    # decoded mono float32 samples used by every arm, without PCM16 serialization.
    import torch
    import torchaudio
    from pyannote.core import Annotation
    if not len(waveform):
        return Annotation(uri="experiment")
    original = torchaudio.load
    segmentation = pipeline.get_segmentations
    class NoSpeech(Exception):
        pass

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def capture(*args, **kwargs):
        result = segmentation(*args, **kwargs)
        if not np.isfinite(result.data).all():
            raise ValueError("non-finite DiariZen local activity")
        if cache_dir is not None:
            rf = pipeline._segmentation.model._receptive_field
            sw = result.sliding_window
            np.savez_compressed(cache_dir / "diarizen-local.npz", data=result.data,
                                window=[sw.start, sw.duration, sw.step],
                                receptive_field=[rf.start, rf.duration, rf.step])
        if not np.any(result.data):
            raise NoSpeech
        return result
    pipeline.get_segmentations = capture
    try:
        torchaudio.load = lambda *args, **kwargs: (torch.from_numpy(waveform).unsqueeze(0), 16000)
        return pipeline("predecoded-local-waveform", sess_name="experiment")
    except NoSpeech:
        # Official reconstruct cannot allocate a negative number of clusters.
        # A segmentation with no activity has a well-defined empty annotation.
        return Annotation(uri="experiment")
    finally:
        torchaudio.load = original
        pipeline.get_segmentations = segmentation


def raster(annotation, samples):
    speakers = annotation.labels()
    probs = np.zeros((int(np.ceil(samples/1280)), max(1, len(speakers))), dtype=np.float32)
    for turn, _, who in annotation.itertracks(yield_label=True):
        probs[max(0, int(turn.start/lab.FRAME)):int(np.ceil(turn.end/lab.FRAME)), speakers.index(who)] = 1
    return probs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ident", nargs="?", default="hina-07")
    p.add_argument("--probe-cpu", action="store_true")
    args = p.parse_args()
    import vram_guard
    if not args.probe_cpu:
        vram_guard.install()
    import torch
    from finesub.speech.runtime.device import cuda_usable
    if not args.probe_cpu and not cuda_usable():
        raise RuntimeError("CUDA unavailable; GPU evaluation must not silently use CPU")
    torch.set_num_threads(4)
    start = time.perf_counter()
    pipeline = load_pipeline(cpu=args.probe_cpu)
    if args.probe_cpu:
        print("CPU load OK", pipeline._segmentation.model.specifications, flush=True)
        return
    import soundfile as sf
    waveform, sr = sf.read(lab.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000
    waveform = waveform.mean(axis=1)
    directory = OUT / args.ident
    directory.mkdir(parents=True, exist_ok=True)
    annotation = infer(pipeline, waveform, cache_dir=directory)
    probs = raster(annotation, len(waveform))
    stable, _, grid, people, *_ = lab.rm.episode(args.ident, lab.rm.sources()[args.ident])
    np.savez_compressed(directory / "diarizen.npz", probs=probs)
    (directory / "diarizen.rttm").write_text(annotation.to_rttm(), encoding="utf-8")
    payload = dict(arm="diarizen", metrics=lab.score(probs, stable, grid, people),
        bcubed=identity_score(probs, stable, grid), elapsed=time.perf_counter()-start,
        identity=dict(audio=lab.digest(lab.source(args.ident)), script=lab.digest(__file__), model=provenance()),
        watchdog_peak_gib=vram_guard.peak_gib(), allocated_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
    (directory / "diarizen.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.ident, payload["metrics"]["fixed"], payload["bcubed"], flush=True)


if __name__ == "__main__":
    main()
