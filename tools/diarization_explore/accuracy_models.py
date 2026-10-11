"""Local heterogeneous models for accuracy_lab: community-1 and WavLM SV.

No remote inference. Download model weights separately, then run with Hub offline.
WavLM is a speaker embedding model, not a VAD: its outputs explicitly use Sortformer VAD.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import accuracy_lab as lab

MODELS = lab.WT / "tmp" / "diarization-models"


def save(ident, arm, probs, seconds, extra=None):
    import torch
    import vram_guard
    stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
    result = lab.score(probs, stable, grid, people)
    directory = lab.OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{arm}.npz", probs=probs)
    (directory / f"{arm}.json").write_text(json.dumps({"arm": arm, "metrics": result,
        "identity": {"audio": lab.digest(lab.source(ident)), "script": lab.digest(__file__)},
        "elapsed": seconds, "allocated_peak_gib": torch.cuda.max_memory_allocated()/1024**3,
        "watchdog_peak_gib": vram_guard.peak_gib(), "extra": extra or {}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{ident} {arm} fixed={result['fixed']:.4f} crossfit={result['crossfit']:.4f} "
          f"coverage={result['coverage']:.4f} tracks={result['tracks']} elapsed={seconds:.1f}", flush=True)


def community(ident, waveform):
    import torch
    # Installed torchcodec hangs loading its Windows DLL (faulthandler confirmed).
    # pyannote explicitly supports predecoded waveform input without this optional decoder.
    sys.modules["torchcodec"] = None
    # community-1 uses its own WeSpeaker embedding, not the optional SpeechBrain
    # adapter; the installed SpeechBrain expects removed torchaudio backend APIs.
    sys.modules["speechbrain"] = None
    sys.modules["nemo"] = None
    from pyannote.audio import Pipeline
    start = time.perf_counter()
    pipeline = Pipeline.from_pretrained(MODELS / "community-1")
    pipeline.segmentation_batch_size = 8
    pipeline.embedding_batch_size = 8
    pipeline.to(torch.device("cuda"))
    progress = {}

    def hook(step, artifact, file=None, total=None, completed=None):
        now = time.perf_counter()
        if now - progress.get(step, 0) > 30 or completed == total:
            print(f"{ident} community {step} {completed}/{total}", flush=True)
            progress[step] = now

    output = pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": 16000}, hook=hook)
    elapsed = time.perf_counter() - start
    raw = output.serialize()
    directory = lab.OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "community-raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    for arm, annotation in (("community", output.speaker_diarization),
                            ("community_exclusive", output.exclusive_speaker_diarization)):
        speakers = annotation.labels()
        probs = np.zeros((int(np.ceil(len(waveform) / 1280)), max(1, len(speakers))), dtype=np.float32)
        for turn, _, who in annotation.itertracks(yield_label=True):
            probs[max(0, int(turn.start / lab.FRAME)):int(np.ceil(turn.end / lab.FRAME)), speakers.index(who)] = 1
        save(ident, arm, probs, elapsed, {"speakers": speakers})


def wavlm(ident, waveform):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    start = time.perf_counter()
    path = lab.OUT / ident / "wavlm-embeddings.npz"
    base = np.load(lab.OUT / ident / "base.npz")["probs"]
    frames = len(base)
    # 3 s centered windows, every 1.52 s (19 frames), bounded memory and consistent frame placement.
    centers = np.arange(19, frames, 19)
    centers = np.array([t for t in centers if base[max(0, t-9):min(frames, t+10)].max(axis=1).mean() > 0.5])
    if path.exists():
        old = np.load(path)
        if str(old["audio"]) != lab.digest(lab.source(ident)):
            raise ValueError("embedding cache audio changed")
        embeddings, centers = old["embeddings"], old["centers"]
    else:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODELS / "wavlm", local_files_only=True)
        model = WavLMForXVector.from_pretrained(MODELS / "wavlm", local_files_only=True).eval().cuda()
        values = []
        with torch.inference_mode():
            for pos in range(0, len(centers), 8):
                clips = [waveform[max(0, t*1280-24000):min(len(waveform), t*1280+24000)]
                         for t in centers[pos:pos+8]]
                inputs = extractor(clips, sampling_rate=16000, padding=True, return_tensors="pt")
                emb = model(**{k: v.cuda() for k, v in inputs.items()}).embeddings
                values.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
                if pos % 160 == 0:
                    print(f"{ident} WavLM embeddings {pos}/{len(centers)}", flush=True)
        embeddings = np.concatenate(values)
        np.savez_compressed(path, embeddings=embeddings, centers=centers, audio=lab.digest(lab.source(ident)))
        del model
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - start
    # Automatic AHC, no oracle speaker count. Thresholds are development-only choices.
    from sklearn.cluster import AgglomerativeClustering
    near = np.searchsorted(centers, np.arange(frames))
    right = np.minimum(near, len(centers)-1)
    left = np.maximum(near-1, 0)
    nearest = np.where(abs(centers[left]-np.arange(frames)) <= abs(centers[right]-np.arange(frames)), left, right)
    for threshold in (0.15, 0.25, 0.35):
        cluster = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                           distance_threshold=threshold).fit_predict(embeddings)
        probs = np.zeros((frames, int(cluster.max())+1), dtype=np.float32)
        probs[np.arange(frames), cluster[nearest]] = base.max(axis=1)
        save(ident, f"wavlm_{threshold}", probs, elapsed, {"threshold": threshold, "vad": "Sortformer base"})


def worker(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    if not cuda_usable():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    signal, sr = sf.read(lab.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000
    waveform = signal.mean(axis=1)
    (community if args.model == "community" else wavlm)(args.ident, waveform)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["community", "wavlm"])
    p.add_argument("ident", nargs="?", default="hina-02")
    p.add_argument("--batch", nargs="+")
    args = p.parse_args()
    if args.batch:
        inventory = json.loads((lab.OUT / "inventory.json").read_text(encoding="utf-8"))
        qualified = set(inventory["valid"])
        for ident in args.batch:
            if ident not in qualified:
                print(f"SKIP {ident}: audio/reference duration mismatch", flush=True)
                continue
            with (lab.OUT / f"{ident}-{args.model}.log").open("a", encoding="utf-8") as log:
                print(f"START {args.model} {ident}", flush=True)
                subprocess.run([sys.executable, "-u", __file__, args.model, ident],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"END {args.model} {ident}", flush=True)
    else:
        worker(args)


if __name__ == "__main__":
    main()
