"""Experiment: speaker recognition aided by Sortformer segmentation.

For each Sortformer speaker label, sample segments -> ECAPA-TDNN embeddings
(speechbrain spkrec-ecapa-voxceleb) -> measure within/between speaker distance,
silhouette, and whether a fresh agglomerative clustering on the embeddings
reproduces the Sortformer labels (purity).

Usage:
    python speaker_id_experiment.py <sortformer.json> <audio.wav> [--per-spk N] [--min-sec 1.0]
"""
import argparse
import json
import random
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import soundfile as sf

# torchaudio 2.11 removed list_audio_backends; speechbrain 1.0.3 still calls it.
# We feed waveforms via encode_batch, so a soundfile-only backend is sufficient.
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.interfaces import LocalStrategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sortformer_json")
    p.add_argument("audio")
    p.add_argument("--per-spk", type=int, default=30, help="segments sampled per speaker")
    p.add_argument("--min-sec", type=float, default=1.0, help="min segment duration to use")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_segments(path):
    return json.load(open(path, encoding="utf-8"))["segments"]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    segs = load_segments(args.sortformer_json)
    by_spk = defaultdict(list)
    for s in segs:
        if s["end"] - s["start"] >= args.min_sec:
            by_spk[s["label"]].append(s)
    print(f"segments: {len(segs)}, usable(>={args.min_sec}s): {sum(len(v) for v in by_spk.values())}")

    # sample balanced set
    sampled = []
    for lbl in sorted(by_spk):
        pool = by_spk[lbl]
        k = min(args.per_spk, len(pool))
        chosen = random.sample(pool, k)
        sampled.extend((lbl, s) for s in chosen)
        print(f"  {lbl}: {len(pool)} usable, sample {k}")
    print(f"total to embed: {len(sampled)}")

    # load audio once
    sig, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    sig = sig[:, 0] if sig.shape[1] > 1 else sig[:, 0]
    print(f"audio: {len(sig)/sr:.1f}s @ {sr}Hz")

    t0 = time.time()
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="C:/Users/Carl/.cache/speechbrain/spkrec-ecapa-voxceleb",
        local_strategy=LocalStrategy.COPY,
    )
    classifier.eval()
    print(f"model load: {time.time()-t0:.1f}s")

    labels, embs = [], []
    for lbl, s in sampled:
        start = int(s["start"] * sr)
        end = min(int(s["end"] * sr), len(sig))
        if end - start < sr * 0.4:
            continue
        wav = sig[start:end]
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = classifier.encode_batch(wav_t).squeeze().cpu().numpy()
        labels.append(lbl)
        embs.append(emb)
    embs = np.stack(embs)
    labels = np.array(labels)
    print(f"embedded: {embs.shape}")

    # normalize
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    dist = 1.0 - embs @ embs.T  # cosine distance

    # within / between distances (excluding diagonal) — cross-group pairwise
    spk_set = sorted(set(labels))
    idx_of = {l: np.where(labels == l)[0] for l in spk_set}
    print("\n=== distance matrix (mean cosine dist) ===")
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in spk_set))
    within, between = [], []
    for a in spk_set:
        row = []
        for b in spk_set:
            ia, ib = idx_of[a], idx_of[b]
            d = dist[np.ix_(ia, ib)]  # (|ia|, |ib|)
            if a == b:
                d = d[~np.eye(len(ia), dtype=bool)]
            d = d.mean()
            row.append(d)
            if a == b:
                within.append(d)
            else:
                between.append(d)
        print(f"{a:>10}" + "".join(f"{d:>10.3f}" for d in row))
    print(f"\nwithin mean={np.mean(within):.3f}  between mean={np.mean(between):.3f}  "
          f"gap={(np.mean(between)-np.mean(within)):.3f}")

    # silhouette
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(embs, labels, metric="cosine")
    print(f"silhouette (Sortformer labels as truth): {sil:.3f}")

    # fresh agglomerative clustering -> purity vs Sortformer labels
    from sklearn.cluster import AgglomerativeClustering
    k = len(spk_set)
    pred = AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine").fit_predict(embs)
    # map pred -> majority sortformer label
    from collections import Counter
    mapping = {}
    for c in range(k):
        idx = np.where(pred == c)[0]
        counter = Counter(labels[idx])
        mapping[c] = counter.most_common(1)[0][0]
    purity = np.mean([mapping[pred[i]] == labels[i] for i in range(len(labels))])
    print(f"embedding-only clustering purity vs Sortformer labels: {purity*100:.1f}% (k={k})")

    out = args.sortformer_json.replace(".json", f".embedding-per-spk{args.per_spk}.json")
    with open(out, "w") as f:
        json.dump({"labels": [str(x) for x in labels],
                   "embs": embs.tolist(),
                   "silhouette": float(sil),
                   "purity": float(purity),
                   "within_mean": float(np.mean(within)),
                   "between_mean": float(np.mean(between))}, f)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
