"""Test anchor-profile classification as an alternative to onepass labels.

Anchor profile from kaguya60 raw vocal (speaker_0 segments, cross-pipeline
reference) -> classify each segment of anchor_pollution.wav by min cosine
distance to profile centroids. Goal: keep anchor segments (recall) while
rejecting F4-type pollution that onepass merges with high confidence.
"""
import json

import numpy as np
import soundfile as sf
import torch

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.interfaces import LocalStrategy


def main():
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="C:/Users/Carl/.cache/speechbrain/spkrec-ecapa-voxceleb",
        local_strategy=LocalStrategy.COPY,
    )
    clf.eval()

    def embed_sig(sig, st, en, sr):
        ia, ib = int(st * sr), int(en * sr)
        with torch.no_grad():
            e = clf.encode_batch(torch.from_numpy(sig[ia:ib]).unsqueeze(0)).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    # 1) anchor profile from kaguya60 raw vocal
    kag_segs = json.load(open("kaguya60.v2.1.json"))["segments"]
    sig_k, sr = sf.read("kaguya60-vocal-16k.wav", dtype="float32")
    anchors = []
    n = 0
    for s in kag_segs:
        if s["label"] == "speaker_0" and s["end"] - s["start"] >= 4 and n < 8:
            anchors.append(embed_sig(sig_k, s["start"] + 0.5, s["end"] - 0.5, sr))
            n += 1
    A = np.stack(anchors)
    print(f"anchor profile: {len(anchors)} refs from kaguya60 raw vocal")

    # 2) classify Sortformer segments of anchor_pollution
    tl = json.load(open("anchor_pollution_timeline.json"))
    segs = json.load(open("anchor_pollution.v2.1.json"))["segments"]
    sig_p, srp = sf.read("anchor_pollution.wav", dtype="float32")
    results = []
    for s in sorted(segs, key=lambda x: x["start"]):
        mid = (s["start"] + s["end"]) / 2
        st, en = max(s["start"], mid - 1.0), min(s["end"], mid + 1.0)
        if en - st < 0.8:
            st, en = s["start"], s["end"]
        e = embed_sig(sig_p, st, en, srp)
        d = float(np.min(1 - A @ e))
        real = None
        for t in tl:
            if t[0] <= mid < t[1]:
                real = t[2]
                break
        results.append((s["label"], d, real, [round(s["start"], 1), round(s["end"], 1)]))

    print("\n=== per-segment anchor distance ===")
    for lbl, d, real, seg in results:
        tag = "ANCHOR" if real == "ANCHOR" else ("SAME" if real == "F5_kag1" else f"OTHER({real})")
        print(f"  onepass={lbl:9s} anchor_dist={d:.3f}  {tag:14s} [{seg[0]:5.1f}-{seg[1]:5.1f}]")

    anchor_d = [r[1] for r in results if r[2] == "ANCHOR"]
    other_d = [r[1] for r in results if r[2] not in ("ANCHOR", "F5_kag1")]
    same_d = [r[1] for r in results if r[2] == "F5_kag1"]
    print(f"\nanchor segs dist: median={np.median(anchor_d):.3f}, range={min(anchor_d):.3f}-{max(anchor_d):.3f}")
    print(f"other-frag dist: {[round(d, 3) for d in other_d]}")
    print(f"same-host-other-state dist: {[round(d, 3) for d in same_d]}")
    print("\nthreshold sweep:")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
        tp = sum(1 for d in anchor_d if d <= thr)
        fp = sum(1 for d in other_d if d <= thr)
        s_tp = sum(1 for d in same_d if d <= thr)
        print(f"  thr={thr:.2f}: anchor kept {tp}/{len(anchor_d)}, other-frag wrongly kept {fp}/{len(other_d)}, same-host kept {s_tp}/{len(same_d)}")


if __name__ == "__main__":
    main()
