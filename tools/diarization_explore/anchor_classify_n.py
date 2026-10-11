"""N-anchor classification test (mini scenario-3 check).

Profiles: host (kaguya60 sp0), nepolabo sp1, yui sp2. Classify each segment of
anchor_pollution.wav by nearest profile. Goal: each real speaker is assigned
to its own anchor (when the anchor exists), regardless of the 4-slot limit.
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

    def build_profile(segs_json, label, audio_path, max_refs=8):
        segs = json.load(open(segs_json))["segments"]
        sig, sr = sf.read(audio_path, dtype="float32")
        embs = []
        n = 0
        for s in segs:
            if s["label"] == label and s["end"] - s["start"] >= 4 and n < max_refs:
                embs.append(embed_sig(sig, s["start"] + 0.5, s["end"] - 0.5, sr))
                n += 1
        P = np.stack(embs)
        return P, (sig, sr)

    # three anchor profiles (cross-pipeline references)
    P_host, _ = build_profile("kaguya60.v2.1.json", "speaker_0", "kaguya60-vocal-16k.wav")
    P_nep1, _ = build_profile("nepolabo.v2.1.json", "speaker_1", "nepolabo-vocal-16k.wav")
    P_yui2, _ = build_profile("yui.v2.1.json", "speaker_2", "yui-vocal-16k.wav")
    profiles = {"HOST(kaguya)": P_host, "NEP1": P_nep1, "YUI2": P_yui2}
    print("profiles:", {k: v.shape for k, v in profiles.items()})

    # classify each real fragment of anchor_pollution
    tl = json.load(open("anchor_pollution_timeline.json"))
    sig_p, srp = sf.read("anchor_pollution.wav", dtype="float32")
    print("\n=== per-fragment nearest-anchor (min dist to any profile ref) ===")
    for st, en, real in tl:
        mid = (st + en) / 2
        seg_st, seg_en = mid - 0.8, mid + 0.8
        e = embed_sig(sig_p, seg_st, seg_en, srp)
        dists = {k: float(np.min(1 - P @ e)) for k, P in profiles.items()}
        best = min(dists, key=dists.get)
        tag = "ANCHOR" if real == "ANCHOR" else ("SAME" if real == "F5_kag1" else real)
        print(f"  [{st:5.1f}-{en:5.1f}] real={tag:10s} -> nearest={best:10s} dists="
              + ", ".join(f"{k}={v:.2f}" for k, v in sorted(dists.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
