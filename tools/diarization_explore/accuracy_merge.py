"""Independent WavLM verification of community identities, with overlap cannot-links.

Only high-purity community windows contribute to prototypes. No reference labels
or speaker counts enter construction. Development thresholds are explicit arms.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score


def merge_identities(activity, embeddings, centers, threshold):
    n = activity.shape[1]
    members = [[] for _ in range(n)]
    for vector, center in zip(embeddings, centers):
        window = activity[max(0, center-19):min(len(activity), center+19)]
        share = window.mean(axis=0)
        order = np.argsort(share)[::-1]
        if share[order[0]] >= 0.7 and (n == 1 or share[order[1]] < 0.15):
            members[order[0]].append(vector)
    prototypes = []
    eligible = []
    for values in members:
        eligible.append(len(values) >= 3)
        mean = np.mean(values, axis=0) if values else np.zeros(embeddings.shape[1])
        prototypes.append(mean/max(np.linalg.norm(mean), 1e-12))
    prototypes = np.asarray(prototypes)
    distance = 1 - prototypes @ prototypes.T
    # At least 1.6 s of simultaneous activity forbids a same-person merge.
    cannot = (activity.T @ activity) >= 20
    groups = [[i] for i in range(n)]
    accepted = []
    while True:
        candidates = []
        for i, a in enumerate(groups):
            for j in range(i+1, len(groups)):
                b = groups[j]
                if not all(eligible[k] for k in a+b) or cannot[np.ix_(a, b)].any():
                    continue
                cost = float(distance[np.ix_(a, b)].mean())
                if cost <= threshold:
                    candidates.append((cost, i, j))
        if not candidates:
            break
        cost, i, j = min(candidates)
        accepted.append({"left": groups[i].copy(), "right": groups[j].copy(), "distance": cost})
        groups[i] += groups.pop(j)
    result = np.stack([activity[:, group].max(axis=1) for group in groups], axis=1)
    return result, {"groups": groups, "merges": accepted, "prototype_windows": [len(v) for v in members]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=[i for i in lab.DEV if i != "hina-12"])
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.15, 0.2])
    args = p.parse_args()
    for ident in args.ids:
        directory = lab.OUT / ident
        embedding_file = np.load(directory / "wavlm-embeddings.npz")
        audio_hash = lab.digest(lab.source(ident))
        community_meta = json.loads((directory / "community.json").read_text(encoding="utf-8"))
        if str(embedding_file["audio"]) != audio_hash or community_meta["identity"]["audio"] != audio_hash:
            raise ValueError("Cross-model inputs must match the current audio")
        activity = np.load(directory / "community.npz")["probs"]
        stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        for threshold in args.thresholds:
            arm = f"merge_wavlm_{threshold}"
            probs, evidence = merge_identities(activity, embedding_file["embeddings"], embedding_file["centers"], threshold)
            metrics = lab.score(probs, stable, grid, people)
            bcubed = identity_score(probs, stable, grid)
            np.savez_compressed(directory / f"{arm}.npz", probs=probs)
            (directory / f"{arm}.json").write_text(json.dumps({"arm": arm, "metrics": metrics,
                "bcubed": bcubed, "threshold": threshold, "identity": {"audio": audio_hash,
                "script": lab.digest(__file__)}, "evidence": evidence}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(ident, arm, f"fixed={metrics['fixed']:.4f} F1={bcubed['f1']:.4f} tracks={probs.shape[1]}/{activity.shape[1]}")


if __name__ == "__main__":
    main()
