"""Cross-model identity consolidation with overlap cannot-link constraints.

The independent witness must explain >= purity of each anchor track's active time
and provide >= 10 s support. Never use reference labels or oracle speaker counts.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score
from accuracy_v2 import OUT, DEV


def consolidate(anchor, witness, purity=.97, support=125, cannot_link=20):
    n = min(len(anchor), len(witness))
    a, w = anchor[:n] > .5, witness[:n] > .5
    # Single-speaker witness evidence avoids endorsing a pair merely because
    # witness frequently predicts both; denominator includes ALL anchor activity.
    exclusive = w * (w.sum(axis=1, keepdims=True) == 1)
    table = a.astype(np.float64).T @ exclusive
    durations = a.sum(axis=0)
    owner = table.argmax(axis=1)
    confidence = table.max(axis=1) / np.maximum(durations, 1)
    valid = (confidence >= purity) & (table.max(axis=1) >= support)
    overlap = a.astype(np.int64).T @ a.astype(np.int64)
    groups = [[i] for i in range(a.shape[1])]
    # Deterministic complete-link merging; transitive chains cannot evade a
    # cannot-link between any two members of a resulting identity.
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            if not groups[i] or not groups[j]:
                continue
            members = groups[i]+groups[j]
            if all(valid[k] for k in members) and len({owner[k] for k in members}) == 1:
                if all(overlap[x, y] <= cannot_link for x in groups[i] for y in groups[j]):
                    groups[i] += groups[j]
                    groups[j] = []
    groups = [g for g in groups if g]
    return np.stack([anchor[:, g].max(axis=1) for g in groups], axis=1), groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=DEV)
    args = p.parse_args()
    for ident in args.ids:
        paths = [lab.OUT/ident/"community.npz", OUT/ident/"diarizen.npz"]
        parents = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for p in paths]
        audio = lab.digest(lab.source(ident))
        if any(parent["identity"]["audio"] != audio for parent in parents):
            raise ValueError(f"Consolidation audio fingerprint mismatch: {ident}")
        community, dz = [np.load(p)["probs"] for p in paths]
        stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        for direction, anchor, witness in (("community_by_dz", community, dz), ("dz_by_community", dz, community)):
            for purity in (.90, .97):
                arm = f"{direction}_{purity}"
                probs, groups = consolidate(anchor, witness, purity)
                result = dict(arm=arm, groups=groups, metrics=lab.score(probs, stable, grid, people),
                              bcubed=identity_score(probs, stable, grid),
                              identity=dict(script=lab.digest(__file__), audio=audio,
                                            parents={p.stem: lab.digest(p) for p in paths}))
                np.savez_compressed(OUT/ident/f"{arm}.npz", probs=probs)
                (OUT/ident/f"{arm}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(ident, arm, groups, result["metrics"]["fixed"], result["bcubed"]["f1"])


if __name__ == "__main__":
    main()
