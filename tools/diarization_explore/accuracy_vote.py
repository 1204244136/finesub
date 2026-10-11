"""Two global models, with local Sortformer evidence resolving disagreements.

No reference mapping, no four-person global limit. Unmatched DiariZen identities
are retained; uncertain local alignment falls back to the community anchor.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
import accuracy_lab as lab
from accuracy_v2 import OUT, DEV
from accuracy_identity import identity_score


def align_union(anchor, other, minimum_iou=.1):
    n = len(anchor)
    other = np.pad(other[:n], ((0, max(0, n-len(other))), (0, 0)))
    overlap = anchor.astype(np.float64).T @ other
    union = anchor.sum(axis=0)[:, None]+other.sum(axis=0)[None, :]-overlap
    iou = overlap/np.maximum(union, 1e-9)
    rows, cols = linear_sum_assignment(-iou)
    mapping = {int(c): int(r) for r, c in zip(rows, cols) if iou[r, c] >= minimum_iou}
    for col in range(other.shape[1]):
        if col not in mapping:
            mapping[col] = anchor.shape[1] + sum(v >= anchor.shape[1] for v in mapping.values())
    width = max(anchor.shape[1], max(mapping.values(), default=-1)+1)
    a = np.pad(anchor, ((0, 0), (0, width-anchor.shape[1])))
    b = np.zeros_like(a)
    for col, destination in mapping.items():
        b[:, destination] = other[:, col]
    return a, b


def arbitrate(community, diarizen, sortformer, window=750):
    a, b = align_union(community, diarizen)
    reference = np.maximum(a, b)
    n = len(a)
    sf = np.pad(sortformer[:n], ((0, max(0, n-len(sortformer))), (0, 0)))
    projection = np.zeros_like(a)
    votes = np.zeros(n)
    # Non-overlapping central decisions use ±30 s evidence for matching.
    for lo in range(0, n, window//2):
        hi = min(n, lo+window//2)
        left, right = max(0, lo-window//2), min(n, hi+window//2)
        source = sf[left:right]
        target = reference[left:right]
        evidence = (source > .5).astype(np.float64).T @ target
        rows, cols = linear_sum_assignment(-evidence)
        for row, col in zip(rows, cols):
            purity = evidence[row, col]/max(evidence[row].sum(), 1)
            if evidence[row, col] >= 25 and purity >= .8:
                projection[lo:hi, col] = sf[lo:hi, row]
        votes[lo:hi] = projection[lo:hi].max(axis=1) > .5
    result = a.copy()
    valid = votes.astype(bool)
    result[valid] = (a[valid]+b[valid]+projection[valid])/3
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=DEV)
    p.add_argument("--anchor", choices=["community", "diarizen"], default="community")
    args = p.parse_args()
    cross_program = [i.startswith("onetonton-") for i in args.ids]
    if any(cross_program):
        if not all(cross_program):
            p.error("Run development and Onetonton corpora in separate invocations")
        from accuracy_cross import configure
        configure()
    for ident in args.ids:
        paths = [lab.OUT/ident/"community.npz", OUT/ident/"diarizen.npz", OUT/ident/"high_right80.npz"]
        parents = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for p in paths]
        if len({p["identity"]["audio"] for p in parents}) != 1:
            raise ValueError("ensemble audio mismatch")
        arrays = [np.load(p)["probs"] for p in paths]
        if args.anchor == "diarizen":
            arrays[0], arrays[1] = arrays[1], arrays[0]
        arm = "local_vote" if args.anchor == "community" else "local_vote_dz_anchor"
        probs = arbitrate(*arrays)
        stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        result = dict(arm=arm, metrics=lab.score(probs, stable, grid, people),
            bcubed=identity_score(probs, stable, grid), identity=dict(audio=parents[0]["identity"]["audio"],
            script=lab.digest(__file__), parents={p.stem: lab.digest(p) for p in paths}))
        np.savez_compressed(OUT/ident/f"{arm}.npz", probs=probs)
        (OUT/ident/f"{arm}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(ident, result["metrics"]["fixed"], result["bcubed"]["f1"], flush=True)


if __name__ == "__main__":
    main()
