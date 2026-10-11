"""Use community-1 identities and Sortformer local timing, with no reference access.

Unlike forcing community's arbitrary speaker count into four global slots, this
maps each local Sortformer slot into a community identity and retains community
where local evidence is ambiguous. All thresholds are development hypotheses.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score


def local_project(sortformer, community, *, window=750, minimum_purity=0.8, blend=0.5):
    frames = len(community)
    sf = np.pad(sortformer[:frames], ((0, max(0, frames-len(sortformer))), (0, 0)))
    projected = np.zeros_like(community)
    votes = np.zeros(frames, dtype=np.float32)
    for lo in range(0, frames, window//2):
        hi = min(frames, lo+window)
        a, b = sf[lo:hi], community[lo:hi]
        evidence = (a > 0.5).astype(np.float32).T @ b
        for slot in range(a.shape[1]):
            row = evidence[slot]
            if row.sum() <= 0:
                continue
            who = int(row.argmax())
            if row[who]/row.sum() < minimum_purity or row[who] < 25:
                continue
            projected[lo:hi, who] += a[:, slot]
        votes[lo:hi] += 1
    projected /= np.maximum(votes[:, None], 1)
    projected = np.minimum(projected, 1.0)
    # Preserve original identity when local mapping cannot explain any speaker.
    valid = projected.max(axis=1) > 0.1
    result = community.copy()
    result[valid] = (1-blend)*community[valid] + blend*projected[valid]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=lab.DEV)
    p.add_argument("--sf", default="base")
    args = p.parse_args()
    for ident in args.ids:
        directory = lab.OUT / ident
        if not (directory / "community.npz").exists():
            continue
        community = np.load(directory / "community.npz")["probs"]
        sf = np.load(directory / f"{args.sf}.npz")["probs"]
        stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        for blend in (0.35, 0.65):
            arm = f"hybrid_{args.sf}_{blend}"
            probs = local_project(sf, community, blend=blend)
            metrics = lab.score(probs, stable, grid, people)
            bcubed = identity_score(probs, stable, grid)
            np.savez_compressed(directory / f"{arm}.npz", probs=probs)
            (directory / f"{arm}.json").write_text(json.dumps({"arm": arm, "metrics": metrics,
                "bcubed": bcubed, "blend": blend, "identity": {"audio": lab.digest(lab.source(ident)),
                "script": lab.digest(__file__)}, "parents": [args.sf, "community"]},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(ident, arm, metrics["fixed"], bcubed["f1"])


if __name__ == "__main__":
    main()
