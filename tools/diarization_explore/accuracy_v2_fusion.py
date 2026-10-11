"""Frozen same-family high-context ensemble, without reference construction."""
import json

import numpy as np
import accuracy_lab as lab
from accuracy_v2 import OUT, DEV
from accuracy_report import ensemble
from accuracy_identity import identity_score


def main():
    for ident in DEV:
        paths = [lab.OUT/ident/"high.npz", OUT/ident/"high_right80.npz", OUT/ident/"high_refresh.npz"]
        parents = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8")) for p in paths]
        if len({p["identity"]["audio"] for p in parents}) != 1:
            raise ValueError("audio mismatch")
        arrays = [np.load(p)["probs"] for p in paths]
        stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        for method in ("mean", "median", "medoid"):
            arm = f"high_fuse_{method}"
            probs = ensemble(arrays, method)
            payload = dict(arm=arm, metrics=lab.score(probs, stable, grid, people),
                bcubed=identity_score(probs, stable, grid),
                identity=dict(audio=parents[0]["identity"]["audio"], script=lab.digest(__file__),
                              parents={p.stem: lab.digest(p) for p in paths}))
            np.savez_compressed(OUT/ident/f"{arm}.npz", probs=probs)
            (OUT/ident/f"{arm}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(ident, arm, payload["metrics"]["fixed"], payload["bcubed"]["f1"], flush=True)


if __name__ == "__main__":
    main()
