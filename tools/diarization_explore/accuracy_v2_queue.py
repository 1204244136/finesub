"""One model/file per fresh process, serialized across both evaluation corpora."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import accuracy_lab as lab
import accuracy_v2 as v2


def configure(ident):
    if ident.startswith("onetonton-"):
        from accuracy_cross import configure as cross
        cross()


def worker(ident, model):
    configure(ident)
    if model in v2.ARMS:
        v2.worker(ident, model)
        return
    if model in ("base", "high"):
        lab.worker(SimpleNamespace(ident=ident, arms=[model]))
    elif model == "community":
        import accuracy_models
        accuracy_models.worker(SimpleNamespace(ident=ident, model=model))
    elif model in ("diarizen", "dz_community"):
        import accuracy_diarizen
        import accuracy_recluster
        sys.argv = [__file__, ident]
        (accuracy_diarizen.main if model == "diarizen" else accuracy_recluster.main)()
        return
    else:
        raise ValueError(model)
    from accuracy_identity import identity_score
    path = lab.OUT/ident/f"{model}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    stable, _, grid, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
    result["bcubed"] = identity_score(np.load(path.with_suffix(".npz"))["probs"], stable, grid)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=v2.DEV)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(args.ids[0], args.models[0])
        return
    for ident in args.ids:
        for model in args.models:
            print("START", ident, model, flush=True)
            with (v2.OUT/f"{ident}-{model}-queue.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", __file__, "--worker", "--ids", ident,
                                "--models", model], stdout=log, stderr=subprocess.STDOUT, check=True)
            print("END", ident, model, flush=True)


if __name__ == "__main__":
    main()
