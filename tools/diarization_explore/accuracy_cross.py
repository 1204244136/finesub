"""Frozen candidates on five preselected episodes of the Onetonton corpus.

Adapter only: reuse the same waveform decoder, inference, assignment and scores.
No thresholds are fitted on this corpus. Run each model/file in its own process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import accuracy_lab as lab
import accuracy_models as models
import refined_ass as ra
import relabel_eval as ev

EVALUATION_IDS = [f"onetonton-{i:02}" for i in (1, 4, 8, 12, 15)]
STRESS_IDS = ["onetonton-14"]  # Prior Sortformer audit found a large mid-file identity drift.


def configure():
    ra.ROOT = lab.WT / "data" / "refined-multi" / "姐姐咚咚"
    lab.OUT = lab.OUT / "cross-program"
    lab.OUT.mkdir(parents=True, exist_ok=True)
    lab.source = lambda ident: lab.WT / "out" / ident / f"{ident}-vocal.ogg"
    lab.rm.sources = lambda: {f"onetonton-{i:02}": None for i in range(1, 16)}
    lab.rm.episode = lambda ident, _: ev.episode(int(ident.rsplit("-", 1)[1]))


def worker(ident, model):
    import soundfile as sf
    n = int(ident.rsplit("-", 1)[1])
    reference = max(b for _, b, who, text in ra.events(ra.episodes()[n-1]) if who != ra.NOTE and text)
    duration = sf.info(lab.source(ident)).duration
    if duration < 0.9 * reference:
        raise ValueError(f"Audio/reference duration mismatch: {duration}/{reference}")
    if model == "community":
        models.worker(SimpleNamespace(ident=ident, model=model))
        arms = ["community", "community_exclusive"]
    else:
        arms = ["base", "ema", "high", "split"]
        lab.worker(SimpleNamespace(ident=ident, arms=arms))
    import numpy as np
    from accuracy_identity import identity_score
    stable, _, grid, *_ = lab.rm.episode(ident, None)
    for arm in arms:
        path = lab.OUT / ident / f"{arm}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["bcubed"] = identity_score(np.load(path.with_suffix(".npz"))["probs"], stable, grid)
        payload["corpus_adapter"] = lab.digest(__file__)
        payload["duration_qualification"] = {"audio": duration, "reference": reference}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ident")
    p.add_argument("--model", choices=["sortformer", "community"])
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    configure()
    ids = EVALUATION_IDS + STRESS_IDS
    if args.report:
        from accuracy_report import summary
        result = {"preselected_five": summary(EVALUATION_IDS), "known_drift_stress": summary(STRESS_IDS)}
        (lab.OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.ident:
        if not args.model:
            p.error("--model is required with --ident")
        worker(args.ident, args.model)
    else:
        for ident in ids:
            for model in ("sortformer", "community"):
                print(f"START {ident} {model}", flush=True)
                with (lab.OUT / f"{ident}-{model}.log").open("a", encoding="utf-8") as log:
                    subprocess.run([sys.executable, "-u", str(Path(__file__)), "--ident", ident,
                                    "--model", model], stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"END {ident} {model}", flush=True)


if __name__ == "__main__":
    main()
