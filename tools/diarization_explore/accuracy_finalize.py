"""Archive compact measurements and provenance after all frozen comparisons finish.

Large probability tensors, models and source media remain local. This snapshot
retains per-file scalar results, confidence intervals and source fingerprints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score
from accuracy_report import summary

ARMS = ("base", "ema", "high", "split", "community")


def annotate(ids):
    records = {}
    for ident in ids:
        stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        audio = lab.digest(lab.source(ident))
        records[ident] = {}
        for arm in ARMS:
            path = lab.OUT / ident / f"{arm}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["identity"]["audio"] != audio:
                raise ValueError(f"Audio fingerprint changed: {ident} {arm}")
            probabilities = np.load(path.with_suffix(".npz"))["probs"]
            # Score all frozen candidates against this same reference snapshot;
            # inference caches do not depend on editorial reference-file changes.
            record["metrics"] = lab.score(probabilities, stable, grid, people)
            record["bcubed"] = identity_score(probabilities, stable, grid)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            records[ident][arm] = {k: record[k] for k in ("identity", "elapsed", "allocated_peak_gib", "watchdog_peak_gib", "bcubed")}
            records[ident][arm]["metrics"] = {k: v for k, v in record["metrics"].items()
                                             if k not in ("row_correct", "weights", "predictions")}
    return records


def main():
    inventory = json.loads((lab.OUT / "inventory.json").read_text(encoding="utf-8"))
    valid = sorted(inventory["valid"])
    dev = sorted(set(valid) & set(lab.DEV))
    holdout = sorted(set(valid) - set(dev))
    result = {"provenance": json.loads((lab.OUT / "run-provenance.json").read_text(encoding="utf-8")),
              "inventory": inventory, "files": annotate(valid)}
    result["samples"] = {"development": dev, "holdout": holdout, "all_qualified": valid}
    result["summaries"] = {name: summary(ids) for name, ids in result["samples"].items()}
    result["summaries"]["original_scope_sensitivity_75_with_invalid_hina12"] = summary(sorted(valid + ["hina-12"]))
    result["source_fingerprints"] = {i: {"reference": lab.digest(lab.rm.sources()[i]),
        "stable": lab.digest(lab.WT / "out" / "refined-2609" / i / f"{i}-stable.json")} for i in valid}
    result["scripts"] = {p.name: lab.digest(p) for p in lab.HERE.glob("accuracy_*.py")}
    result["scripts"]["candidate_run.py"] = lab.digest(lab.HERE / "candidate_run.py")
    result["models"] = {f"{prefix}/{p.relative_to(root).as_posix()}": lab.digest(p)
        for prefix, root in (("sortformer", lab.ARTIFACTS), ("downloaded", lab.WT / "tmp" / "diarization-models"))
        for p in root.rglob("*") if p.is_file() and p.suffix in (".ts", ".bin", ".safetensors", ".npz", ".yaml")}
    import accuracy_cross
    accuracy_cross.configure()
    cross = accuracy_cross.EVALUATION_IDS + accuracy_cross.STRESS_IDS
    result["cross_program_files"] = annotate(cross)
    result["cross_program_fingerprints"] = {i: {"reference": lab.digest(accuracy_cross.ra.episodes()[int(i.rsplit("-", 1)[1])-1]),
        "stable": lab.digest(lab.WT / "out" / i / f"{i}-stable.json")} for i in cross}
    result["summaries"]["cross_program"] = summary(accuracy_cross.EVALUATION_IDS)
    result["summaries"]["known_drift_stress"] = summary(accuracy_cross.STRESS_IDS)
    destination = lab.HERE / "results" / "accuracy-20260916.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
