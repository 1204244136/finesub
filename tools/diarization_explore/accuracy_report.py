"""Summarize cached experiments and evaluate truth-free ensemble construction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import accuracy_lab as lab


#: Below this many files a resampled mean is not an interval, whatever the quantiles print.
#: Two files give three distinct resample means, three give ten -- the "95%" is an artefact of the
#: grid, not a coverage statement (audit 2026-09-18, section 2.9). The repository already holds the
#: same rule for A/B work at n >= 5. ⚠ `results/diarizen-v3-20260917.json` was written before this
#: guard and still carries n=2 and n=3 intervals; report those as per-file values instead.
MIN_FILES_FOR_INTERVAL = 5


def bootstrap_mean_ci(values):
    if len(values) < MIN_FILES_FOR_INTERVAL:
        return None  # Too few files to resample: report the per-file values and the win/loss split.
    rng = np.random.default_rng(20260916)
    return np.quantile(rng.choice(values, size=(5000, len(values))).mean(axis=1), [0.025, 0.975]).tolist()


def ensemble(arrays, method="mean"):
    aligned = np.stack([arrays[0], *[lab.align(arrays[0], a) for a in arrays[1:]]])
    if method == "median":
        return np.median(aligned, axis=0)
    if method == "medoid":
        costs = np.abs(aligned[:, None] - aligned[None, :]).mean(axis=(2, 3)).sum(axis=1)
        return aligned[int(costs.argmin())]
    return aligned.mean(axis=0)


def fusion(ids, variants):
    for ident in ids:
        directory = lab.OUT / ident
        data = lab.rm.episode(ident, lab.rm.sources()[ident])
        stable, _, grid, people, *_ = data
        for name, (arms, method) in variants.items():
            paths = [directory / f"{a}.npz" for a in arms]
            if not all(p.exists() for p in paths):
                continue
            arrays = [np.load(p)["probs"] for p in paths]
            result = lab.score(ensemble(arrays, method), stable, grid, people)
            parents = [json.loads((directory / f"{a}.json").read_text(encoding="utf-8")) for a in arms]
            if len({p["identity"]["audio"] for p in parents}) != 1:
                raise ValueError("Ensemble inputs are not the same audio")
            oracle = np.any([p["metrics"]["row_correct"] for p in parents], axis=0)
            result["oracle_fixed"] = float(np.average(oracle, weights=parents[0]["metrics"]["weights"]))
            (directory / f"{name}.json").write_text(json.dumps({"arm": name, "metrics": result,
                "parents": arms, "method": method}, ensure_ascii=False, indent=2), encoding="utf-8")


def summary(ids):
    rows = {}
    for ident in ids:
        directory = lab.OUT / ident
        for path in directory.glob("*.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            if "metrics" in result:
                rows.setdefault(path.stem, {})[ident] = result["metrics"]
    output = {}
    for arm, episodes in sorted(rows.items()):
        # Adding another candidate must not change this arm's resampling draws.
        valid = sorted(set(episodes) & set(rows.get("base", {})))
        d = np.array([episodes[i]["fixed"] - rows["base"][i]["fixed"] for i in valid])
        if not len(d):
            continue
        ci = bootstrap_mean_ci(d)
        few = [episodes[i]["fixed"] - rows["base"][i]["fixed"] for i in valid if episodes[i]["people"] <= 4]
        many = [episodes[i]["fixed"] - rows["base"][i]["fixed"] for i in valid if episodes[i]["people"] > 4]
        metrics = {k: float(np.mean([episodes[i][k] for i in valid])) for k in
                   ("fixed", "crossfit", "coverage", "strict", "lenient", "labels")}
        output[arm] = dict(n=len(valid), **metrics, delta=float(d.mean()), ci=ci,
                           few_delta=float(np.mean(few)) if few else None,
                           many_delta=float(np.mean(many)) if many else None,
                           per_file={i: float(v) for i, v in zip(valid, d)})
        for group_name, members in (("few", [i for i in valid if episodes[i]["people"] <= 4]),
                                    ("many", [i for i in valid if episodes[i]["people"] > 4])):
            if members:
                b = np.array([rows["base"][i]["strict"] for i in members])
                s = np.array([episodes[i]["strict"] for i in members])
                draws = np.random.default_rng(20260916).integers(0, len(members), size=(5000, len(members)))
                boot_b, boot_s = b[draws].mean(axis=1), s[draws].mean(axis=1)
                output[arm][group_name + "_n"] = len(members)
                output[arm][group_name + "_strict_relative_error_reduction"] = float((s.mean()-b.mean())/max(1-b.mean(), 1e-12))
                output[arm][group_name + "_strict_relative_error_reduction_ci"] = (np.quantile(
                    (boot_s-boot_b)/np.maximum(1-boot_b, 1e-12), [0.025, 0.975]).tolist() if len(members) > 1 else None)
                output[arm][group_name + "_strict_delta"] = float((s-b).mean())
        identity_deltas = []
        for ident in valid:
            a = json.loads((lab.OUT / ident / f"{arm}.json").read_text(encoding="utf-8"))
            b = json.loads((lab.OUT / ident / "base.json").read_text(encoding="utf-8"))
            if "bcubed" in a and "bcubed" in b:
                identity_deltas.append(a["bcubed"]["f1"]-b["bcubed"]["f1"])
        if identity_deltas:
            output[arm]["bcubed_delta"] = float(np.mean(identity_deltas))
            output[arm]["bcubed_n"] = len(identity_deltas)
            output[arm]["bcubed_delta_ci"] = bootstrap_mean_ci(identity_deltas)
            output[arm]["bcubed_f1"] = float(np.mean([
                json.loads((lab.OUT / i / f"{arm}.json").read_text(encoding="utf-8"))["bcubed"]["f1"]
                for i in valid if "bcubed" in json.loads((lab.OUT / i / f"{arm}.json").read_text(encoding="utf-8"))]))
        output[arm]["worst_files"] = sorted(output[arm]["per_file"].items(), key=lambda item: item[1])[:5]
        output[arm]["wins_ties_losses"] = [int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())]
        interval = f"[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]" if ci else "n/a"
        print(f"{arm:18} n={len(valid):2} fixed={metrics['fixed']:.4f} cross={metrics['crossfit']:.4f} "
              f"delta={100*d.mean():+6.2f} pp CI={interval} "
              f"few={100*np.mean(few) if few else float('nan'):+.2f} "
              f"many={100*np.mean(many) if many else float('nan'):+.2f}")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=lab.DEV)
    p.add_argument("--fusion", action="store_true")
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    ids = args.ids
    if ids in (["all"], ["holdout"]):
        qualified = json.loads((lab.OUT / "inventory.json").read_text(encoding="utf-8"))["valid"]
        ids = sorted(i for i in qualified if ids == ["all"] or i not in lab.DEV)
    if args.fusion:
        fusion(ids, {
            "fuse_refresh": (["base", "refresh", "ema"], "mean"),
            "median_refresh": (["base", "refresh", "ema"], "median"),
            "medoid_refresh": (["base", "refresh", "ema"], "medoid"),
            "fuse_shift": (["base", "shift87", "micro"], "mean"),
        })
    result = summary(ids)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
