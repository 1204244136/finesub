"""Compact paired archive for the second round; no GPU required."""
from __future__ import annotations

import json

import numpy as np
import accuracy_lab as lab
from accuracy_v2 import OUT, DEV, ARMS
from accuracy_report import bootstrap_mean_ci

CONFIRM = [f"onetonton-{i:02}" for i in (2, 3, 5, 6, 7, 9, 10, 11, 13)]


def read(ident, arm):
    root = lab.OUT/("cross-program" if ident.startswith("onetonton-") else "")
    path = root/ident/f"{arm}.json" if arm in ("base", "high", "community") else OUT/ident/f"{arm}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    return {"metrics": {k: v for k, v in result["metrics"].items() if k not in ("row_correct", "weights", "predictions")},
            **{k: result[k] for k in ("bcubed", "identity", "watchdog_peak_gib", "allocated_peak_gib", "groups") if k in result}}


def summarize(ids, arms, baseline):
    result = {}
    for arm in arms:
        pairs = [(i, read(i, arm), read(i, baseline)) for i in ids]
        pairs = [(i, a, b) for i, a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        for ident, a, b in pairs:
            if a["identity"]["audio"] != b["identity"]["audio"]:
                raise ValueError(f"audio mismatch: {ident} {arm}/{baseline}")
        delta = [a["metrics"]["fixed"]-b["metrics"]["fixed"] for _, a, b in pairs]
        f1_delta = [a["bcubed"]["f1"]-b["bcubed"]["f1"] for _, a, b in pairs]
        result[arm] = dict(n=len(pairs), fixed=float(np.mean([a["metrics"]["fixed"] for _, a, _ in pairs])),
            bcubed_f1=float(np.mean([a["bcubed"]["f1"] for _, a, _ in pairs])),
            baseline=baseline, fixed_delta=float(np.mean(delta)), fixed_ci=bootstrap_mean_ci(delta),
            f1_delta=float(np.mean(f1_delta)), f1_ci=bootstrap_mean_ci(f1_delta),
            per_file={i: a for i, a, _ in pairs})
        r = result[arm]
        print(f"{arm:26} n={r['n']} fixed={r['fixed']:.4f} F1={r['bcubed_f1']:.4f} "
              f"delta={r['fixed_delta']*100:+.2f} CI={r['fixed_ci']}")
    return result


def summarize_overlap(records):
    from accuracy_overlap import ALI, CALLS
    result = {}
    metrics = ("der", "miss", "false_alarm", "confusion", "overlap_precision",
               "overlap_recall", "overlap_all_speakers_recall", "three_plus_all_recall")
    for name, ids in (("alimeeting", ALI), ("callhome_jpn", CALLS)):
        complete = [i for i in ids if all(a in records.get(i, {})
                    for a in ("high", "community", "diarizen"))]
        group = dict(expected=len(ids), n=len(complete), ids=complete, arms={})
        for ident in complete:
            fingerprints = [records[ident][a]["identity"] for a in ("high", "community", "diarizen")]
            if any((f["audio"], f["reference"]) != (fingerprints[0]["audio"], fingerprints[0]["reference"])
                   for f in fingerprints[1:]):
                raise ValueError(f"Overlap inputs mismatch: {ident}")
        for arm in ("high", "community", "diarizen"):
            group["arms"][arm] = {}
            for variant in ("metrics", "production_metrics"):
                values = [records[i][arm].get(variant, records[i][arm]["metrics"]) for i in complete]
                summary = {}
                for metric in metrics:
                    valid = [r[metric] for r in values if r[metric] is not None]
                    summary[metric] = dict(n=len(valid), mean=float(np.mean(valid)) if valid else None)
                group["arms"][arm][variant] = summary
        group["diarizen_minus_community"] = {}
        for metric in metrics:
            pairs = [(records[i]["diarizen"]["metrics"][metric], records[i]["community"]["metrics"][metric])
                     for i in complete]
            delta = [a-b for a, b in pairs if a is not None and b is not None]
            group["diarizen_minus_community"][metric] = dict(n=len(delta),
                mean=float(np.mean(delta)) if delta else None,
                ci=bootstrap_mean_ci(delta) if delta else None)
        result[name] = group
    return result


def development_complementarity():
    """Post-hoc labelled error analysis, never a deployable fusion score."""
    rows = {}
    for ident in DEV:
        paths = (lab.OUT/ident/"community.json", OUT/ident/"diarizen.json")
        c, d = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        if c["identity"]["audio"] != d["identity"]["audio"]:
            raise ValueError(f"Complementarity input mismatch: {ident}")
        weights = np.asarray(c["metrics"]["weights"])
        np.testing.assert_array_equal(weights, d["metrics"]["weights"])
        a, b = [np.asarray(r["metrics"]["row_correct"], dtype=bool) for r in (c, d)]
        rows[ident] = {key: float(weights[mask].sum()/weights.sum()) for key, mask in (
            ("both_correct", a & b), ("community_only_correct", a & ~b),
            ("diarizen_only_correct", ~a & b), ("both_wrong", ~a & ~b),
            ("labelled_row_oracle", a | b))}
    return dict(per_file=rows, mean={k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))},
                caveat="Post-hoc DEV only. Reference-aligned row oracle is an optimistic diagnostic, not an inference algorithm or validated accuracy.")


def main():
    arms = ["base", "high", "community", *ARMS, "diarizen", "dz_community",
            "local_vote", "local_vote_dz_anchor", "community_native", "diarizen_native", "dz_community_native",
            *[f"high_fuse_{m}" for m in ("mean", "median", "medoid")],
            *[f"{d}_{p}" for d in ("community_by_dz", "dz_by_community") for p in (.9, .97)]]
    result = dict(development=summarize(DEV, arms, "high"),
                  development_vs_community=summarize(DEV, ["diarizen", "dz_community"], "community"),
                  confirmation=summarize(CONFIRM, arms, "high"),
                  confirmation_vs_community=summarize(CONFIRM, ["diarizen", "dz_community"], "community"),
                  confirmation_vs_base=summarize(CONFIRM, ["diarizen"], "base"),
                  exploratory_fusion_vs_diarizen=summarize(CONFIRM, ["local_vote_dz_anchor"], "diarizen"),
                  exploratory_native_vs_diarizen=summarize(CONFIRM, ["diarizen_native"], "diarizen"),
                  protocol="diarization-accuracy-v2-20260916", frame_seconds=lab.FRAME,
                  caveat="Subtitle attribution proxies, not DER. Prior 67-file holdout is now reused regression data.")
    result["development_complementarity"] = development_complementarity()
    result["overlap"] = {}
    for directory in (OUT/"overlap").glob("*"):
        if directory.is_dir():
            result["overlap"][directory.name] = {}
            for path in directory.glob("*.json"):
                value = json.loads(path.read_text(encoding="utf-8"))
                result["overlap"][directory.name][path.stem] = {k: v for k, v in value.items() if k not in ("reference", "hypothesis")}
    result["overlap_summary"] = summarize_overlap(result["overlap"])
    result["evaluation_settings"] = dict(bootstrap_draws=5000, bootstrap_seed=20260916,
        confirmation_expected=len(CONFIRM), overlap_collar_seconds=0, overlap_uem="full audio",
        equal_file_weight=True, gpu_guard_gib=8, inference_uses_reference_labels=False,
        exploratory_fusion="Designed on DEV after partial confirmation baselines were visible; not a fully blind primary comparison.",
        public_corpora="AliMeeting array downmix, within a corpus used by DiariZen training. CALLHOME Japanese utterance-level timestamps. Neither is an official challenge score.")
    result["historical"] = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                            for p in (OUT/"historical/nepolabo").glob("*.json")}
    result["single_speaker_control"] = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                            for p in (OUT/"historical/kaguya60").glob("*.json")}
    # Verify that reused first-round scores still refer to the same editorial
    # snapshot. Inference never uses these reference files.
    original = json.loads((lab.HERE/"results/accuracy-20260916.json").read_text(encoding="utf-8"))
    result["source_fingerprints"] = {}
    for ident in DEV:
        current = dict(reference=lab.digest(lab.rm.sources()[ident]),
                       stable=lab.digest(lab.source(ident).with_name(f"{ident}-stable.json")))
        if current != original["source_fingerprints"][ident]:
            raise ValueError(f"Reference snapshot changed: rescore both rounds before comparing {ident}")
        result["source_fingerprints"][ident] = current
    import accuracy_cross
    # Select references without changing lab.OUT or inference bindings.
    accuracy_cross.ra.ROOT = lab.WT/"data/refined-multi/姐姐咚咚"
    for ident in CONFIRM:
        n = int(ident.rsplit("-", 1)[1])
        result["source_fingerprints"][ident] = dict(
            reference=lab.digest(accuracy_cross.ra.episodes()[n-1]),
            stable=lab.digest(lab.WT/"out"/ident/f"{ident}-stable.json"))
    result["scripts"] = {p.name: lab.digest(p) for p in lab.HERE.glob("accuracy_*.py")}
    result["scripts"]["candidate_diarizen.py"] = lab.digest(lab.HERE/"candidate_diarizen.py")
    destination = lab.HERE/"results/accuracy-v2-20260916.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
