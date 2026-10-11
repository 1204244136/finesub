"""Summarize round-three cached results; no inference or tuning."""
from __future__ import annotations

import argparse
import json

import numpy as np
import accuracy_lab as lab
from accuracy_dz3 import OUT
from accuracy_report import bootstrap_mean_ci
from accuracy_v2 import DEV

MANY = [f"hina-{i:02}" for i in (6, 8, 13, 14, 32, 40, 63, 64, 65, 67, 68, 73, 74)]
FOUR = [f"hina-{i:02}" for i in (10, 30, 37, 57, 62, 69)]


def reference_density(grid, duration):
    """Diagnostic only: subtitle reference occupancy, not acoustic ground truth."""
    cumulative = np.vstack([np.zeros((1, grid.shape[1])), np.cumsum(grid, axis=0)])
    width, hop = round(duration / lab.FRAME), round(duration * .1 / lab.FRAME)
    starts = np.arange(0, max(1, len(grid)-width+1), max(1, hop))
    stops = np.minimum(len(grid), starts+width)
    counts = ((cumulative[stops]-cumulative[starts])*lab.FRAME >= .4).sum(axis=1)
    return dict(windows=len(starts), over4_fraction=float((counts > 4).mean()), max_people=int(counts.max()))


def scalar(record):
    result = {k: v for k, v in record.items() if k not in ("metrics", "native")}
    result["metrics"] = {k: v for k, v in record["metrics"].items()
                         if k not in ("row_correct", "weights", "predictions")}
    if "native" in record:
        result["native"] = scalar(record["native"])
    return result


def summarize(files, ids, arms):
    result = {}
    for arm in arms:
        complete = [i for i in ids if arm in files.get(i, {})]
        if not complete:
            continue
        values = [files[i][arm] for i in complete]
        result[arm] = dict(n=len(values), ids=complete,
            fixed=float(np.mean([v["metrics"]["fixed"] for v in values])),
            crossfit=float(np.mean([v["metrics"]["crossfit"] for v in values])),
            identity_precision=float(np.mean([v["bcubed"]["precision"] for v in values])),
            identity_recall=float(np.mean([v["bcubed"]["recall"] for v in values])),
            tracks=float(np.mean([v["metrics"]["tracks"] for v in values])),
            f1=float(np.mean([v["bcubed"]["f1"] for v in values])))
        for baseline in ("base", "high", "community", "w16_default"):
            paired = [i for i in complete if baseline in files[i]]
            if not paired or baseline == arm:
                continue
            delta = {}
            for metric, group, key in (("fixed", "metrics", "fixed"), ("f1", "bcubed", "f1")):
                diffs = np.array([files[i][arm][group][key] - files[i][baseline][group][key] for i in paired])
                delta[metric] = dict(mean=float(diffs.mean()), ci95=bootstrap_mean_ci(diffs),
                                     wins=int((diffs > 0).sum()), ties=int((diffs == 0).sum()))
            result[arm]["vs_" + baseline] = dict(n=len(paired), **delta)
    return result


def public_results():
    from accuracy_dz3_meetings import MEETINGS, VOX
    from accuracy_overlap import score_intervals
    files = {}
    for ident in MEETINGS + VOX:
        files[ident] = {}
        paths = list((OUT / ident).glob("w*.json"))
        paths += list((OUT / "overlap" / ident).glob("*.json"))
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            if "hypothesis" not in record:
                continue
            record["metrics"] = score_intervals(record["reference"], record["hypothesis"], record["duration"])
            record["reference_people"] = len({r[2] for r in record["reference"]})
            record["hypothesis_people"] = len({r[2] for r in record["hypothesis"]})
            files[ident][path.stem] = {k: v for k, v in record.items() if k not in ("reference", "hypothesis")}
    summaries = {}
    for group, ids in (("new_four_person_meetings", MEETINGS), ("vox_5_to_13_people", VOX)):
        summaries[group] = {}
        arms = sorted({a for i in ids for a in files[i]})
        for arm in arms:
            completed = [i for i in ids if arm in files[i]]
            summary = dict(n=len(completed), ids=completed)
            for metric in ("der", "miss", "false_alarm", "confusion", "overlap_all_speakers_recall",
                           "three_plus_all_recall", "four_plus_all_recall"):
                values = [files[i][arm]["metrics"][metric] for i in completed
                          if files[i][arm]["metrics"][metric] is not None
                          and (metric != "overlap_all_speakers_recall"
                               or files[i][arm]["metrics"]["overlap_seconds"] > 0)]
                summary[metric] = dict(n=len(values), mean=float(np.mean(values)) if values else None)
            for baseline in ("high", "community", "w16_default"):
                paired = [i for i in completed if baseline in files[i]]
                if paired and baseline != arm:
                    diffs = [files[i][arm]["metrics"]["der"]-files[i][baseline]["metrics"]["der"] for i in paired]
                    summary["vs_"+baseline] = dict(n=len(paired), der_delta=float(np.mean(diffs)),
                                                  ci95=bootstrap_mean_ci(diffs), wins=sum(v < 0 for v in diffs))
            summaries[group][arm] = summary
    return dict(files=files, summaries=summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    archived = json.loads((lab.HERE / "results/accuracy-20260916.json").read_text(encoding="utf-8"))
    files, arms, fingerprints, density = {}, set(), {}, {}
    for ident in DEV + MANY + FOUR:
        files[ident] = {a: archived["files"][ident][a] for a in ("base", "high", "community")}
        for path in sorted((OUT / ident).glob("w*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            files[ident][path.stem] = scalar(record)
            arms.add(path.stem)
        fingerprints[ident] = dict(audio=lab.digest(lab.source(ident)), reference=lab.digest(lab.rm.sources()[ident]),
            stable=lab.digest(lab.WT / "out/refined-2609" / ident / f"{ident}-stable.json"))
        expected = archived["source_fingerprints"][ident]
        if any(fingerprints[ident][k] != expected[k] for k in ("reference", "stable")):
            raise ValueError(f"Reference snapshot changed: {ident}")
        if fingerprints[ident]["audio"] != files[ident]["base"]["identity"]["audio"]:
            raise ValueError(f"Baseline audio changed: {ident}")
        if args.write:
            _, _, grid, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
            density[ident] = {str(w): reference_density(grid, w) for w in (8, 16)}
    groups = dict(development=DEV, heldout_over4=MANY, heldout_four=FOUR, heldout_all=MANY+FOUR)
    summaries = {name: summarize(files, ids, ["base", "high", "community", *sorted(arms)])
                 for name, ids in groups.items()}
    native_files = {i: {a: v["native"] for a, v in records.items() if "native" in v}
                    for i, records in files.items()}
    native_summaries = {name: summarize(native_files, ids, sorted(arms)) for name, ids in groups.items()}
    output = dict(groups=groups, files=files, fingerprints=fingerprints, summaries=summaries,
                  native_summaries=native_summaries,
                  reference_window_density=density, public=public_results())
    for group, rows in summaries.items():
        print(group)
        for arm, row in rows.items():
            print(f"  {arm:25} n={row['n']:2} fixed={row['fixed']*100:.3f} F1={row['f1']*100:.3f}")
    if args.write:
        from accuracy_dz3_meetings import MEETINGS, VOX
        required = [OUT / i / f"{a}.json" for i in MANY + FOUR + MEETINGS + VOX
                    for a in ("w16_default", "w8_default", "w16_median5")]
        required += [OUT / "overlap" / i / f"{a}.json" for i in MEETINGS + VOX for a in ("high", "community")]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Validation incomplete: {len(missing)} missing files")
        # A complete directory is insufficient: reject stale or mixed inputs.
        from accuracy_dz3_meetings import dataset
        for ident in MANY + FOUR + MEETINGS + VOX:
            public = ident in MEETINGS + VOX
            audio, truth = dataset(ident)[:2] if public else (lab.source(ident), lab.rm.sources()[ident])
            audio_hash, reference_hash = lab.digest(audio), lab.digest(truth)
            records = output["public"]["files"][ident] if public else files[ident]
            for arm in ("w16_default", "w8_default", "w16_median5"):
                identity = records[arm]["identity"]
                if identity["audio"] != audio_hash:
                    raise ValueError(f"Stale audio: {ident}/{arm}")
                if identity["duration"] != (8 if arm == "w8_default" else 16):
                    raise ValueError(f"Wrong geometry: {ident}/{arm}")
                if public and identity["reference"] != reference_hash:
                    raise ValueError(f"Stale public reference: {ident}/{arm}")
            if public:
                for arm in ("high", "community"):
                    identity = records[arm]["identity"]
                    if identity["audio"] != audio_hash or identity["reference"] != reference_hash:
                        raise ValueError(f"Stale public baseline: {ident}/{arm}")
        output["frozen_protocol"] = json.loads((OUT / "frozen-validation.json").read_text())
        frozen = output["frozen_protocol"]
        for name, ids in (("subtitles_over4", MANY), ("subtitles_four", FOUR),
                          ("meetings", MEETINGS), ("vox", VOX)):
            if frozen[name] != ids:
                raise ValueError(f"Validation membership differs from frozen protocol: {name}")
        for arm, expected in frozen["development"].items():
            if any(not np.isclose(summaries["development"][arm][key], value, rtol=0, atol=1e-12)
                   for key, value in expected.items()):
                raise ValueError(f"Development scores changed after freezing: {arm}")
        from accuracy_dz3_diagnose import collect, rare_local_diagnostics
        output["speaker_recall_diagnostics"] = collect()
        output["four_global_identities_der_lower_bound"] = {}
        for ident, records in output["speaker_recall_diagnostics"].items():
            seconds = sorted((s["seconds"] for s in records["w16_default"]), reverse=True)
            # Scoring-only capacity bound: even perfect activity cannot recover
            # identities beyond the four longest speakers under one-to-one mapping.
            output["four_global_identities_der_lower_bound"][ident] = sum(seconds[4:])/sum(seconds)
        output["posthoc_rare_local_diagnostics"] = rare_local_diagnostics()
        output["exact_default_replay"] = json.loads((OUT / "default-replay-check.json").read_text())
        scripts = [*lab.HERE.glob("accuracy_dz3*.py"), lab.HERE / "accuracy_diarizen.py",
                   lab.HERE / "candidate_diarizen.py"]
        output["scripts"] = {p.name: lab.digest(p) for p in scripts}
        output["resources"] = {p.parent.parent.name + "/" + p.parent.name: json.loads(p.read_text())
                               for p in OUT.glob("*/w*/resources.json")}
        output["validation_resources"] = {}
        import soundfile as sf
        for group, ids in (("subtitles", MANY + FOUR), ("public", MEETINGS + VOX)):
            lengths = [sf.info(dataset(i)[0] if group == "public" else lab.source(i)).duration for i in ids]
            resources = {w: [output["resources"][f"{i}/w{w}"] for i in ids] for w in (16, 8)}
            output["validation_resources"][group] = dict(n=len(ids), audio_seconds=sum(lengths),
                windows={str(w): dict(elapsed_seconds=sum(r["elapsed"] for r in rows),
                    real_time_factor=sum(r["elapsed"] for r in rows)/sum(lengths),
                    max_watchdog_gib=max(r["watchdog_peak_gib"] for r in rows),
                    max_allocated_gib=max(r["allocated_peak_gib"] for r in rows))
                    for w, rows in resources.items()},
                median_paired_elapsed_ratio_8_over_16=float(np.median([
                    b["elapsed"]/a["elapsed"] for a, b in zip(resources[16], resources[8])])))
        destination = lab.HERE / "results/diarizen-v3-20260917.json"
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
