"""Plan 5n.2: the final verdict on VoxConverse test (never touched before 5n.2).

All test files <=12 min with >=5 speakers (57) and with 2-4 speakers (56), on the raw
wav and on the production separator's `-vocal.ogg`; decided on the ogg. Arms:
onepass, split-trig5, split-trig5 + the pairwise veto frozen in 5n.1. Primary metric:
identity errors (lines where the reference has exactly one speaker and we put exactly
one label). Writes per-file results + git HEAD + times to data/diarization/results/.
"""

from __future__ import annotations

import json
import subprocess
import time
import zipfile

import numpy as np

import over4
import over4_anchor as anchor
import over4_v2 as v2
from calibrate import parse_rttm
from over4_breakdown import breakdown
from over4_verify import same_person

TEST = over4.VOX / "voxconverse-master" / "test"
FRAME, THRESHOLD = over4.FRAME, over4.THRESHOLD


def select():
    rows = []
    for rttm in sorted(TEST.glob("*.rttm")):
        reference = parse_rttm(rttm)
        if reference:
            rows.append((rttm.stem, len({s.speaker for s in reference}), max(s.end for s in reference) / 60))
    return ([r for r in rows if r[1] >= 5 and r[2] <= 12], [r for r in rows if 2 <= r[1] <= 4 and r[2] <= 12])


def raw_path(name):
    path = over4.WORK / "vox-test" / f"{name}.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(over4.DATA / "public" / "voxconverse_test_wav.zip") as archive:
            path.write_bytes(archive.read(f"voxconverse_test_wav/{name}.wav"))
    return path


def ogg_path(name):
    path = over4.WORK / "vox-test-sep" / f"{name}-vocal.ogg"
    if not path.exists():
        from finesub.speech.preprocessing.separator.separation import run_vocal_separation

        path.parent.mkdir(parents=True, exist_ok=True)
        run_vocal_separation(raw_path(name), output_path=path)
    return path


def truth(name, frames):
    reference = parse_rttm(TEST / f"{name}.rttm")
    speakers = sorted({s.speaker for s in reference})
    grid = np.zeros((frames, len(speakers)), dtype=bool)
    for s in reference:
        grid[int(s.start / FRAME): int(np.ceil(s.end / FRAME)), speakers.index(s.speaker)] = True
    return reference, grid


def score(probs, reference, grid):
    mapped = v2.mapping(probs, grid)
    lines = v2.line_scores(probs, reference, mapped)
    counts = breakdown(probs, reference, grid)
    frames = v2.frame_scores(probs, grid, mapped)
    return {"lines": len(lines), "lines_right": sum(r for r, _ in lines),
            "identity": counts["identity"], "identity_wrong": counts["identity_wrong"],
            "single_frames": frames["single_frames"], "single_right": frames["single_right"]}


def run_file(runner, name, kind):
    import soundfile

    path = raw_path(name) if kind == "raw" else ogg_path(name)
    signal, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    assert rate == 16000, rate
    waveform = signal.mean(axis=1)
    key = f"test-{name}-{kind}"
    p1 = anchor.cached(runner, key, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
    owner = np.where((p1 > THRESHOLD).any(axis=1), p1.argmax(axis=1), -1)

    def veto(index, unit, stats):
        identities = {f"keep{unit['host']}": stats["keep_frames"][unit["host"]]}
        identities.update({f"track{g}": np.flatnonzero(owner == g) for g in range(p1.shape[1]) if g != unit["host"]})
        return any(same_person(runner, f"{key}-u{index}-{label}", waveform, frames, unit["frames"])
                   for label, frames in identities.items() if frames.size * FRAME >= 5)

    config = v2.CONFIGS["split-trig5"]
    stats_s, stats_v = {}, {}
    arms = {"onepass": p1,
            "split": v2.split(runner, key, waveform, p1, config, stats_s),
            "veto": v2.split(runner, key, waveform, p1, config, stats_v, veto)}
    reference, grid = truth(name, len(p1))
    out = {arm: score(probs, reference, grid) for arm, probs in arms.items()}
    out["meta"] = {"triggered": stats_s.get("triggered", False), "units": stats_s.get("units", 0),
                   "vetoed": stats_v.get("vetoed", 0)}
    return out


def ratio(rows, arm, right, total):
    return sum(r[arm][right] for r in rows) / max(sum(r[arm][total] for r in rows), 1)


def bootstrap_identity(rows, arm, draws=2000, seed=11):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        base = 1 - ratio(sample, "onepass", "identity_wrong", "identity")
        other = 1 - ratio(sample, arm, "identity_wrong", "identity")
        deltas.append(other - base)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    started = time.time()
    runner = over4.make_runner()
    many, few = select()
    results = []
    for group, rows in (("many", many), ("few", few)):
        for name, speakers, _ in rows:
            for kind in ("raw", "ogg"):
                results.append({"group": group, "name": name, "speakers": speakers, "kind": kind, **run_file(runner, name, kind)})
            print(group, name, "done", flush=True)
    summary = {"head": head, "started": started, "finished": time.time(), "files": {"many": len(many), "few": len(few)}}
    for kind in ("raw", "ogg"):
        for group in ("many", "few"):
            rows = [r for r in results if r["kind"] == kind and r["group"] == group]
            entry = {}
            for arm in ("onepass", "split", "veto"):
                wrong = ratio(rows, arm, "identity_wrong", "identity")
                entry[arm] = {"identity_error": round(wrong, 4),
                              "lines": round(ratio(rows, arm, "lines_right", "lines"), 4),
                              "single_frames": round(ratio(rows, arm, "single_right", "single_frames"), 4)}
            base = entry["onepass"]["identity_error"]
            for arm in ("split", "veto"):
                entry[arm]["identity_error_reduction"] = round((base - entry[arm]["identity_error"]) / max(base, 1e-9), 3)
                entry[arm]["identity_accuracy_delta_ci95"] = [round(x, 4) for x in bootstrap_identity(rows, arm)]
                entry[arm]["lines_delta"] = round(entry[arm]["lines"] - entry["onepass"]["lines"], 4)
            entry["triggered"] = sum(r["meta"]["triggered"] for r in rows)
            entry["vetoed_units"] = sum(r["meta"]["vetoed"] for r in rows)
            entry["units"] = sum(r["meta"]["units"] for r in rows)
            summary[f"{kind}/{group}"] = entry
    print(json.dumps(summary, indent=1))
    record = over4.DATA / "results" / f"over4_final-test-{time.strftime('%Y%m%d-%H%M%S')}.json"
    record.write_text(json.dumps({"summary": summary, "files": results}), encoding="utf-8")
    print("record ->", record)


if __name__ == "__main__":
    main()
