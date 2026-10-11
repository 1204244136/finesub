"""Plan 5m: do the conclusions survive the input production actually reads?

Every number before 5m was measured on raw, unseparated, lossless audio; `--diarize`
reads the separated `-vocal.ogg`. Three inputs on the 5i development files:
  raw   the original VoxConverse wav
  ogg   the production separator's ASR delivery (16 kHz mono Vorbis)
  flac  its lossless delivery, resampled to 16 kHz mono in float
Scored exactly as 5k (over4_v2's functions). No perturbations: 5k.1 measured a
noise band of zero at these lengths.
"""

from __future__ import annotations

import json
import subprocess
import time
from fractions import Fraction

import numpy as np

import over4
import over4_anchor as anchor
import over4_v2 as v2
from calibrate import parse_rttm

KINDS = ("raw", "ogg", "flac")


def separated(name, suffix):
    path = over4.WORK / "vox-sep" / f"{name}-vocal{suffix}"
    if not path.exists():
        from finesub.speech.preprocessing.separator.separation import run_vocal_separation

        path.parent.mkdir(parents=True, exist_ok=True)
        run_vocal_separation(over4.audio(name), output_path=path)
    return path


def load(name, kind):
    import soundfile
    from scipy.signal import resample_poly

    path = over4.audio(name) if kind == "raw" else separated(name, f".{kind}")
    signal, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if rate != 16000:
        ratio = Fraction(16000, rate)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator).astype(np.float32)
    return signal, rate


def scores(probs, reference, grid):
    mapped = v2.mapping(probs, grid)
    lines = v2.line_scores(probs, reference, mapped)
    return {**v2.frame_scores(probs, grid, mapped), "lines": len(lines), "lines_right": sum(r for r, _ in lines)}


def main():
    import argparse

    from finesub.speech.diarization.chunking import SortformerGeometry
    from finesub.speech.diarization.runner import SortformerRunner

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="score just these files (a partial report from cached runs)")
    arguments = parser.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    started = time.time()
    runner = over4.make_runner()  # the default (production) geometry
    high = SortformerRunner(over4.WORK / "export_v2", device="cuda", geometry=SortformerGeometry(**over4.GEOMETRIES["high"]))
    config = v2.CONFIGS["split-trig5"]
    many, few = over4.select()
    if arguments.only:
        many = [row for row in many if row[0] in arguments.only]
        few = [row for row in few if row[0] in arguments.only]
    split_rows, geometry_rows, rates = {k: [] for k in KINDS}, {k: [] for k in KINDS}, {}
    for group, rows in (("many", many), ("few", few)):
        for name, speakers, _ in rows:
            reference = parse_rttm(over4.VOX / "voxconverse-master" / "dev" / f"{name}.rttm")
            for kind in KINDS:
                waveform, rate = load(name, kind)
                rates[kind] = rate
                key = f"{name}-{kind}"
                p1 = anchor.cached(runner, key, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
                p1h = anchor.cached(high, key, "onepass-hq", lambda: {"probs": over4.run(high, waveform)})["probs"]
                grid = over4.truth(name, len(p1))
                base = {"group": group, "name": name, "speakers": speakers, "perturbation": "plain"}
                split_rows[kind].append({**base, "onepass": scores(p1, reference, grid),
                                         "split": scores(v2.split(runner, key, waveform, p1, config), reference, grid)})
                geometry_rows[kind].append({**base, "onepass": scores(p1, reference, grid),
                                            "split": scores(p1h, reference, over4.truth(name, len(p1h)))})
            print(f"{group} {name} done", flush=True)

    summary = {"head": head, "started": started, "finished": time.time(), "source_rates": rates,
               "files": {"many": len(many), "few": len(few)}}
    for kind in KINDS:
        for label, rows in (("split", split_rows[kind]), ("geometry_high_minus_default", geometry_rows[kind])):
            for group in ("many", "few"):
                if not any(r["group"] == group for r in rows):
                    continue
                for field in ("line", "single"):
                    a = v2.pooled(rows, group, "onepass", field)
                    b = v2.pooled(rows, group, "split", field)
                    lo, hi = v2.bootstrap(rows, group, field)
                    summary[f"{kind}/{label}/{group}/{field}"] = {
                        "base": round(a, 4), "other": round(b, 4), "delta": round(b - a, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    print(json.dumps(summary, indent=1))
    tag = "partial" if arguments.only else "full"
    record = over4.DATA / "results" / f"over4_input-{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"summary": summary, "split": split_rows, "geometry": geometry_rows}), encoding="utf-8")
    print("record ->", record)


if __name__ == "__main__":
    main()
