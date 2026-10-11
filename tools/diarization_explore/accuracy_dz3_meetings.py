"""Additional four-person meetings, using frozen DiariZen research arms."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile

import accuracy_dz3 as dz
import accuracy_overlap as overlap

MEETINGS = ["R8003_M8001", "R8007_M8011"]
VOX = ["vox-bmsyn", "vox-jjvkx", "vox-aiqwk", "vox-auzru", "vox-aggyz", "vox-bvqnu"]
original_dataset = overlap.dataset


def dataset(ident):
    if not ident.startswith("vox-"):
        return original_dataset(ident)
    name = ident.removeprefix("vox-")
    if ident not in VOX:
        raise ValueError("VoxConverse id is outside the frozen sample")
    truth = overlap.PUBLIC / "voxconverse/voxconverse-master/test" / f"{name}.rttm"
    audio = dz.OUT / "vox-audio" / f"{name}.wav"
    if not audio.exists():
        audio.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(overlap.PUBLIC / "voxconverse_test_wav.zip") as archive:
            audio.write_bytes(archive.read(f"voxconverse_test_wav/{name}.wav"))
    rows = []
    for line in truth.read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            rows.append((float(f[3]), float(f[3])+float(f[4]), f[7]))
    return audio, truth, rows


def save(ident, name, annotation, samples, provenance):
    _, truth, reference = dataset(ident)
    hypothesis = overlap.annotation_rows(annotation)
    duration = samples / 16000
    result = dict(arm=name, identity=provenance | {"reference": dz.lab.digest(truth),
        "scorer": dz.lab.digest(__file__), "metric": dz.lab.digest(overlap.__file__)},
        duration=duration, metrics=overlap.score_intervals(reference, hypothesis, duration),
        reference=reference, hypothesis=hypothesis)
    directory = dz.OUT / ident
    (directory / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (directory / f"{name}.rttm").write_text(annotation.to_rttm(), encoding="utf-8")
    print(ident, name, result["metrics"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=MEETINGS)
    p.add_argument("--variants", nargs="+", default=["default"])
    p.add_argument("--worker", action="store_true")
    p.add_argument("--durations", nargs="+", type=int, choices=[8, 16], default=[16])
    p.add_argument("--model", choices=["diarizen", "community", "high"], default="diarizen")
    args = p.parse_args()
    dz.OUT.mkdir(exist_ok=True)
    overlap.dataset = dataset
    if args.worker:
        if args.model == "diarizen":
            dz.lab.source = lambda ident: dataset(ident)[0]
            dz.save_annotation = save
            dz.worker(args.ids[0], args.durations[0], args.variants)
        else:
            overlap.OUT = dz.OUT
            overlap.worker(args.ids[0], args.model)
        return
    for ident in args.ids:
        jobs = [("high", 16), ("community", 16)] + [("diarizen", d) for d in args.durations]
        for model, duration in jobs:
            print("START", ident, model, duration, flush=True)
            with (dz.OUT / f"{ident}-{model}-w{duration}.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", __file__, "--worker", "--ids", ident,
                    "--model", model, "--durations", str(duration), "--variants", *args.variants],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            print("END", ident, model, duration, flush=True)


if __name__ == "__main__":
    main()
