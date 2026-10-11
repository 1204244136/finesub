"""Run the frozen third-round validation, one GPU job at a time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys

import accuracy_dz3 as dz
from accuracy_dz3_meetings import MEETINGS, VOX, dataset
from accuracy_dz3_report import MANY, FOUR


def run(script, arguments, label, result, audio):
    if result.exists():
        record = json.loads(result.read_text())
        if record["identity"]["audio"] != dz.lab.digest(audio):
            raise ValueError(f"Validation cache audio mismatch: {result}")
        print("CACHED", label, flush=True)
        return
    print("START", label, datetime.now(timezone.utc).isoformat(), flush=True)
    with (dz.OUT / f"validation-{label}.log").open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(dz.lab.HERE / script), *arguments],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if not result.exists():
        raise RuntimeError(f"Successful process did not produce {result}")
    print("END", label, datetime.now(timezone.utc).isoformat(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["all", "public", "subtitles", "boundaries"], default="all")
    args = p.parse_args()
    frozen = json.loads((dz.OUT / "frozen-validation.json").read_text())
    if frozen["candidate"] != "w8_default" or frozen["boundary_candidate"] != "w16_median5":
        raise ValueError("This queue is for the frozen window and boundary candidates")
    if args.phase in ("all", "public"):
        for ident in MEETINGS + VOX:
            for model, duration in (("high", 16), ("community", 16), ("diarizen", 16), ("diarizen", 8)):
                result = (dz.OUT / ident / f"w{duration}_default.json" if model == "diarizen"
                          else dz.OUT / "overlap" / ident / f"{model}.json")
                run("accuracy_dz3_meetings.py", ["--worker", "--ids", ident, "--model", model,
                    "--durations", str(duration), "--variants", "default"],
                    f"{ident}-{model}-w{duration}", result, dataset(ident)[0])
    if args.phase in ("all", "subtitles"):
        for ident in MANY + FOUR:
            for duration in (16, 8):
                run("accuracy_dz3.py", ["--worker", "--ids", ident, "--durations", str(duration),
                    "--variants", "default"], f"{ident}-w{duration}",
                    dz.OUT / ident / f"w{duration}_default.json", dz.lab.source(ident))
    if args.phase in ("all", "boundaries"):
        ids = MANY + FOUR + MEETINGS + VOX
        missing = [i for i in ids if not (dz.OUT / i / "w16_median5.json").exists()]
        if missing:
            print("START frozen median5 reconstruction", flush=True)
            with (dz.OUT / "validation-median5.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", str(dz.lab.HERE / "accuracy_dz3_boundaries.py"),
                    "--ids", *missing, "--widths", "5"], stdout=log, stderr=subprocess.STDOUT, check=True)
            print("END frozen median5 reconstruction", flush=True)


if __name__ == "__main__":
    main()
