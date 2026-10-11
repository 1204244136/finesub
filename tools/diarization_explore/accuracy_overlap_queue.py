"""Serialized frozen overlap/history evaluations with per-job logs."""
from __future__ import annotations

import argparse
import subprocess
import sys

import accuracy_lab as lab
from accuracy_v2 import OUT
from accuracy_overlap import ALI, CALLS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", action="store_true")
    args = p.parse_args()
    if args.history:
        jobs = [(f"nepolabo-{arm}", "accuracy_history.py", [arm]) for arm in ("high", "community", "diarizen")]
        jobs += [(f"kaguya60-{arm}", "accuracy_history.py", [arm, "--single-speaker-control"])
                 for arm in ("high", "community", "diarizen")]
    else:
        jobs = [(f"{ident}-{arm}", "accuracy_overlap.py", [ident, arm])
                for ident in ALI+CALLS for arm in ("high", "community", "diarizen")]
    for label, script, arguments in jobs:
        print("START", label, flush=True)
        with (OUT/f"{label}-overlap.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(lab.HERE/script), *arguments],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print("END", label, flush=True)


if __name__ == "__main__":
    main()
