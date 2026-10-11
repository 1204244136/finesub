"""Serial, resumable GPU queue. One file/model process at a time, failure stops the queue."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import accuracy_lab as lab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids-json", required=True)
    p.add_argument("--arms", nargs="+", default=["ema", "high", "split"])
    p.add_argument("--community", action="store_true")
    args = p.parse_args()
    ids = json.loads(open(args.ids_json, encoding="utf-8").read())
    for number, ident in enumerate(ids, 1):
        jobs = [("sortformer", "accuracy_lab.py", ["worker", ident, "--arms", *args.arms])]
        if args.community:
            output = lab.OUT / ident / "community.json"
            cache_ok = False
            if output.exists() and output.with_suffix(".npz").exists():
                identity = json.loads(output.read_text(encoding="utf-8")).get("identity", {})
                cache_ok = identity.get("audio") == lab.digest(lab.source(ident)) and identity.get("script") == lab.digest(lab.HERE / "accuracy_models.py")
            if not cache_ok:
                jobs.append(("community", "accuracy_models.py", ["community", ident]))
        for name, script, options in jobs:
            print(f"[{number}/{len(ids)}] START {ident} {name}", flush=True)
            with (lab.OUT / f"{ident}-{name}-queue.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", str(lab.HERE / script), *options],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"[{number}/{len(ids)}] END {ident} {name}", flush=True)


if __name__ == "__main__":
    main()
