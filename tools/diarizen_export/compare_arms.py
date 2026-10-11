"""Pair the port's readings against this chain's own noise, per file.

The port reading (`check_port.py`) and the noise band (`noise_arm.py`) are both "how far does the
answer move", measured the same way on the same material. Putting them side by side per file is
the comparison that means something; the two medians on their own hide whether they move together.

⚠ Deliberately NOT a pass/fail. The pre-registered threshold was derived from a band that turned
out to be n=1 and from a different corpus, and a threshold re-derived from a band measured after
the fact is weaker evidence than a pre-registered one. This prints the pairing and leaves the
reading to a person.

    python tools/diarizen_export/compare_arms.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPORT_DIR = Path(__file__).resolve().parents[2] / "tmp" / "diarizen-export"


def load(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the step that produces it first")
    return {row["ident"]: row for row in json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=str(EXPORT_DIR / "port-results.json"))
    parser.add_argument("--noise", default=str(EXPORT_DIR / "noise-arm.json"))
    args = parser.parse_args()

    port, noise = load(Path(args.port)), load(Path(args.noise))
    shared = sorted(set(port) & set(noise))

    print(f"{'file':>10}  {'port':>7}  {'noise':>7}  {'ratio':>7}  speakers (port / noise)")
    worse = []
    for ident in shared:
        a, b = port[ident]["label_change_pct"], noise[ident]["label_change_pct"]
        ratio = b / a if a else float("inf")
        same_port = port[ident]["speakers_ours"] == port[ident]["speakers_reference"]
        same_noise = noise[ident]["speakers_clean"] == noise[ident]["speakers_nudged"]
        flag = "" if ratio >= 1 else "   <- port moves MORE than the noise here"
        if ratio < 1:
            worse.append(ident)
        print(f"{ident:>10}  {a:>6.3f}%  {b:>6.3f}%  {ratio:>6.2f}x  "
              f"{'same' if same_port else 'CHANGED':>7} / {'same' if same_noise else 'CHANGED'}{flag}")

    if shared:
        pa = [port[i]["label_change_pct"] for i in shared]
        nb = [noise[i]["label_change_pct"] for i in shared]
        print(f"\npaired on {len(shared)} files")
        print(f"  port  median {np.median(pa):.3f}%  max {max(pa):.3f}%")
        print(f"  noise median {np.median(nb):.3f}%  max {max(nb):.3f}%")
        print(f"  the port moves less on {len(shared) - len(worse)}/{len(shared)} files"
              + (f"; more on {', '.join(worse)}" if worse else ""))
        counts = sum(port[i]["speakers_ours"] == port[i]["speakers_reference"] for i in shared)
        noise_counts = sum(noise[i]["speakers_clean"] == noise[i]["speakers_nudged"]
                           for i in shared)
        print(f"  speaker count unchanged: port {counts}/{len(shared)}, "
              f"noise {noise_counts}/{len(shared)}")

    extra = sorted(set(noise) - set(port))
    if extra:
        values = [noise[i]["label_change_pct"] for i in extra]
        print(f"\nnoise-only files ({len(extra)}): median {np.median(values):.3f}%  "
              f"max {max(values):.3f}%  -- {', '.join(extra)}")

    every = [row["label_change_pct"] for row in noise.values()]
    print(f"\nthe band, n={len(every)}: median {np.median(every):.3f}%  "
          f"P90 {np.percentile(every, 90):.3f}%  max {max(every):.3f}%")
    print("  it replaces a single reading of 1.36% taken on onetonton-14 (probe 7.5),")
    print("  which was both n=1 and a different corpus than anything measured here.")


if __name__ == "__main__":
    main()
