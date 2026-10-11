"""C1: does `refresh` survive on the geometry production actually runs?

`refresh` (cache update alpha = 1) was the one knob the closeout flagged as possibly killed by
the wrong rule: on the development set it read **F1 +2.51 pp at fixed -0.22**, and the freeze
threshold ("must beat high on both") dropped it. The closeout's own note says that threshold was
right for choosing ONE arm to confirm and wrong as a keep/drop verdict.

⚠ But every arm in `accuracy_v2.ARMS` is built on **HIGH**, and production runs **BASE**. So
`base + refresh` -- the only combination that could improve what ships -- had never been measured.
This measures it.

**Pre-registered criterion** (closeout §1a, written before this ran): F1 rises significantly AND
`fixed`'s 95% lower bound is no worse than -1.0 pp => net gain. Not "wins on both".

⚠ `base_geom` is included as a control even though a frozen `base` arm exists, so both sides of
every pair come from the same script on the same day. It reproduces the frozen arm bit-identically
on onetonton-02, which is the check that the geometry is spelled right.

    python tools/diarization_explore/c1_base_refresh.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np  # noqa: E402

from accuracy_report import bootstrap_mean_ci  # noqa: E402

OUT = HERE.parents[1] / "tmp" / "diarization-accuracy-v2"
GROUPS = {
    "onetonton (held-out confirmation, 3 people)": [
        f"onetonton-{n:02d}" for n in (2, 3, 5, 6, 7, 9, 10, 11, 13)],
    "hina + nagi-bd (development)": [
        "hina-01", "hina-02", "hina-03", "hina-04", "hina-07", "hina-11", "nagi-bd"],
}
#: The pre-registered gate, from the closeout. `fixed` may lose a little if F1 clearly gains.
FIXED_FLOOR = -1.0


def read(ident: str, arm: str) -> dict | None:
    path = OUT / ident / f"{arm}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare(name: str, idents: list[str]) -> dict:
    pairs = [(read(i, "base_geom"), read(i, "base_refresh")) for i in idents]
    usable = [(i, a, b) for i, (a, b) in zip(idents, pairs) if a and b]
    print(f"\n{name}: {len(usable)} files, refresh minus base")
    summary = {"n": len(usable)}
    for label, block, key in (("fixed", "metrics", "fixed"), ("B3 F1", "bcubed", "f1")):
        deltas = [100 * (b[block][key] - a[block][key]) for _, a, b in usable]
        interval = bootstrap_mean_ci(deltas)
        band = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "(n < 5)"
        better = sum(1 for d in deltas if d > 0)
        print(f"  {label:<7}{np.mean(deltas):+7.2f} pp {band:>22}"
              f"   refresh better on {better}/{len(deltas)}")
        summary[label] = {"mean": float(np.mean(deltas)), "ci95": interval, "better": better}
    f1, fixed = summary["B3 F1"], summary["fixed"]
    gained = bool(f1["ci95"] and f1["ci95"][0] > 0)
    floored = bool(fixed["ci95"] and fixed["ci95"][0] >= FIXED_FLOOR)
    summary["verdict"] = "net gain" if (gained and floored) else "no"
    print(f"  pre-registered gate: F1 significantly up? {gained}. "
          f"fixed lower bound >= {FIXED_FLOOR}? {floored}.  => {summary['verdict'].upper()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(HERE / "results" / "c1-base-refresh-20260920.json"))
    args = parser.parse_args()

    payload = {name: compare(name, idents) for name, idents in GROUPS.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
