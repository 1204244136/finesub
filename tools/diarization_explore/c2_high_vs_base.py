"""C2: give `high` the `base` control it never had -- from readings already on disk.

The closeout records `high` as **未判定、缺对照**: the direct high-against-base comparisons were
two, one each way, and the three corpora that made `high` look good (AliMeeting, Japanese
CALLHOME, nepolabo) **only ever ran high**, so they say "high beats community/DiariZen" and
nothing at all about high against base.

⚠ Owner 2026-09-20 said "虽然我总觉得以前量过了", and for one of the groups that is right: the
frozen `accuracy-v2-20260916.json` carries per-file `base` AND `high` on the nine-episode
confirmation set. Nobody had read the pair out. This does, offline, with the same paired
bootstrap the rest of the closeout uses -- no GPU, no new inference.

What it cannot do is invent the missing arms: a group with no `base` run stays unjudged, and the
script says so per group rather than quietly reporting the groups it happens to have.

    python tools/diarization_explore/c2_high_vs_base.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from accuracy_report import bootstrap_mean_ci  # noqa: E402

FROZEN = HERE / "results" / "accuracy-v2-20260916.json"

#: The two arms this script exists to compare, and the metrics the closeout judges on.
PAIR = ("base", "high")
METRICS = (("fixed", "metrics", "fixed"), ("B3 F1", "bcubed", "f1"))


def paired(group: dict, name: str) -> None:
    """Per-file deltas for high minus base, wherever both arms ran on the same file."""
    if not all(arm in group for arm in PAIR):
        missing = [arm for arm in PAIR if arm not in group]
        print(f"{name}: no {'/'.join(missing)} arm -- ⚠ UNJUDGED, not 'high wins'")
        return
    files = sorted(set(group["base"].get("per_file", {}))
                   & set(group["high"].get("per_file", {})))
    if not files:
        print(f"{name}: the two arms share no file")
        return

    print(f"\n{name}: {len(files)} files, high minus base (negative = base is better)")
    for label, block, key in METRICS:
        deltas = [100 * (group["high"]["per_file"][f][block][key]
                         - group["base"]["per_file"][f][block][key]) for f in files]
        interval = bootstrap_mean_ci(deltas)
        band = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "(n < 5: no interval)"
        wins = sum(1 for d in deltas if d > 0)
        mean = sum(deltas) / len(deltas)
        verdict = ("high better" if interval and interval[0] > 0 else
                   "base better" if interval and interval[1] < 0 else "⚠ crosses zero")
        print(f"  {label:<7}{mean:+7.2f} pp {band:>22}   high better on {wins}/{len(files)}"
              f"   -> {verdict}")


def overlap_pair(name: str, idents) -> None:
    """The three corpora that only ever ran `high`, now that `base` exists for them too.

    ⚠ A different metric family, and it must not be read like the one above. These runs score
    continuous-time DER against RTTM/TextGrid turns, so **lower is better** and the sign flips:
    a positive delta here means `high` is WORSE. The `fixed`/B³ pair above cannot be computed on
    them at all -- there are no subtitle rows to fix.
    """
    from accuracy_v2 import OUT

    rows = []
    for ident in idents:
        directory = OUT / "overlap" / ident
        both = [directory / f"{arm}.json" for arm in PAIR]
        if not all(path.exists() for path in both):
            continue
        base, high = (json.loads(path.read_text(encoding="utf-8")) for path in both)
        rows.append((ident, base["metrics"], high["metrics"]))
    if not rows:
        print(f"\n{name}: no paired runs -- ⚠ UNJUDGED")
        return

    print(f"\n{name}: {len(rows)} files, high minus base (⚠ DER family: POSITIVE = high is worse)")
    for key in ("der", "miss", "false_alarm", "confusion"):
        deltas = [100 * (high[key] - base[key]) for _, base, high in rows]
        interval = bootstrap_mean_ci(deltas)
        band = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "(n < 5: no interval)"
        better = sum(1 for delta in deltas if delta < 0)
        verdict = ("high better" if interval and interval[1] < 0 else
                   "base better" if interval and interval[0] > 0 else "⚠ crosses zero")
        print(f"  {key:<12}{sum(deltas) / len(deltas):+7.2f} pp {band:>22}   "
              f"high better on {better}/{len(rows)}   -> {verdict}")


def historical_pair() -> None:
    """nepolabo: one file, 20 hand-labelled windows, so a point accuracy and no interval."""
    from accuracy_v2 import OUT

    directory = OUT / "historical" / "nepolabo"
    scores = {}
    for arm in PAIR:
        path = directory / f"{arm}.json"
        if path.exists():
            scores[arm] = json.loads(path.read_text(encoding="utf-8"))["point_accuracy"]
    if len(scores) < 2:
        print("\nnepolabo: no paired runs -- ⚠ UNJUDGED")
        return
    print(f"\nnepolabo: base {scores['base']:.1%}  high {scores['high']:.1%}  "
          f"(high minus base {100 * (scores['high'] - scores['base']):+.1f} pp)")
    print("  ⚠ ONE file and 20 labelled windows -- a direction, never a verdict. It cannot carry "
          "an interval and must not be pooled with the corpora above.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen", default=str(FROZEN))
    args = parser.parse_args()

    payload = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    for name in ("development", "confirmation"):
        group = payload.get(name)
        if isinstance(group, dict):
            paired(group, name)

    from accuracy_overlap import ALI, CALLS

    overlap_pair("AliMeeting", ALI)
    overlap_pair("CALLHOME Japanese", CALLS)
    historical_pair()


if __name__ == "__main__":
    main()
