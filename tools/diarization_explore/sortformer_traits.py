"""Plan 5r.2 「摸清 Sortformer 本身」: what the streaming model's own structure does to our readings.

Two questions, both answered from artifacts already on disk -- no inference, no GPU, no downloads. Reads
tmp/refined-2609/agree.json (the `refined_multi.py agree` output) and each file's own -diar.json.

  length  Is "first track == human falls with headcount" about the HEADCOUNT or about the DURATION?
          NVIDIA says the arrival-order speaker cache degrades on long recordings, and a 6-person show is
          also a long one, so the recorded numbers (3 people 85%, 6 people 68%) confound the two.
  slots   Where does the 4-speaker model put its four slots? Below four people the spare slots have to hold
          something; above four, people share. The question is whether the decline is slot pressure at all.

Run from the diarization worktree root.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

AGREE = Path("tmp/refined-2609/agree.json")
RUNS = Path("out/refined-2609")


def rows():
    out = []
    for r in json.loads(AGREE.read_text(encoding="utf-8")):
        diar = RUNS / r["id"] / f"{r['id']}-diar.json"
        if not diar.exists():
            continue
        tracks = json.loads(diar.read_text(encoding="utf-8"))["tracks"]
        if not tracks:
            continue
        share = defaultdict(float)
        for t in tracks:
            share[t["track"]] += t["end"] - t["start"]
        total = sum(share.values())
        out.append({**r, "hours": max(t["end"] for t in tracks) / 3600.0,
                    "shares": sorted((v / total for v in share.values()), reverse=True)})
    return out


def _rank(v):
    v = np.asarray(v, float)
    order = np.argsort(np.argsort(v)).astype(float)
    out = np.zeros(len(v))
    for u in np.unique(v):
        m = v == u
        out[m] = order[m].mean()
    return out


def spearman(x, y):
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def partial(x, y, z):
    """Spearman of x,y with z's rank effect removed from both -- which of headcount/duration survives."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def length():
    data = rows()
    strict = [r["strict"] for r in data]
    people = [r["people"] for r in data]
    hours = [r["hours"] for r in data]
    print(f"{len(data)} files, {sum(hours):.1f} h total, duration {min(hours):.2f}-{max(hours):.2f} h, "
          f"headcount {min(people)}-{max(people)}")
    print(f"\nspearman(strict, people) = {spearman(strict, people):+.3f}")
    print(f"spearman(strict, hours)  = {spearman(strict, hours):+.3f}")
    print(f"spearman(people, hours)  = {spearman(people, hours):+.3f}   <- the confound")
    print(f"\npartial(strict, people | hours) = {partial(strict, people, hours):+.3f}")
    print(f"partial(strict, hours | people) = {partial(strict, hours, people):+.3f}")

    print("\nstrict median by headcount x duration (n in brackets); '.' = empty cell")
    edges = [0, 0.75, 1.25, 2.0, 99]
    names = ["<45m", "45-75m", "1.25-2h", ">2h"]
    print(f"{'people':<8}" + "".join(f"{n:>12}" for n in names) + f"{'all':>12}")
    for p in sorted(set(people)):
        cells = []
        for lo, hi in zip(edges, edges[1:]):
            v = [r["strict"] for r in data if r["people"] == p and lo <= r["hours"] < hi]
            cells.append(f"{np.median(v):.0%} [{len(v)}]" if v else ".")
        v = [r["strict"] for r in data if r["people"] == p]
        print(f"{p:<8}" + "".join(f"{c:>12}" for c in cells) + f"{f'{np.median(v):.0%} [{len(v)}]':>12}")

    for p in sorted(set(people)):
        grp = [r for r in data if r["people"] == p]
        if len(grp) < 8:
            continue
        h = [r["hours"] for r in grp]
        print(f"within {p} people (n={len(grp)}, {min(h):.2f}-{max(h):.2f} h): "
              f"spearman(strict, hours) = {spearman([r['strict'] for r in grp], h):+.3f}")
    print("\n⚠ duration only reaches 1.41 h here; NVIDIA's 'long recordings' are a different order of "
          "magnitude, so this does NOT test their claim -- we have no long material.")


def slots():
    data = rows()
    print(f"tracks emitted, over {len(data)} files: "
          f"{sorted({r['tracks'] for r in data})}  <- the export is a 4-speaker model")
    print("\nper-file track time share, median by headcount (sorted desc within a file)")
    print(f"{'people':<8}{'n':>4}   " + "".join(f"{f'slot{i + 1}':>9}" for i in range(4))
          + f"{'spare >5%':>12}{'strict':>9}")
    for p in sorted({r["people"] for r in data}):
        grp = [r for r in data if r["people"] == p]
        arr = np.array([r["shares"] + [0.0] * (4 - len(r["shares"])) for r in grp])
        med = np.median(arr, axis=0)
        # "spare" slots are the ones past the headcount: below 4 people they cannot be anybody real
        spare = arr[:, p:] if p < 4 else np.zeros((len(arr), 0))
        n_spare = int((spare > 0.05).any(axis=1).sum()) if spare.size else 0
        print(f"{p:<8}{len(grp):>4}   " + "".join(f"{v:>8.1%} " for v in med)
              + f"{f'{n_spare}/{len(grp)}':>12}{np.median([r['strict'] for r in grp]):>9.0%}")

    print("\nregime, by whether the four slots were enough")
    for name, sel in (("slots to spare (people < 4)", lambda r: r["people"] < 4),
                      ("exactly enough (people == 4)", lambda r: r["people"] == 4),
                      ("people share slots (> 4)", lambda r: r["people"] > 4)):
        grp = [r for r in data if sel(r)]
        if grp:
            print(f"  {name:<30} n={len(grp):<3} strict median {np.median([r['strict'] for r in grp]):.0%}")
    below = [r for r in data if r["people"] <= 4]
    print(f"\n⚠ Within the 'never exhausted' half (n={len(below)}) strict still falls 93% -> 85% -> 80% across "
          "2/3/4 people, so the decline is NOT slot pressure; running out of slots is a second, later loss.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("length", "slots"))
    {"length": length, "slots": slots}[parser.parse_args().command]()
