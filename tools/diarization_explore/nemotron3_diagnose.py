"""Where does Nemotron-3 lose `fixed` on a file? Per-person and per-track breakdown, CPU only.

For each reference-covered row (the rows `fixed` scores): who the reference says, which track the
arm labelled first, and that track's whole-file owner. Prints, per arm:

* per person: row seconds, share correct, share unlabelled, and the tracks their rows went to
* per track: owner, and the owner of each 10-minute block (a track that changes hands mid-file
  shows a different owner per block -- the Sortformer failure mode of findings 4.2 / 4.3)

    python tools/diarization_explore/nemotron3_diagnose.py --corpus hina hina-68 hina-06
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

import nemotron3_cost as nc

FRAME = 0.08
BLOCK = 600.0


def rows_of(probs, stable, grid):
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    tracks = segments_from_activity(probs.tolist())
    labels = [label.to_dict() for label in assign_all(stable, tracks)]
    active = np.zeros((len(grid), probs.shape[1]), dtype=bool)
    for t in tracks:
        active[int(t.start / FRAME):int(np.ceil(t.end / FRAME)), t.track] = True
    table = active.T.astype(np.float64) @ grid.astype(np.float64)
    owner = np.where(table.sum(axis=1) > 0, table.argmax(axis=1), -1)
    out = []
    for s, label in zip(stable, labels):
        lo, hi = int(s["start"] / FRAME), max(int(s["start"] / FRAME) + 1, int(np.ceil(s["end"] / FRAME)))
        ref = grid[lo:hi].sum(axis=0)
        if ref.any():
            ids = label.get("tracks", [])
            out.append((s["start"], s["end"] - s["start"], int(ref.argmax()), ids[0] if ids else None))
    return out, owner


def report(ident, people, arm, probs, stable, grid):
    rows, owner = rows_of(probs, stable, grid)
    total = sum(w for _, w, _, _ in rows)
    print(f"\n{ident}  {arm}   fixed {sum(w for _, w, p, t in rows if t is not None and owner[t] == p) / total:.1%}")
    by_person = defaultdict(Counter)
    for _, w, p, t in rows:
        by_person[p]["T" if t is None else t] += w
    for p in sorted(by_person, key=lambda p: -sum(by_person[p].values())):
        c = by_person[p]
        sec = sum(c.values())
        right = sum(w for t, w in c.items() if t != "T" and owner[t] == p)
        spread = ", ".join(f"{'-' if t == 'T' else f't{t}->{people[owner[t]] if owner[t] >= 0 else '?'}'} {w / sec:.0%}"
                           for t, w in c.most_common(4))
        print(f"  {people[p]:<12} {sec:7.0f}s ({sec / total:4.0%})  correct {right / sec:4.0%}   {spread}")
    blocks = defaultdict(lambda: defaultdict(Counter))
    for start, w, p, t in rows:
        if t is not None:
            blocks[t][int(start // BLOCK)][p] += w
    for t in sorted(blocks):
        seq = " ".join(people[b.most_common(1)[0][0]][:6] if b else "." for b in
                       (blocks[t].get(i, Counter()) for i in range(int(max(blocks[t])) + 1)))
        print(f"  track {t}: owner {people[owner[t]] if owner[t] >= 0 else '?':<8} per 10 min: {seq}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", choices=("onetonton", "hina"), default="hina")
    p.add_argument("--arms", nargs="*", default=["diarizen", "nemotron3"])
    p.add_argument("idents", nargs="+")
    args = p.parse_args()
    nc.corpus_keys(args.corpus)
    for ident in args.idents:
        key = ident if args.corpus == "hina" else int(ident.rsplit("-", 1)[1])
        loaded = nc.load_episode(args.corpus, key)
        if loaded is None:
            continue
        ident, stable, grid, people, _, arms = loaded
        for arm in args.arms:
            report(ident, people, arm, arms[arm], stable, grid)


if __name__ == "__main__":
    main()
