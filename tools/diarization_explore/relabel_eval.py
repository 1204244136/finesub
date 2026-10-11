"""Plan 5r A / 5r.2: the `spk` relabel rule on 姐姐咚咚, measured against the fansub's speaker-styled axis.

Run from the checkout root. Reads `out/onetonton-NN/onetonton-NN-{stable,diar}.json` (the production path,
`--stage stable --diarize acoustics --language ja`) and the refined axis through `refined_ass`.
The human speaker of a segment is the fansub person whose full lines cover most of its span; a track's
owner is the episode's majority person (⚠ that assumption breaks where a track changes owner mid-episode --
`--owners`; #14 is the worst case, plan 5r.2).

  --sweep     #1-#6, descriptive (theta was chosen on these): first track == human, human within the label
              set, labels per segment -- production and the three relabel modes over theta
  --criteria  #7-#15, the pre-registered A criteria at coverage 0.5: lenient drop <= 1.0 pp, labels per
              segment drop >= 0.15, mis-deletion <= 5% (count basis); also reported without #14
  --cost      ⚠ SUPERSEDED, kept for the record: it prices "present" by time coverage and double-charges a
              multi-speaker segment (audit 2026-09-14 §3.1-3.2). Use --strong.
  --strong    edit cost per theta under the owner's current pricing (2026-09-14/15), the same ruler
              `refined_multi.py relabel` uses so the two materials compare: delete 1, move 2, a missed
              second speaker `add` only on strong evidence (their refined line STARTS inside the segment),
              and the merge itself `merge` per extra person. add=2 / merge=3 is the owner's answer.
  --owners    per 300 s window, a track's majority person (track sounds >= 30 s, person >= 60% of its
              frames); prints every change between two such windows
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refined_ass as ra  # noqa: E402
import speaker_relabel as rl  # noqa: E402

FRAME = 0.08
THETAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)


def tracks_of(segment):
    speaker = segment.get("speaker") or {}
    return speaker.get("tracks") or ([speaker["track"]] if speaker.get("track") is not None else [])


def episode(n):
    """(stable segments, timeline, frame grid of fansub people, people, track owner) or None."""
    stem = f"onetonton-{n:02d}"
    out = Path(f"out/{stem}")
    if not (out / f"{stem}-diar.json").exists():
        return None
    stable = json.loads((out / f"{stem}-stable.json").read_text(encoding="utf-8"))["segments"]
    timeline = json.loads((out / f"{stem}-diar.json").read_text(encoding="utf-8"))
    lines = [(a, b, w) for a, b, w, t in ra.events(ra.episodes()[n - 1]) if w != ra.NOTE and t]
    people = sorted({w for *_, w in lines})
    end = max([b for _, b, _ in lines] + [s["end"] for s in stable]) + 2
    grid = np.zeros((int(end / FRAME) + 2, len(people)), dtype=bool)
    for a, b, w in lines:
        grid[int(a / FRAME):int(np.ceil(b / FRAME)), people.index(w)] = True
    frames = defaultdict(Counter)
    for t in timeline["tracks"]:
        span = grid[int(t["start"] / FRAME):int(np.ceil(t["end"] / FRAME))].sum(axis=0)
        for i, c in enumerate(span):
            frames[t["track"]][people[i]] += int(c)
    owner = {k: c.most_common(1)[0][0] for k, c in frames.items()}
    return stable, timeline, grid, people, owner


def span_cover(segment, grid):
    a = int(segment["start"] / FRAME)
    b = max(a + 1, int(np.ceil(segment["end"] / FRAME)))
    return grid[a:b].sum(axis=0), b - a


def agreement(segments, grid, people, owner):
    strict = lenient = total = labels = 0.0
    for s in segments:
        tr = tracks_of(s)
        if not tr:
            continue
        cover, _ = span_cover(s, grid)
        if not cover.any():
            continue
        human = people[int(cover.argmax())]
        d = s["end"] - s["start"]
        total += d
        labels += d * len(tr)
        strict += d * (owner.get(tr[0]) == human)
        lenient += d * (human in {owner.get(t) for t in tr})
    return strict / total, lenient / total, labels / total


def sweep():
    variants = [("words", t) for t in (0.8, 1.01)] + [("lead", t) for t in (0.5, 0.6, 0.7, 0.8, 0.9)] \
        + [("coverage", t) for t in (0.3, 0.4, 0.5, 0.6, 0.8)]
    rows, stats = defaultdict(list), defaultdict(Counter)
    for n in range(1, 7):
        data = episode(n)
        if data is None:
            continue
        stable, timeline, grid, people, owner = data
        rows["production"].append(agreement(stable, grid, people, owner))
        for mode, theta in variants:
            relabelled, st = rl.relabel_segments(stable, timeline, theta, mode)
            rows[(mode, theta)].append(agreement(relabelled, grid, people, owner))
            stats[(mode, theta)] += st
    names = {"words": "按字数重排+减", "lead": "首选不变·按字数减", "coverage": "首选不变·按覆盖减"}
    print("#1-#6, by duration, episodes weighted equally: first track == human / human in label set / labels per segment")
    mean = lambda key: [float(np.mean([r[i] for r in rows[key]])) for i in range(3)]
    s, l, k = mean("production")
    print(f"{'生产现状':<22} {s:6.1%} / {l:6.1%} / {k:.2f}")
    for mode, theta in variants:
        s, l, k = mean((mode, theta))
        st = stats[(mode, theta)]
        print(f"{names[mode]:<14} θ={theta:<5} {s:6.1%} / {l:6.1%} / {k:.2f} ｜ multi {st['multi']}, reduced {st['reduced']}, kept {st['kept']}")


def criteria():
    per, reduced_total, misdeleted = {}, Counter(), Counter()
    for n in range(7, 16):
        data = episode(n)
        if data is None:
            print(f"  #{n}: no timeline")
            continue
        stable, timeline, grid, people, owner = data
        relabelled, _ = rl.relabel_segments(stable, timeline, 0.5, "coverage")
        for before, after in zip(stable, relabelled):
            tb, ta = tracks_of(before), tracks_of(after)
            if len(tb) < 2 or len(ta) != 1:
                continue
            cover, _ = span_cover(before, grid)
            if not cover.any():
                continue
            human = people[int(cover.argmax())]
            reduced_total[n] += 1
            deleted = {owner.get(t) for t in tb if t not in ta}
            misdeleted[n] += human in deleted and owner.get(ta[0]) != human
        per[n] = (agreement(stable, grid, people, owner), agreement(relabelled, grid, people, owner))
        p, r = per[n]
        print(f"  #{n}: first {p[0]:.1%}->{r[0]:.1%}  lenient {p[1]:.1%}->{r[1]:.1%}  labels {p[2]:.2f}->{r[2]:.2f}  "
              f"reduced {reduced_total[n]}, mis-deleted {misdeleted[n]} ({misdeleted[n] / max(1, reduced_total[n]):.1%})")
    for label, keep in (("all", lambda n: True), ("without #14", lambda n: n != 14)):
        eps = [n for n in per if keep(n)]
        if not eps:
            continue
        mean = lambda side, i: float(np.mean([per[n][side][i] for n in eps]))
        d_len = (mean(0, 1) - mean(1, 1)) * 100
        d_lab = mean(0, 2) - mean(1, 2)
        mis_n, red_n = sum(misdeleted[n] for n in eps), sum(reduced_total[n] for n in eps)
        mis = mis_n / red_n if red_n else 0.0
        checks = [d_len <= 1.0, d_lab >= 0.15, mis <= 0.05]
        print(f"{label} ({len(eps)} episodes): lenient {mean(0, 1):.1%}->{mean(1, 1):.1%} (drop {d_len:.2f} pp) "
              f"{'PASS' if checks[0] else 'FAIL'}; labels {mean(0, 2):.2f}->{mean(1, 2):.2f} (drop {d_lab:.2f}) "
              f"{'PASS' if checks[1] else 'FAIL'}; mis-deletion {mis_n}/{red_n} = {mis:.1%} {'PASS' if checks[2] else 'FAIL'}")


def edit_cost(segments, grid, people, owner, present_share=None, add=0):
    """delete 1 / move 2; with `present_share`, also charge a present secondary person left unlabelled."""
    dele = move = missing = 0
    for s in segments:
        tr = tracks_of(s)
        if not tr:
            continue
        cover, width = span_cover(s, grid)
        if not cover.any():
            continue
        human = people[int(cover.argmax())]
        labels = [owner.get(t) for t in tr]
        move += human not in labels
        if present_share is None:
            dele += len(labels) - 1
            continue
        present = {people[i] for i, v in enumerate(cover) if v >= present_share * width} | {human}
        dele += sum(1 for who in labels if who not in present) + len(labels) - len(set(labels))
        missing += len(present - set(labels) - {human})
    return dele, move, dele + 2 * move + add * missing


def cost():
    data = {n: episode(n) for n in range(1, 16)}
    data = {n: d for n, d in data.items() if d is not None}
    relabelled = {n: {th: rl.relabel_segments(d[0], d[1], th, "coverage")[0] for th in THETAS} for n, d in data.items()}
    for n, d in data.items():
        same = all(tracks_of(a) == tracks_of(b) for a, b in zip(d[0], relabelled[n][0.2]))
        print(f"#{n}: theta 0.2 label sets identical to production: {same}")
    groups = (("dev #1-#6", lambda n: n <= 6), ("holdout #7-#15", lambda n: n > 6))
    for group, keep in groups:
        eps = [n for n in data if keep(n)]
        print(f"\n{group} ({len(eps)} episodes), delete 1 / move 2: delete / move / cost")
        base = np.sum([edit_cost(data[n][0], *data[n][2:]) for n in eps], axis=0)
        print(f"  {'production':<12} {int(base[0]):5d} / {int(base[1]):4d} / {int(base[2]):5d}")
        for th in THETAS:
            v = np.sum([edit_cost(relabelled[n][th], *data[n][2:]) for n in eps], axis=0)
            print(f"  theta {th:<6.2f} {int(v[0]):5d} / {int(v[1]):4d} / {int(v[2]):5d}")
    for present_share in (0.3, 0.5):
        for add in (1, 2):
            print(f"\npresent >= {present_share:.0%} of the segment, missing second speaker costs {add}")
            for group, keep in groups:
                eps = [n for n in data if keep(n)]
                row = [sum(edit_cost(relabelled[n][th], *data[n][2:], present_share, add)[2] for n in eps) for th in THETAS]
                best = int(np.argmin(row))
                gap = (row[THETAS.index(0.5)] - row[best]) / row[best]
                print(f"  {group:<15} " + "  ".join(f"{th:.2f}:{int(v)}" for th, v in zip(THETAS, row))
                      + f"   best {THETAS[best]:.2f}, theta 0.5 +{gap:.1%}")


def strong():
    """Edit cost under the owner's 2026-09-14/15 pricing, on 姐姐咚咚 #1-#15.

    Same ruler as `refined_multi.py relabel`, so the two materials are comparable: a missed second speaker counts
    only on strong evidence (that person's refined line STARTS inside the segment), and the merge itself -- one ASR
    line swallowing two people -- costs 3 per extra person. The merge term does not depend on theta (relabelling
    never moves a boundary), so it only dilutes the gaps; it is priced here for arms that DO change segmentation.
    """
    import refined_multi as rm  # function-local: refined_multi imports this module

    data = {n: episode(n) for n in range(1, 16)}
    data = {n: d for n, d in data.items() if d is not None}
    starts = {n: sorted((a, w) for a, _, w in
                        [(a, b, w) for a, b, w, t in ra.events(ra.episodes()[n - 1]) if w != ra.NOTE and t])
              for n in data}
    relabelled = {n: {th: rl.relabel_segments(d[0], d[1], th, "coverage")[0] for th in THETAS} for n, d in data.items()}
    for group, keep in (("dev #1-#6", lambda n: n <= 6), ("holdout #7-#15", lambda n: n > 6)):
        eps = [n for n in data if keep(n)]
        merges = sum(rm.strong_cost(relabelled[n][0.5], *data[n][2:], starts[n], 2)[1] for n in eps)
        print(f"\n{group} ({len(eps)} episodes), {merges} strong-evidence merges")
        for add in (0, 1, 2):
            for merge in (0, 3):
                row = [sum(rm.strong_cost(relabelled[n][th], *data[n][2:], starts[n], add, merge=merge)[0] for n in eps)
                       for th in THETAS]
                best = int(np.argmin(row))
                print(f"  missed {add} / merge {merge}: " + "  ".join(f"{th}:{int(v)}" for th, v in zip(THETAS, row))
                      + f"  best {THETAS[best]}, 0.5 +{(row[THETAS.index(0.5)] - row[best]) / row[best]:.1%}")


def owners(window=300.0, min_seconds=30.0, min_share=0.60):
    for n in range(1, 16):
        data = episode(n)
        if data is None:
            continue
        _, timeline, grid, people, _ = data
        counts, frames = defaultdict(lambda: defaultdict(Counter)), Counter()
        for t in timeline["tracks"]:
            for i in range(int(t["start"] / FRAME), min(int(np.ceil(t["end"] / FRAME)), len(grid))):
                w = int(i * FRAME // window)
                frames[(t["track"], w)] += 1
                for p in np.flatnonzero(grid[i]):
                    counts[t["track"]][w][people[p]] += 1
        total = sum(frames.values())
        changes = []
        for track in sorted(counts):
            held = []
            for w in sorted(counts[track]):
                tot = frames[(track, w)]
                who, c = counts[track][w].most_common(1)[0]
                if tot * FRAME >= min_seconds and c / tot >= min_share:
                    held.append((w, who, c / tot))
            share = sum(v for (tr, _), v in frames.items() if tr == track) / total
            for (_, a, sa), (w1, b, sb) in zip(held, held[1:]):
                if a != b:
                    changes.append(f"s{track}({share:.0%}) {a}{sa:.0%}->{b}{sb:.0%} @{int(w1 * window)}s")
        print(f"#{n:2d}: {'; '.join(changes) if changes else 'no owner change'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    names = ("sweep", "criteria", "cost", "strong", "owners")
    for name in names:
        group.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    {"sweep": sweep, "criteria": criteria, "cost": cost, "strong": strong, "owners": owners}[
        next(k for k in names if getattr(args, k))]()
