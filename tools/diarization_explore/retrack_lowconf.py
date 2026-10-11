"""How much of production's wrong-track labels a low-confidence cut would catch (owner 2026-10-04).

Every LABELLED row of production's assignment (Nemotron-3 + vbx, `assign_all`) carries the share of
the row each track covers. Two confidences from that: the leader's share, and the relative margin
(leader - runner-up) / leader (`review` is margin <= 0.15). A row is WRONG when its leading track's
majority person is not the row's person (truth: the one person whose refined line starts in the row,
else the person covering most of it) -- the `fixed` idea, per row.

Two kinds of error are set apart, because no per-row cut is meant to catch them:
  switched  the track's majority person in the +-150 s around the row differs from its global one
            (a track that changed hands midway, findings "中途换人")
  merged    the row sits in a track whose whole-file purity is < 0.8 (several people in one track)

    python tools/diarization_explore/retrack_lowconf.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import retrack_voices as rv  # noqa: E402

WINDOW = 150.0
MERGED = 0.80


def voices_material():
    import nemotron3_cost as nc
    import refined_multi as rm
    nc.corpus_keys("hina")
    sources = rm.sources()
    out = []
    for ident in rv.EPISODES:
        stable, _, grid, people, _, lines = rm.episode(ident, sources[ident])
        starts = sorted((a, w) for a, _, w in lines)
        tracks = json.loads((lr.ABSTAIN / ident / "nemotron3-diar.json").read_text(encoding="utf-8"))["tracks"]
        out.append((ident, "放送局", stable, lr._assign(stable, tracks), grid, people, starts))
    return out


def rows(material):
    import relabel_eval as ev
    for ident, corpus, stable, speakers, grid, people, starts in material:
        records = []
        for row, speaker in zip(stable, speakers):
            d = speaker.to_dict()
            tracks = d.get("tracks") or []
            if not tracks:
                continue
            starters = {w for a, w in starts if float(row["start"]) - 0.3 <= a < float(row["end"])}
            cover, _ = ev.span_cover(row, grid)
            truth = next(iter(starters)) if len(starters) == 1 else (people[int(cover.argmax())] if cover.any() else None)
            if truth is None:
                continue
            shares = sorted(d.get("coverage", {}).values(), reverse=True)
            leader = shares[0] if shares else 0.0
            runner = shares[1] if len(shares) > 1 else 0.0
            records.append(dict(track=tracks[0], truth=truth, leader=leader, margin=(leader - runner) / leader if leader else 0.0,
                                 centre=(float(row["start"]) + float(row["end"])) / 2,
                                 seconds=float(row["end"]) - float(row["start"])))
        # Owners from the rows themselves (row-level, time-weighted), global and local.
        totals = defaultdict(Counter)
        for r in records:
            totals[r["track"]][r["truth"]] += r["seconds"]
        owner = {k: c.most_common(1)[0][0] for k, c in totals.items()}
        purity = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in totals.items()}
        centres = np.array([r["centre"] for r in records])
        for r in records:
            near = [records[j] for j in np.flatnonzero(np.abs(centres - r["centre"]) <= WINDOW) if records[j]["track"] == r["track"]]
            local = Counter()
            for n in near:
                local[n["truth"]] += n["seconds"]
            r["wrong"] = owner[r["track"]] != r["truth"]
            r["kind"] = ("merged" if purity[r["track"]] < MERGED else
                         "switched" if local and local.most_common(1)[0][0] != owner[r["track"]] else "ordinary")
            r["corpus"] = corpus
            yield r


def main():
    voices = voices_material()  # ⚠ before anything loads onetonton (it rebinds refined_multi.sources)
    w2 = [(i, c, s, sp, g, p, st) for i, c, s, sp, _, g, p, st, _ in lr.fill_material()]
    for name, material in (("hina + 姐姐咚咚（W2 31 份）", w2), ("放送局 9 集", voices)):
        records = list(rows(material))
        errors = Counter(r["kind"] for r in records if r["wrong"])
        ordinary = [r for r in records if r["kind"] == "ordinary"]
        wrong = sum(r["wrong"] for r in ordinary)
        print(f"\n== {name}: {len(records)} labelled rows with truth, wrong {sum(r['wrong'] for r in records)} "
              f"({sum(r['wrong'] for r in records) / len(records):.1%}); errors by kind {dict(errors)}")
        print(f"   ordinary rows {len(ordinary)}, wrong {wrong} ({wrong / len(ordinary):.1%}) -- the cuts below are on these")
        for label, key, cuts in (("margin <=", "margin", (0.15, 0.3, 0.5)), ("leader share <", "leader", (0.5, 0.7, 0.9))):
            for cut in cuts:
                flagged = [r for r in ordinary if (r[key] <= cut if key == "margin" else r[key] < cut)]
                rest = [r for r in ordinary if r not in flagged]
                caught = sum(r["wrong"] for r in flagged)
                print(f"   {label} {cut:<4}: flags {len(flagged) / len(ordinary):5.1%} of rows, error rate {caught / max(1, len(flagged)):5.1%} "
                      f"(rest {sum(r['wrong'] for r in rest) / max(1, len(rest)):4.1%}), catches {caught}/{wrong} = {caught / max(1, wrong):5.1%} of errors")


if __name__ == "__main__":
    main()
