"""Production's assignment by leader share: where rows sit, where the wrong labels sit (owner 2026-10-05).

`assign.AssignmentParameters`: a track must cover `min_share` = 0.20 of a row to be named at all; below
that the row is unlabelled. This tabulates every row with a truth (W2 31 files + 放送局 9) by the
leading track's share -- 0 for rows no track reaches -- and counts, per band, the rows, the wrong
labels, and (for unlabelled rows) the misses. Truth and track owners as in `retrack_lowconf.py`;
wrong labels from structurally broken tracks (merged / switched) are counted apart.

    python tools/diarization_explore/retrack_threshold.py
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import retrack_lowconf as rl  # noqa: E402

BANDS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def all_rows(material):
    import relabel_eval as ev
    for ident, corpus, stable, speakers, grid, people, starts in material:
        records = []
        for row, speaker in zip(stable, speakers):
            d = speaker.to_dict()
            starters = {w for a, w in starts if float(row["start"]) - 0.3 <= a < float(row["end"])}
            cover, _ = ev.span_cover(row, grid)
            truth = next(iter(starters)) if len(starters) == 1 else (people[int(cover.argmax())] if cover.any() else None)
            if truth is None:
                continue
            shares = sorted(d.get("coverage", {}).values(), reverse=True)
            tracks = d.get("tracks") or []
            records.append(dict(track=tracks[0] if tracks else None, truth=truth, leader=shares[0] if shares else 0.0,
                                seconds=float(row["end"]) - float(row["start"]),
                                centre=(float(row["start"]) + float(row["end"])) / 2))
        labelled = [r for r in records if r["track"] is not None]
        totals = defaultdict(Counter)
        for r in labelled:
            totals[r["track"]][r["truth"]] += r["seconds"]
        owner = {k: c.most_common(1)[0][0] for k, c in totals.items()}
        purity = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in totals.items()}
        for r in records:
            if r["track"] is None:
                r["state"] = "unlabelled"
            elif owner[r["track"]] == r["truth"]:
                r["state"] = "right"
            else:
                r["state"] = "wrong-merged" if purity[r["track"]] < rl.MERGED else "wrong"
            yield r


def main():
    voices = rl.voices_material()  # ⚠ before onetonton loads
    w2 = [(i, c, s, sp, g, p, st) for i, c, s, sp, _, g, p, st, _ in lr.fill_material()]
    for name, material in (("W2 31 份（hina 16 + 姐姐咚咚 15）", w2), ("放送局 9 集", voices)):
        rows = list(all_rows(material))
        n = len(rows)
        states = Counter(r["state"] for r in rows)
        wrong_all = states["wrong"] + states["wrong-merged"]
        print(f"\n== {name}: {n} rows with a truth | labelled {(n - states['unlabelled']) / n:.1%} | "
              f"wrong labels {wrong_all} ({wrong_all / n:.1%} of all rows; {states['wrong-merged']} in merged tracks) | "
              f"unlabelled {states['unlabelled']} ({states['unlabelled'] / n:.1%})")
        print(f"   {'leader share':<14}{'rows':>8}{'cum rows':>10}{'wrong in band':>15}{'cum share of wrong':>20}{'(excl. merged)':>16}")
        cum_rows = cum_wrong = cum_ordinary = 0
        ordinary = states["wrong"]
        for lo, hi in zip(BANDS, BANDS[1:]):
            band = [r for r in rows if lo <= r["leader"] < hi]
            wrong = sum(r["state"].startswith("wrong") for r in band)
            cum_rows += len(band)
            cum_wrong += wrong
            cum_ordinary += sum(r["state"] == "wrong" for r in band)
            labelled = [r for r in band if r["state"] != "unlabelled"]
            rate = f"{wrong / len(labelled):.0%} of {len(labelled)}" if labelled else "unlabelled"
            print(f"   [{lo:.1f}, {min(hi, 1.0):.1f}){'':<3}{len(band) / n:>8.1%}{cum_rows / n:>10.1%}{rate:>15}"
                  f"{cum_wrong / max(1, wrong_all):>20.1%}{cum_ordinary / max(1, ordinary):>16.1%}")


if __name__ == "__main__":
    main()
