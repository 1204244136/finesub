"""Timestamp precision, on two references:

T1  human word timing: `data/disfluency-gold/BV1cqLR6hEp3/BV1cqLR6hEp3-fixed.srt`, the hand-corrected word
    cues behind `tools/wt_refine_validation/disfluency_gold.json` (8.9 min). Scored at every word and at
    phrase edges (a word with >= PHRASE_GAP of silence before it = onset / after it = offset) -- the edges
    are where a subtitle starts and ends, and where the reference is least ambiguous.
T2  fansub line edges on the 星穹铁道 episodes (`score.reference`): the fansub's timing conventions shift
    every arm alike, so read the arms against each other and against production.

Matching is one-to-one: reference and hypothesis edges pair greedily by distance within MAX_D, so an arm
that emits more edges (mai2 times every character) cannot let one edge answer several reference edges.
Phrase edges are matched against the hypothesis's own phrase edges (same >= PHRASE_GAP rule on its tokens).
Reported: signed median (+ = hypothesis late), |d| p50 / p90 and share within 0.1 / 0.3 s over matched
reference edges; `none` = reference edges left unmatched; `prec` = hypothesis edges (inside the reference's
time range) that found a reference edge. Tokens are words where the arm times them, else its segments.

Readings: `report.py timing t1` / `report.py timing t2 <items>`.
"""

from __future__ import annotations

import bisect

import numpy as np

from . import score as S
from .common import MAIN

MAX_D = 1.0
PHRASE_GAP = 0.3
FIXED = MAIN / "data" / "disfluency-gold" / "BV1cqLR6hEp3" / "BV1cqLR6hEp3-fixed.srt"


def srt(path) -> list[dict]:
    blocks = path.read_text(encoding="utf-8-sig").strip().split("\n\n")
    out = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        a, e = (sum(float(x) * m for x, m in zip(t.strip().replace(",", ".").split(":"), (3600, 60, 1)))
                for t in lines[1].split("-->"))
        out.append({"start": a, "end": e, "text": "".join(lines[2:])})
    return sorted(out, key=lambda x: x["start"])


def edges(units: list[dict], *, phrase: bool) -> tuple[list[float], list[float]]:
    """Onsets and offsets of time-sorted units; `phrase`: only those with >= PHRASE_GAP of silence beside them."""
    units = sorted(units, key=lambda u: u["start"])
    if not phrase:
        return [u["start"] for u in units], [u["end"] for u in units]
    on = [u["start"] for i, u in enumerate(units) if i == 0 or u["start"] - units[i - 1]["end"] >= PHRASE_GAP]
    off = [u["end"] for i, u in enumerate(units) if i + 1 == len(units) or units[i + 1]["start"] - u["end"] >= PHRASE_GAP]
    return on, off


def stats(ref_pts: list[float], hyp_pts: list[float]) -> dict:
    hyp = sorted(hyp_pts)
    cand = []
    for i, t in enumerate(ref_pts):
        lo, hi = bisect.bisect_left(hyp, t - MAX_D), bisect.bisect_right(hyp, t + MAX_D)
        cand += [(abs(hyp[j] - t), i, j) for j in range(lo, hi)]
    used_r, used_h, ds = set(), set(), []
    for _, i, j in sorted(cand):
        if i not in used_r and j not in used_h:
            used_r.add(i)
            used_h.add(j)
            ds.append(hyp[j] - ref_pts[i])
    a = np.abs(ds)
    span = (min(ref_pts) - MAX_D, max(ref_pts) + MAX_D)
    in_span = sum(1 for h in hyp if span[0] <= h <= span[1])
    return {"n": len(ref_pts), "signed": float(np.median(ds)), "p50": float(np.median(a)), "p90": float(np.quantile(a, .9)),
            "w01": float((a <= 0.1).mean()), "w03": float((a <= 0.3).mean()),
            "unmatched": 1 - len(ds) / len(ref_pts), "prec": len(used_h) / max(1, in_span)}


def hyp_edges(arm: dict, *, phrase: bool = False) -> tuple[list[float], list[float]]:
    return edges([t for t in S.tokens(arm) if S.norm(t.get("text", ""))], phrase=phrase)


def row(label: str, which: str, s: dict) -> str:
    return (f"{label:<18} {which:<7} n={s['n']:5d}  signed {s['signed']:+.3f}  |d| p50 {s['p50']:.3f} p90 {s['p90']:.3f}  "
            f"<=0.1s {s['w01']:5.1%}  <=0.3s {s['w03']:5.1%}  none {s['unmatched']:5.1%}  prec {s['prec']:5.1%}")
