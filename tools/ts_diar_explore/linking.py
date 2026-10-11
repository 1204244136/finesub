"""Which labels of two windows are the same person, from the seconds they co-speak where the windows overlap.

One definition for both users: `episode.stitch` (greedy, the arm as the API gives it) and
`merge_windows.py` (greedy or Hungarian, as one of several merging methods). Intervals are clipped to the
overlap region before summing, so a turn that began before the region still counts for the part inside it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def overlap(x: dict, y: dict) -> float:
    """Seconds two timed units share."""
    return max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))


def co_seconds(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(x1, y1) - max(x0, y0)) for x0, x1 in a for y0, y1 in b)


def by_label(units: list[dict], lo: float = float("-inf"), hi: float = float("inf")) -> dict[str, list[tuple[float, float]]]:
    """label -> its intervals clipped to [lo, hi); units without a label are skipped."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for u in units:
        a, b = max(u["start"], lo), min(u["end"], hi)
        if u.get("speaker") is not None and b > a:
            out[u["speaker"]].append((a, b))
    return out


def link(prev: dict[str, list], cur: dict[str, list], *, hungarian: bool = False, min_link: float = 0.0) -> list[tuple[str, str]]:
    """(prev label, cur label) pairs, one-to-one, by co-speaking seconds: greedy largest-first, or Hungarian;
    a pair needs more than `min_link` seconds (Hungarian: at least)."""
    p, c = list(prev), list(cur)
    if not p or not c:
        return []
    m = np.array([[co_seconds(prev[x], cur[y]) for y in c] for x in p])
    if hungarian:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-m)
        return [(p[r], c[q]) for r, q in zip(rows, cols) if m[r, q] >= max(min_link, 1e-9)]
    out, used_p, used_c = [], set(), set()
    for r, q in sorted(np.ndindex(m.shape), key=lambda rc: -m[rc]):
        if m[r, q] > min_link and r not in used_p and q not in used_c:
            out.append((p[r], c[q]))
            used_p.add(r)
            used_c.add(q)
    return out
