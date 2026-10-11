"""Overlapped speech as its own dimension (owner 2026-09-24), on AliMeeting Eval: far-field Chinese meetings,
2-4 people, per-speaker transcripts with times (TextGrid) -- the one set here with TEXT under overlap.

(a) attribution   `score.speaker` on the reference intervals, overall and on overlapped ones (another person's
                  interval shares >= 0.3 s)
(b) detection     10 ms frames: overlap recall = frames where the reference has >= 2 speakers and the arm has
                  >= 2 labels active; false overlap = arm >= 2 where the reference has <= 1; miss = reference
                  speech where the arm has no label
(c) text          reference intervals split by class (>= 50 % of the interval overlapped by someone else vs
                  clean); per class, character recall = multiset intersection of reference and hypothesis
                  characters (tokens whose midpoint falls inside the stretch), taken per connected stretch
                  of the class's intervals and summed, over reference characters. Order-free on purpose: under overlap the two speakers' words interleave and no
                  single reference order is right. `hyp/ref` guards against a model that recalls by babbling.

⚠ AliMeeting is Chinese far-field; findings 2.6 already shows Nemotron-3 detects less of it than DiariZen.

Readings: `report.py overlap <items> [--span 0 1490]` (Pro stops at 1505 s on R8007).
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from . import score as S
from .common import ALI

FRAME = 0.01


def reference(item: str) -> list[dict]:
    text = (ALI / "textgrid" / f"{item[4:]}.TextGrid").read_text(encoding="utf-8")
    out = []
    for tier in re.split(r"\n\s*item \[\d+\]:", text)[1:]:
        name = re.search(r'name = "([^"]*)"', tier).group(1)
        for a, b, t in re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', tier):
            if t.strip():
                out.append({"start": float(a), "end": float(b), "speaker": name, "text": t.strip()})
    out.sort(key=lambda r: r["start"])
    for i, x in enumerate(out):
        others = [y for y in out[max(0, i - 30):i + 30] if y is not x and y["speaker"] != x["speaker"]]
        x["overlapped"] = any(S._overlap(x, y) >= 0.3 for y in others)
        spans = sorted((max(x["start"], y["start"]), min(x["end"], y["end"])) for y in others
                       if S._overlap(x, y) > 0)
        covered, reach = 0.0, x["start"]
        for a, b in spans:  # union, not sum: two backchannels over one stretch count once
            if b > reach:
                covered += b - max(a, reach)
                reach = b
        x["ovl_frac"] = covered / max(1e-6, x["end"] - x["start"])
    return out


def clip_arm(arm: dict, lo: float, hi: float) -> dict:
    inside = lambda u: lo <= (u["start"] + u["end"]) / 2 < hi  # noqa: E731
    return {**arm, "words": [w for w in arm["words"] if inside(w)] if arm.get("words") else arm.get("words"),
            "segments": [s for s in arm["segments"] if inside(s)]}


def detection(ref: list[dict], arm: dict) -> dict:
    end = max(max(r["end"] for r in ref), max((s["end"] for s in arm["segments"]), default=0)) + 1
    n = int(end / FRAME) + 1
    per_spk: dict[str, np.ndarray] = {}
    for r in ref:
        per_spk.setdefault(r["speaker"], np.zeros(n, bool))[int(r["start"] / FRAME):int(r["end"] / FRAME)] = True
    rc = sum(v.astype(int) for v in per_spk.values())
    per_lab: dict[str, np.ndarray] = {}
    for s in arm["segments"]:
        if s.get("speaker"):
            per_lab.setdefault(s["speaker"], np.zeros(n, bool))[int(s["start"] / FRAME):int(s["end"] / FRAME)] = True
    hc = sum(v.astype(int) for v in per_lab.values()) if per_lab else np.zeros(n, int)
    ro, ho = rc >= 2, hc >= 2
    return {"ref_overlap_share": ro.sum() / max(1, (rc >= 1).sum()),
            "overlap_recall": (ro & ho).sum() / max(1, ro.sum()),
            "false_overlap": (ho & ~ro).sum() / max(1, ho.sum()) if ho.any() else float("nan"),
            "miss": ((rc >= 1) & (hc == 0)).sum() / max(1, (rc >= 1).sum())}


def islands(rows: list[dict]) -> list[tuple[float, float, list[dict]]]:
    """Connected stretches of the given intervals (overlapping ones join)."""
    out: list[tuple[float, float, list[dict]]] = []
    for r in sorted(rows, key=lambda r: r["start"]):
        if out and r["start"] < out[-1][1]:
            a, b, rs = out[-1]
            out[-1] = (a, max(b, r["end"]), rs + [r])
        else:
            out.append((r["start"], r["end"], [r]))
    return out


def text_recall(ref: list[dict], arm: dict) -> dict:
    """Per class, character multiset recall computed island by island: a hypothesis character only counts
    against reference characters of the same connected stretch (pooling a whole session would let a common
    character match anywhere)."""
    toks = S.tokens(arm)
    mids = np.array([(t["start"] + t["end"]) / 2 for t in toks]) if toks else np.array([])
    out = {}
    for cls, keep in (("clean", lambda r: r["ovl_frac"] < 0.5), ("overlap", lambda r: r["ovl_frac"] >= 0.5)):
        hit = total = hyp = 0
        for a, b, rows in islands([r for r in ref if keep(r)]):
            rchars = Counter(S.norm("".join(r["text"] for r in rows)))
            inside = (mids >= a) & (mids < b)
            hchars = Counter(S.norm("".join(t["text"] for t, k in zip(toks, inside) if k)))
            hit += sum((rchars & hchars).values())
            total += sum(rchars.values())
            hyp += sum(hchars.values())
        out[cls] = {"recall": hit / max(1, total), "hyp_ref": hyp / max(1, total), "ref_chars": total}
    return out
