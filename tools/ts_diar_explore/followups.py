"""The two follow-up experiments of the survey (owner 2026-09-25), on data already collected -- no API calls.

text-source   mai2 as production's text source. Production's subtitle lines (cuts, times) stay as they are;
              each mai2 word goes to the production line its midpoint falls in (in a gap: the nearest line),
              and the line's text becomes those words. Segmentation and timing are production's by
              construction, so the only question is whether words land in the right line. Measured as
              line-level CER against the fansub text on production's own line spans, for production's
              Whisper text and for the hybrid, plus the share of mai2 characters that fell in a gap.
              (The alternative -- mai2's words through production's DP -- is `gold_runs.py` `mai2/sbd+*`.)

pro-gate      MOSS Pro for speakers only when the episode is small. Gate: persons estimated from a reference
              timeline as the tracks holding >= GATE_SHARE of its labelled speech; Pro when the estimate is
              <= GATE_MAX, else the reference. A coverage check falls back to the reference when Pro's
              segments cover less than COVER_MIN of the reference's speech (Pro stops silently on dense
              meetings). Printed: gate accuracy on every item with a known cast, then the gated system's B-cubed
              next to Pro alone and the references on the items Pro exists for.

    python -m tools.ts_diar_explore.followups text-source hina-72 hina-70 ...
    python -m tools.ts_diar_explore.followups pro-gate hina-72 hina-70 ... [--reference diarizen]
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter

import numpy as np
from rapidfuzz.distance import Levenshtein

from . import arms as A
from . import score as S
from .report import reference

GATE_SHARE = 0.03
GATE_MAX = 4
COVER_MIN = 0.9


def assign(lines: list[dict], words: list[dict]) -> tuple[list[str], int]:
    """Per line, the text of the words assigned to it; and how many characters fell in a gap."""
    starts = [l["start"] for l in lines]
    texts, gap_chars = [""] * len(lines), 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        i = bisect.bisect_right(starts, mid) - 1
        if 0 <= i < len(lines) and mid < lines[i]["end"]:
            k = i
        else:
            gap_chars += len(S.norm(w["text"]))
            cand = [j for j in (i, i + 1) if 0 <= j < len(lines)]
            k = min(cand, key=lambda j: min(abs(mid - lines[j]["start"]), abs(mid - lines[j]["end"])))
        texts[k] += w["text"]
    return texts, gap_chars


def line_cer(lines: list[dict], texts: list[str], ref: list[dict]) -> float:
    """Edits over reference characters, line by line: each production line against the fansub text whose
    midpoints fall in its span (the same grouping for every arm, so the grouping noise cancels in a comparison)."""
    mids = sorted(((r["start"] + r["end"]) / 2, S.norm(r["text"])) for r in ref if S.norm(r["text"]))
    keys = [m for m, _ in mids]
    edits = chars = 0
    for line, text in zip(lines, texts):
        a, b = bisect.bisect_left(keys, line["start"]), bisect.bisect_left(keys, line["end"])
        r = "".join(t for _, t in mids[a:b])
        edits += Levenshtein.distance(r, S.norm(text))
        chars += len(r)
    return edits / max(1, chars)


def cmd_text_source(items: list[str]) -> None:
    print(f"{'item':<9} {'episode CER prod / mai2':>24} {'line CER prod / hybrid':>24} {'mai2 chars in gaps':>19}")
    diffs = []
    for item in items:
        if not (A.exists(item, "prod") and A.exists(item, "mai2-diar")):
            continue
        ref, prod, mai2 = S.reference(item), A.load(item, "prod"), A.load(item, "mai2-diar")
        lines = prod["lines"]
        prod_texts, _ = assign(lines, prod["words"])
        hyb_texts, gap = assign(lines, mai2["words"])
        ep = S.cer(ref, prod, 0, 1e9)["cer"], S.cer(ref, mai2, 0, 1e9)["cer"]
        lc = line_cer(lines, prod_texts, ref), line_cer(lines, hyb_texts, ref)
        total = sum(len(S.norm(w["text"])) for w in mai2["words"])
        diffs.append((ep[1] - ep[0], lc[1] - lc[0]))
        print(f"{item:<9} {ep[0]:11.3f} / {ep[1]:.3f} {lc[0]:14.3f} / {lc[1]:.3f} {gap / max(1, total):18.1%}")
    d = np.array(diffs)
    print(f"mean change mai2 - prod: episode CER {d[:, 0].mean():+.3f}, line CER {d[:, 1].mean():+.3f}")


def estimate_persons(timeline: list[dict]) -> int:
    dur = Counter()
    for s in timeline:
        dur[s["speaker"]] += s["end"] - s["start"]
    total = sum(dur.values())
    return sum(1 for v in dur.values() if total and v / total >= GATE_SHARE)


def coverage(pro: list[dict], timeline: list[dict]) -> float:
    """Share of the reference timeline's speech seconds that some Pro segment covers (1 s slack)."""
    spans: list[list[float]] = []
    for a, b in sorted((s["start"] - 1, s["end"] + 1) for s in pro):  # union, so overlapping segments count once
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    covered = total = 0.0
    for s in timeline:
        total += s["end"] - s["start"]
        for a, b in spans:
            if b > s["start"] and a < s["end"]:
                covered += min(b, s["end"]) - max(a, s["start"])
    return covered / max(1e-9, total)


def cmd_pro_gate(items: list[str], ref_arm: str, span: tuple[float, float] | None) -> None:
    lo, hi = span or (0.0, 1e9)
    print(f"gate: tracks with >= {GATE_SHARE:.0%} of {ref_arm}'s speech, Pro when <= {GATE_MAX}; "
          f"fallback when Pro covers < {COVER_MIN:.0%}")
    print(f"{'item':<16} {'persons':>7} {'est':>4} {'gate':>5} {'cover':>6}   B3: {ref_arm:<9} {'Pro':>6} {'gated':>6}")
    rows = []
    for item in items:
        ref = reference(item)
        persons = len({r["speaker"] for r in ref if r["speaker"] and lo <= r["start"] < hi})
        timeline = [s for s in A.load(item, ref_arm)["segments"] if lo <= s["start"] < hi]
        est = estimate_persons(timeline)
        gate_pro = est <= GATE_MAX
        ref_b3 = S.speaker(ref, A.load(item, ref_arm), lo, hi)["b3_f1"]
        has_pro = A.exists(item, "mosspro", span=span) if span else A.exists(item, "mosspro")
        if not has_pro:
            print(f"{item:<16} {persons:7d} {est:4d} {'Pro' if gate_pro else 'ref':>5} {'':>6}   {ref_b3:9.3f}")
            continue
        pro = A.load(item, "mosspro", span=span) if span else A.load(item, "mosspro")
        cov = coverage(pro["segments"], timeline)
        pro_b3 = S.speaker(ref, pro, lo, hi)["b3_f1"]
        gated = pro_b3 if gate_pro and cov >= COVER_MIN else ref_b3
        rows.append((ref_b3, pro_b3, gated))
        print(f"{item:<16} {persons:7d} {est:4d} {'Pro' if gate_pro else 'ref':>5} {cov:6.1%}   {ref_b3:9.3f} {pro_b3:6.3f} {gated:6.3f}")
    if rows:
        r = np.array(rows)
        print(f"mean over {len(rows)} items with Pro: {ref_arm} {r[:, 0].mean():.3f}  Pro {r[:, 1].mean():.3f}  gated {r[:, 2].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("text-source")
    p.add_argument("items", nargs="+")
    p = sub.add_parser("pro-gate")
    p.add_argument("items", nargs="+")
    p.add_argument("--reference", default="diarizen", choices=["diarizen", "n3vbx", "n3"])
    p.add_argument("--span", nargs=2, type=float, metavar=("A", "B"))
    a = ap.parse_args()
    if a.cmd == "text-source":
        cmd_text_source(a.items)
    else:
        cmd_pro_gate(a.items, a.reference, tuple(a.span) if a.span else None)


if __name__ == "__main__":
    main()
