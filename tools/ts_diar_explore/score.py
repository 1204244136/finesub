"""One ruler for every arm: the fansub's Japanese lines (`-日文` styles of the 星穹铁道 bilingual axes).

Every arm (`arms.py`: the models, production, the reference timelines) is reduced to text-with-time and
labelled intervals, then scored against the same reference lines, so production is recomputed with this code
rather than quoted from `diarization-findings.md` (those numbers use a different ruler and do not compare).

Text   CER over the window: reference lines (by midpoint) vs hypothesis tokens (by midpoint), NFKC, with
       punctuation, symbols and whitespace removed. ⚠ The reference is a fansub text, not a verbatim
       transcript (fillers dropped, some rephrasing): read CER as a ranking, not an absolute error rate.
Speaker per reference line with a person, the hypothesis label = the label overlapping the line longest.
       `owner`  each label -> its majority person over the file; line correct if owner(label) == person
                (the `fixed` idea on reference lines: forgives over-splitting)
       `b3`     B-cubed F1 of labels vs persons over lines (punishes over-splitting and merging)
       `unheard` reference lines no label overlaps at all -- counted as misses (own singleton label,
                never owned), so every arm is scored on the same lines
Lines  edge P/R/F1 of hypothesis subtitle lines against the fansub's lines (`segmentation`).

Readings: `report.py text | lines | speakers`.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from .common import source_audio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diarization_explore"))
import refined_multi as rm  # noqa: E402

_JA_STYLE = re.compile(r"-日文\d*(\s*-\s*复制)?$")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def reference(item: str) -> list[dict]:
    """Japanese lines: start, end, person (None for non-speaker styles such as 全员), text.

    Files without Japanese lines (the 原神 radio axes are Chinese-only) fall back to the translated lines:
    their timing and person are usable for the speaker score, their text is not Japanese -- `has_ja`
    tells the caller to skip CER."""
    ass = source_audio(item).with_suffix(".ass")
    rows = [line.split(",", 9) for line in ass.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            if line.startswith("Dialogue:")]
    has_ja = any(_JA_STYLE.search(parts[3].strip()) for parts in rows)
    out = []
    for parts in rows:
        style = parts[3].strip()
        if has_ja and not _JA_STYLE.search(style):
            continue
        text = re.sub(r"\{[^}]*\}", "", parts[9]).replace("\\N", " ").strip()
        a, b = rm.seconds(parts[1]), rm.seconds(parts[2])
        if b > a and text:
            out.append({"start": a, "end": b, "speaker": rm.speaker(style), "text": text if has_ja else ""})
    out.sort(key=lambda r: r["start"])
    for i, x in enumerate(out):  # overlapped: another person's line shares >= 0.3 s with this one
        x["overlapped"] = any(y["speaker"] and y["speaker"] != x["speaker"] and _overlap(x, y) >= 0.3
                              for y in out[max(0, i - 15):i + 15] if y is not x)
    return out


def tokens(arm: dict) -> list[dict]:
    """Timed text units: words when the arm has them, else its segments."""
    return arm["words"] if arm.get("words") else arm["segments"]


def labelled(arm: dict) -> list[dict]:
    """Intervals that carry a speaker label (segments for every arm; production: DiariZen tracks)."""
    return [s for s in arm["segments"] if s.get("speaker")]


def _inside(x: dict, a: float, b: float) -> bool:
    mid = (x["start"] + x["end"]) / 2
    return a <= mid < b


def cer(ref: list[dict], arm: dict, a: float, b: float) -> dict:
    r = norm("".join(x["text"] for x in ref if _inside(x, a, b)))
    h = norm("".join(x["text"] for x in tokens(arm) if _inside(x, a, b)))
    counts = Counter(op.tag for op in Levenshtein.editops(r, h))
    return {"cer": sum(counts.values()) / max(1, len(r)), "ref_chars": len(r), "hyp_chars": len(h),
            "sub": counts["replace"], "del": counts["delete"], "ins": counts["insert"]}


def _overlap(x: dict, y: dict) -> float:
    return max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))


def line_overlaps(ref: list[dict], arm: dict, a: float, b: float) -> tuple[list[dict], list[Counter]]:
    """Reference lines in [a, b) with a speaker, and per line the seconds each label overlaps it (in the order
    the labels' segments first reach the line)."""
    lab = sorted(labelled(arm), key=lambda s: s["start"])
    lines = [x for x in ref if x["speaker"] and _inside(x, a, b)]
    out, j = [], 0
    for line in lines:
        while j < len(lab) and lab[j]["end"] < line["start"] - 30:
            j += 1
        best = Counter()
        for s in lab[j:]:
            if s["start"] > line["end"]:
                break
            ov = _overlap(line, s)
            if ov > 0:
                best[s["speaker"]] += ov
        out.append(best)
    return lines, out


def pairs_from(lines: list[dict], overlaps: list[Counter], group: dict | None = None) -> tuple[list[tuple], int]:
    """Per line (person, label, seconds, overlapped): the label -- or, with `group`, the group of labels -- with
    the most overlap, or a singleton `∅k` when nothing overlaps; plus the count of such unheard lines. Grouping
    sums before choosing, as relabelling the arm and scoring it would."""
    pairs, unheard = [], 0
    for line, ov in zip(lines, overlaps):
        if group:
            summed = Counter()
            for label, sec in ov.items():
                summed[group.get(label, label)] += sec
            ov = summed
        if not ov:
            # Unheard (no label overlaps the line) is a miss, not a skip: dropping it would score arms on
            # different line sets and spare exactly the hard lines. It becomes its own singleton label, so
            # B-cubed charges it against the person's recall, and it never counts as correctly owned.
            unheard += 1
            pairs.append((line["speaker"], f"∅{unheard}", line["end"] - line["start"], line.get("overlapped", False)))
            continue
        pairs.append((line["speaker"], ov.most_common(1)[0][0], line["end"] - line["start"], line.get("overlapped", False)))
    return pairs, unheard


def b3(pairs: list[tuple]) -> tuple[float, float, float]:
    """B-cubed over lines (unweighted), (precision, recall, F1): precision within label clusters, recall within
    person clusters. ⚠ An unheard line's singleton label has precision 1, and a person with one line recall 1:
    B-cubed undercharges misses, so read it with the unheard count and `owner` (which scores them wrong)."""
    n = len(pairs)
    lc = Counter(p[1] for p in pairs)
    pc = Counter(p[0] for p in pairs)
    both = Counter((p[0], p[1]) for p in pairs)
    prec = sum(both[(p[0], p[1])] / lc[p[1]] for p in pairs) / n
    rec = sum(both[(p[0], p[1])] / pc[p[0]] for p in pairs) / n
    return prec, rec, 2 * prec * rec / (prec + rec)


def speaker(ref: list[dict], arm: dict, a: float, b: float) -> dict:
    lines, overlaps = line_overlaps(ref, arm, a, b)
    pairs, unheard = pairs_from(lines, overlaps)
    if not pairs:
        return {"lines": len(lines), "unheard": unheard}
    by_label = defaultdict(Counter)
    for person, label, d, _ in pairs:
        by_label[label][person] += d
    owner = {k: (None if k.startswith("∅") else c.most_common(1)[0][0]) for k, c in by_label.items()}
    total = sum(d for _, _, d, _ in pairs)
    owner_acc = sum(d for p, l, d, _ in pairs if owner[l] == p) / total
    ov = [(p, l) for p, l, _, o in pairs if o]
    overlap_acc = sum(owner[l] == p for p, l in ov) / len(ov) if ov else float("nan")
    prec, rec, f1 = b3(pairs)
    labels = sum(1 for l in by_label if not l.startswith("∅"))
    return {"lines": len(lines), "unheard": unheard, "labels": labels, "persons": len({p for p, *_ in pairs}),
            "owner": owner_acc, "overlap_lines": len(ov), "overlap_owner": overlap_acc, "b3_p": prec, "b3_r": rec, "b3_f1": f1}


def _edges(lines: list[dict], a: float, b: float, merge: float = 0.25, starts_only: bool = False) -> list[float]:
    """Line starts and ends inside the window; an end and the next start closer than `merge` are one edge.
    `starts_only`: the cut decision without the tail -- for arms whose word END times are not real."""
    keys = ("start",) if starts_only else ("start", "end")
    pts = sorted(x[k] for x in lines for k in keys if a <= x[k] < b)
    out: list[float] = []
    for t in pts:
        if out and t - out[-1] < merge:
            out[-1] = (out[-1] + t) / 2
        else:
            out.append(t)
    return out


def _f1(r: list[float], h: list[float], tau: float) -> tuple[float, float, float]:
    used, hit = set(), 0
    for t in h:
        best = min((j for j in range(len(r)) if j not in used and abs(r[j] - t) <= tau),
                   key=lambda j: abs(r[j] - t), default=None)
        if best is not None:
            used.add(best)
            hit += 1
    p, rc = hit / max(1, len(h)), hit / max(1, len(r))
    return (2 * p * rc / (p + rc) if p + rc else 0.0), p, rc


def segmentation(ref: list[dict], lines: list[dict], a: float, b: float, taus=(0.3, 0.6)) -> dict:
    """Edge P/R/F1 of hypothesis subtitle lines against the fansub's lines, plus the shape of the lines."""
    r, h = _edges(ref, a, b), _edges(lines, a, b)
    out: dict = {"ref_edges": len(r), "hyp_edges": len(h)}
    out["start_f1@0.3"] = _f1(_edges(ref, a, b, starts_only=True), _edges(lines, a, b, starts_only=True), 0.3)[0]
    for tau in taus:
        out[f"f1@{tau}"], out[f"p@{tau}"], out[f"r@{tau}"] = _f1(r, h, tau)
    inside = [x for x in lines if _inside(x, a, b)]
    durs = sorted(x["end"] - x["start"] for x in inside)
    chars = sorted(len(norm(x.get("text", ""))) for x in inside)
    if durs:
        out.update(lines=len(inside), dur_median=durs[len(durs) // 2], chars_median=chars[len(chars) // 2],
                   over_8s=sum(d > 8 for d in durs) / len(durs), over_36c=sum(c > 36 for c in chars) / len(chars))
    return out
