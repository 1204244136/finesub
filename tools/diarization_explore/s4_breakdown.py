"""Plan 5h.12: per character -- right / homophone / misheard / missed -- and who got the credit.

Reads what s4_projection / s4_live / s4_local already decoded; no model runs here.

  --spot-a     print spot-check A (60 reference sentences, fixed seed) for reading by hand
  --spot-b     print spot-check B (50 hypothesis segments, fixed seed)
  --score      the full breakdown

Reference side: every character of an overlapped reference sentence gets one status from
each source (the baseline words in its window; an S4 arm's text for the spans touching it),
and an arm's status is the best over baseline + that arm. Attribution is a separate axis,
per source, never pooled. Hypothesis side: every character an S4 arm wrote in a real span.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np

import s4_headroom as ref
import s4_live as live
import s4_llm as llm
import s4_projection as proj

try:  # homophones need it; without it they fold into "misheard" (plan 5h.12)
    from pypinyin import lazy_pinyin
except ImportError:  # pragma: no cover - depends on the environment
    lazy_pinyin = None
if ref.LANGUAGE != "zh":  # English stand-in code points, Japanese kana / kanji: pinyin means nothing
    lazy_pinyin = None

RANK = {"miss": 0, "misheard": 1, "homophone": 2, "right": 3}
DIGITS = "零一二三四五六七八九"
UNITS = ("", "十", "百", "千", "万")
NEAR = 1.0  # a span's references: those within +-1 s, as in 5h.3 / 5h.4
SEED = 20260911


def hanzi(number: str) -> str:
    if len(number) > 5 or (len(number) > 1 and number[0] == "0"):
        return "".join(DIGITS[int(d)] for d in number)
    out, zero = "", False
    for position, digit in zip(range(len(number) - 1, -1, -1), number):
        if digit == "0":
            zero = True
            continue
        if zero and out:
            out += "零"
        zero = False
        out += DIGITS[int(digit)] + UNITS[position]
    out = out or "零"
    return out[1:] if out.startswith("一十") else out


def normalize(text: str) -> str:
    return re.sub(r"\d+", lambda m: hanzi(m.group()), ref.normalize(text))


def keys(text: str) -> list[str]:
    return [lazy_pinyin(c)[0] if lazy_pinyin else c for c in text]


def blocks(r: str, h: str, masked: set[int]) -> list[tuple[int, int, int]]:
    """Monotonic matches of r in h: blocks of >=2, singles only between two such blocks."""
    rk = keys(r)
    hk = [object() if j in masked else key for j, key in enumerate(keys(h))]
    raw = [(b.a, b.b, b.size) for b in difflib.SequenceMatcher(None, rk, hk, autojunk=False).get_matching_blocks() if b.size]
    big = [n >= 2 for _, _, n in raw]
    return [b for index, b in enumerate(raw) if big[index] or (any(big[:index]) and any(big[index + 1:]))]


def align(h: str, refs: list[dict]) -> dict:
    """Greedy claiming of one hypothesis text by nearby references.

    Returns per reference id its statuses and, per status, the hypothesis index that
    produced it; and per hypothesis index what it turned out to be.
    """
    order = sorted(refs, key=lambda row: -sum(n for _, _, n in blocks(row["norm"], h, set())))
    claimed: dict[int, int] = {}  # h index -> ref id
    result = {"refs": {}, "near": {}}
    kept = {}
    for row in order:
        found = blocks(row["norm"], h, set(claimed))
        kept[row["id"]] = found
        for i, j, n in found:
            for t in range(n):
                claimed[j + t] = row["id"]
    for row in order:
        r = row["norm"]
        status, source = ["miss"] * len(r), [None] * len(r)
        found = kept[row["id"]]
        for i, j, n in found:
            for t in range(n):
                status[i + t] = "right" if r[i + t] == h[j + t] else "homophone"
                source[i + t] = j + t
        free = lambda j: 0 <= j < len(h) and j not in claimed  # noqa: E731

        def pair(r_gap, h_gap):
            # A pair in a gap that turns out to be the same character (or sound) is
            # that, not a mishearing: "房子租好了" / "房子床好" keeps its 好.
            for ri, hj in zip(r_gap, h_gap):
                same = r[ri] == h[hj]
                sound = not same and lazy_pinyin is not None and keys(r[ri]) == keys(h[hj])
                status[ri] = "right" if same else "homophone" if sound else "misheard"
                source[ri] = hj
                result["near"].setdefault(hj, row["id"])

        if found:
            first_i, first_j, _ = found[0]
            run, j = [], first_j - 1
            while free(j) and len(run) < first_i:
                run.append(j)
                j -= 1
            pair(list(range(first_i - 1, -1, -1)), run)
            for (i1, j1, n1), (i2, j2, _) in zip(found, found[1:]):
                pair(list(range(i1 + n1, i2)), [j for j in range(j1 + n1, j2) if free(j)])
            last_i, last_j, last_n = found[-1]
            run, j = [], last_j + last_n
            while free(j) and len(run) < len(r) - last_i - last_n:
                run.append(j)
                j += 1
            pair(list(range(last_i + last_n, len(r))), run)
        result["refs"][row["id"]] = (status, source)
    result["claimed"] = claimed
    return result


class Session:
    def __init__(self, name: str):
        from finesub.speech.diarization.stage import labels_for, read_timeline

        self.name = name
        self.rows = [dict(row, norm=normalize(row["text"])) for row in ref.score(name)]
        every = ref.reference(name)
        self.all_refs = [dict(row, id=f"x{i}", norm=normalize(row["text"])) for i, row in enumerate(every)]
        for row in self.rows:  # the scored rows are a subset of every row; share one id space
            row["id"] = f"r{row['id']}"
        self.refs = self.rows + [row for row in self.all_refs if len(row["norm"]) < ref.MIN_CHARS]
        payload = json.loads((proj.ALI / name / f"{name}-stable.json").read_text(encoding="utf-8"))
        timeline = read_timeline(proj.ALI / name / f"{name}-diar.json")
        segments = payload["segments"]
        self.labels = [tuple(s.extras.get("tracks", ())) for s in labels_for(segments, timeline)]
        # (time, char, segment index). Sorted by time with the reading order as the
        # tie-break -- every character of a word shares its time, and sorting on the
        # character itself would shuffle the word.
        ordered = []
        for index, segment in enumerate(segments):
            for word in segment.get("words") or ():
                for c in normalize(word["word"]):
                    ordered.append(((word["start"] + word["end"]) / 2, len(ordered), c, index))
        self.chars = [(t, c, seg) for t, _, c, seg in sorted(ordered)]
        self.tracks = proj.timeline(name)
        self.speaker_track = self._mapping()
        self.spans = [s for c in llm.chunks(name) for s in c["spans"]]
        optional = lambda file: json.loads((proj.ALI / name / file).read_text(encoding="utf-8")) if (proj.ALI / name / file).exists() else {}  # noqa: E731
        self.qwen = optional("qwen.json")  # absent on AMI (5h.20 decodes only the time-cut arms)
        self.live = optional("live.json")
        self.projection = {(round(r["start"], 2), round(r["end"], 2)): r
                           for r in json.loads((proj.ALI / name / "projection.json").read_text(encoding="utf-8"))}

    def _mapping(self) -> dict[str, int]:
        from scipy.optimize import linear_sum_assignment

        frames = int(max(t["end"] for t in self.tracks) / proj.FRAME) + 2
        active = proj.activity(self.tracks, frames)
        speakers = sorted({row["spk"] for row in self.all_refs})
        truth = np.zeros((frames, len(speakers)), dtype=bool)
        for row in self.all_refs:
            truth[int(row["start"] / proj.FRAME): int(np.ceil(row["end"] / proj.FRAME)), speakers.index(row["spk"])] = True
        single = truth.sum(axis=1) == 1
        table = np.array([[np.sum(single & truth[:, s] & active[:, k]) for k in range(active.shape[1])] for s in range(len(speakers))])
        rows, cols = linear_sum_assignment(-table)
        return {speakers[r]: int(c) for r, c in zip(rows, cols) if table[r, c] > 0}

    def near(self, start: float, end: float, pad: float) -> list[dict]:
        return [row for row in self.refs if row["start"] < end + pad and row["end"] > start - pad]

    def base(self, row: dict):
        window = [(c, seg) for t, c, seg in self.chars if row["start"] - ref.PAD <= t <= row["end"] + ref.PAD]
        h = "".join(c for c, _ in window)
        return h, [seg for _, seg in window], align(h, self.near(row["start"], row["end"], ref.PAD))

    def span_texts(self, span: dict) -> dict[str, list[tuple[object, str, str]]]:
        """arm -> [(track or None, text, anchor text)] for one span."""
        sid = f"{span['start']:.2f}-{span['end']:.2f}"
        # The baseline cut to the audio the Live / Qwen clips heard (span +-0.3 s): the
        # head-to-head column, where neither side is a union with the other.
        heard = "".join(c for t, c, _ in self.chars if span["start"] - live.SPAN_PAD <= t <= span["end"] + live.SPAN_PAD)
        out = {"base-span": [(None, heard, "")],
               "qwen-plain": [(None, normalize(self.qwen.get(f"{sid}/L0/0") or ""), "")]}
        for arm, results in (("qwen-anchored", self.qwen), ("live-anchored", self.live)):
            out[arm] = []
            for job, text in results.items():
                if job.startswith(f"{sid}/L1/"):
                    k = int(job.rsplit("/", 1)[1])
                    anchor = results.get(f"{sid}/A/{k}") or ""
                    out[arm].append((k, normalize(live.strip_anchor(text or "", anchor)), normalize(anchor)))
        record = self.projection.get((round(span["start"], 2), round(span["end"], 2)))
        if record:
            out["whisper-masked"] = [(int(k), normalize(proj.region_text(w, span["start"], span["end"])), "")
                                     for k, w in record["masked"].items()]
        return out


def deleak(text: str, anchor: str) -> str:
    """Drop every >=2-char stretch of text that the anchor's own transcript also has (5h.10)."""
    if not anchor:
        return text
    drop = set()
    for b in difflib.SequenceMatcher(None, text, anchor, autojunk=False).get_matching_blocks():
        if b.size >= 2:
            drop.update(range(b.a, b.a + b.size))
    return "".join(c for j, c in enumerate(text) if j not in drop)


ARMS = ("qwen-plain", "qwen-anchored", "live-anchored", "whisper-masked")


def collect(session: Session):
    """Everything the report needs for one session."""
    span_align = {}
    for span in session.spans:
        for arm, items in session.span_texts(span).items():
            for k, text, anchor in items:
                text = deleak(text, anchor)
                span_align[(span["start"], span["end"], arm, k)] = (text, align(text, session.near(span["start"], span["end"], NEAR)))
    base_status = {}
    for row in session.refs:
        h, segs, aligned = session.base(row)
        base_status[row["id"]] = (h, segs, aligned)
    return span_align, base_status


def pair_members(session: Session):
    regions = list(session.projection.values())
    for a, b in ref.pairs(session.rows):
        middle = (max(a["start"], b["start"]) + min(a["end"], b["end"])) / 2
        if not any(r["start"] <= middle <= r["end"] for r in regions):
            continue
        worse, better = sorted((a, b), key=lambda m: m["recall"])
        yield "worse", worse
        yield "better", better


def touching(session: Session, row: dict):
    return [s for s in session.spans if s["kind"] == "real" and s["start"] < row["end"] + ref.PAD and s["end"] > row["start"] - ref.PAD]


def by_span(span_align):
    index = defaultdict(list)
    for (s, e, arm, k), value in span_align.items():
        index[(s, e)].append((arm, k, value))
    return index


def statuses(session, index, base_status, row, arm=None):
    status = list(base_status[row["id"]][2]["refs"][row["id"]][0])
    if arm:
        for span in touching(session, row):
            for a, _, (_, aligned) in index[(span["start"], span["end"])]:
                if a == arm and row["id"] in aligned["refs"]:
                    for position, value in enumerate(aligned["refs"][row["id"]][0]):
                        if RANK[value] > RANK[status[position]]:
                            status[position] = value
    return status


def alone(session, index, row, arm):
    """One source on its own, best over the spans touching the row; no union with anything."""
    status = ["miss"] * len(row["norm"])
    for span in touching(session, row):
        for a, _, (_, aligned) in index[(span["start"], span["end"])]:
            if a == arm and row["id"] in aligned["refs"]:
                for position, value in enumerate(aligned["refs"][row["id"]][0]):
                    if RANK[value] > RANK[status[position]]:
                        status[position] = value
    return status


def coarse(value: str) -> str:
    return "right" if value == "right" else "miss" if value == "miss" else "wrong"


def credit(session, row, track_of_source, status, source) -> Counter:
    """Attribution of the characters one source got (right / homophone / misheard)."""
    out = Counter()
    own = session.speaker_track.get(row["spk"])
    for value, h_index in zip(status, source):
        if value == "miss":
            continue
        tracks = track_of_source(h_index)
        if own is None:
            out["no track for this speaker"] += 1
        elif tracks is None:
            out["unattributed"] += 1
        elif not tracks:
            out["unlabelled"] += 1
        elif own in tracks:
            out["right speaker"] += 1
            out["of which multi-label"] += len(tracks) > 1
        else:
            out["wrong speaker"] += 1
    return out


def fmt(counter: Counter, order) -> str:
    total = sum(counter[k] for k in order) or 1
    return "  ".join(f"{k} {counter[k] / total:6.1%}" for k in order) + f"  (n={total})"


def score():
    ref_side = defaultdict(Counter)       # (side, arm) -> status counts
    attribution = defaultdict(Counter)    # source -> credit counts
    hyp_side = defaultdict(Counter)       # arm -> hypothesis char kinds
    decoys = defaultdict(Counter)
    head = defaultdict(Counter)           # (side, source) -> status counts, each source alone
    moves = defaultdict(Counter)          # (side, arm) -> (baseline cut to the span, arm alone)
    for name in ref.SESSIONS:
        session = Session(name)
        span_align, base_status = collect(session)
        index = by_span(span_align)
        print(name, "speaker -> track", session.speaker_track, flush=True)
        for row in session.rows:
            if row["cover"] <= ref.CLEAN:
                ref_side[("clean", "base")].update(base_status[row["id"]][2]["refs"][row["id"]][0])
        for side, row in pair_members(session):
            ref_side[(side, "base")].update(statuses(session, index, base_status, row))
            for arm in ARMS:
                ref_side[(side, arm)].update(statuses(session, index, base_status, row, arm))
            cut = alone(session, index, row, "base-span")
            head[(side, "base-span")].update(cut)
            for arm in ARMS:
                mine = alone(session, index, row, arm)
                head[(side, arm)].update(mine)
                moves[(side, arm)].update(zip(map(coarse, cut), map(coarse, mine)))
            h, segs, aligned = base_status[row["id"]]
            status, source = aligned["refs"][row["id"]]
            attribution["base"] += credit(session, row, lambda j: session.labels[segs[j]], status, source)
            for span in touching(session, row):
                for arm, k, (_, al) in index[(span["start"], span["end"])]:
                    if arm != "base-span" and row["id"] in al["refs"]:
                        st, src = al["refs"][row["id"]]
                        attribution[arm] += credit(session, row, lambda j, k=k: None if k is None else (k,), st, src)
        by_id = {row["id"]: row for row in session.refs}
        for (s, e, arm, k), (text, al) in span_align.items():
            if arm == "base-span":
                continue
            kind = next(sp["kind"] for sp in session.spans if (sp["start"], sp["end"]) == (s, e))
            for j in range(len(text)):
                if j in al["claimed"]:
                    rid = al["claimed"][j]
                    position = al["refs"][rid][1].index(j)
                    was = base_status[rid][2]["refs"][rid][0][position]
                    label = "repeat" if was == "right" else "new"
                    if k is not None:
                        owner = session.speaker_track.get(by_id[rid]["spk"])
                        label += " (own)" if owner == k else " (someone else's)"
                elif j in al["near"]:
                    label = "near-miss"
                else:
                    label = "nothing near"
                (decoys if kind == "decoy" else hyp_side)[arm][label] += 1
    print("\nREFERENCE SIDE (characters of overlapped reference sentences; arms = base + that arm)")
    homophone = "" if lazy_pinyin else "  [no pypinyin: homophones counted as misheard]"
    print(homophone)
    order = ("right", "homophone", "misheard", "miss")
    for side in ("clean", "worse", "better"):
        for arm in ("base",) + (ARMS if side != "clean" else ()):
            if (side, arm) in ref_side:
                print(f"  {side:<6} {arm:<15} {fmt(ref_side[(side, arm)], order)}")
    print("\nATTRIBUTION (characters a source got, per source)")
    for source, counter in attribution.items():
        print(f"  {source:<15} " + "  ".join(f"{k} {v}" for k, v in counter.most_common()))
    print("\nHYPOTHESIS SIDE (characters each arm wrote in real spans)")
    for arm, counter in hyp_side.items():
        print(f"  {arm:<15} " + "  ".join(f"{k} {v / sum(counter.values()):.1%}" for k, v in counter.most_common()) + f"  (n={sum(counter.values())})")
    print("DECOYS")
    for arm, counter in decoys.items():
        print(f"  {arm:<15} " + "  ".join(f"{k} {v}" for k, v in counter.most_common()))
    print("\nHEAD TO HEAD (same audio: the baseline cut to the spans vs each arm alone; no union)")
    for side in ("worse", "better"):
        for arm in ("base-span",) + ARMS:
            print(f"  {side:<6} {arm:<15} {fmt(head[(side, arm)], order)}")
    print("  moves, baseline-in-span -> arm alone (right / wrong = homophone+misheard / miss):")
    for side in ("worse", "better"):
        for arm in ARMS:
            m = moves[(side, arm)]
            print(f"  {side:<6} {arm:<15} " + "  ".join(f"{a}->{b} {m[(a, b)]}" for a in ("right", "wrong", "miss") for b in ("right", "wrong", "miss")))


FILLERS = set("嗯啊呃唉哎哦噢呀吧儿")


def posthoc():
    """Not pre-registered: how much of "missed" is fillers / erhua, and the sentence-level view."""
    chars = defaultdict(Counter)
    sentences = defaultdict(Counter)
    for name in ref.SESSIONS:
        session = Session(name)
        _, base_status = collect(session)
        members = [("clean", row) for row in session.rows if row["cover"] <= ref.CLEAN] + list(pair_members(session))
        for side, row in members:
            status = base_status[row["id"]][2]["refs"][row["id"]][0]
            for c, value in zip(row["norm"], status):
                chars[(side, "filler" if c in FILLERS else "word")][value] += 1
            heard = [v for c, v in zip(row["norm"], status) if c not in FILLERS]
            if heard:
                kind = ("all missed" if all(v == "miss" for v in heard)
                        else "some misheard" if "misheard" in heard else "right + missed only")
                sentences[side][kind] += 1
    for side in ("clean", "worse", "better"):
        for part in ("filler", "word"):
            counter = chars[(side, part)]
            print(f"  {side:<6} {part:<6} {fmt(counter, ('right', 'homophone', 'misheard', 'miss'))}")
        print(f"  {side:<6} sentences (fillers ignored): {fmt(sentences[side], ('all missed', 'some misheard', 'right + missed only'))}")


def clean_miss():
    """Not pre-registered: what the 12.9% missed on clean sentences is made of (owner's question)."""
    wide_pad, kinds, edges, total = 1.5, Counter(), Counter(), 0
    for name in ref.SESSIONS:
        session = Session(name)
        segments = json.loads((proj.ALI / name / f"{name}-stable.json").read_text(encoding="utf-8"))["segments"]
        for row in session.rows:
            if row["cover"] > ref.CLEAN:
                continue
            h, _, aligned = session.base(row)
            status = aligned["refs"][row["id"]][0]
            r = row["norm"]
            total += len(r)
            wide = "".join(c for t, c, _ in session.chars if row["start"] - wide_pad <= t <= row["end"] + wide_pad)
            i = 0
            while i < len(r):
                if status[i] != "miss":
                    i += 1
                    continue
                j = i
                while j < len(r) and status[j] == "miss":
                    j += 1
                run, n = r[i:j], j - i
                if all(c in FILLERS for c in run):
                    kind = "filler / erhua"
                elif run in (r[max(0, i - n):i], r[j:j + n]):
                    kind = "repeat (disfluency)"
                elif n >= 2 and run in h:
                    kind = "written, credited to a neighbour sentence"
                elif n >= 2 and run in wide:
                    kind = "written, just outside the window"
                elif i == 0 or j == len(r):
                    kind = "sentence edge, not found"
                    if n >= 4 and n < len(r):
                        # Characters carry no times: spread the sentence evenly over its span.
                        per = (row["end"] - row["start"]) / len(r)
                        lo, hi = row["start"] + i * per, row["start"] + j * per
                        if any(lo - 0.2 <= t <= hi + 0.2 for t, _, _ in session.chars):
                            edges["words there, something else written"] += n
                        elif any(s["start"] < hi and s["end"] > lo for s in segments):
                            edges["a segment covers it, no words"] += n
                        else:
                            edges["no segment at all (VAD / segmentation)"] += n
                else:
                    kind = f"middle, not found ({'1 char' if n == 1 else '2+ chars'})"
                kinds[kind] += n
                i = j
    print(f"clean characters {total}, missed {sum(kinds.values())}")
    for kind, n in kinds.most_common():
        print(f"  {kind:<45} {n:5d}  {n / total:6.1%}")
    print("  edge runs of 4+ characters:", dict(edges))


def show(row, h, status, source):
    marks = {"right": "=", "homophone": "~", "misheard": "x", "miss": "_"}
    return (f"    ref  {row['norm']}\n    got  {''.join(marks[s] for s in status)}\n"
            f"    hyp  {h}\n    pairs {' '.join(f'{c}>{h[j]}' for c, s, j in zip(row['norm'], status, source) if s == 'misheard')}")


def spot_a():
    rng = random.Random(SEED)
    pool = {"worse": [], "better": []}
    for name in ref.SESSIONS:
        session = Session(name)
        _, base_status = collect(session)
        for side, row in pair_members(session):
            h, segs, aligned = base_status[row["id"]]
            status, source = aligned["refs"][row["id"]]
            pool[side].append((name, row, h, status, source))
    for side, n in (("worse", 40), ("better", 20)):
        for index, (name, row, h, status, source) in enumerate(rng.sample(pool[side], n)):
            print(f"A-{side}-{index:02d} {name} {row['start']:.2f}-{row['end']:.2f} spk {row['spk']}")
            print(show(row, h, status, source))


def spot_b():
    rng = random.Random(SEED + 1)
    pool = {"nothing near": [], "someone else's": []}
    for name in ref.SESSIONS:
        session = Session(name)
        span_align, base_status = collect(session)
        by_id = {row["id"]: row for row in session.refs}
        for (s, e, arm, k), (text, al) in span_align.items():
            if not arm.startswith("qwen"):
                continue
            kind = next(sp["kind"] for sp in session.spans if (sp["start"], sp["end"]) == (s, e))
            if kind != "real":
                continue
            refs = session.near(s, e, NEAR)
            context = " | ".join(f"{r['spk']}:{r['norm']}" for r in refs)
            run = "".join(c if (j not in al["claimed"] and j not in al["near"]) else "·" for j, c in enumerate(text))
            if len(run.replace("·", "")) >= 3:
                pool["nothing near"].append(f"{name} {s:.2f}-{e:.2f} {arm} k={k}\n    text  {text}\n    none  {run}\n    refs  {context}")
            if k is not None:
                other = [j for j, rid in al["claimed"].items() if session.speaker_track.get(by_id[rid]["spk"]) != k]
                if len(other) >= 2:
                    mark = "".join(c if j in other else "·" for j, c in enumerate(text))
                    pool["someone else's"].append(f"{name} {s:.2f}-{e:.2f} {arm} k={k} (k's speaker: "
                                                  f"{[p for p, t in session.speaker_track.items() if t == k]})\n    text  {text}\n    other {mark}\n    refs  {context}")
    for kind, items in pool.items():
        for index, item in enumerate(rng.sample(items, min(25, len(items)))):
            print(f"B-{kind}-{index:02d} {item}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-a", action="store_true")
    parser.add_argument("--spot-b", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--posthoc", action="store_true", help="not pre-registered: fillers and sentence-level view")
    parser.add_argument("--clean-miss", action="store_true", help="not pre-registered: what clean sentences lose")
    arguments = parser.parse_args()
    if arguments.clean_miss:
        clean_miss()
    if arguments.spot_a:
        spot_a()
    if arguments.spot_b:
        spot_b()
    if arguments.score:
        score()
    if arguments.posthoc:
        posthoc()


if __name__ == "__main__":
    main()
