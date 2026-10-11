"""Plan 5l.6: on a multi-label line, which label owns the text -- and can production signals tell?

Reads the AliMeeting runs s4_breakdown reads (stable, -diar.json, TextGrid); no model runs.
One sample per (line, label). Target: that label's speaker owns >=2 characters of the
line's text (right or homophone, 5h.12 greedy alignment against the references within
+-0.5 s). Signals: the coverage share `assign` already computes, its rank, and the share
of the line's word midpoints inside the track's activity (and inside it alone).
"""

from __future__ import annotations

import bisect
import json
from collections import Counter, defaultdict
from statistics import mean

import s4_breakdown as b
import s4_headroom as ref
import s4_projection as proj

MIN_CHARS = 2
DEMOTE_MISS = 0.10
SIGNALS = ("share", "inside", "only")


class Activity:
    def __init__(self, tracks):
        self.spans = defaultdict(list)
        for t in tracks:
            self.spans[t["track"]].append((t["start"], t["end"]))
        for spans in self.spans.values():
            spans.sort()
        self.starts = {k: [s for s, _ in v] for k, v in self.spans.items()}

    def on(self, k, t) -> bool:
        i = bisect.bisect_right(self.starts.get(k, []), t) - 1
        return i >= 0 and self.spans[k][i][1] >= t


def samples(name):
    from finesub.speech.diarization.stage import labels_for, read_timeline

    session = b.Session(name)
    by_id = {row["id"]: row for row in session.refs}
    segments = json.loads((proj.ALI / name / f"{name}-stable.json").read_text(encoding="utf-8"))["segments"]
    labels = labels_for(segments, read_timeline(proj.ALI / name / f"{name}-diar.json"))
    activity = Activity(session.tracks)
    every_track = sorted(activity.spans)
    words = defaultdict(list)
    for t, c, seg in session.chars:
        words[seg].append((t, c))
    rows, lines = [], Counter()
    for index, (segment, label) in enumerate(zip(segments, labels)):
        tracks = tuple(label.extras.get("tracks", ()))
        text = "".join(c for _, c in words[index])
        if not tracks or not text:
            continue
        aligned = b.align(text, session.near(segment["start"], segment["end"], ref.PAD))
        owned = Counter()
        for rid, (status, _) in aligned["refs"].items():
            owned[by_id[rid]["spk"]] += sum(s in ("right", "homophone") for s in status)
        owners = {session.speaker_track.get(spk) for spk, n in owned.items() if n >= MIN_CHARS}
        if not any(n >= MIN_CHARS for n in owned.values()):
            lines["no owner found (excluded)"] += 1
            continue
        lines["multi-label" if len(tracks) > 1 else "single-label"] += 1
        coverage = label.extras.get("coverage", {})
        ranked = sorted(tracks, key=lambda k: -coverage.get(str(k), 0.0))
        mids = [t for t, _ in words[index]]
        for rank, k in enumerate(ranked):
            inside = [activity.on(k, t) for t in mids]
            alone = [on and not any(activity.on(o, t) for o in every_track if o != k) for on, t in zip(inside, mids)]
            rows.append({"session": name, "line": index, "labels": len(tracks), "rank": rank,
                         "share": coverage.get(str(k), 0.0), "inside": mean(inside), "only": mean(alone),
                         "owns": k in owners, "owners_labelled": bool(owners & set(tracks)), "review": label.review})
    return rows, lines


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def report(rows, lines):
    print("lines:", dict(lines))
    multi = defaultdict(list)
    for r in rows:
        if r["labels"] > 1:
            multi[(r["session"], r["line"])].append(r)
    cases = Counter()
    for items in multi.values():
        first = next(r for r in items if r["rank"] == 0)["owns"]
        rest = any(r["owns"] for r in items if r["rank"] > 0)
        cases[{(True, True): "both own text", (True, False): "only the first", (False, True): "only a second",
                (False, False): "no label owns it"}[(first, rest)]] += 1
    n = sum(cases.values())
    print(f"multi-label lines ({n}):", {k: f"{v} ({v / n:.1%})" for k, v in cases.most_common()})
    one = [items for items in multi.values() if sum(r["owns"] for r in items) == 1]
    right = sum(next(r for r in items if r["owns"])["rank"] == 0 for items in one)
    print(f"  exactly one label owns text: the top-share label is it in {right}/{len(one)} = {right / max(len(one), 1):.1%}")
    single = [r for r in rows if r["labels"] == 1]
    print(f"single-label lines: the label owns text in {sum(r['owns'] for r in single)}/{len(single)}")
    labelled = [r for r in rows if r["labels"] > 1]
    seconds = [r for r in labelled if r["rank"] > 0]
    for signal in SIGNALS:
        a_all = auc([r[signal] for r in labelled if r["owns"]], [r[signal] for r in labelled if not r["owns"]])
        a_sec = auc([r[signal] for r in seconds if r["owns"]], [r[signal] for r in seconds if not r["owns"]])
        print(f"  AUC {signal:<6} every label on multi-label lines {a_all:.3f} | second labels only {a_sec:.3f}")
    print(f"  second labels: {len(seconds)}, of which own text {sum(r['owns'] for r in seconds)}")
    # Not pre-registered: production already flags lines whose two shares are close.
    for flag in (True, False):
        group = [items for items in multi.values() if items[0]["review"] is flag]
        wrong_first = sum(not next(r for r in items if r["rank"] == 0)["owns"] for items in group)
        print(f"  review={flag}: multi-label lines {len(group)}, first label does not own the text in {wrong_first} "
              f"({wrong_first / max(len(group), 1):.1%})")
    print("OPERATING POINT (second labels; threshold from two sessions, scored on the third)")
    for signal in SIGNALS:
        for held in ref.SESSIONS:
            train = [r for r in seconds if r["session"] != held]
            test = [r for r in seconds if r["session"] == held]
            owning = sorted(r[signal] for r in train if r["owns"])
            # highest t with at most DEMOTE_MISS of the owning second labels below it
            t = owning[int(DEMOTE_MISS * len(owning))] if owning else 0.0
            demoted = [r for r in test if r[signal] < t]
            wrong = sum(r["owns"] for r in demoted)
            owning_test = sum(r["owns"] for r in test)
            print(f"  {signal:<6} held out {held}: t={t:.3f} demoted {len(demoted)}/{len(test)} = {len(demoted) / max(len(test), 1):.1%}; "
                  f"of those actually owning text {wrong} ({wrong / max(owning_test, 1):.1%} of owning seconds)")


def main():
    rows, lines = [], Counter()
    for name in ref.SESSIONS:
        r, l = samples(name)
        rows += r
        lines += l
        print(name, "samples", len(r), flush=True)
    report(rows, lines)


if __name__ == "__main__":
    main()
