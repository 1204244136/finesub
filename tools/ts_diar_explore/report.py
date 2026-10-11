"""Every reading of the survey write-up, from one command. Arms are the names in `arms.ARMS`; each subcommand prints
the per-item rows of every listed arm that exists, then a **paired mean** over the items where all listed
arms exist (the items are named), so arms are never averaged over different episode sets.

    python -m tools.ts_diar_explore.report speakers hina-72 hina-70 ... [--arms ...] [--span A B]
    python -m tools.ts_diar_explore.report speakers hina-72 ... --windows [--window-source mosspro-win]
    python -m tools.ts_diar_explore.report text hina-72 hina-70 ... [--arms ...]
    python -m tools.ts_diar_explore.report lines hina-72 [--arms ...]              # subtitle line edges
    python -m tools.ts_diar_explore.report overlap ali-R8001_M8004 ... [--arms ...] [--span 0 1490]
    python -m tools.ts_diar_explore.report timing t1 | t2 hina-72 ...

`--span A B` scores only [A, B): whole-episode arms restricted to it, plus each arm's `episode.py --span A B`
run where one exists (shown as `<arm>·span`). Metric definitions live with the metrics: `score.py` (text,
speaker, line edges), `overlap.py` (detection, text under overlap), `timing.py` (edge matching).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from . import arms as A
from . import overlap as O
from . import score as S
from . import timing as T

ALL = 0.0, 1e9


def reference(item: str) -> list[dict]:
    return O.reference(item) if item.startswith("ali-") else S.reference(item)


def candidates(item: str, names: list[str], span: tuple[float, float] | None) -> list[tuple[str, dict]]:
    """(column, arm) for every listed arm that exists: the whole-episode run, and with a span also the span run."""
    out = []
    for n in names:
        if A.exists(item, n):
            out.append((n, A.load(item, n)))
        if span and A.ARMS[n].kind == "model" and A.exists(item, n, span=span):
            out.append((f"{n}·span", A.load(item, n, span=span)))
    return out


def paired(rows: dict[str, dict[str, dict]], keys: list[str], fmt) -> None:
    """rows: item -> column -> metrics. Mean of each key over the items every column has."""
    cols = sorted({c for r in rows.values() for c in r}, key=lambda c: list(A.ARMS).index(c.split("·")[0]) * 2 + ("·" in c))
    items = [i for i, r in rows.items() if all(c in r for c in cols)]
    if len(rows) < 2 or not items:
        return
    print(f"paired mean over {len(items)} items ({' '.join(items)}):")
    for c in cols:
        print(f"  {c:<18} " + fmt({k: float(np.mean([rows[i][c][k] for i in items])) for k in keys}))


def cmd_speakers(a) -> None:
    fmt = lambda s: f"owner {s['owner'] * 100:5.1f}  B3 {s['b3_f1']:.3f}"  # noqa: E731
    rows = defaultdict(dict)
    lo, hi = a.span or ALL
    for item in a.items:
        ref = reference(item)
        for col, arm in candidates(item, a.arms, a.span):
            s = S.speaker(ref, arm, lo, hi)
            if "b3_f1" not in s:
                continue
            rows[item][col] = s
            print(f"{item:<16} {col:<18} {fmt(s)}  labels {s['labels']:2d} / persons {s['persons']}  unheard {s['unheard']}/{s['lines']}")
        print()
    paired(rows, ["owner", "b3_f1"], fmt)


def cmd_windows(a) -> None:
    """Each window of a windowed arm scored on its own reference lines -- that window by its raw output, every
    other arm restricted to the same stretch -- so cross-window identity plays no part."""
    src = a.window_source
    own = f"{src} window"
    rows = defaultdict(dict)
    for item in a.items:
        ref = reference(item)
        others = [(c, arm) for c, arm in candidates(item, a.arms, None) if c != src]
        moss = {round(s): w for s, _, w in A.windows(item, "moss")} if src != "moss" else {}
        for k, (lo, hi, w) in enumerate(A.windows(item, src)):
            cands = [(own, w)] + ([("moss window", moss[round(lo)])] if round(lo) in moss else [])
            for col, arm in cands + others:
                s = S.speaker(ref, arm, lo, hi)
                if "b3_f1" in s:
                    rows[(item, k)][col] = (s["owner"], s["b3_f1"])
    for col in sorted({c for r in rows.values() for c in r}, key=lambda c: (c != own, c)):
        common = [r for r in rows.values() if col in r and own in r]
        own_b3 = np.mean([r[own][1] for r in common])
        better = sum(r[col][1] > r[own][1] for r in common)
        worse = sum(r[col][1] < r[own][1] for r in common)
        print(f"{col:<20} {len(common):3d} windows  owner {np.mean([r[col][0] for r in common]) * 100:5.1f}  "
              f"B3 {np.mean([r[col][1] for r in common]):.3f}  ({own} {own_b3:.3f}; better on {better}, worse on {worse})")


def cmd_text(a) -> None:
    fmt = lambda c: f"CER {c['cer']:.3f}"  # noqa: E731
    rows = defaultdict(dict)
    lo, hi = a.span or ALL
    for item in a.items:
        ref = S.reference(item)
        for col, arm in candidates(item, [n for n in a.arms if A.ARMS[n].text], a.span):
            c = S.cer(ref, arm, lo, hi)
            if not c["ref_chars"]:
                continue
            rows[item][col] = c
            print(f"{item:<16} {col:<18} {fmt(c)}  (S{c['sub']} D{c['del']} I{c['ins']} / {c['ref_chars']}, "
                  f"hyp/ref {c['hyp_chars'] / c['ref_chars']:.3f})")
        print()
    paired(rows, ["cer"], fmt)


def cmd_lines(a) -> None:
    """Line edges of each arm as the model gives them, and -- where `seg_arm.py` has run on it -- of the subtitle
    lines the production DP + SBD cut from its words (`<arm>·DP+SBD`, `episodes/<item>/lines/<stem>-sbd.json`)."""
    lo, hi = a.span or ALL
    for item in a.items:
        ref = [x for x in S.reference(item) if S.norm(x["text"])]
        cols = [("fansub (self)", {"segments": ref})]
        for col, arm in candidates(item, [n for n in a.arms if A.ARMS[n].text], a.span):
            cols.append((col, arm))
            sbd = A.EPISODES / item / "lines" / f"{A.stem(col.split('·')[0])}-sbd.json"
            if "·" not in col and A.ARMS[col].kind == "model" and sbd.exists():
                cols.append((f"{col}·DP+SBD", json.loads(sbd.read_text(encoding="utf-8"))))
        for col, arm in cols:
            s = S.segmentation(ref, arm.get("lines") or arm["segments"], lo, hi)
            if "lines" not in s:
                print(f"{item:<8} {col:<22} no lines in [{lo:g}, {hi:g})")
                continue
            print(f"{item:<8} {col:<22} lines {s['lines']:4d}  dur~{s['dur_median']:.1f}s chars~{s['chars_median']}  "
                  f">8s {s['over_8s']:.1%} >36c {s['over_36c']:.1%}  F1@0.3 {s['f1@0.3']:.3f} (P {s['p@0.3']:.2f} "
                  f"R {s['r@0.3']:.2f})  F1@0.6 {s['f1@0.6']:.3f}  startF1@0.3 {s['start_f1@0.3']:.3f}")


def cmd_overlap(a) -> None:
    lo, hi = a.span or ALL
    print(f"{'item':<16} {'arm':<16} {'owner':>6} {'B3':>6} {'ovl-own':>8} {'ovlRec':>7} {'falseOv':>8} "
          f"{'miss':>6} {'txt clean':>10} {'txt ovl':>9}")
    for item in a.items:
        ref = [r for r in O.reference(item) if r["start"] >= lo and r["end"] <= hi]
        for col, arm in candidates(item, a.arms, None):
            arm = O.clip_arm(arm, lo, hi)
            s, d = S.speaker(ref, arm, *ALL), O.detection(ref, arm)
            txt = O.text_recall(ref, arm) if A.ARMS[col].text else None
            t = (f"{txt['clean']['recall']:6.1%}({txt['clean']['hyp_ref']:.2f}) {txt['overlap']['recall']:6.1%}"
                 f"({txt['overlap']['hyp_ref']:.2f})" if txt else "")
            print(f"{item:<16} {col:<16} {s['owner']:6.1%} {s['b3_f1']:6.3f} {s['overlap_owner']:8.1%} "
                  f"{d['overlap_recall']:7.1%} {d['false_overlap']:8.1%} {d['miss']:6.1%} {t}")
        share = O.detection(ref, {"segments": []})["ref_overlap_share"]
        print(f"{'':<16} ref overlap share {share:.1%}, {sum(r['ovl_frac'] >= 0.5 for r in ref)}/{len(ref)} intervals mostly overlapped")


def cmd_timing(a) -> None:
    if a.which == "t1":
        ref, item = T.srt(T.FIXED), "gold-BV1cqLR6hEp3"
        arms = candidates(item, [n for n in a.arms if A.ARMS[n].text], None)
        print(f"T1 human word cues: {len(ref)} words, {T.FIXED.name}")
        for phrase in (False, True):
            on, off = T.edges(ref, phrase=phrase)
            for col, arm in arms:
                if A.ARMS[col].words or phrase:  # segment-only arms: phrase edges only
                    hon, hoff = T.hyp_edges(arm, phrase=phrase)
                    print(T.row(col, "phrase-on" if phrase else "word-on", T.stats(on, hon)))
                    print(T.row(col, "phrase-off" if phrase else "word-off", T.stats(off, hoff)))
            print()
        return
    pooled = defaultdict(lambda: ([], []))
    names = [n for n in a.arms if A.ARMS[n].text]
    items = [it for it in a.items if all(A.exists(it, n) for n in names)]
    for it in items:
        ref = [r for r in S.reference(it) if S.norm(r["text"])]
        on, off = [r["start"] for r in ref], [r["end"] for r in ref]
        for n in names:
            hon, hoff = T.hyp_edges(A.load(it, n))
            pooled[n][0].append(T.stats(on, hon))
            pooled[n][1].append(T.stats(off, hoff))
    print(f"T2 fansub line edges, items every arm has: {' '.join(items)} (per-item stats averaged)")
    for n in names:
        for side, which in ((0, "line-on"), (1, "line-off")):
            ss = pooled[n][side]
            avg = {k: float(np.mean([s[k] for s in ss])) for k in ss[0]}
            avg["n"] = int(sum(s["n"] for s in ss))
            print(T.row(n, which, avg))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = list(A.ARMS)
    for cmd, default in (("speakers", [n for n in names if n != "prod"]), ("text", names), ("lines", names),
                         ("overlap", [n for n in names if n != "prod"])):
        p = sub.add_parser(cmd)
        p.add_argument("items", nargs="+")
        p.add_argument("--arms", nargs="+", choices=names, default=default)
        p.add_argument("--span", nargs=2, type=float, metavar=("A", "B"))
        if cmd == "speakers":
            p.add_argument("--windows", action="store_true")
            p.add_argument("--window-source", default="moss", choices=[n for n, x in A.ARMS.items() if x.windowed])
    p = sub.add_parser("timing")
    p.add_argument("which", choices=["t1", "t2"])
    p.add_argument("items", nargs="*")
    p.add_argument("--arms", nargs="+", choices=names, default=["prod", "mai2-diar", "g35t", "moss"])
    a = ap.parse_args()
    if getattr(a, "span", None):
        a.span = tuple(a.span)
    {"speakers": lambda: cmd_windows(a) if a.windows else cmd_speakers(a), "text": lambda: cmd_text(a),
     "lines": lambda: cmd_lines(a), "overlap": lambda: cmd_overlap(a), "timing": lambda: cmd_timing(a)}[a.cmd]()


if __name__ == "__main__":
    main()
