"""Post-hoc (not a gate): a third truth for P2 -- whose **words** each ASR row carries, by character alignment.

Only the 11 hina files whose reference is a bilingual track (hina-61, hina-63..74) can do this: they carry a Japanese
twin of every line under the speaker's style (`三月七-日文`), and our ASR is Japanese; everywhere else the reference is a
Chinese translation. ⚠ The twins are the fansub's text, not ours (timings 0.1-0.2 s off, our ASR's mishearings
corrected), but their timing and first draft probably came from some other ASR pipeline: the speaker label per line is
the human part. So this is a third reading, not a verdict.

Per row: the row's normalised characters are aligned (difflib matching blocks of >= MIN_BLOCK characters) against the
Japanese lines within +-WINDOW s, concatenated in time order; matched characters vote for the line's speaker.
Text truth = the majority speaker when >= COVER of the row's characters matched and that speaker holds > half of them;
otherwise undetermined. Owners stay as P2 fitted them (fixed map, no refit).

    python tools/diarization_explore/p2_text_truth.py
"""
from __future__ import annotations

import difflib
import pickle
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf  # noqa: E402

WINDOW = 5.0
#: The references with a Japanese twin of every line (hina-66 and hina-69 are not in this material).
BILINGUAL = ("hina-61", "hina-63", "hina-64", "hina-65", "hina-67", "hina-68", "hina-70", "hina-71", "hina-72", "hina-73", "hina-74")
MIN_BLOCK = 2
COVER = 0.5
GROUP = {"全员", "众人"}  # several people at once: no single speaker


def norm(text):
    text = re.sub(r"\{[^}]*\}", "", text).replace(r"\N", "")
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text))


def japanese_lines(ident):
    import refined_multi as rm
    out = []
    for line in rm.sources()[ident].read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        style = parts[3].strip()
        if not re.search(r"-日文\d*$", style):
            continue
        who, text = rm.speaker(style), norm(parts[9])
        if who and who not in GROUP and text:
            out.append((rm.seconds(parts[1]), rm.seconds(parts[2]), who, text))
    return sorted(out)


def text_truth(row, lines):
    """(person or None, share of the row's characters matched)."""
    t = norm(str(row.get("text", "")))
    if not t:
        return None, 0.0
    near = [x for x in lines if x[0] < float(row["end"]) + WINDOW and x[1] > float(row["start"]) - WINDOW]
    if not near:
        return None, 0.0
    joined = "".join(x[3] for x in near)
    owner = [x[2] for x in near for _ in x[3]]
    votes = Counter()
    for a, b, size in difflib.SequenceMatcher(None, t, joined, autojunk=False).get_matching_blocks():
        if size >= MIN_BLOCK:
            votes.update(owner[b:b + size])
    matched = sum(votes.values())
    cover = matched / len(t)
    if cover < COVER:
        return None, cover
    who, n = votes.most_common(1)[0]
    return (who if n * 2 > matched else None), cover


def main():
    # ⚠ owners fixed to the covered-truth map for all three truths (no refit per truth)
    rows, _ = pf.load_rows("covered")
    files = [f for f in rows if f in BILINGUAL]
    texts = {}
    for ident in sorted(files):
        stable = pf.reference(ident)[0]
        lines = japanese_lines(ident)
        for r in rows[ident]:
            if r["state"] == "ordinary":
                texts[(ident, r["index"])] = text_truth(stable[r["index"]], lines)
    (pf.CACHE / "truth_text.pkl").write_bytes(pickle.dumps(texts))
    for name, part_files in (("dev", [f for f in files if f in pf.SPLIT["dev"]]), ("held-out", [f for f in files if f in pf.SPLIT["heldout"]])):
        rs = [r for f in part_files for r in rows[f] if r["state"] == "ordinary"]
        if not rs:
            continue
        txt = lambda r: texts[(r["file"], r["index"])][0]  # noqa: E731
        onset = lambda r: set(r["truths"]["onset"])  # noqa: E731
        cov = lambda r: set(r["truths"]["covered"])  # noqa: E731
        determined = [r for r in rs if txt(r) is not None]
        print(f"== {name}: {', '.join(part_files)} -- {len(rs)} ordinary rows, text truth determined on {len(determined) / len(rs):.1%} ==")
        agree = [r for r in determined if onset(r) == cov(r)]
        print(f"validation (onset == covered, {len(agree)} rows): text truth inside that set {np.mean([txt(r) in onset(r) for r in agree]):.1%}")
        split = [r for r in determined if onset(r) != cov(r)]
        if split:
            side = lambda r: ("both" if txt(r) in onset(r) and txt(r) in cov(r) else "onset" if txt(r) in onset(r)  # noqa: E731
                              else "covered" if txt(r) in cov(r) else "neither")
            c = Counter(side(r) for r in split)
            print(f"onset != covered ({len(split)} rows, {len([r for r in rs if onset(r) != cov(r)])} before 'determined'): text truth sides with "
                  + ", ".join(f"{k} {v} ({v / len(split):.0%})" for k, v in c.most_common()))
            for label, sub in (("sent (margin <= 0.5)", [r for r in split if r["margin"] <= 0.5]), ("not sent", [r for r in split if r["margin"] > 0.5])):
                if sub:
                    c = Counter(side(r) for r in sub)
                    print(f"   {label}: {len(sub)} rows -- " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        undetermined = [r for r in rs if txt(r) is None]
        print(f"undetermined rows: median chars {np.median([r['chars'] for r in undetermined]) if undetermined else 0:.0f} "
              f"(determined {np.median([r['chars'] for r in determined]):.0f})")
        # step-1 readings under each truth, on the rows the text truth determines
        for label, truth in (("onset", onset), ("covered (main)", cov), ("text", lambda r: {txt(r)})):
            wrong = np.array([r["owner"][r["leader"]] not in truth(r) for r in determined])
            sent = np.array([r["margin"] <= 0.5 for r in determined])
            print(f"   truth {label:16s}: error rate {wrong.mean():.1%}; margin <= 0.5 sends {sent.mean():.1%} of rows, "
                  f"error rate there {wrong[sent].mean():.1%}, catches {wrong[sent].sum() / max(1, wrong.sum()):.1%} of errors")


if __name__ == "__main__":
    main()
