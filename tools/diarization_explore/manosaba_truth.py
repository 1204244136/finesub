"""Speaker truth for the 魔裁 story clips from on-screen nameplates (owner 2026-10-05), and a diarization run scored on it.

Material: three 20-min no-commentary clips of 『魔法少女ノ魔女裁判』 (YouTube 9quE-XDgUBg, 1:00-1:20 / 5:00-5:20 /
10:00-10:20), `assets/manosaba/manosaba-{a,b,c}.mp4`. FlowOCR (stage 1-2, its `dev` checkout) read the text box
(`*-region00-subtitle.srt`) and the nameplate above it into `<flowocr>/out/manosaba/`. The nameplate track
(`*-nameplate.srt`) misses whole scenes that its region (`*-region01-subtitle.srt`) did read (clip b 482-611 s, for
one), so plates come from both. In this game every character line is voiced and carries a nameplate; narration has none
and is not voiced (owner).

Truth per subtitle cue: the character whose display name best matches the nameplate shown with it (fuzzy, OCR slips
like 桜羽工 for 桜羽エマ), or None for narration. A cue's span is cut at the next cue's start (display time runs past
the speech, especially at the end -- owner). Truth per ASR row: the speaker of the cue covering most of the row; rows
mostly over narration or over nothing are left out (the coverage line says how many, and where).

Scoring a run (`--runs`, default the production run in `out/manosaba/`): each track's owner is its majority person by
seconds (many-to-one, rewards splitting a person over tracks), and a one-to-one owner by Hungarian matching on the same
seconds; both are reported over labelled rows and with unlabelled (missed) rows counted wrong. With two runs, the second
is compared to the first on the same rows (row bootstrap; rows within a track are not independent, so it is narrow).

    python tools/diarization_explore/manosaba_truth.py
    python tools/diarization_explore/manosaba_truth.py --runs out/manosaba out/manosaba-diarizen
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron3_probe as n3  # noqa: E402

FLOWOCR = n3.MAIN.parent / "flowocr"
NAMES = FLOWOCR / "data" / "corpus" / "mocai" / "script-raw" / "CharacterNames.bytes"
OCR = FLOWOCR / "out" / "manosaba"
RUNS = n3.MAIN / "out" / "manosaba"
PLATES = ("nameplate.srt", "region01-subtitle.srt")
CLIPS = ("manosaba-a", "manosaba-b", "manosaba-c")
NAME_MIN = 0.5      # fuzzy ratio for a nameplate to count as a character name
COVER_MIN = 0.5     # share of an ASR row a cue must cover to give the row its speaker
GAP_REPORT = 20.0   # report stretches this long with ASR rows but no truth


def names():
    """{jp display name: id}; Creature* duplicates fold into the base id."""
    out, jp = {}, None
    for line in NAMES.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("; ") and "localization" not in line:
            jp = line[2:].strip()
        elif ":" in line and jp:
            ident = line.split(":", 1)[0].strip()
            out.setdefault(jp, re.sub(r"^Creature", "", ident))
            jp = None
    return out


def _seconds(stamp):
    h, m, s, ms = re.split("[:,]", stamp.strip())[:4]
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def srt(path):
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.strip().splitlines()
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        a, b = (_seconds(s) for s in lines[1].split("-->"))
        out.append((a, b, "\n".join(lines[2:]).strip()))
    return out


def speaker_of(text, table):
    best = max(table, key=lambda jp: SequenceMatcher(None, jp, text).ratio())
    return table[best] if SequenceMatcher(None, best, text).ratio() >= NAME_MIN else None


def cues(clip, table):
    """[(start, end_cut, text, speaker or None)]."""
    subs = srt(OCR / f"{clip}-region00-subtitle.srt")
    # a plate that reads as no name (a region01 cue that caught box text too) must not outvote one that does
    plates = [(pa, pb, who) for name in PLATES for pa, pb, pt in srt(OCR / f"{clip}-{name}") if (who := speaker_of(pt, table))]
    out = []
    for i, (a, b, text) in enumerate(subs):
        end = min(b, subs[i + 1][0]) if i + 1 < len(subs) else b
        over = [(min(b, pb) - max(a, pa), who) for pa, pb, who in plates if min(b, pb) > max(a, pa)]
        out.append((a, end, text, max(over)[1] if over and max(over)[0] >= 0.5 * (b - a) else None))
    return out


def production(clip, runs=None):
    """(stable segments, assigned speakers) of a run, labels recomputed from its diar.json."""
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import TrackSegment
    runs = runs or RUNS
    stable = json.loads((runs / f"{clip}-stable.json").read_text(encoding="utf-8"))["segments"]
    tracks = json.loads((runs / f"{clip}-diar.json").read_text(encoding="utf-8"))["tracks"]
    speakers = assign_all(stable, [TrackSegment(track=int(t["track"]), start=float(t["start"]), end=float(t["end"])) for t in tracks])
    return stable, speakers


def rows_with_truth(clip, table, runs=None):
    """[{index, start, end, text, truth, leader}] for ASR rows over a character's cue."""
    cs = cues(clip, table)
    stable, speakers = production(clip, runs)
    out = []
    for i, (row, sp) in enumerate(zip(stable, speakers)):
        s, e = float(row["start"]), float(row["end"])
        if e <= s:
            continue
        cover = Counter()
        for a, b, _, who in cs:
            cover[who] += max(0.0, min(e, b) - max(s, a))
        if not cover:
            continue
        who, sec = cover.most_common(1)[0]
        if who is None or sec < COVER_MIN * (e - s):
            continue
        tracks = sp.to_dict().get("tracks") or []
        out.append({"index": i, "start": s, "end": e, "text": str(row.get("text", "")), "truth": who,
                    "leader": tracks[0] if tracks else None})
    return out


def owners(rows):
    """(many-to-one, one-to-one) {track: person} by seconds over labelled rows."""
    from scipy.optimize import linear_sum_assignment
    weight = defaultdict(Counter)
    for r in rows:
        if r["leader"] is not None:
            weight[r["leader"]][r["truth"]] += r["end"] - r["start"]
    many = {k: c.most_common(1)[0][0] for k, c in weight.items()}
    tracks, people = sorted(weight), sorted({r["truth"] for r in rows})
    cost = np.array([[weight[t][p] for p in people] for t in tracks])
    ti, pi = linear_sum_assignment(-cost)
    return many, {tracks[i]: people[j] for i, j in zip(ti, pi)}, weight


def coverage(clip, table, runs=None):
    """(ASR rows, rows with truth, [(start, end, rows)] stretches >= GAP_REPORT of rows without truth)."""
    stable, _ = production(clip, runs)
    have = {r["index"] for r in rows_with_truth(clip, table, runs)}
    gaps, run = [], []
    for i, row in enumerate(stable):
        if i in have:
            if run and float(stable[run[-1]]["end"]) - float(stable[run[0]]["start"]) >= GAP_REPORT:
                gaps.append((float(stable[run[0]]["start"]), float(stable[run[-1]]["end"]), len(run)))
            run = []
        else:
            run.append(i)
    if run and float(stable[run[-1]]["end"]) - float(stable[run[0]]["start"]) >= GAP_REPORT:
        gaps.append((float(stable[run[0]]["start"]), float(stable[run[-1]]["end"]), len(run)))
    return len(stable), len(have), gaps


def score(runs, table):
    """{(clip, index): (right one-to-one, right many-to-one)}, missed rows wrong; prints the per-clip lines."""
    out, total = {}, Counter()
    for clip in CLIPS:
        rows = rows_with_truth(clip, table, runs)
        many, one, weight = owners(rows)
        labelled = [r for r in rows if r["leader"] is not None]
        wrong = [r for r in labelled if many[r["leader"]] != r["truth"]]
        wrong1 = [r for r in labelled if one.get(r["leader"]) != r["truth"]]
        owned = set(many.values())
        n_rows, n_truth, gaps = coverage(clip, table, runs)
        people = Counter(r["truth"] for r in rows)
        print(f"{clip}: ASR rows {n_rows}, with truth {n_truth}; stretches >= {GAP_REPORT:.0f} s without truth "
              f"{[(round(a), round(b), n) for a, b, n in gaps]}")
        print(f"   people {len(people)} ({len(owned)} own a track), tracks {len(weight)}; labelled {len(labelled)}, missed {len(rows) - len(labelled)}")
        print(f"   error many-to-one {len(wrong) / len(labelled):.1%} (missed counted wrong {(len(wrong) + len(rows) - len(labelled)) / len(rows):.1%}), "
              f"one-to-one {len(wrong1) / len(labelled):.1%} ({(len(wrong1) + len(rows) - len(labelled)) / len(rows):.1%}); "
              f"wrong rows whose person owns a track {sum(r['truth'] in owned for r in wrong)}/{len(wrong)}")
        for k in sorted(weight, key=lambda k: -sum(weight[k].values())):
            print(f"   s{k} -> {many[k]} (purity {weight[k][many[k]] / sum(weight[k].values()):.2f}); seconds {dict((p, round(s)) for p, s in weight[k].most_common())}")
        for r in rows:
            out[(clip, r["index"])] = (one.get(r["leader"]) == r["truth"], r["leader"] is not None and many[r["leader"]] == r["truth"])
        total.update(rows=len(rows), labelled=len(labelled), wrong=len(wrong), wrong1=len(wrong1))
    missed = total["rows"] - total["labelled"]
    print(f"all: labelled {total['labelled']}/{total['rows']}; error many-to-one {total['wrong'] / total['labelled']:.1%} "
          f"(missed wrong {(total['wrong'] + missed) / total['rows']:.1%}), one-to-one {total['wrong1'] / total['labelled']:.1%} "
          f"({(total['wrong1'] + missed) / total['rows']:.1%})")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, default=[RUNS], help="one run folder, or two to compare (second minus first)")
    args = parser.parse_args()
    table = names()
    runs_list = [r if r.is_absolute() else n3.MAIN / r for r in args.runs]
    if len(runs_list) == 2:
        # rows are paired by (clip, index): both runs must carry the same ASR rows
        for clip in CLIPS:
            a, b = (json.loads((r / f"{clip}-stable.json").read_text(encoding="utf-8"))["segments"] for r in runs_list)
            key = lambda s: (float(s["start"]), float(s["end"]), str(s.get("text", "")))  # noqa: E731
            if [key(s) for s in a] != [key(s) for s in b]:
                sys.exit(f"{clip}: the two runs have different ASR rows; they cannot be compared row by row")
    results = []
    for runs in runs_list:
        print(f"== {runs}")
        results.append(score(runs, table))
    if len(results) == 2:
        keys = sorted(set(results[0]) & set(results[1]))
        rng = np.random.default_rng(1)
        for j, label in ((0, "one-to-one"), (1, "many-to-one")):
            d = np.array([results[1][k][j] - results[0][k][j] for k in keys], float)
            sims = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            print(f"second - first, {label}, missed wrong, {len(keys)} rows: {100 * d.mean():+.1f} pp "
                  f"[{100 * np.percentile(sims, 2.5):+.1f}, {100 * np.percentile(sims, 97.5):+.1f}] (row bootstrap)")


if __name__ == "__main__":
    main()
