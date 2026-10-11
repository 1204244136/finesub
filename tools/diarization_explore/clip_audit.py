"""Plan 5r.1: the two 7-minute preview clips through the full pipeline, one output dir per arm.

Run from the checkout root. Clips: `assets/onetonton-02-clip.mp4` (姐姐咚咚 #2 from 27:30) and
`assets/nepolabo-clip.mp4` (30:30); arms live in `out/<clip>-<arm>/` (`prod`, `relabel`, the split run in
`out/<clip>/`, `b1`, `b2`, and the text-only DeepSeek runs with a `-ds` / `-ds41` suffix). Each row is one
line of `-annotated.csv` -- what the correction round delivered, merges included.

  --cost           edit cost of the corrected rows against the fansub axis (姐姐咚咚 only): deleting a
                   wrong label 1, moving a line 2 (owner 2026-09-12); the track owners are fixed by hand
                   because the roles layer swapped s0 / s2 on this clip. Also time-overlapping cue pairs.
  --b12[=SUFFIX]   what correction did with the re-decoded overlap candidates for prod/b1/b2 + SUFFIX
                   (none for agy; `--b12=-ds41` for DeepSeek V4.1 -- the `=` keeps argparse from reading
                   the suffix as a flag): text similarity to production inside vs outside the overlap
                   windows (outside is run noise), originals dropped, candidates used
  --detail         with --b12: B1's dropped originals and every row that took a candidate
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preview_ass as pa  # noqa: E402
import refined_ass as ra  # noqa: E402
from finesub.llm.output_protocol import split_annotated_row  # noqa: E402

CLIPS = ("onetonton-02-clip", "nepolabo-clip")
CLIP_START, CLIP_LENGTH = 1650.0, 420.0  # 姐姐咚咚 #2 27:30, 7 minutes
OWNER = {0: "青木佑磨", 1: "樱川惠", 2: "小日向美香"}
FRAME = 0.08
COST_ARMS = [("生产现状 agy", "prod"), ("改标签 r=0.5", "relabel"), ("拆分后 agy", ""), ("B1 插候选行 agy", "b1"),
             ("B2 候选区块 agy", "b2"), ("生产现状 DS V4", "prod-ds"), ("生产现状 DS V4.1", "prod-ds41"),
             ("B1 DS V4.1", "b1-ds41"), ("B2 DS V4.1", "b2-ds41")]


def load(clip, arm):
    """(source segments by id, delivered rows) for one arm, or None when that arm has no output."""
    name = f"{clip}-{arm}" if arm else clip
    d = Path(f"out/{name}")
    if not (d / f"{name}-annotated.csv").exists():
        return None
    segments = json.loads((d / f"{name}-stable.json").read_text(encoding="utf-8"))["segments"]
    ids = {str(s.get("id") or i): s for i, s in enumerate(segments, start=1)}
    rows = []
    for line in (d / f"{name}-annotated.csv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = split_annotated_row(line)
        covered = [i for i in pa.ids_in(fields.position) if i in ids] if fields else []
        if not covered:
            continue
        rows.append({"ids": covered, "start": min(ids[i]["start"] for i in covered), "end": max(ids[i]["end"] for i in covered),
                     "tracks": pa.tracks_in(fields.spk)[0], "text": fields.corrected, "note": fields.note,
                     "cand": sum(bool(ids[i].get("candidate")) for i in covered),
                     "orig": sum(not ids[i].get("candidate") for i in covered)})
    return ids, rows


def overlapping_pairs(rows):
    rows = sorted(rows, key=lambda r: r["start"])
    return sum(1 for i, a in enumerate(rows) for b in rows[i + 1:] if b["start"] < a["end"] - 0.05)


def cost():
    lines = [(a - CLIP_START, b - CLIP_START, w) for a, b, w, t in ra.events(ra.episodes()[1])
             if w != ra.NOTE and t and CLIP_START <= a < CLIP_START + CLIP_LENGTH]
    people = sorted({w for *_, w in lines})
    grid = np.zeros((int((CLIP_LENGTH + 20) / FRAME), len(people)), bool)
    for a, b, w in lines:
        grid[int(max(a, 0) / FRAME):int(np.ceil(min(b, CLIP_LENGTH + 19) / FRAME)), people.index(w)] = True
    print("姐姐咚咚 clip, corrected rows: rows / delete / move / cost (delete 1 + move 2) / rows needing no edit | overlapping cue pairs")
    for label, arm in COST_ARMS:
        loaded = load(CLIPS[0], arm)
        if loaded is None:
            continue
        n = dele = move = perfect = 0
        for r in loaded[1]:
            if not r["tracks"]:
                continue
            a = int(r["start"] / FRAME)
            cover = grid[a:max(a + 1, int(np.ceil(r["end"] / FRAME)))].sum(0)
            if not cover.any():
                continue
            human = people[int(cover.argmax())]
            labels = [OWNER.get(t) for t in r["tracks"]]
            n += 1
            dele += len(labels) - 1
            if human in labels:
                perfect += len(labels) == 1
            else:
                move += 1
        print(f"  {label:<18} {n} / {dele} / {move} / {dele + 2 * move} / {perfect} | {overlapping_pairs(loaded[1])}")
    print("nepolabo clip (no truth): rows / multi-label rows | overlapping cue pairs")
    for label, arm in COST_ARMS:
        loaded = load(CLIPS[1], arm)
        if loaded is not None:
            rows = loaded[1]
            print(f"  {label:<18} {len(rows)} / {sum(len(r['tracks']) > 1 for r in rows)} | {overlapping_pairs(rows)}")

    # Owner 2026-09-15: a missed second speaker costs 2 on strong evidence, and the merge itself costs 3.
    # ⚠ The merge term is the reason this rescore exists: it is constant across THETA arms (relabelling never
    # moves a boundary) but NOT across arms that re-cut the rows -- T4 and B1 split merges, so the 5r.1 table
    # was scored on a ruler that could not see what they buy. Same min-set-edit shape as refined_multi.strong_cost.
    starts = sorted((a, w) for a, _, w in lines)
    keys = [a for a, _ in starts]
    print("\nowner pricing (delete 1, move 2, strong-evidence missed 2, merge 3): "
          "rows / label edits / merges left / total")
    for label, arm in COST_ARMS:
        loaded = load(CLIPS[0], arm)
        if loaded is None:
            continue
        n = edits = merges = 0
        for r in loaded[1]:
            if not r["tracks"]:
                continue
            a = int(r["start"] / FRAME)
            cover = grid[a:max(a + 1, int(np.ceil(r["end"] / FRAME)))].sum(0)
            if not cover.any():
                continue
            human = people[int(cover.argmax())]
            lo, hi = bisect.bisect_left(keys, r["start"] - 0.3), bisect.bisect_left(keys, r["end"])
            target = {human} | {w for _, w in starts[lo:hi]}
            labels = [OWNER.get(t) for t in r["tracks"]]
            extra, missing = len(set(labels) - target), len(target - set(labels))
            n += 1
            edits += min(2 * k + (extra - k) + 2 * (missing - k) for k in range(min(extra, missing) + 1))
            edits += len(labels) - len(set(labels))
            merges += len(target) - 1
        print(f"  {label:<18} {n} / {edits} / {merges} / {edits + 3 * merges}")


def b12(suffix, detail):
    joined = lambda rows: "".join(r["text"] for r in rows)
    for clip in CLIPS:
        candidates = json.loads(Path(f"out/{clip}-prod/{clip}-prod-overlap-candidates.json").read_text(encoding="utf-8"))
        windows = sorted({tuple(c["window"]) for c in candidates})
        inside = lambda r: any(r["end"] > a and r["start"] < b for a, b in windows)
        _, prod = load(clip, f"prod{suffix}")
        pin, pout = [r for r in prod if inside(r)], [r for r in prod if not inside(r)]
        print(f"\n######## {clip}{suffix}: {len(windows)} overlap windows, {sum(b - a for a, b in windows):.0f} s of 420 s")
        for arm in (f"b1{suffix}", f"b2{suffix}"):
            ids, rows = load(clip, arm)
            covered = {i for r in rows for i in r["ids"]}
            cand = {i for i, s in ids.items() if s.get("candidate")}
            rin, rout = [r for r in rows if inside(r)], [r for r in rows if not inside(r)]
            sim_in = SequenceMatcher(None, joined(pin), joined(rin), autojunk=False).ratio()
            sim_out = SequenceMatcher(None, joined(pout), joined(rout), autojunk=False).ratio()
            line = (f"== {arm}: rows {len(rows)} (prod {len(prod)}); similarity to prod inside {sim_in:.3f} / outside {sim_out:.3f}; "
                    f"originals dropped {len(set(ids) - cand - covered)}/{len(set(ids) - cand)}")
            if cand:
                line += (f"; candidates used {len(cand & covered)}/{len(cand)} (rows of candidates only "
                         f"{sum(1 for r in rows if r['cand'] and not r['orig'])}, merged into originals {sum(1 for r in rows if r['cand'] and r['orig'])})")
            notes = sum(1 for r in rows if any(k in r["note"] for k in ("候选", "重转")))
            print(line + f"; notes mentioning candidates {notes}")
            if not (detail and cand):
                continue
            print("   dropped originals: text | share of its characters found in rows within 3 s")
            for i, s in ids.items():
                if s.get("candidate") or i in covered:
                    continue
                text = s["text"].strip("。 ")
                near = "".join(r["text"] for r in rows if r["end"] > s["start"] - 3 and r["start"] < s["end"] + 3)
                match = SequenceMatcher(None, text, near, autojunk=False)
                kept = sum(b.size for b in match.get_matching_blocks()) / max(1, len(text))
                print(f"     {s['start']:6.1f} 「{text[:30]}」 {kept:.0%}")
            print("   rows that took candidates: original / candidate / delivered")
            for r in rows:
                if r["cand"]:
                    o = "".join(ids[i]["text"] for i in r["ids"] if not ids[i].get("candidate"))
                    c = "".join(ids[i]["text"] for i in r["ids"] if ids[i].get("candidate"))
                    print(f"     {r['start']:6.1f} s{r['tracks']} 原「{o[:30]}」候「{c[:30]}」出「{r['text'][:36]}」")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cost", action="store_true")
    group.add_argument("--b12", nargs="?", const="", metavar="SUFFIX")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()
    cost() if args.cost else b12(args.b12, args.detail)
