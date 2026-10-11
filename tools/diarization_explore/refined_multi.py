"""Refined multi-speaker axes against our timeline, beyond 姐姐咚咚 (owner 2026-09-15: "全部新素材补上 sortformer，结合精修分析").

Sources (main checkout, local):
  data/refined-multi/希娜小姐应援会/<title>.ass  -- not produced by finesub; style = the *character* a host voices
  data/manually-refined-subs/2026-09反馈/nagi-bd/nagi-bd-refined.ass  -- a finesub run refined by a fansub, style = person
Runs: out/refined-2609/<id>/<id>-{stable,diar}.json (`--stage stable --diarize acoustics --language ja`), ids in
tmp/refined-2609/ids.json. Speaker = style name with decorations removed; non-dialogue styles are dropped by the same
heuristic list as the 希娜 inventory (so the speaker set is approximate). Bilingual 希娜 files: the Chinese line only.

Everything reuses relabel_eval's definitions: a segment's human = the person whose refined lines cover most of its
span; a track's owner = the file's majority person (⚠ breaks where a track changes owner; `owners` shows where).

  python refined_multi.py agree      # per file: first track == human, human in label set, labels per segment
  python refined_multi.py owners     # 300 s windows, >= 30 s of track, >= 60% one person: owner changes
  python refined_multi.py overlap    # refined cross-speaker overlap time found as >= 2 active tracks
  python refined_multi.py relabel    # 5r A criteria at 0.5 and the strong-evidence edit cost per theta
  (add --only ID ... to restrict; --json OUT to dump the per-file rows)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import relabel_eval as ev  # noqa: E402
import speaker_relabel as rl  # noqa: E402


def _main_checkout(path: Path) -> Path:
    """A worktree's data/ is not the main one; the refined material lives in the main checkout."""
    parts = path.parts
    return Path(*parts[: parts.index(".worktrees")]) if ".worktrees" in parts else path


MAIN = _main_checkout(WT)
IDS = WT / "tmp" / "refined-2609" / "ids.json"
FRAME = ev.FRAME
NON_SPEAKER = re.compile(
    r"注释|水印|watermark|STAFF|staff|声明|标题|提示|选项|大字|答案|名片|赞助|温馨提醒|非台词|采访字幕|右侧字幕|底部|顶部|战斗知识|系统解锁|"
    r"闪现文字|邀约|临时|^Default$|^default$|官方注释|折大之声|全员|众人|嘉宾三人|旁白|小剧场|剧情对话|主要角色对话|路人角色对话|行动中对话|路人|导播|观众"
)


def speaker(style: str) -> str | None:
    if NON_SPEAKER.search(style):
        return None
    s = re.sub(r"\s*-\s*(上|复制)$", "", style.strip())
    s = re.sub(r"-(中文|日文)\d*$", "", s)
    return re.sub(r"\s+正文$", "", s).strip() or None


def seconds(stamp: str) -> float:
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def refined_lines(ass: Path):
    """(start, end, person) for dialogue lines with a speaker; bilingual Japanese twins and ※ notes dropped."""
    out = []
    for line in ass.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        style = parts[3].strip()
        if re.search(r"-日文\d*$", style):
            continue
        text = re.sub(r"\{[^}]*\}", "", parts[9]).strip()
        who = speaker(style)
        if who is None or not text or text.startswith("※"):
            continue
        a, b = seconds(parts[1]), seconds(parts[2])
        if b > a:
            out.append((a, b, who))
    return out


def sources():
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    hina = MAIN / "data" / "refined-multi" / "希娜小姐应援会"
    out = {}
    for ident, info in ids.items():
        if ident.startswith("hina-"):
            out[ident] = hina / f"{info['title']}.ass"
        elif ident == "nagi-bd":
            out[ident] = MAIN / "data" / "manually-refined-subs" / "2026-09反馈" / "nagi-bd" / "nagi-bd-refined.ass"
    return out


#: Which run directory `episode` reads. `--run-root` points it at an alternative arm (e.g. the
#: un-separated-audio timelines in out/refined-2609-nosep) so the same scorer serves both.
RUN_ROOT = WT / "out" / "refined-2609"


def episode(ident: str, ass: Path):
    """relabel_eval.episode's tuple for one of our runs, or None when the run is not there yet."""
    run = RUN_ROOT / ident
    if not (run / f"{ident}-diar.json").exists():
        return None
    stable = json.loads((run / f"{ident}-stable.json").read_text(encoding="utf-8"))["segments"]
    timeline = json.loads((run / f"{ident}-diar.json").read_text(encoding="utf-8"))
    lines = refined_lines(ass)
    people = sorted({w for *_, w in lines})
    if len(people) < 2:
        return None
    end = max([b for _, b, _ in lines] + [s["end"] for s in stable] + [t["end"] for t in timeline["tracks"]]) + 2
    grid = np.zeros((int(end / FRAME) + 2, len(people)), dtype=bool)
    for a, b, w in lines:
        grid[int(a / FRAME):int(np.ceil(b / FRAME)), people.index(w)] = True
    frames = defaultdict(Counter)
    for t in timeline["tracks"]:
        span = grid[int(t["start"] / FRAME):int(np.ceil(t["end"] / FRAME))].sum(axis=0)
        for i, c in enumerate(span):
            frames[t["track"]][people[i]] += int(c)
    owner = {k: c.most_common(1)[0][0] for k, c in frames.items() if sum(c.values())}
    return stable, timeline, grid, people, owner, lines


def selected(only):
    for ident, ass in sources().items():
        if only and ident not in only:
            continue
        data = episode(ident, ass)
        if data is not None:
            yield ident, data


def cmd_agree(only):
    rows = []
    for ident, (stable, timeline, grid, people, owner, lines) in selected(only):
        strict, lenient, labels = ev.agreement(stable, grid, people, owner)
        rows.append({"id": ident, "people": len(people), "tracks": len({t["track"] for t in timeline["tracks"]}),
                     "strict": strict, "lenient": lenient, "labels": labels})
        print(f"{ident:<9} people {len(people)} tracks {rows[-1]['tracks']}  first==human {strict:6.1%}  human in set {lenient:6.1%}  labels/seg {labels:.2f}")
    if rows:
        for key in ("strict", "lenient", "labels"):
            values = [r[key] for r in rows]
            print(f"  {key}: median {np.median(values):.3f}  mean {np.mean(values):.3f}  min {min(values):.3f}")
        for k in sorted({r["people"] for r in rows}):
            sub = [r["strict"] for r in rows if r["people"] == k]
            print(f"  people={k}: {len(sub)} files, first==human median {np.median(sub):.1%}")
    return rows


def cmd_owners(only, window=300.0, min_seconds=30.0, min_share=0.60):
    rows = []
    for ident, (stable, timeline, grid, people, owner, lines) in selected(only):
        counts, frames = defaultdict(lambda: defaultdict(Counter)), Counter()
        for t in timeline["tracks"]:
            for i in range(int(t["start"] / FRAME), min(int(np.ceil(t["end"] / FRAME)), len(grid))):
                w = int(i * FRAME // window)
                frames[(t["track"], w)] += 1
                for p in np.flatnonzero(grid[i]):
                    counts[t["track"]][w][people[p]] += 1
        total = sum(frames.values()) or 1
        changes = []
        for track in sorted(counts):
            held = []
            for w in sorted(counts[track]):
                tot = frames[(track, w)]
                if not counts[track][w]:
                    continue
                who, c = counts[track][w].most_common(1)[0]
                if tot * FRAME >= min_seconds and c / tot >= min_share:
                    held.append((w, who, c / tot))
            share = sum(v for (tr, _), v in frames.items() if tr == track) / total
            for (_, a, _), (w1, b, _) in zip(held, held[1:]):
                if a != b:
                    changes.append({"track": track, "track_share": round(share, 3), "from": a, "to": b, "at": int(w1 * window)})
        main = [c for c in changes if c["track_share"] >= 0.15]
        rows.append({"id": ident, "changes": changes, "main_track_changes": len(main)})
        print(f"{ident:<9} " + ("; ".join(f"s{c['track']}({c['track_share']:.0%}) {c['from']}->{c['to']} @{c['at']}s" for c in changes) or "no owner change"))
    if rows:
        print(f"files with a change on a track holding >= 15% of track time: {sum(1 for r in rows if r['main_track_changes'])}/{len(rows)}")
    return rows


def cmd_overlap(only):
    rows = []
    for ident, (stable, timeline, grid, people, owner, lines) in selected(only):
        active = np.zeros((len(grid), 4), dtype=bool)
        for t in timeline["tracks"]:
            active[int(t["start"] / FRAME):min(len(grid), int(np.ceil(t["end"] / FRAME))), t["track"]] = True
        both = grid.sum(axis=1) >= 2
        multi = active.sum(axis=1) >= 2
        ref = int(both.sum())
        hit = int((both & multi).sum())
        rows.append({"id": ident, "refined_overlap_s": ref * FRAME, "found_s": hit * FRAME,
                     "ours_overlap_s": int(multi.sum()) * FRAME})
        print(f"{ident:<9} refined overlap {ref * FRAME:7.1f}s  found as >=2 tracks {hit / max(1, ref):6.1%}  ours total {multi.sum() * FRAME:7.1f}s")
    ref = sum(r["refined_overlap_s"] for r in rows)
    if ref:
        print(f"pooled: {sum(r['found_s'] for r in rows) / ref:.1%} of {ref / 60:.1f} min refined overlap")
    return rows


def cmd_relabel(only):
    rows, reduced_all, mis_all = [], 0, 0
    thetas = ev.THETAS
    costs = defaultdict(lambda: np.zeros(len(thetas)))
    merges_all = [0]
    for ident, (stable, timeline, grid, people, owner, lines) in selected(only):
        relabelled = {th: rl.relabel_segments(stable, timeline, th, "coverage")[0] for th in thetas}
        before, after = ev.agreement(stable, grid, people, owner), ev.agreement(relabelled[0.5], grid, people, owner)
        reduced = mis = 0
        for b, a in zip(stable, relabelled[0.5]):
            tb, ta = ev.tracks_of(b), ev.tracks_of(a)
            if len(tb) < 2 or len(ta) != 1:
                continue
            cover, _ = ev.span_cover(b, grid)
            if not cover.any():
                continue
            human = people[int(cover.argmax())]
            reduced += 1
            mis += human in {owner.get(t) for t in tb if t not in ta} and owner.get(ta[0]) != human
        reduced_all += reduced
        mis_all += mis
        starts = sorted((a, w) for a, _, w in lines)
        for k, th in enumerate(thetas):
            for add in (0, 1, 2):
                costs[add][k] += strong_cost(relabelled[th], grid, people, owner, starts, add)[0]
        merges_all[0] += strong_cost(relabelled[0.5], grid, people, owner, starts, 2)[1]
        rows.append({"id": ident, "lenient_drop_pp": 100 * (before[1] - after[1]), "labels_drop": before[2] - after[2],
                     "reduced": reduced, "misdeleted": mis})
        print(f"{ident:<9} lenient {before[1]:.1%}->{after[1]:.1%}  labels {before[2]:.2f}->{after[2]:.2f}  mis-deleted {mis}/{reduced}")
    if rows:
        print(f"5r A at 0.5 over {len(rows)} files (equal weight): lenient drop {np.mean([r['lenient_drop_pp'] for r in rows]):.2f} pp (<= 1.0), "
              f"labels drop {np.mean([r['labels_drop'] for r in rows]):.2f} (>= 0.15), mis-deletion {mis_all}/{reduced_all} = {mis_all / max(1, reduced_all):.1%} (<= 5%)")
        for add in (0, 1, 2):
            row = costs[add]
            best = int(np.argmin(row))
            print(f"  strong-evidence missed speaker costs {add}: " + "  ".join(f"{th}:{int(v)}" for th, v in zip(thetas, row))
                  + f"  best {thetas[best]}, 0.5 +{(row[thetas.index(0.5)] - row[best]) / row[best]:.1%}")
        # owner 2026-09-15: the merge itself costs 3. Constant across theta (see strong_cost), so it only
        # dilutes the gaps -- printed so the dilution is visible and so segmentation-changing arms can use it.
        merge_total = 3 * merges_all[0]
        row = costs[2] + merge_total
        best = int(np.argmin(row))
        print(f"  + merge cost 3 x {merges_all[0]} strong-evidence merges = {merge_total}: "
              + "  ".join(f"{th}:{int(v)}" for th, v in zip(thetas, row))
              + f"  best {thetas[best]}, 0.5 +{(row[thetas.index(0.5)] - row[best]) / row[best]:.1%}")
    return rows


def strong_cost(segments, grid, people, owner, starts, add, slack=0.3, merge=0, price_unlabelled=False):
    """Owner 2026-09-14: a missed second speaker costs only on strong evidence -- a refined line of that person starts
    inside the segment. Minimum set edit: delete 1, add `add`, substitute (move) 2.

    `merge` (owner 2026-09-15) prices the merge itself: an ASR line that swallowed two people has to be split before
    any label is right, once per extra person. ⚠ It is a function of the segments, not of the labels, and
    `relabel_segments` never moves a boundary -- so it is the SAME constant in every theta arm and cannot move the
    optimum. It only bites when an arm changes the segmentation (T4, B1); returned separately for that reason.

    ⚠⚠ `price_unlabelled` decides what a row the arm left with NO label costs. Skipping it -- the historical
    behaviour, still the default -- is sound in this function's original home: `relabel_eval --strong` compares theta
    arms over one fixed segmentation, and what an arm declines to label it declines for reasons the ruler is not
    asked about. It is NOT sound for comparing two BACKENDS with different coverage, which is what W2/W2b/W3/W6/Q11
    then did with it: deleting every label makes the total 0, so abstention is free and an arm that labels less looks
    cheaper (2026-09-20 audit, finding 3). With the flag on, such a row is priced by the SAME minimisation rather
    than by a new rule -- no labels means `extra = 0` and `missing = len(target)`, so it costs `add` per person the
    reference says is there. It also restores the merge term to being a true segment property, which the skip had
    quietly broken (4214 against 4222 on 姐姐咚咚).

    ⚠ Measured before switching anything over: on 姐姐咚咚 #1-#15 it moves the DiariZen-minus-Sortformer edit-cost
    gap by about -3.6 per file against a mean of -139.3, because Sortformer abstains more (46 rows against 19). The
    direction of every published comparison survives; the ruler still had to be fixed.
    """
    import bisect

    keys = [a for a, _ in starts]
    total = merges = 0
    for s in segments:
        tr = ev.tracks_of(s)
        if not tr and not price_unlabelled:
            continue
        cover, _ = ev.span_cover(s, grid)
        if not cover.any():
            continue
        human = people[int(cover.argmax())]
        lo, hi = bisect.bisect_left(keys, s["start"] - slack), bisect.bisect_left(keys, s["end"])
        target = {human} | {w for _, w in starts[lo:hi]}
        labels = [owner.get(t) for t in tr]
        extra, missing = len(set(labels) - target), len(target - set(labels))
        total += min(2 * k + (extra - k) + add * (missing - k) for k in range(min(extra, missing) + 1))
        total += len(labels) - len(set(labels))
        merges += len(target) - 1
    return total + merge * merges, merges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("agree", "owners", "overlap", "relabel"))
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--json")
    parser.add_argument("--run-root", help="read timelines from here instead of out/refined-2609")
    args = parser.parse_args()
    if args.run_root:
        RUN_ROOT = Path(args.run_root)
    result = {"agree": cmd_agree, "owners": cmd_owners, "overlap": cmd_overlap, "relabel": cmd_relabel}[args.command](set(args.only))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
