"""Owner question 2026-09-12: how often does a track carry OTHER people's speech, versus one
person being split over two tracks?

Per track, over frames where truth has someone speaking and the track sounds:
  own      the track's owner (most frames) is speaking, alone or with others
  foreign  the owner is silent and someone else is speaking   <- "混入别人"
Per file, the split cost: speech time on tracks whose owner already owns a longer track.

  python track_mixing.py ali|ali-holdout|ami|ami-holdout|callhome-ja|vox|refined
(the S4 corpora need DIAR_S4_CORPUS set to the same name; vox reads the onepass cache.)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / "tools" / "diarization_explore"
sys.path.insert(0, str(HERE))
FRAME = 0.08
#: An extra track this pure is one person's duplicate; below it, a slot several people share.
#: Descriptive cut, not calibrated: on 姐姐咚咚 the extras sit at 58-72% or 85-88% (#3 at 78%).
PURE = 0.80


def measure(tracks, truth, claim=None):
    """tracks: [(track, start, end)]; truth: [(speaker, start, end)]."""
    frames = int(max([e for _, _, e in tracks] + [e for _, _, e in truth] + [1.0]) / FRAME) + 2
    speakers = sorted({s for s, _, _ in truth})
    grid = np.zeros((frames, len(speakers)), dtype=bool)
    for s, a, b in truth:
        end = a + claim if claim else b
        grid[int(a / FRAME): int(np.ceil(end / FRAME)), speakers.index(s)] = True
    ids = sorted({k for k, _, _ in tracks})
    active = np.zeros((frames, len(ids)), dtype=bool)
    for k, a, b in tracks:
        active[int(a / FRAME): int(np.ceil(b / FRAME)), ids.index(k)] = True
    speech = grid.any(axis=1)
    # Frames where only this one of our tracks sounds. A truth that leaves out interjections
    # (the fansub axis drops fillers, plan 6) turns our track sounding over an unwritten
    # interjection into "foreign"; restricting to solo frames drops those whenever the main
    # speaker's track is sounding too. It also drops real mixing during overlaps -- a lower
    # bound, reported beside the plain figure (audit 5q round 3).
    solo = active.sum(axis=1) == 1
    rows = []
    for i, k in enumerate(ids):
        on = active[:, i] & speech
        n = int(on.sum())
        if not n:
            continue
        owner = int(grid[on].sum(axis=0).argmax())
        own = int(grid[on, owner].sum())
        alone = on & solo
        rows.append({"track": k, "owner": speakers[owner], "seconds": n * FRAME,
                     "own": own * FRAME, "foreign": (n - own) * FRAME,
                     "solo_seconds": int(alone.sum()) * FRAME,
                     "solo_foreign": int((alone & ~grid[:, owner]).sum()) * FRAME,
                     # only this track sounds and the truth has nobody: our false alarms plus any
                     # speech the truth never wrote. Against a verbatim truth it is the false-alarm
                     # floor; the excess on a fansub axis reads its unwritten lines (audit 5q round 4)
                     "solo_unwritten": int((active[:, i] & solo & ~speech).sum()) * FRAME})
    longest = {}
    for r in rows:
        longest[r["owner"]] = max(longest.get(r["owner"], 0.0), r["seconds"])
    extra = sum(r["seconds"] for r in rows if r["seconds"] < longest[r["owner"]])
    # With more tracks than people the argmax owner hands someone two tracks by pigeonhole, so
    # "someone was split" is not an observation. What is: whether that extra track is one person
    # (a real duplicate) or several people sharing a short slot (audit 5q round 3).
    extra_tracks = [{"track": r["track"], "owner": r["owner"], "seconds": r["seconds"], "purity": r["own"] / r["seconds"]}
                    for r in rows if r["seconds"] < longest[r["owner"]]]
    split_people = len([o for o in longest if sum(r["owner"] == o for r in rows) > 1])
    return {"tracks": rows, "extra_seconds": extra, "extra_tracks": extra_tracks, "split_people": split_people,
            "people": len(speakers), "track_count": len(ids)}


def reshape(truth, chop=None, snap=False):
    """Make exact truth look like the refined axis, to price what that axis does to a measure.

    snap: each utterance's end pushed to the next utterance's start (any speaker), never pulled
          back -- the fansub axis's 87% "end == next start" (plan 5p.1).
    chop: utterances cut into lines this many seconds long (the fansub median is 2.1 s).
    """
    rows = sorted(truth, key=lambda row: row[1])
    if snap:
        rows = [(s, a, max(b, rows[i + 1][1]) if i + 1 < len(rows) else b) for i, (s, a, b) in enumerate(rows)]
    if chop:
        pieces = []
        for s, a, b in rows:
            t = a
            while t < b:
                pieces.append((s, t, min(t + chop, b)))
                t += chop
        rows = pieces
    return rows


def s4_corpus(claim=None, chop=None, snap=False):
    import s4_headroom as ref
    import s4_projection as proj
    out = []
    for name in ref.SESSIONS:
        path = proj.ALI / name / f"{name}-diar.json"
        if not path.exists():
            continue
        tracks = [(t["track"], t["start"], t["end"]) for t in json.loads(path.read_text(encoding="utf-8"))["tracks"]]
        truth = reshape([(r["spk"], r["start"], r["end"]) for r in ref.reference(name)], chop, snap)
        out.append((name, measure(tracks, truth, claim=claim)))
    return out


def vox():
    import over4
    import over4_final as final
    from calibrate import parse_rttm
    from finesub.speech.diarization.segments import segments_from_activity
    cache = over4.WORK / "over4-cache" / "default"
    out = []
    for pattern, rttm_dir in (("test-*-ogg-onepass.npz", final.TEST),
                              ("*-plain-onepass.npz", over4.VOX / "voxconverse-master" / "dev")):
        for path in sorted(cache.glob(pattern)):
            name = path.name.removeprefix("test-").removesuffix("-ogg-onepass.npz").removesuffix("-plain-onepass.npz")
            probs = np.load(path, allow_pickle=True)["probs"]
            tracks = [(t.track, t.start, t.end) for t in segments_from_activity(np.asarray(probs, dtype=float).tolist())]
            truth = [(s.speaker, s.start, s.end) for s in parse_rttm(rttm_dir / f"{name}.rttm")]
            out.append((name, measure(tracks, truth)))
    return out


def refined():
    import refined_ass as ra
    root = Path(__file__).resolve().parents[2] / "out"
    out = []
    for n, path in enumerate(ra.episodes(), start=1):
        diar = root / f"onetonton-{n:02d}" / f"onetonton-{n:02d}-diar.json"
        if not diar.exists():
            continue
        tracks = [(t["track"], t["start"], t["end"]) for t in json.loads(diar.read_text(encoding="utf-8"))["tracks"]]
        truth = [(who, a, b) for a, b, who, text in ra.events(path) if who != ra.NOTE and text]
        # Full line spans. Their ends are snapped to the next line (plan 5p.1), which mostly fills
        # silence, so mixing reads as an upper bound. The "first 1 s after each start" window
        # 5p.5 used reads 18-26% mixing on meeting corpora whose full truth says 1.6-7.2%, so it
        # cannot tell a clean track from a dirty one (5p.5 re-correction, 5q).
        out.append((f"#{n}", measure(tracks, truth)))
    return out


def summarise(label, files):
    seconds = sum(r["seconds"] for _, m in files for r in m["tracks"])
    foreign = sum(r["foreign"] for _, m in files for r in m["tracks"])
    extra = sum(m["extra_seconds"] for _, m in files)
    per_file = sorted(sum(r["foreign"] for r in m["tracks"]) / max(sum(r["seconds"] for r in m["tracks"]), 1e-9) for _, m in files)
    worst = [r["foreign"] / r["seconds"] for _, m in files for r in m["tracks"] if r["seconds"] >= 20]
    split_files = sum(1 for _, m in files if m["split_people"])
    print(f"{label:<12} files {len(files):>3}  混入别人（本人不在而别人在说）占轨时长 {foreign / max(seconds, 1e-9):6.1%}"
          f"  逐文件中位 {per_file[len(per_file) // 2]:6.1%}  ≥20 s 的轨里混入 >30% 的 {sum(w > 0.3 for w in worst) / max(len(worst), 1):5.1%}"
          f" | 拆轨（同一人第二条及以后的轨）占 {extra / max(seconds, 1e-9):6.1%}  有人被拆的文件 {split_files}/{len(files)}")
    solo_seconds = sum(r["solo_seconds"] for _, m in files for r in m["tracks"])
    solo_foreign = sum(r["solo_foreign"] for _, m in files for r in m["tracks"])
    unwritten = sum(r["solo_unwritten"] for _, m in files for r in m["tracks"])
    extras = [e for _, m in files for e in m["extra_tracks"]]
    duplicate = sum(e["seconds"] for e in extras if e["purity"] >= PURE)
    shared = sum(e["seconds"] for e in extras if e["purity"] < PURE)
    # Not a strict lower bound: solo frames drop an unwritten interjection only while the main
    # speaker's track also sounds. An unwritten line inside a gap the axis snapped shut still
    # counts, and --snap cannot model speech hidden there (audit 5q round 4).
    print(f"{'':<12} 混入，只算我们仅一条轨在响的帧（排掉与主句重叠的漏写插话；不是严格下界；占轨时长 {solo_seconds / max(seconds, 1e-9):.0%}）"
          f"{solo_foreign / max(solo_seconds, 1e-9):6.1%}  ｜我们仅一条轨在响而真值没人说话的帧 {unwritten / max(unwritten + solo_seconds, 1e-9):5.1%}")
    print(f"{'':<12}"
          f" | 多出来的轨：一人的复制（纯度 ≥ {PURE:.0%}）占 {duplicate / max(seconds, 1e-9):5.1%}，几人共用的短槽占 {shared / max(seconds, 1e-9):5.1%}"
          f"  [{', '.join(f'{e['owner'][:2]} {e['seconds']:.0f}s {e['purity']:.0%}' for e in extras)}]")
    return {"label": label, "files": len(files), "foreign": foreign / max(seconds, 1e-9), "extra": extra / max(seconds, 1e-9),
            "per_file_median": per_file[len(per_file) // 2], "split_files": split_files,
            "foreign_solo": solo_foreign / max(solo_seconds, 1e-9), "solo_share": solo_seconds / max(seconds, 1e-9),
            "solo_unwritten_share": unwritten / max(unwritten + solo_seconds, 1e-9),
            "extra_duplicate": duplicate / max(seconds, 1e-9), "extra_shared": shared / max(seconds, 1e-9)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("which")
    parser.add_argument("--claim", type=float, default=None, help="S4 corpora: credit each truth line only this many seconds from its start")
    parser.add_argument("--chop", type=float, default=None, help="S4 corpora: cut truth utterances into lines this long first")
    parser.add_argument("--snap", action="store_true", help="S4 corpora: push each truth end to the next utterance start")
    arguments = parser.parse_args()
    which = arguments.which
    shaped = arguments.claim or arguments.chop or arguments.snap
    if which == "vox":
        files = vox()
    elif which == "refined":
        files = refined()
    else:
        files = s4_corpus(arguments.claim, arguments.chop, arguments.snap)
    label = which + (f" chop{arguments.chop:g}" if arguments.chop else "") + (f" claim{arguments.claim:g}" if arguments.claim else "") + (" snap" if arguments.snap else "")
    result = summarise(label, files)
    if which == "vox":
        # the <= 4-person control with the most files; its overlap is lighter than the other corpora
        # (solo frames ~91% of track time vs 56-79%), so read it beside them, not in place of them
        result["groups"] = {name: summarise(name, [(n, m) for n, m in files if test(m["people"])])
                            for name, test in (("vox 2-4", lambda p: 2 <= p <= 4), ("vox >=5", lambda p: p >= 5))}
    suffix = label.replace(" ", "-") if shaped else which
    target = Path(__file__).resolve().parents[2] / "tmp" / "rescore-5q" / f"mixing-{suffix}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"summary": result, "files": files}, ensure_ascii=False, indent=1), encoding="utf-8")
