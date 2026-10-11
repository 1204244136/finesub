"""W5: are DiariZen's extra second labels real overlap, or just more tracks bumping into each other?

The merge term is 63% of the edit cost, so whether a second label is a gain matters more than
another point of attribution. Protocol frozen in `docs/archive/diarization-closeout-plan.md` section 7
BEFORE this ran.

The trap is that "two tracks are active at once" rises automatically with the track count, the same
way per-track oracle naming inflated `fixed` (audit 2.1). So the primary reading deduplicates
simultaneous tracks by their whole-file majority owner first: one person split across two tracks
that sound together is NOT overlap. That is findings 6.1's "excluding what only one person's two
tracks hold up", at frame level, and it is invariant to track count by construction --
`test_w5_overlap.py` pins that by splitting a track in two and requiring the numbers not to move.

Three readings per arm:

  detect_id   >=2 distinct owners active AND those owners are among the people actually speaking
  detect_any  >=2 distinct owners active, without asking whether they are the right people
  raw         >=2 tracks active, no deduplication -- kept only to reproduce findings 6.1

`detect_id` is the product question, since a second label naming the wrong person is not a gain;
`detect_any` beside it says whether a miss was a detection failure or an identification failure.

    python tools/diarization_explore/w5_overlap.py selfcheck   # must reproduce 50.4% / 411.8 min
    python tools/diarization_explore/w5_overlap.py compare --write
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402
import backend_cost as bc  # noqa: E402
import relabel_eval as ev  # noqa: E402
from accuracy_report import bootstrap_mean_ci  # noqa: E402

FRAME = lab.FRAME
#: Files holding less than this much reference overlap stay out of the file mean (they stay in the
#: pooled total). 星穹铁道's axis barely writes overlap, and a near-zero denominator swamps a mean.
MIN_REF_OVERLAP_S = 60.0
#: `accuracy_cross.configure()` rebinds these to Onetonton, and a later hina pass in the same
#: process would then score nothing (the same trap w6_tail hit). Captured before anything patches.
_PRISTINE_RM = (lab.rm.sources, lab.rm.episode)
DZ3 = lab.WT / "tmp" / "diarization-dz3"
W3 = lab.WT / "tmp" / "diarization-w3"


def activity_from_tracks(tracks, frames, n_tracks=None):
    """(frames, tracks) boolean co-activity map from (track, start, end) triples."""
    ids = sorted({int(t[0]) for t in tracks})
    index = {t: i for i, t in enumerate(ids)}
    out = np.zeros((frames, max(1, n_tracks or len(ids))), dtype=bool)
    for track, start, end in tracks:
        lo, hi = int(start / FRAME), min(int(np.ceil(end / FRAME)), frames)
        if hi > lo:
            out[lo:hi, index[int(track)]] = True
    return out


def activity_from_probs(probs, frames):
    """Production's own path: smooth the frame probabilities into segments, then lay them out."""
    from finesub.speech.diarization.segments import segments_from_activity

    tracks = [(s.track, s.start, s.end) for s in segments_from_activity(probs.tolist())]
    return activity_from_tracks(tracks, frames)


def rttm_tracks(path: Path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            rows.append((f[7], float(f[3]), float(f[3]) + float(f[4])))
    speakers = sorted({who for who, *_ in rows})
    return [(speakers.index(who), a, b) for who, a, b in rows if b > a]


def owners_of(active, grid):
    """Each track's whole-file majority reference person -- the same mapping `fixed` uses."""
    out = {}
    for k in range(active.shape[1]):
        sel = active[:, k] & grid.any(axis=1)
        if not sel.any():
            continue
        counts = Counter()
        for person in range(grid.shape[1]):
            counts[person] = int((sel & grid[:, person]).sum())
        best, n = counts.most_common(1)[0]
        if n:
            out[k] = best
    return out


def measure(active, grid):
    """The three readings of the protocol, per file."""
    ref_count = grid.sum(axis=1)
    both, single = ref_count >= 2, ref_count == 1
    owners = owners_of(active, grid)

    distinct = np.zeros(len(active), dtype=int)
    subset = np.zeros(len(active), dtype=bool)
    raw = active.sum(axis=1) >= 2
    for i in np.flatnonzero(active.any(axis=1)):
        people = {owners[k] for k in np.flatnonzero(active[i]) if k in owners}
        distinct[i] = len(people)
        if len(people) >= 2:
            subset[i] = all(grid[i, person] for person in people)
    pair = distinct >= 2

    def share(mask, among):
        return float((mask & among).sum()) / float(among.sum()) if among.any() else float("nan")

    return {
        "ref_overlap_s": float(both.sum()) * FRAME,
        "ref_single_s": float(single.sum()) * FRAME,
        "detect_id": share(pair & subset, both),
        "detect_any": share(pair, both),
        "false_id": share(pair, single),
        "raw_detect": share(raw, both),
        "raw_false": share(raw, single),
        "tracks": int(active.shape[1]),
    }


def sortformer_arm(timeline, frames):
    return activity_from_tracks(
        [(t["track"], t["start"], t["end"]) for t in timeline["tracks"]], frames)


def cmd_selfcheck(_args):
    """Reproduce findings 6.1's 50.4% over 411.8 minutes, or the wiring is wrong."""
    import refined_multi as rm

    hit = ref = 0.0
    files = 0
    for ident, data in rm.selected(()):
        stable, timeline, grid, people, owner, lines = data
        active = sortformer_arm(timeline, len(grid))
        m = measure(active, grid)
        if np.isnan(m["raw_detect"]):
            continue
        files += 1
        ref += m["ref_overlap_s"]
        hit += m["raw_detect"] * m["ref_overlap_s"]
    print(f"Sortformer, {files} files: raw >=2 tracks over reference overlap = {hit / ref:.1%} "
          f"of {ref / 60:.1f} min")
    print("findings 6.1 says 50.4% of 411.8 min -- these must agree")


def arms_for(corpus, key):
    """(data, {arm: activity}) for one file, or None when a backend's cache is missing."""
    if corpus == "hina":
        loaded = bc.hina_episode(key)
        if loaded is None:
            return None
        (stable, timeline, grid, people, _), _, _ = loaded
        rttm = DZ3 / key / "w16_default.rttm"
        probs = DZ3 / key / "w16_default.npz"
    else:
        data = ev.episode(key)
        if data is None:
            return None
        stable, timeline, grid, people, _ = data
        ident = f"onetonton-{key:02d}"
        rttm = bc.V2_OUT / ident / "diarizen.rttm"
        probs = bc.V2_OUT / ident / "diarizen.npz"
    if not rttm.exists() or not probs.exists():
        return None
    frames = len(grid)
    arms = {
        "sortformer": sortformer_arm(timeline, frames),
        "diarizen_native": activity_from_tracks(rttm_tracks(rttm), frames),
        "diarizen_raster": activity_from_probs(np.load(probs)["probs"][:frames], frames),
    }
    if np.array_equal(arms["diarizen_native"], arms["diarizen_raster"]):
        raise ValueError(f"{key}: native and raster co-activity are identical -- suspect the wiring "
                         "before reporting (see W3's all-zero bug)")
    return grid, arms


def cmd_compare(args):
    payload = {}
    for corpus, keys in (("hina", None), ("onetonton", list(range(1, 16)))):
        if corpus == "hina":
            lab.rm.sources, lab.rm.episode = _PRISTINE_RM
            frozen = json.loads((HERE / "results" / "diarizen-v3-20260917.json").read_text(encoding="utf-8"))
            keys = frozen["groups"]["heldout_all"]
        else:
            from accuracy_cross import configure

            configure()
        rows = []
        for key in keys:
            got = arms_for(corpus, key)
            if got is None:
                continue
            grid, arms = got
            rows.append({"file": str(key),
                         **{arm: measure(active, grid) for arm, active in arms.items()}})
        if not rows:
            raise SystemExit(f"{corpus}: scored nothing -- check the wiring, do not report a blank pass")
        payload[corpus] = report(corpus, rows)
        payload[corpus]["rows"] = rows
    if args.write:
        out = bc.V2_OUT / "w5-overlap.json"
        out.write_text(json.dumps({**payload, "script": lab.digest(__file__)}, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")


def report(corpus, rows):
    kept = [r for r in rows if r["sortformer"]["ref_overlap_s"] >= MIN_REF_OVERLAP_S]
    dropped = len(rows) - len(kept)
    print(f"\n=== {corpus}: {len(rows)} files, {len(kept)} with >= {MIN_REF_OVERLAP_S:.0f}s of "
          f"reference overlap ({dropped} kept out of the file mean) ===")
    print(f"{'arm':18}{'detect_id':>11}{'detect_any':>12}{'false_id':>10}{'raw det':>9}{'raw false':>11}{'tracks':>8}")
    for arm in ("sortformer", "diarizen_native", "diarizen_raster"):
        vals = {k: float(np.mean([r[arm][k] for r in kept])) for k in
                ("detect_id", "detect_any", "false_id", "raw_detect", "raw_false", "tracks")}
        print(f"{arm:18}{vals['detect_id']*100:>10.2f}%{vals['detect_any']*100:>11.2f}%"
              f"{vals['false_id']*100:>9.2f}%{vals['raw_detect']*100:>8.2f}%"
              f"{vals['raw_false']*100:>10.2f}%{vals['tracks']:>8.1f}")
    out = {"files": len(rows), "scored": len(kept), "dropped_low_overlap": dropped}
    print(f"\n  paired, diarizen_native minus sortformer ({len(kept)} files):")
    for label, key in (("detect_id", "detect_id"), ("detect_any", "detect_any"),
                       ("false_id", "false_id")):
        diffs = [(r["diarizen_native"][key] - r["sortformer"][key]) * 100 for r in kept]
        ci = bootstrap_mean_ci(diffs)
        band = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "(n < 5)"
        print(f"    {label:<12}{np.mean(diffs):+8.2f} pp {band}")
        out[label] = {"mean": float(np.mean(diffs)), "ci95": ci}
    detect, false = out["detect_id"], out["false_id"]
    if detect["ci95"] and false["ci95"]:
        if detect["ci95"][0] > 0 and false["ci95"][1] <= 1.0:
            verdict = "真收益：检出′ 下界 > 0 且虚报′ 上界 <= +1.0 pp"
        elif detect["mean"] > 0 and false["ci95"][0] > 1.0:
            verdict = "以虚报换检出，不算收益"
        else:
            verdict = "未能区分"
    else:
        verdict = "未能区分（样本不足以给区间）"
    print(f"\n  => {verdict}")
    out["verdict"] = verdict
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("selfcheck", "compare"))
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    {"selfcheck": cmd_selfcheck, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
