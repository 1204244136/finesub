"""Plan 5o.5: where the two acoustic hints' values should sit, against speaker truth.

Tracks are what production writes: `-diar.json` for AliMeeting / AMI (the S4 runs) and the
three streams, and VoxConverse's cached onepass probabilities smoothed with production's
`segments_from_activity` defaults. Truth: TextGrid (AliMeeting), AMI words, RTTM.

  --collect SOURCE   ali | ami | vox | streams -> tmp/rescore-5o4/hints-SOURCE.json
                     (ali / ami read the S4 layout: run ami with DIAR_S4_CORPUS=ami)
  --decide           read every collected file, apply 5o.5's rules, print the verdict
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from finesub.speech.diarization.segments import TrackSegment, pair_co_activity, segments_from_activity, track_seconds

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "tmp" / "rescore-5o4"
FRAME = 0.08
PURITY = 0.70
RATIOS = (0.05, 0.10, 0.15, 0.20)
FLOORS = (0.0, 5.0, 10.0, 20.0)
SHARES = (0.02, 0.05)


def truth_grid(rows, frames):
    """rows: (speaker, start, end) -> speakers, bool[frames, speakers]."""
    speakers = sorted({s for s, _, _ in rows})
    grid = np.zeros((frames, len(speakers)), dtype=bool)
    for speaker, start, end in rows:
        grid[int(start / FRAME): int(np.ceil(end / FRAME)), speakers.index(speaker)] = True
    return speakers, grid


def describe(name, tracks, truth_rows):
    """One file: every track pair (shorter seconds, co-activity, same / different / mixed)
    and the per-track seconds for the capacity rules."""
    frames = int(max([t.end for t in tracks] + [e for _, _, e in truth_rows] + [1.0]) / FRAME) + 2
    speakers, grid = truth_grid(truth_rows, frames)
    active = np.zeros((frames, 1 + max((t.track for t in tracks), default=0)), dtype=bool)
    for t in tracks:
        active[int(t.start / FRAME): int(np.ceil(t.end / FRAME)), t.track] = True
    owner = {}
    for k in range(active.shape[1]):
        on = active[:, k]
        if on.sum() == 0 or grid.shape[1] == 0:
            continue
        shares = (grid[on]).sum(axis=0) / on.sum()
        owner[k] = (speakers[int(shares.argmax())], float(shares.max()))
    seconds = track_seconds(tracks)
    pairs = []
    for (a, b), value in pair_co_activity(tracks).items():
        if a not in owner or b not in owner:
            continue
        pure = owner[a][1] >= PURITY and owner[b][1] >= PURITY
        kind = ("same" if owner[a][0] == owner[b][0] else "different") if pure else "mixed"
        pairs.append({"pair": f"s{a}-s{b}", "shorter": min(seconds[a], seconds[b]), "co": value, "kind": kind})
    union = float(active.any(axis=1).sum() * FRAME)
    return {"name": name, "speakers": len(speakers), "track_seconds": sorted(seconds.values(), reverse=True),
            "speech_seconds": union, "pairs": pairs}


def read_tracks(path):
    return [TrackSegment(int(t["track"]), float(t["start"]), float(t["end"]))
            for t in json.loads(Path(path).read_text(encoding="utf-8"))["tracks"]]


def collect(source):
    rows = []
    if source in ("ali", "ami"):
        import s4_headroom as ref
        import s4_projection as proj

        assert ref.CORPUS == ("ami" if source == "ami" else "alimeeting"), "set DIAR_S4_CORPUS to match"
        for name in ref.SESSIONS:
            truth = [(r["spk"], r["start"], r["end"]) for r in ref.reference(name)]
            rows.append(describe(name, read_tracks(proj.ALI / name / f"{name}-diar.json"), truth))
    elif source == "vox":
        import over4
        import over4_final as final
        from calibrate import parse_rttm

        cache = over4.WORK / "over4-cache" / "default"
        for group, pattern, rttm_dir in (("test-ogg", "test-*-ogg-onepass.npz", final.TEST),
                                         ("dev-plain", "*-plain-onepass.npz", over4.VOX / "voxconverse-master" / "dev")):
            for path in sorted(cache.glob(pattern)):
                name = path.name.removeprefix("test-").removesuffix("-ogg-onepass.npz").removesuffix("-plain-onepass.npz")
                probs = np.load(path, allow_pickle=True)["probs"]
                tracks = segments_from_activity(np.asarray(probs, dtype=float).tolist())
                truth = [(s.speaker, s.start, s.end) for s in parse_rttm(rttm_dir / f"{name}.rttm")]
                rows.append({"group": group, **describe(name, tracks, truth)})
    elif source == "streams":
        roles = Path(os.environ.get("DIAR_EXPLORE_WORK", HERE.parents[1] / "tmp" / "diarization_explore")) / "roles"
        known = {"kaguya60": 1, "nepolabo": 4, "yui": None}  # yui: streamer + a film's many voices
        for name, people in known.items():
            tracks = read_tracks(roles / f"{name}-diar.json")
            row = describe(name, tracks, [])  # no truth: only the capacity rules and the raw co-activity are read
            row.update(speakers=people, co={f"s{a}-s{b}": round(v, 4) for (a, b), v in pair_co_activity(tracks).items()})
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (OUT / f"hints-{source}.json").write_text(json.dumps({"head": head, "rows": rows}, indent=1), encoding="utf-8")
    print(source, len(rows), "files")


def capacity(row, rule):
    seconds = row["track_seconds"]
    if len(seconds) < 4:
        return False
    if rule == "20s":
        return seconds[3] >= 20.0
    return seconds[3] >= rule * row["speech_seconds"]


def decide():
    data = {p.stem.removeprefix("hints-"): json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("hints-*.json"))}
    print("heads:", {k: v["head"][:8] for k, v in data.items()})
    pairs = [dict(p, source=k) for k, v in data.items() if k != "streams" for r in v["rows"] for p in r["pairs"]]
    same = [p for p in pairs if p["kind"] == "same"]
    diff = [p for p in pairs if p["kind"] == "different"]
    print(f"\nnot_same: pairs same {len(same)} / different {len(diff)} / mixed {sum(p['kind'] == 'mixed' for p in pairs)}")
    by_source = defaultdict(lambda: [0, 0])
    for p in pairs:
        by_source[p["source"]][p["kind"] == "same"] += p["kind"] != "mixed"
    print("  per source (different, same):", dict(by_source))
    table = []
    for floor, ratio in product(FLOORS, RATIOS):
        flag = lambda p: p["co"] > ratio and p["shorter"] >= floor  # noqa: E731
        false_same = sum(map(flag, same)) / max(len(same), 1)
        recall = sum(map(flag, diff)) / max(len(diff), 1)
        table.append((floor, ratio, false_same, recall))
        print(f"  floor {floor:4.0f} s  ratio {ratio:4.0%}: same flagged {false_same:6.1%}  different flagged {recall:6.1%}")
    if len(same) < 10:
        print(f"  => fewer than 10 same-person pairs ({len(same)}): no value is set (5o.5)")
    else:
        ok = [row for row in table if row[2] <= 0.05]
        if ok:
            floor, ratio, false_same, recall = min(ok, key=lambda r: (r[0], r[1]))
            print(f"  => floor {floor:.0f} s, ratio {ratio:.0%}: same flagged {false_same:.1%}, different flagged {recall:.1%}")
        else:
            print("  => no rule keeps same-person pairs under 5%")
    print("\ncapacity_warning: trigger rate per group")
    groups = defaultdict(list)
    for k, v in data.items():
        for r in v["rows"]:
            if k == "vox":
                groups[f"vox {r['group']} {'>=5' if r['speakers'] >= 5 else '2-4' if r['speakers'] >= 2 else '1'}"].append(r)
            elif k == "ami":
                groups["ami (4 people)"].append(r)
            elif k == "ali":
                groups[f"ali ({r['speakers']} people)"].append(r)
            else:
                groups[f"stream {r['name']} ({r['speakers']})"].append(r)
    rules = ("20s",) + SHARES
    verdict = {}
    for rule in rules:
        line = "  ".join(f"{g}: {sum(capacity(r, rule) for r in rs)}/{len(rs)}" for g, rs in sorted(groups.items()))
        print(f"  {str(rule):>5}: {line}")
        many = [r for g, rs in groups.items() if g.startswith("vox") and g.endswith(">=5") for r in rs]
        few = [r for g, rs in groups.items() if g.startswith("vox") and g.endswith("2-4") for r in rs]
        verdict[rule] = (sum(capacity(r, rule) for r in many) / max(len(many), 1), sum(capacity(r, rule) for r in few) / max(len(few), 1))
    for rule, (many, few) in verdict.items():
        print(f"  {str(rule):>5}: >=5 {many:.1%}  2-4 {few:.1%}  {'PASS' if many >= 0.40 and few <= 0.05 else 'fail'}")
    passing = [rule for rule, (many, few) in verdict.items() if many >= 0.40 and few <= 0.05]
    print("  =>", f"switch to {passing[0]}" if passing and passing[0] != "20s" else "keep 20 s" if "20s" in passing
          else "no rule passes: report the fourth track's share, drop the boolean and the runtime warning")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", choices=("ali", "ami", "vox", "streams"))
    parser.add_argument("--decide", action="store_true")
    arguments = parser.parse_args()
    if arguments.collect:
        collect(arguments.collect)
    if arguments.decide:
        decide()


if __name__ == "__main__":
    main()
