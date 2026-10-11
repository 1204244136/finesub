"""Plan 5k.4 (post hoc, descriptive): what the 5k.2 numbers are made of.

Answers the second review of the branch, from the 5k.2 final run's cache (no new
model runs, same files, plain perturbation):
  - how often the 5 s trigger (what the algorithm used) and the 20 s one fire;
  - per triggered file: split-off units, identities after the merge pass, and the
    reference head count -- the merge pass is one stream with four slots, so
    orphan identities are capped at four;
  - line errors split into *identity* errors (reference has exactly one speaker and
    we put exactly one label) and everything else (boundary / set errors). Only the
    first kind is what the split can fix, and what a proofreader fixes per track.
Not an acceptance: 5k.2's verdict stands; this is the input to the next pre-registration.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import over4
import over4_anchor as anchor
import over4_v2 as v2
from calibrate import parse_rttm, true_speakers_over


def line_sets(probs, reference, mapped):
    from finesub.speech.diarization.assign import assign_subtitle
    from finesub.speech.diarization.segments import segments_from_activity

    speakers = sorted({s.speaker for s in reference})
    timeline = segments_from_activity(probs.tolist())
    for segment in reference:
        truth = {speakers.index(s) for s in true_speakers_over(segment, reference)}
        tracks = assign_subtitle(segment.start, segment.end, timeline).extras.get("tracks", [])
        yield truth, {mapped.get(int(k), -1 - int(k)) for k in tracks}


def breakdown(probs, reference, grid):
    mapped = v2.mapping(probs, grid)
    counts = Counter()
    for truth, ours in line_sets(probs, reference, mapped):
        kind = "identity" if len(truth) == 1 and len(ours) == 1 else "other"
        counts[kind] += 1
        counts[f"{kind}_wrong"] += truth != ours
    return counts


def main():
    import soundfile

    runner = over4.make_runner()
    config = v2.CONFIGS["split-trig5"]
    many, few = v2.remainder()
    out = {}
    for group, rows in (("many", many), ("few", few)):
        totals = {"onepass": Counter(), "split": Counter()}
        triggers = Counter()
        files = []
        for name, speakers, _ in rows:
            waveform, _ = soundfile.read(str(over4.audio(name)), dtype="float32", always_2d=True)
            waveform = waveform.mean(axis=1)
            key = f"{name}-plain"
            p1 = anchor.cached(runner, key, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
            stats = {}
            p2 = v2.split(runner, key, waveform, p1, config, stats)
            triggers["t5"] += v2.slots_full(p1, 5.0)
            triggers["t20"] += v2.slots_full(p1, 20.0)
            reference = parse_rttm(over4.VOX / "voxconverse-master" / "dev" / f"{name}.rttm")
            grid = over4.truth(name, len(p1))
            for arm, probs in (("onepass", p1), ("split", p2)):
                totals[arm].update(breakdown(probs, reference, grid))
            if stats["triggered"]:
                identities = int(((p2 > over4.THRESHOLD).sum(axis=0) * over4.FRAME >= 2.0).sum())
                files.append({"name": name, "speakers": speakers, "units": stats["units"],
                              "groups": stats["groups"], "identities": identities})
        report = {"files": len(rows), "trigger_5s": triggers["t5"], "trigger_20s": triggers["t20"]}
        for arm, c in totals.items():
            report[arm] = {
                "identity_lines": c["identity"], "identity_wrong": c["identity_wrong"],
                "other_lines": c["other"], "other_wrong": c["other_wrong"],
            }
        o, s = report["onepass"], report["split"]
        report["identity_error_reduction"] = round(
            (o["identity_wrong"] - s["identity_wrong"]) / max(o["identity_wrong"], 1), 3)
        report["other_error_change"] = s["other_wrong"] - o["other_wrong"]
        report["triggered_files"] = files
        out[group] = report
        print(group, json.dumps({k: v for k, v in report.items() if k != "triggered_files"}))
        by_people = {}
        for f in files:
            by_people.setdefault(min(f["speakers"], 9), []).append(f)
        for people, fs in sorted(by_people.items()):
            print(f"   {people}{'+' if people == 9 else ''} people: {len(fs)} triggered; identities after split "
                  f"{[f['identities'] for f in fs]}; merged groups {[f['groups'] for f in fs]}; units {[f['units'] for f in fs]}")
    record = over4.DATA / "results" / "over4_breakdown-5k2.json"
    record.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("record ->", record)


if __name__ == "__main__":
    main()
