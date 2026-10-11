"""Third review, items 2-4: re-score 5k.2 and 5n.3 from the cache (no new passes expected).

  2. identity error on a *fixed* population: every line whose reference has exactly one
     speaker; wrong unless our label set is exactly that speaker. The old count took only
     lines where we also put exactly one label, so a second label moved a hard line out of
     the denominator instead of counting as an error.
  3. the 5n.1 verifier on test: of the units it vetoed, how many were true orphans (the
     unit's single-speaker frames belong mostly to another reference speaker than the
     host's kept frames do).
  4. 5n.3 by head count: the merge pass is one four-slot stream, so orphan identities cap.

Writes per-file rows + git HEAD + times to data/diarization/results/.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter

import numpy as np

import over4
import over4_anchor as anchor
import over4_final as final
import over4_v2 as v2
from calibrate import parse_rttm
from over4_breakdown import line_sets
from over4_verify import same_person

FRAME, THRESHOLD = over4.FRAME, over4.THRESHOLD
BINS = ((5, 6), (7, 9), (10, 15), (16, 99))


def fixed_identity(probs, reference, grid):
    mapped = v2.mapping(probs, grid)
    out = Counter()
    for truth, ours in line_sets(probs, reference, mapped):
        if len(truth) != 1:
            continue
        out["lines"] += 1
        out["wrong"] += ours != truth
        out["multi_label"] += len(ours) > 1
        out["old_population"] += len(ours) == 1
        out["old_wrong"] += len(ours) == 1 and ours != truth
    return dict(out)


def majority(frames, grid):
    frames = frames[(frames < len(grid))]
    single = frames[grid[frames].sum(axis=1) == 1]
    if single.size == 0:
        return None
    return int(np.bincount(grid[single].argmax(axis=1), minlength=grid.shape[1]).argmax())


def test_file(runner, name, kind):
    import soundfile

    path = final.raw_path(name) if kind == "raw" else final.ogg_path(name)
    signal, _ = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = signal.mean(axis=1)
    key = f"test-{name}-{kind}"
    p1 = anchor.cached(runner, key, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
    reference, grid = final.truth(name, len(p1))
    owner = np.where((p1 > THRESHOLD).any(axis=1), p1.argmax(axis=1), -1)
    decisions = []

    def veto(index, unit, stats):
        identities = {f"keep{unit['host']}": stats["keep_frames"][unit["host"]]}
        identities.update({f"track{g}": np.flatnonzero(owner == g) for g in range(p1.shape[1]) if g != unit["host"]})
        vetoed = any(same_person(runner, f"{key}-u{index}-{label}", waveform, frames, unit["frames"])
                     for label, frames in identities.items() if frames.size * FRAME >= 5)
        host, orphan = majority(np.asarray(stats["keep_frames"][unit["host"]]), grid), majority(np.asarray(unit["frames"]), grid)
        decisions.append({"vetoed": bool(vetoed), "true_orphan": None if host is None or orphan is None else host != orphan})
        return vetoed

    config = v2.CONFIGS["split-trig5"]
    arms = {"onepass": p1, "split": v2.split(runner, key, waveform, p1, config, {}),
            "veto": v2.split(runner, key, waveform, p1, config, {}, veto)}
    return {arm: fixed_identity(probs, reference, grid) for arm, probs in arms.items()}, decisions


def dev_file(runner, name):
    import soundfile

    waveform, _ = soundfile.read(str(over4.audio(name)), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    key = f"{name}-plain"
    p1 = anchor.cached(runner, key, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
    p2 = v2.split(runner, key, waveform, p1, v2.CONFIGS["split-trig5"], {})
    reference = parse_rttm(over4.VOX / "voxconverse-master" / "dev" / f"{name}.rttm")
    grid = over4.truth(name, len(p1))
    return {arm: fixed_identity(probs, reference, grid) for arm, probs in (("onepass", p1), ("split", p2))}


def rate(rows, arm, wrong="wrong", total="lines"):
    return sum(r[arm][wrong] for r in rows) / max(sum(r[arm][total] for r in rows), 1)


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    started = time.time()
    runner = over4.make_runner()
    out = {"head": head, "started": started}
    many, few = v2.remainder()
    dev = [{"group": g, "name": n, "speakers": s, **dev_file(runner, n)} for g, rows in (("many", many), ("few", few)) for n, s, _ in rows]
    for group in ("many", "few"):
        rows = [r for r in dev if r["group"] == group]
        out[f"5k2/{group}"] = {arm: {"fixed_error": round(rate(rows, arm), 4), "old_error": round(rate(rows, arm, "old_wrong", "old_population"), 4),
                                     "lines": sum(r[arm]["lines"] for r in rows), "old_lines": sum(r[arm]["old_population"] for r in rows)}
                               for arm in ("onepass", "split")}
        print("5k.2", group, json.dumps(out[f"5k2/{group}"]), flush=True)
    t_many, t_few = final.select()
    test, decisions = [], Counter()
    for group, rows in (("many", t_many), ("few", t_few)):
        for name, speakers, _ in rows:
            for kind in ("raw", "ogg"):
                scores, made = test_file(runner, name, kind)
                test.append({"group": group, "name": name, "speakers": speakers, "kind": kind, **scores})
                if kind == "ogg":
                    for d in made:
                        decisions[(d["vetoed"], d["true_orphan"])] += 1
    for kind in ("raw", "ogg"):
        for group in ("many", "few"):
            rows = [r for r in test if r["kind"] == kind and r["group"] == group]
            out[f"5n3/{kind}/{group}"] = {arm: {"fixed_error": round(rate(rows, arm), 4), "old_error": round(rate(rows, arm, "old_wrong", "old_population"), 4),
                                                "lines": sum(r[arm]["lines"] for r in rows), "old_lines": sum(r[arm]["old_population"] for r in rows)}
                                          for arm in ("onepass", "split", "veto")}
            print("5n.3", kind, group, json.dumps(out[f"5n3/{kind}/{group}"]), flush=True)
    by_heads = {}
    for low, high in BINS:
        rows = [r for r in test if r["kind"] == "ogg" and low <= r["speakers"] <= high]
        if rows:
            o, s = rate(rows, "onepass"), rate(rows, "split")
            by_heads[f"{low}-{high}"] = {"files": len(rows), "onepass": round(o, 4), "split": round(s, 4),
                                          "reduction": round((o - s) / max(o, 1e-9), 3)}
    out["5n3/ogg/by_heads"] = by_heads
    print("by heads", json.dumps(by_heads), flush=True)
    out["5n1_on_test_ogg"] = {f"vetoed={v} true_orphan={t}": n for (v, t), n in sorted(decisions.items(), key=str)}
    print("verifier on test (ogg):", out["5n1_on_test_ogg"], flush=True)
    out["finished"] = time.time()
    record = over4.DATA / "results" / f"over4_rescore-{time.strftime('%Y%m%d-%H%M%S')}.json"
    record.write_text(json.dumps({"summary": out, "dev": dev, "test": test}), encoding="utf-8")
    print("record ->", record)


if __name__ == "__main__":
    main()
