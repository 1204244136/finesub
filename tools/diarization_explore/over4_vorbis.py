"""Plan 5m.2: isolate Vorbis. Raw wav vs the same wav through the production encoder.

`encode_asr_delivery` is the `--no-separate` path; VoxConverse is already 16 kHz, so
the only difference between the two inputs is the Vorbis encode. onepass only
(the default production path), default geometry, on the 5k remainder. The raw arm
reuses the 5k.2 cache ("<name>-plain").
"""

from __future__ import annotations

import json
import subprocess
import time

import over4
import over4_anchor as anchor
import over4_v2 as v2
from calibrate import parse_rttm


def vorbis(name):
    path = over4.WORK / "vox-ogg" / f"{name}-vocal.ogg"
    if not path.exists():
        from finesub.speech.preprocessing.separator.separation import encode_asr_delivery

        encode_asr_delivery(over4.audio(name), path)
    return path


def main():
    import soundfile

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    started = time.time()
    runner = over4.make_runner()
    many, few = v2.remainder()
    rows = []
    for group, files in (("many", many), ("few", few)):
        for name, speakers, _ in files:
            raw, _ = soundfile.read(str(over4.audio(name)), dtype="float32", always_2d=True)
            raw = raw.mean(axis=1)
            ogg, rate = soundfile.read(str(vorbis(name)), dtype="float32", always_2d=True)
            assert rate == 16000, rate
            ogg = ogg.mean(axis=1)
            p_raw = anchor.cached(runner, f"{name}-plain", "onepass", lambda: {"probs": over4.run(runner, raw)})["probs"]
            p_ogg = anchor.cached(runner, f"{name}-vorbis", "onepass", lambda: {"probs": over4.run(runner, ogg)})["probs"]
            reference = parse_rttm(over4.VOX / "voxconverse-master" / "dev" / f"{name}.rttm")
            scored = {}
            for arm, probs in (("onepass", p_raw), ("split", p_ogg)):  # "split" slot = the ogg arm
                grid = over4.truth(name, len(probs))
                mapped = v2.mapping(probs, grid)
                lines = v2.line_scores(probs, reference, mapped)
                scored[arm] = {**v2.frame_scores(probs, grid, mapped), "lines": len(lines),
                               "lines_right": sum(r for r, _ in lines)}
            rows.append({"group": group, "name": name, "speakers": speakers, "perturbation": "plain", **scored})
    summary = {"head": head, "started": started, "finished": time.time(), "arms": "onepass=raw wav, split=vorbis ogg"}
    for group in ("many", "few"):
        for field in ("single", "line"):
            a, b = v2.pooled(rows, group, "onepass", field), v2.pooled(rows, group, "split", field)
            lo, hi = v2.bootstrap(rows, group, field)
            summary[f"{group}/{field}"] = {"raw": round(a, 4), "ogg": round(b, 4), "delta": round(b - a, 4),
                                           "ci95": [round(lo, 4), round(hi, 4)]}
    worst = sorted(
        ((r["name"], r["group"], r["onepass"]["single_right"] / max(r["onepass"]["single_frames"], 1),
          r["split"]["single_right"] / max(r["split"]["single_frames"], 1)) for r in rows),
        key=lambda x: x[3] - x[2])[:6]
    summary["largest_drops"] = [(n, g, round(a, 3), round(b, 3)) for n, g, a, b in worst]
    print(json.dumps(summary, indent=1))
    record = over4.DATA / "results" / f"over4_vorbis-{time.strftime('%Y%m%d-%H%M%S')}.json"
    record.write_text(json.dumps({"summary": summary, "files": rows}), encoding="utf-8")
    print("record ->", record)


if __name__ == "__main__":
    main()
