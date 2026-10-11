"""Freeze the Nemotron-3 round into `results/nemotron3-20260924.json` (CPU only, no inference).

Everything is recomputed from the per-file outputs under the worktree's `tmp/` so the record and
the scripts agree; hypotheses and probabilities stay in `tmp/`, only per-file metrics come in.

    python tools/diarization_explore/nemotron3_freeze.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import nemotron3_probe as n3  # noqa: E402
import over8  # noqa: E402

OUT = HERE / "results" / "nemotron3-20260924.json"
PUBLIC_METHODS = ["n3", "vbx", "vbx-prod1.0", "vbx-step0.8-window8.0", "vbx-ahc0.45", "recur", "recurvbx",
                  "enroll", "enrollmerge", "sortformer", "diarizen"]
OURS_METHODS = ["n3", "vbx", "vbx-step0.8-window8.0", "recur"]

#: The formal timing session asr-playground-20260924-120953 (hw report of that name), one pass per
#: backend on the same waveform, load excluded, fp32, RTX 5070 Ti. Pass 1 is cold. DiariZen's pass 1
#: on onetonton-01 overlapped a Defender disturbance and is void; its pass 2 is clean.
SPEED = {
    "onetonton-01 (2135.5 s)": {"nemotron3": [6.350, 3.062], "sortformer": [17.829, 16.593],
                                "diarizen": [None, 71.072]},
    "hina-40 (4897.3 s)": {"nemotron3": [11.696, 8.100], "sortformer": [26.229, 25.548],
                           "diarizen": [169.353, 168.514]},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def main():
    record = {
        "model": {"repo": "nvidia/Nemotron-3-Diarization", "revision": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
                  "config": n3.OFFLINE, "nemo_speech_commit": "cf724ac337d1ebc7d0dda1e23fb80916f52927a5",
                  "lhotse": "2.0.0a6"},
        "scripts": {p.name: digest(p) for p in sorted(HERE.glob("nemotron3_*.py")) + [HERE / "over8.py"]},
    }
    public = {}
    for ident in n3.CALLS + n3.ALI + n3.VOX:
        path = n3.OUT / ident / "result.json"
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            keys = ("der", "miss", "false_alarm", "overlap_precision", "overlap_recall", "overlap_all_speakers_recall")
            arms = {"nemotron3_native": {"der": r["native"]["der"]},
                    "nemotron3_ours": {k: r["ours"][k] for k in keys}}
            for arm in ("high", "community", "diarizen"):
                c = n3.cached(ident, arm) or {}
                if c:
                    m = c.get("production_metrics", c["metrics"])
                    arms[arm] = {k: m[k] for k in keys}
            public[ident] = {"arms": arms, "reference_speakers": r["reference_speakers"],
                             "tracks": r["tracks_native"], "duration": r["duration"],
                             "gpu_max_allocated_gib": r["max_allocated_gib"], "gpu_peak_gib": r["watchdog_peak_gib"]}
    record["public_der"] = public
    for corpus in ("onetonton", "hina"):
        cost = json.loads((n3.OUT / f"cost-{corpus}.json").read_text(encoding="utf-8"))
        record[f"prereg_cost_{corpus}"] = {
            "paired": cost["paired"],
            "rows": {r["episode"]: {arm: {"edit": r[arm]["cost"]["total"], "fixed": r[arm]["fixed"],
                                          "f1": r[arm]["f1"], "lead10": r[arm]["leading_counts"]["10"]
                                          if "10" in r[arm]["leading_counts"] else r[arm]["leading_counts"][10]}
                                    for arm in ("sortformer", "diarizen", "nemotron3")}
                     for r in cost["rows"]},
        }
    for name in ("R8003_M8001", "hina-40"):
        path = n3.OUT / name / "micro.json"
        if path.exists():
            record.setdefault("perturbation", {})[name] = json.loads(path.read_text(encoding="utf-8"))["arms"]
    record["speed_seconds_per_pass"] = SPEED
    record["cpu"] = json.loads((n3.OUT / "1642" / "cpu.json").read_text(encoding="utf-8"))
    record["threshold_sweep"] = json.loads((n3.OUT / "sweep.json").read_text(encoding="utf-8"))
    # Whole-file NeMo inference on our material: the peak grows with duration (one STFT over the file).
    peaks = {}
    for line in (n3.WT / "tmp" / "nemotron3-cost-infer.log").read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[1] == "frames":
            frames = int(parts[2].strip("(,"))
            peaks[parts[0]] = {"minutes": round(frames * 0.01 / 60, 1), "peak_gib": float(parts[-1])}
    record["gpu_peak_by_duration"] = peaks
    over = {}
    for dataset in ("vox", "mocai", "ali"):
        quiet(over8.score, PUBLIC_METHODS, [dataset])
        over[dataset] = json.loads((over8.OUT / f"score-{dataset}.json").read_text(encoding="utf-8"))
    # ⚠ hina before 姐姐咚咚: `accuracy_cross.configure()` (run for 姐姐咚咚) rebinds
    # `refined_multi.sources` to the 姐姐咚咚 list, after which every hina file would be skipped.
    for dataset in ("hina", "onetonton"):
        quiet(over8.score_ours, OURS_METHODS, dataset)
        over[dataset] = json.loads((over8.OUT / f"score-{dataset}.json").read_text(encoding="utf-8"))
    record["over8"] = {d: {m: rows for m, rows in t.items() if rows} for d, t in over.items()}
    record["over8_screen"] = {"voxs": over8.idents("voxs")}
    OUT.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
