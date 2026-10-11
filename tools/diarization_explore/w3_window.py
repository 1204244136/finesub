"""W3: does DiariZen's 8 s window survive the material closest to the product?

Thresholds were frozen in `docs/archive/diarization-closeout-plan.md` section 2 BEFORE this ran:
8 s against 16 s on 姐姐咚咚 #1-#15, the paired 95% lower bound of both `fixed` and B-cubed F1 at
or above -1.5 pp (section 2a says why 1.5), `leading` paired mean not rising, `covering` reported
as description. Native boundaries get re-read on the same fifteen at a mean of at least -0.3 pp.

    python tools/diarization_explore/w3_window.py infer          # GPU, one process per file
    python tools/diarization_explore/w3_window.py score --write  # offline

⚠ The 16 s arm is NOT re-run: it is the cache the probe already produced
(`tmp/diarization-accuracy-v2/<ident>/diarizen.npz`), bit-exact across processes (probe 7.1), so
re-running it would only spend GPU to reproduce the same array.

⚠ Native boundaries cost no inference. They are a different way of turning the SAME annotation
into tracks -- the official intervals instead of the 80 ms raster plus production smoothing -- so
both window sizes' native arms come from the saved RTTM offline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402
import cost_metrics as cm  # noqa: E402
import relabel_eval as ev  # noqa: E402
from accuracy_identity import identity_score  # noqa: E402
from accuracy_report import bootstrap_mean_ci  # noqa: E402
from accuracy_v2 import OUT as V2_OUT  # noqa: E402

OUT = lab.WT / "tmp" / "diarization-w3"
EPISODES = list(range(1, 16))
WINDOW = 8


def ident_of(n: int) -> str:
    return f"onetonton-{n:02d}"


def worker(ident: str, duration: int) -> None:
    import vram_guard

    vram_guard.install()
    import soundfile as sf
    import torch
    from accuracy_cross import configure
    from accuracy_diarizen import infer, load_pipeline, provenance, raster
    from finesub.speech.runtime.device import cuda_usable

    configure()
    if not cuda_usable():
        raise RuntimeError("CUDA unavailable; this GPU experiment does not silently fall back")
    torch.set_num_threads(4)
    directory = OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    wave, sr = sf.read(lab.source(ident), dtype="float32", always_2d=True)
    assert sr == 16000
    signal = wave.mean(axis=1)
    annotation = infer(load_pipeline(inference_overrides={"seg_duration": duration}), signal)
    probs = raster(annotation, len(signal))
    np.savez_compressed(directory / f"w{duration}.npz", probs=probs)
    (directory / f"w{duration}.rttm").write_text(annotation.to_rttm(), encoding="utf-8")
    (directory / f"w{duration}.json").write_text(json.dumps(dict(
        ident=ident, duration=duration, elapsed=time.perf_counter() - start,
        identity=dict(audio=lab.digest(lab.source(ident)), model=provenance()),
        watchdog_peak_gib=vram_guard.peak_gib(),
    ), indent=2), encoding="utf-8")
    print(f"{ident} w{duration} done, {probs.shape[1]} tracks", flush=True)


def native_tracks(rttm: Path):
    """The model's OWN interval boundaries, as `TrackSegment`s -- never rasterised.

    ⚠ A first version of this built a 0/1 frame map at `lab.FRAME` and handed it to the same path
    as the raster arm. That is the 80 ms grid this arm exists to avoid, so every delta came out
    exactly 0.000 -- a bug that reads like a finding. If this function ever returns something whose
    scores match the raster arm to the last digit, suspect it again rather than reporting it.
    """
    from finesub.speech.diarization.segments import TrackSegment

    rows = []
    for line in rttm.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            rows.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    speakers = sorted({who for *_, who in rows})
    tracks = [TrackSegment(speakers.index(who), a, b) for a, b, who in rows if b > a]
    tracks.sort(key=lambda s: (s.start, s.track))
    return tracks, max(1, len(speakers))


def arms_for(n: int):
    ident = ident_of(n)
    cached16 = V2_OUT / ident / "diarizen.npz"
    fresh8 = OUT / ident / f"w{WINDOW}.npz"
    if not cached16.exists() or not fresh8.exists():
        return None
    return {
        "w16": (np.load(cached16)["probs"], V2_OUT / ident / "diarizen.rttm"),
        "w8": (np.load(fresh8)["probs"], OUT / ident / f"w{WINDOW}.rttm"),
    }


def score_episode(n: int):
    data = ev.episode(n)
    arms = arms_for(n)
    if data is None or arms is None:
        return None
    stable, _, grid, people, _ = data
    import refined_ass as ra

    starts = sorted((a, w) for a, _, w, t in ra.events(ra.episodes()[n - 1]) if w != ra.NOTE and t)
    row = {"episode": n, "people": len(people)}
    from accuracy_native import score_timeline
    from finesub.speech.diarization import segments as seg

    for name, (probs, rttm) in arms.items():
        p = probs[: len(grid)]
        metrics = lab.score(p, stable, grid, people)
        row[f"{name}_raster"] = {
            "fixed": metrics["fixed"], "f1": identity_score(p, stable, grid)["f1"],
            "naming": cm.naming_cost(p, stable, grid),
            "cost": cm.edit_cost(p, stable, grid, people, starts),
        }
        tracks, columns = native_tracks(rttm)
        metrics, bcubed = score_timeline(tracks, columns, stable, grid, people)
        original = seg.segments_from_activity
        try:
            seg.segments_from_activity = lambda *a, **kw: tracks
            placeholder = np.zeros((1, columns))
            row[f"{name}_native"] = {
                "fixed": metrics["fixed"], "f1": bcubed["f1"],
                "naming": cm.naming_cost(placeholder, stable, grid),
                "cost": cm.edit_cost(placeholder, stable, grid, people, starts),
            }
        finally:
            seg.segments_from_activity = original
    return row


def paired(rows, a, b, get):
    diffs = [get(r[a]) - get(r[b]) for r in rows]
    return float(np.mean(diffs)), bootstrap_mean_ci(diffs), diffs


def report(rows, write=False):
    def show(a, b, title, gates):
        print(f"\n{title}  ({len(rows)} episodes, {a} minus {b})")
        out = {}
        for label, get, unit in (
            ("fixed", lambda x: x["fixed"] * 100, "pp"),
            ("B3 F1", lambda x: x["f1"] * 100, "pp"),
            ("leading", lambda x: x["naming"]["leading"], ""),
            ("covering", lambda x: x["naming"]["covering"], ""),
            ("edit cost", lambda x: x["cost"]["total"], ""),
        ):
            mean, ci, diffs = paired(rows, a, b, get)
            band = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "(n < 5)"
            worse = sum(1 for d in diffs if d < 0) if unit == "pp" else sum(1 for d in diffs if d > 0)
            print(f"  {label:<10}{mean:+8.3f}{unit:<3}{band:>20}   worse on {worse}/{len(diffs)}")
            out[label] = {"mean": mean, "ci95": ci}
        for name, ok, why in gates:
            print(f"  [{'PASS' if ok(out) else 'FAIL'}] {name}: {why(out)}")
        return out

    eight = show("w8_raster", "w16_raster", "8 s window", [
        ("fixed lower bound >= -1.5 pp", lambda o: o["fixed"]["ci95"] and o["fixed"]["ci95"][0] >= -1.5,
         lambda o: f"lower bound {o['fixed']['ci95'][0]:+.2f}" if o["fixed"]["ci95"] else "no interval"),
        ("F1 lower bound >= -1.5 pp", lambda o: o["B3 F1"]["ci95"] and o["B3 F1"]["ci95"][0] >= -1.5,
         lambda o: f"lower bound {o['B3 F1']['ci95'][0]:+.2f}" if o["B3 F1"]["ci95"] else "no interval"),
        ("leading does not rise", lambda o: o["leading"]["mean"] <= 0,
         lambda o: f"mean {o['leading']['mean']:+.2f}"),
    ])
    native = show("w16_native", "w16_raster", "native boundaries at 16 s", [
        ("fixed mean >= -0.3 pp", lambda o: o["fixed"]["mean"] >= -0.3,
         lambda o: f"mean {o['fixed']['mean']:+.3f}"),
        ("F1 mean >= -0.3 pp", lambda o: o["B3 F1"]["mean"] >= -0.3,
         lambda o: f"mean {o['B3 F1']['mean']:+.3f}"),
    ])
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "w3-results.json"
        path.write_text(json.dumps({"rows": rows, "eight_vs_sixteen": eight,
                                    "native_vs_raster": native,
                                    "script": lab.digest(__file__)}, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("infer", "score"))
    p.add_argument("--episodes", nargs="*", type=int, default=EPISODES)
    p.add_argument("--duration", type=int, default=WINDOW)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker(ident_of(args.episodes[0]), args.duration)
        return
    if args.command == "infer":
        OUT.mkdir(parents=True, exist_ok=True)
        for n in args.episodes:
            if (OUT / ident_of(n) / f"w{args.duration}.npz").exists():
                print("CACHED", ident_of(n), flush=True)
                continue
            print("START", ident_of(n), flush=True)
            with (OUT / f"{ident_of(n)}-w{args.duration}.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", __file__, "infer", "--worker",
                                "--episodes", str(n), "--duration", str(args.duration)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print("END", ident_of(n), flush=True)
        return
    from accuracy_cross import configure

    configure()
    rows = [r for r in (score_episode(n) for n in args.episodes) if r]
    print(f"scored {len(rows)} episodes")
    report(rows, write=args.write)


if __name__ == "__main__":
    main()
