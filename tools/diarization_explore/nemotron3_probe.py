"""Nemotron-3-Diarization on the public-truth sets the Sortformer/DiariZen rounds used.

One file per process (NeMo eager, fp32, offline 30.4 s config). References are the
rows already frozen in the earlier rounds' cached `high.json`, so every arm is scored
against byte-identical truth with the same `accuracy_overlap.score_intervals`.

Two hypotheses per file:
  native  — NeMo's own postprocessed segments from `diarize()`
  ours    — the 10 ms probabilities through production `segments_from_activity`

Outputs under <worktree>/tmp/nemotron3/<ident>/; `--summary` prints the comparison table,
`--micro` runs the base / replay / x(1+1e-7) / PCM16 stability probe on one file,
`--sweep` the smoothing-threshold sweep, `--cpu IDENT` a rough CPU wall clock (hide the GPU first).

Needs NeMo newer than PyPI 3.0.0 (RoPE TransformerEncoder), without touching the asr env:
NVIDIA-NeMo/Speech@cf724ac `nemo/` unpacked to tmp/nemo-main and `lhotse==2.0.0a6`
pip-installed `--no-deps --target tmp/n3-deps`, both put first on PYTHONPATH.
⚠ Training data includes VoxConverse dev+test and NIST SRE2000 CALLHOME Part 1:
VoxConverse readings are contaminated; CALLHOME Japanese may overlap. AliMeeting Eval is clean.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
# The public corpora and earlier rounds' caches live in the main checkout, not in a worktree.
MAIN = Path(*WT.parts[: WT.parts.index(".worktrees")]) if ".worktrees" in WT.parts else WT
PUBLIC = MAIN / "data/diarization/public"
MODEL = MAIN / "tmp/diarization-models/nemotron3/Nemotron-3-Diarization.nemo"  # a main-checkout resource
OUT = WT / "tmp/nemotron3"
CACHES = [MAIN / "tmp/diarization-accuracy-v2/overlap", MAIN / "tmp/diarization-dz3/overlap"]
CALLS = ["2188", "2235", "2225", "2218", "1967", "2223", "2096", "1642", "2210", "1928"]
ALI = ["R8001_M8004", "R8009_M8018", "R8007_M8010", "R8003_M8001", "R8007_M8011"]
VOX = ["vox-bmsyn", "vox-jjvkx", "vox-aiqwk", "vox-auzru", "vox-aggyz", "vox-bvqnu"]
OFFLINE = dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300, spkcache_len=264)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WT / "src"))


def audio_path(ident):
    if ident.startswith("vox-"):
        return MAIN / "tmp/diarization-dz3/vox-audio" / f"{ident[4:]}.wav"
    corpus = "alimeeting" if ident.startswith("R") else "callhome-jpn"
    return PUBLIC / corpus / "audio" / f"{ident}.wav"


def cached(ident, arm):
    for root in CACHES:
        path = root / ident / f"{arm}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def parse_segments(rows):
    out = []
    for row in rows:
        a, b, who = (row.split() if isinstance(row, str) else row)[:3]
        out.append((float(a), float(b), str(who)))
    return out


def load_model():
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(str(MODEL), map_location="cuda", strict=False)
    model.eval()
    for key, value in OFFLINE.items():
        setattr(model.sortformer_modules, key, value)
    model._check_streaming_parameters()
    return model


def infer(model, wave, sr=16000):
    """(NeMo's own segments, (frames, 8) probabilities at 10 ms) for one mono waveform."""
    import torch
    with torch.inference_mode():
        segments, probs = model.diarize(audio=[wave], sample_rate=sr, batch_size=1,
                                        include_tensor_outputs=True, verbose=False)
    probs = probs[0] if isinstance(probs, (list, tuple)) else probs
    return parse_segments(segments[0]), probs.squeeze(0).float().cpu().numpy()


def worker(ident):
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from accuracy_overlap import score_intervals
    from finesub.speech.diarization.segments import segments_from_activity

    reference = cached(ident, "high")["reference"]
    waveform, sr = sf.read(audio_path(ident), dtype="float32", always_2d=True)
    assert sr == 16000, sr
    waveform = waveform.mean(axis=1)
    duration = len(waveform) / sr

    t0 = time.perf_counter()
    model = load_model()
    load_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    native, probs = infer(model, waveform, sr)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t1
    frame = duration / probs.shape[0]
    ours = [(t.start, t.end, str(t.track)) for t in segments_from_activity(probs.tolist(), frame_seconds=round(frame, 3))]

    directory = OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "probs.npz", probs=probs)
    result = dict(ident=ident, duration=duration, frame_seconds=frame, probs_shape=list(probs.shape),
                  config=OFFLINE, load_s=load_s, infer_s=infer_s, rtfx=duration / infer_s,
                  watchdog_peak_gib=vram_guard.peak_gib(),
                  max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  tracks_native=len({s for *_, s in native}), tracks_ours=len({s for *_, s in ours}),
                  reference_speakers=len({s for *_, s in reference}),
                  native=score_intervals(reference, native, duration),
                  ours=score_intervals(reference, ours, duration),
                  hypothesis_native=native, hypothesis_ours=ours)
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(ident, f"DER native={result['native']['der']:.4f} ours={result['ours']['der']:.4f}",
          f"rtfx={result['rtfx']:.0f} frame={frame:.4f}", flush=True)


def micro(ident):
    """Input-perturbation stability, one process: base / replay / x(1+1e-7) / PCM16 grid.

    `self_der` scores each arm's segments against the base run's segments (base as truth),
    so it measures how much of the timeline was redrawn, independent of the real reference.
    """
    import vram_guard
    vram_guard.install()
    import soundfile as sf
    from accuracy_overlap import score_intervals
    from finesub.speech.diarization.segments import segments_from_activity

    # Our own material has no RTTM; there only the self-comparison is scored.
    reference = (cached(ident, "high") or {}).get("reference")
    import nemotron3_cost as nc
    path = audio_path(ident) if reference else nc.source(ident)
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    duration = len(wave) / sr
    arms = {"base": wave, "replay": wave,
            "micro": (wave.astype(np.float64) * (1 + 1e-7)).astype(np.float32),
            # Not a gain change, so no normalisation anywhere can absorb it (dz_probe's arm).
            "noise": (wave + np.random.default_rng(0).normal(0.0, 1e-7, len(wave))).astype(np.float32),
            "pcm16": (np.round(wave * 32768.0) / 32768.0).astype(np.float32)}
    model = load_model()
    probs, hyps, out = {}, {}, {}
    for arm, signal in arms.items():
        _, probs[arm] = infer(model, signal, sr)
        hyps[arm] = [(t.start, t.end, str(t.track)) for t in
                     segments_from_activity(probs[arm].tolist(), frame_seconds=0.01)]
    for arm in arms:
        n = min(len(probs[arm]), len(probs["base"]))
        out[arm] = dict(bit_exact=bool(np.array_equal(probs[arm], probs["base"])),
                        max_prob_delta=float(np.abs(probs[arm][:n] - probs["base"][:n]).max()),
                        der=score_intervals(reference, hyps[arm], duration)["der"] if reference else None,
                        self_der=score_intervals(hyps["base"], hyps[arm], duration)["der"] if hyps["base"] else None,
                        # A mid-file identity flip shows up as confusion; boundary jitter as miss/FA.
                        self_parts={k: score_intervals(hyps["base"], hyps[arm], duration)[k]
                                    for k in ("miss", "false_alarm", "confusion")} if hyps["base"] else None,
                        tracks=len({s for *_, s in hyps[arm]}))
        print(ident, arm, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out[arm].items()}, flush=True)
    directory = OUT / ident
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "micro.json").write_text(json.dumps(dict(ident=ident, arms=out, hypotheses=hyps), indent=2),
                                          encoding="utf-8")


SWEEP = [(0.5, 0.4), (0.4, 0.3), (0.3, 0.2), (0.2, 0.15)]


def sweep():
    """Is the AliMeeting miss a threshold artefact? Smoothing enter/leave over the cached probs, CPU only.

    VoxConverse is the `over8` screening subset (every 5th >= 9-speaker file).
    """
    import over8
    from accuracy_overlap import score_intervals
    from finesub.speech.diarization.segments import SmoothingParameters, segments_from_activity

    out = {}
    for name, ids in (("ali", ALI), ("voxs", over8.idents("voxs"))):
        data = [(np.load(over8.probs_path(i))["probs"], over8.reference(i)) for i in ids]
        for enter, leave in SWEEP:
            params = SmoothingParameters(enter=enter, leave=leave)
            ms = [score_intervals(ref, [(t.start, t.end, str(t.track)) for t in segments_from_activity(
                probs.tolist(), frame_seconds=0.01, parameters=params)], probs.shape[0] * 0.01) for probs, ref in data]
            row = {k: float(np.mean([m[k] for m in ms]))
                   for k in ("der", "miss", "false_alarm", "confusion", "overlap_recall")}
            out[f"{name} enter={enter} leave={leave}"] = dict(row, n=len(ms))
            print(name, enter, leave, {k: round(100 * v, 2) for k, v in row.items()}, flush=True)
    (OUT / "sweep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cpu(ident):
    """Rough CPU wall clock for the model alone on one file, and agreement with the GPU run."""
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    wave, sr = sf.read(audio_path(ident), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    model = SortformerEncLabelModel.restore_from(str(MODEL), map_location="cpu", strict=False)
    model.eval()
    for key, value in OFFLINE.items():
        setattr(model.sortformer_modules, key, value)
    model._check_streaming_parameters()
    t = time.perf_counter()
    _, probs = infer(model, wave, sr)
    elapsed = time.perf_counter() - t
    gpu = np.load(OUT / ident / "probs.npz")["probs"]
    n = min(len(gpu), len(probs))
    record = dict(ident=ident, audio_seconds=len(wave) / sr, seconds=elapsed, rtf=elapsed / (len(wave) / sr),
                  threads=torch.get_num_threads(), max_prob_delta=float(np.abs(probs[:n] - gpu[:n]).max()),
                  frame_agreement=float(np.mean((probs[:n] > 0.5) == (gpu[:n] > 0.5))))
    (OUT / ident / "cpu.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(record, flush=True)


def summary():
    rows = []
    for ident in CALLS + ALI + VOX:
        path = OUT / ident / "result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        base = {arm: (cached(ident, arm) or {}).get("metrics", {}).get("der") for arm in ("high", "community")}
        # DiariZen: the production-smoothed reading where the round kept one (13 overlap files),
        # else the dz3 round's research-path default (16 s window) — marked with *.
        dz, mark = cached(ident, "diarizen"), " "
        if dz and "production_metrics" in dz:
            base["diarizen"] = dz["production_metrics"]["der"]
        else:
            path = MAIN / "tmp/diarization-dz3" / ident / "w16_default.json"
            if path.exists():
                base["diarizen"], mark = json.loads(path.read_text(encoding="utf-8"))["metrics"]["der"], "*"
        rows.append((ident, r["reference_speakers"], r["tracks_native"], r["native"]["der"], r["ours"]["der"],
                     base.get("high"), base.get("diarizen"), mark, base.get("community"),
                     r["native"]["overlap_all_speakers_recall"], r["rtfx"]))
    fmt = lambda v: "   -  " if v is None else f"{100*v:6.2f}"
    print("ident          ref hyp  n3-nat  n3-our  v2.1hi  dzen     comm   ovl-all  rtfx")
    for ident, ref, hyp, nat, our, hi, dz, mark, cm, ovl, rtfx in rows:
        print(f"{ident:<14} {ref:>3} {hyp:>3} {fmt(nat)} {fmt(our)} {fmt(hi)} {fmt(dz)}{mark} {fmt(cm)}  {fmt(ovl)} {rtfx:5.0f}")
    for label, group in (("callhome", CALLS), ("alimeeting", ALI), ("voxconverse", VOX)):
        sel = [row for row in rows if row[0] in group]
        if sel:
            mean = lambda k: None if any(row[k] is None for row in sel) else float(np.mean([row[k] for row in sel]))
            print(f"mean {label:<11} n={len(sel):<2}  {fmt(mean(3))} {fmt(mean(4))} {fmt(mean(5))} {fmt(mean(6))}  {fmt(mean(8))}  {fmt(mean(9))}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ident", nargs="?")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--micro", action="store_true", help="input-perturbation stability probe on IDENT")
    p.add_argument("--sweep", action="store_true", help="smoothing threshold sweep (AliMeeting, VoxConverse subset)")
    p.add_argument("--cpu", action="store_true", help="CPU wall clock on IDENT (hide the GPU yourself)")
    args = p.parse_args()
    if args.summary:
        summary()
    elif args.sweep:
        sweep()
    elif args.cpu:
        cpu(args.ident)
    elif args.micro:
        micro(args.ident)
    else:
        worker(args.ident)


if __name__ == "__main__":
    main()
