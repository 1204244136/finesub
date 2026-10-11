"""Does DiariZen have Sortformer's two failure modes? Protocol frozen in
`docs/archive/diarization-dz-probes-20260918.md` BEFORE any of this ran.

Sortformer is streaming, and its arrival-order speaker cache gives it two behaviours we measured:
a track can change owner mid-file (findings 4.2 and 4.3), and a 1e-7 nudge on the input redraws the
whole timeline (findings 5.1). Each of those holds up a product commitment -- `-diar.json` is kept
rather than recomputed (plan C6), and A/B on this chain must decode with
`-c:a pcm_f32le` (findings 9). DiariZen clusters the whole file offline and should have neither.
If it does not, both commitments can be relaxed FOR THAT BACKEND. Three rounds never checked.

  python tools/diarization_explore/dz_probe.py replay --ident onetonton-14   # P4 self-check, run first
  python tools/diarization_explore/dz_probe.py stress                        # P1  #14 against the arms on file
  python tools/diarization_explore/dz_probe.py owners                        # P2  track-owner stability, 15 episodes
  python tools/diarization_explore/dz_probe.py micro --ident onetonton-14    # P3  base / x(1+1e-7) / PCM16 grid
  python tools/diarization_explore/dz_probe.py sensitivity --ident onetonton-14   # W7  five arms, BOTH models

⚠ `owners` prints the Sortformer production arm through the SAME function, so the comparison is
equal by construction rather than by two implementations agreeing. Its Sortformer column must
reproduce `relabel_eval.py --owners`; if it does not, this file is wrong, not that one.

⚠ `micro` is a pure gain change, which DiariZen normalises away before the model sees it
(`normalize_waveform` is True for wavlm_large_s80_md, so the waveform goes through F.layer_norm),
while Sortformer's mel front end does not. So that arm CANNOT compare the two -- `sensitivity`
adds arms that reach both and keeps `micro` only as a negative control. See the plan, 7.7 and 8.

⚠ The owner rule is biased toward "no change" for a model that emits more tracks: more tracks are
purer, so each clears the 60% share more easily, and each is quieter, so more windows fail the 30 s
floor and drop out of the test entirely. That is why the qualifying-window count is printed next to
the result -- `0/15` on half as many windows is not the same finding.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402
import relabel_eval as ev  # noqa: E402
from accuracy_v2 import OUT as V2_OUT  # noqa: E402

FRAME = ev.FRAME
EPISODES = range(1, 16)
#: `relabel_eval.owners`' rule, verbatim. Changing these makes the Sortformer column incomparable.
WINDOW, MIN_SECONDS, MIN_SHARE = 300.0, 30.0, 0.60


def diarizen_probs(n: int):
    path = V2_OUT / f"onetonton-{n:02d}" / "diarizen.npz"
    return np.load(path)["probs"] if path.exists() else None


def tracks_from_probs(probs):
    from finesub.speech.diarization.segments import segments_from_activity

    return [(t.track, t.start, t.end) for t in segments_from_activity(probs.tolist())]


def owner_windows(tracks, grid):
    """(track, window) -> (majority person index, share, frames) for windows that qualify.

    One function for both arms; `tracks` is (track, start, end) either way.
    """
    counts, frames = defaultdict(Counter), Counter()
    for track, start, end in tracks:
        for i in range(int(start / FRAME), min(int(np.ceil(end / FRAME)), len(grid))):
            w = int(i * FRAME // WINDOW)
            frames[(track, w)] += 1
            for p in np.flatnonzero(grid[i]):
                counts[(track, w)][p] += 1
    out = {}
    for key, total in frames.items():
        if total * FRAME < MIN_SECONDS or not counts[key]:
            continue
        who, c = counts[key].most_common(1)[0]
        if c / total >= MIN_SHARE:
            out[key] = (who, c / total, total)
    return out


def changes_of(qualified):
    """Every owner change between two consecutive qualifying windows of the same track."""
    by_track = defaultdict(list)
    for (track, w), value in sorted(qualified.items()):
        by_track[track].append((w, value))
    found = []
    for track, series in by_track.items():
        for (_, (a, sa, _)), (w, (b, sb, _)) in zip(series, series[1:]):
            if a != b:
                found.append((track, a, sa, b, sb, w))
    return found


def cmd_owners(_args):
    rows = []
    print(f"{'ep':>4}  {'arm':<10}{'tracks':>7}{'windows':>9}{'qualified':>11}  changes")
    for n in EPISODES:
        data = ev.episode(n)
        if data is None:
            print(f"{n:>4}  -- no production artifacts --")
            continue
        _, timeline, grid, people, _ = data
        probs = diarizen_probs(n)
        arms = {"sortformer": [(t["track"], t["start"], t["end"]) for t in timeline["tracks"]]}
        if probs is not None:
            arms["diarizen"] = tracks_from_probs(probs)
        for arm, tracks in arms.items():
            qualified = owner_windows(tracks, grid)
            found = changes_of(qualified)
            n_tracks = len({t for t, _, _ in tracks})
            spans = int(np.ceil(max(e for _, _, e in tracks) / WINDOW)) if tracks else 0
            rows.append(dict(ep=n, arm=arm, tracks=n_tracks, windows=spans * n_tracks,
                             qualified=len(qualified), changes=len(found)))
            detail = "; ".join(f"s{t} {people[a]}{sa:.0%}->{people[b]}{sb:.0%} @{int(w * WINDOW)}s"
                               for t, a, sa, b, sb, w in found) or "-"
            print(f"{n:>4}  {arm:<10}{n_tracks:>7}{spans * n_tracks:>9}{len(qualified):>11}  {detail}")
    print()
    for arm in ("sortformer", "diarizen"):
        mine = [r for r in rows if r["arm"] == arm]
        if not mine:
            continue
        episodes_with = sum(1 for r in mine if r["changes"])
        print(f"{arm:<11} {episodes_with}/{len(mine)} episodes with an owner change; "
              f"{sum(r['changes'] for r in mine)} changes; "
              f"{sum(r['qualified'] for r in mine)} qualifying windows of "
              f"{sum(r['windows'] for r in mine)}; mean tracks {np.mean([r['tracks'] for r in mine]):.2f}")
    print("\n⚠ Read the qualifying-window counts before the change counts (see module docstring).")


def cmd_stress(args):
    """P1: #14 as a diagnostic. No promotion threshold; n=1 gets no interval.

    `lab.OUT` already ends in `cross-program` once `accuracy_cross.configure()` has run -- appending
    it again here silently found nothing and printed a table with only the DiariZen row.
    """
    print(f"{'arm':<22}{'fixed':>8}{'crossfit':>10}{'B3 F1':>8}{'tracks':>8}")
    directory = lab.OUT / args.ident
    for arm in ("base", "ema", "high", "split", "community", "community_exclusive"):
        path = directory / f"{arm}.json"
        if path.exists():
            p = json.loads(path.read_text(encoding="utf-8"))
            m, b = p["metrics"], p.get("bcubed", {})
            print(f"{arm:<22}{m['fixed']:>8.4f}{m['crossfit']:>10.4f}"
                  f"{b.get('f1', float('nan')):>8.4f}{m['tracks']:>8}")
    path = V2_OUT / args.ident / "diarizen.json"
    if path.exists():
        p = json.loads(path.read_text(encoding="utf-8"))
        m, b = p["metrics"], p["bcubed"]
        print(f"{'diarizen':<22}{m['fixed']:>8.4f}{m['crossfit']:>10.4f}{b['f1']:>8.4f}{m['tracks']:>8}")
    else:
        print("diarizen: not run yet")


def labels_of(probs, stable):
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    return [s.track for s in assign_all(stable, segments_from_activity(probs.tolist()))]


def pad(probs, width):
    if probs.shape[1] >= width:
        return probs
    return np.hstack([probs, np.zeros((len(probs), width - probs.shape[1]), probs.dtype)])


def cmd_micro(args):
    """P3: same file, three inputs, one process -- so the numerical path is held fixed."""
    import vram_guard

    vram_guard.install()
    import soundfile as sf
    import torch
    from accuracy_diarizen import infer, load_pipeline, raster
    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        raise RuntimeError("CUDA unavailable; this probe must not silently run on CPU")
    torch.set_num_threads(4)
    from accuracy_cross import configure

    configure()
    wave, sr = sf.read(lab.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000
    base_wave = wave.mean(axis=1)
    arms = {
        "base": base_wave,
        "replay": base_wave,                                   # P4 inside P3: two runs, one input
        "micro": (base_wave.astype(np.float64) * (1 + 1e-7)).astype(np.float32),
        "pcm16": (np.round(base_wave * 32768.0) / 32768.0).astype(np.float32),
    }
    pipeline = load_pipeline()
    stable, _, _, _, _ = ev.episode(int(args.ident.rsplit("-", 1)[1]))
    out, probs = {}, {}
    for arm, signal in arms.items():
        probs[arm] = raster(infer(pipeline, signal), len(signal))
        out[arm] = dict(tracks=int(probs[arm].shape[1]),
                        input_max_delta=float(np.abs(signal - base_wave).max()))
        print(f"{arm}: {out[arm]['tracks']} tracks, input max delta {out[arm]['input_max_delta']:.3e}", flush=True)
    width = max(p.shape[1] for p in probs.values())
    base = pad(probs["base"], width)
    base_labels = labels_of(base, stable)
    for arm in arms:
        if arm == "base":
            continue
        other = pad(probs[arm], width)[: len(base)]
        bit_exact = other.shape == base.shape and bool(np.array_equal(other, base))
        aligned = lab.align(base, other)
        labels = labels_of(aligned, stable)
        pairs = [(a, b) for a, b in zip(base_labels, labels) if a is not None or b is not None]
        changed = sum(1 for a, b in pairs if a != b)
        out[arm].update(bit_exact=bit_exact, max_prob_delta=float(np.abs(aligned - base).max()),
                        labelled_rows=len(pairs), changed_rows=changed,
                        changed_share=changed / max(len(pairs), 1))
        print(f"{arm} vs base: bit-exact {bit_exact}; max |dp| {out[arm]['max_prob_delta']:.6f}; "
              f"labels changed {changed}/{len(pairs)} = {out[arm]['changed_share']:.2%}", flush=True)
    directory = V2_OUT / args.ident
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dz-probe-micro.json").write_text(
        json.dumps(dict(ident=args.ident, arms=out, script=lab.digest(__file__),
                        audio=lab.digest(lab.source(args.ident))), indent=2), encoding="utf-8")
    print(f"\nwrote {directory / 'dz-probe-micro.json'}")


# --- W7: perturbation sensitivity, both models, one metric -----------------------------------

#: Fixed so the noise arm is the same waveform every run.
NOISE_SEED = 20260918


def perturbations(base_wave, seed=None):
    """The five arms of plan section 8.2, as (name, signal, what changed).

    `micro` is kept as a NEGATIVE CONTROL: it is a pure gain change, which DiariZen's WavLM front
    end normalises away, so a near-zero reading there confirms the mechanism rather than the model.
    `noise` and `shift1` are the arms that survive any gain normalisation.
    """
    rng = np.random.default_rng(seed if seed is not None else NOISE_SEED)
    return {
        "base": base_wave,
        "micro": (base_wave.astype(np.float64) * (1 + 1e-7)).astype(np.float32),
        "noise": (base_wave + rng.normal(0.0, 1e-7, len(base_wave))).astype(np.float32),
        "shift1": np.concatenate([np.zeros(1, base_wave.dtype), base_wave]),
        "pcm16": (np.round(base_wave * 32768.0) / 32768.0).astype(np.float32),
    }


def _label_change(base_probs, other_probs, stable):
    """Share of labelled subtitle rows whose leading track changes, after a one-to-one alignment."""
    width = max(base_probs.shape[1], other_probs.shape[1])
    base = pad(base_probs, width)
    other = pad(other_probs, width)[: len(base)]
    if len(other) < len(base):
        other = np.vstack([other, np.zeros((len(base) - len(other), width), other.dtype)])
    aligned = lab.align(base, other)
    a, b = labels_of(base, stable), labels_of(aligned, stable)
    pairs = [(x, y) for x, y in zip(a, b) if x is not None or y is not None]
    changed = sum(1 for x, y in pairs if x != y)
    return dict(rows=len(pairs), changed=changed, share=changed / max(len(pairs), 1),
                max_delta=float(np.abs(aligned - base).max()),
                tracks=int(other_probs.shape[1]))


def cmd_sensitivity(args):
    """W7: the same five perturbations through both backends, scored identically."""
    import vram_guard

    vram_guard.install()
    import soundfile as sf
    import torch
    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        raise RuntimeError("CUDA unavailable; this probe must not silently run on CPU")
    torch.set_num_threads(4)
    from accuracy_cross import configure

    configure()
    wave, sr = sf.read(lab.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000
    arms = perturbations(wave.mean(axis=1))
    stable, _, _, _, _ = ev.episode(int(args.ident.rsplit("-", 1)[1]))
    out = {}

    if getattr(args, "seeds", None):
        # Repeat ONE arm across seeds: the question stops being "how big is this perturbation" and
        # becomes "how often does a perturbation of this size flip the timeline at all".
        arms = {"base": arms["base"], **{f"noise{s}": perturbations(arms["base"], s)["noise"]
                                         for s in args.seeds}}

    from accuracy_diarizen import infer, load_pipeline, raster

    pipeline = load_pipeline()
    dz = {name: raster(infer(pipeline, signal), len(signal)) for name, signal in arms.items()}
    del pipeline
    torch.cuda.empty_cache()
    sortformer = {name: lab.run_arm(signal, "base") for name, signal in arms.items()}

    for model, probs in (("diarizen", dz), ("sortformer", sortformer)):
        out[model] = {}
        for name in arms:
            if name == "base":
                continue
            out[model][name] = _label_change(probs["base"], probs[name], stable)
            r = out[model][name]
            print(f"{model:11}{name:8} labels changed {r['changed']:4}/{r['rows']} = {r['share']:.2%}"
                  f"  max|delta| {r['max_delta']:.6f}  tracks {r['tracks']}", flush=True)
    directory = V2_OUT / args.ident
    directory.mkdir(parents=True, exist_ok=True)
    name = "dz-probe-w7-seeds.json" if getattr(args, "seeds", None) else "dz-probe-w7.json"
    (directory / name).write_text(json.dumps(
        dict(ident=args.ident, seed=NOISE_SEED, arms=out, script=lab.digest(__file__),
             audio=lab.digest(lab.source(args.ident))), indent=2), encoding="utf-8")
    print(f"wrote {directory / name}")


def cmd_replay(args):
    """P4 on its own: rerun one cached ident and require bit-exact probabilities."""
    import vram_guard

    vram_guard.install()
    import soundfile as sf
    import torch
    from accuracy_diarizen import infer, load_pipeline, raster
    from accuracy_cross import configure

    configure()
    torch.set_num_threads(4)
    cached = diarizen_probs(int(args.ident.rsplit("-", 1)[1]))
    if cached is None:
        raise SystemExit(f"no cached diarizen probs for {args.ident}")
    wave, sr = sf.read(lab.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000
    signal = wave.mean(axis=1)
    fresh = raster(infer(load_pipeline(), signal), len(signal))
    same = fresh.shape == cached.shape and bool(np.array_equal(fresh, cached))
    print(f"{args.ident}: cached {cached.shape} fresh {fresh.shape} bit-exact {same}")
    if not same:
        raise SystemExit("replay is not bit-exact -- the probe group is void until this is explained")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("owners")
    for name in ("stress", "micro", "replay", "sensitivity"):
        q = sub.add_parser(name)
        q.add_argument("--ident", default="onetonton-14")
    sub.choices["sensitivity"].add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="repeat the noise arm with these seeds instead of running all five arms -- one draw "
             "per arm cannot tell a benign perturbation from a draw that happened not to diverge")
    args = p.parse_args()
    if args.command in ("owners", "stress"):
        from accuracy_cross import configure

        configure()
    {"owners": cmd_owners, "stress": cmd_stress, "micro": cmd_micro, "replay": cmd_replay,
     "sensitivity": cmd_sensitivity}[args.command](args)


if __name__ == "__main__":
    main()
