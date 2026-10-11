"""Plan 5i.3: onepass-anchored multi-pass diarization for more than four speakers.

1. onepass -> <=4 global tracks, kept as they are;
2. per-window reruns (fresh stream each window);
3. anchor: each window's local tracks are one-to-one IoU-matched to onepass
   tracks over the whole window; matched ones inherit the onepass identity,
   unmatched long-enough ones are *orphans* and take their frames over;
4. only orphans are linked across windows, into new global identities.

  --dev [--variant NAME]   iterate on the 5i development files (cached probs)
  --test NAME              the held-out split, once, with a frozen variant
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

import over4

FRAME, THRESHOLD = over4.FRAME, over4.THRESHOLD


@dataclass(frozen=True)
class Variant:
    window: float = 60.0
    hop: float = 30.0
    dynamic: bool = False  # cut windows at onepass silences near the nominal boundaries
    anchor_iou: float = 0.3  # local <-> onepass match threshold (whole window)
    orphan_seconds: float = 3.0  # an unmatched local track must be active this long
    link_iou: float = 0.3  # orphan <-> orphan across the overlap
    gate: float = 1.01  # only take frames whose onepass p_max is below this (1.01 = off)


VARIANTS = {
    "base": Variant(),
    "short": Variant(window=30.0, hop=15.0),
    "dynamic": Variant(dynamic=True),
    "strict": Variant(anchor_iou=0.15, orphan_seconds=5.0),
    "gated": Variant(gate=0.95),
    "strict-gated": Variant(anchor_iou=0.15, orphan_seconds=5.0, gate=0.95),
}


def test_split():
    """Same pools as 5i, development files excluded, then an even deterministic pick."""
    many_dev, few_dev = over4.select()
    dev = {name for name, _, _ in many_dev + few_dev}
    rows = []
    from calibrate import parse_rttm

    for rttm in sorted((over4.VOX / "voxconverse-master" / "dev").glob("*.rttm")):
        if rttm.stem in dev:
            continue
        reference = parse_rttm(rttm)
        if reference:
            rows.append((rttm.stem, len({s.speaker for s in reference}), max(s.end for s in reference) / 60))

    def even(items, count):
        return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]

    many = even([r for r in rows if r[1] >= 5 and r[2] <= 12], over4.MANY)
    few = even([r for r in rows if 2 <= r[1] <= 4 and r[2] <= 12], over4.FEW)
    return many, few


def cached(runner, name, key, compute):
    # The geometry is part of the key: the same file under two geometries is two answers.
    path = over4.WORK / "over4-cache" / over4.GEOMETRY / f"{name}-{key}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    value = compute()
    np.savez_compressed(path, **value)
    return value


def boundaries(p1, total, variant):
    """Window [start, end) pairs in seconds."""
    starts = list(np.arange(0.0, max(total - variant.hop, 0.0) + 1e-9, variant.hop)) or [0.0]
    if variant.dynamic:
        silent = ~(p1 > THRESHOLD).any(axis=1)
        snapped = []
        for start in starts:
            frame = int(start / FRAME)
            window = range(max(0, frame - int(5 / FRAME)), min(len(silent), frame + int(5 / FRAME)))
            quiet = [f for f in window if silent[f]]
            snapped.append(min(quiet, key=lambda f: abs(f - frame)) * FRAME if quiet and start > 0 else start)
        starts = sorted(set(snapped))
    return [(s, min(total, s + variant.window)) for s in starts]


def windows_for(runner, name, waveform, rate, p1, variant):
    total = len(waveform) / rate
    spans = boundaries(p1, total, variant)
    key = f"win{variant.window:g}-{variant.hop:g}-{'dyn' if variant.dynamic else 'fix'}"

    def compute():
        probs = [over4.run(runner, waveform[int(s * rate) : int(e * rate)]) for s, e in spans]
        return {"starts": np.array([s for s, _ in spans]), "ends": np.array([e for _, e in spans]),
                "probs": np.array(probs, dtype=object)}

    return cached(runner, name, key, compute)


def anchored(p1, windows, variant):
    from scipy.optimize import linear_sum_assignment

    a1 = p1 > THRESHOLD
    frames = len(p1)
    items = []
    for start, end, probs in zip(windows["starts"], windows["ends"], windows["probs"]):
        first = int(round(start / FRAME))
        probs = probs[: max(0, frames - first)]
        local = probs > THRESHOLD
        span = a1[first : first + len(probs)]
        live = [k for k in range(local.shape[1]) if local[:, k].sum() * FRAME >= 1.0]
        onepass = [g for g in range(span.shape[1]) if span[:, g].any()]
        anchored_to = {}
        if live and onepass:
            iou = np.zeros((len(live), len(onepass)))
            for i, k in enumerate(live):
                for j, g in enumerate(onepass):
                    union = (local[:, k] | span[:, g]).sum()
                    iou[i, j] = (local[:, k] & span[:, g]).sum() / union if union else 0.0
            for i, j in zip(*linear_sum_assignment(-iou)):
                if iou[i, j] >= variant.anchor_iou:
                    anchored_to[live[i]] = onepass[j]
        orphans = [k for k in live if k not in anchored_to and local[:, k].sum() * FRAME >= variant.orphan_seconds]
        items.append({"first": first, "probs": probs, "orphans": orphans, "gid": {}, "center": (start + end) / 2})

    next_gid = p1.shape[1]
    for index, item in enumerate(items):
        if index and item["orphans"]:
            previous = items[index - 1]
            overlap = previous["first"] + len(previous["probs"]) - item["first"]
            if overlap > 0 and previous["gid"]:
                a = previous["probs"][-overlap:] > THRESHOLD
                b = item["probs"][:overlap] > THRESHOLD
                size = min(len(a), len(b))
                before = list(previous["gid"])
                iou = np.zeros((len(before), len(item["orphans"])))
                for i, p in enumerate(before):
                    for j, q in enumerate(item["orphans"]):
                        union = (a[:size, p] | b[:size, q]).sum()
                        iou[i, j] = (a[:size, p] & b[:size, q]).sum() / union if union else 0.0
                for i, j in zip(*linear_sum_assignment(-iou)):
                    if iou[i, j] >= variant.link_iou:
                        item["gid"][item["orphans"][j]] = previous["gid"][before[i]]
        for k in item["orphans"]:
            if k not in item["gid"]:
                item["gid"][k] = next_gid
                next_gid += 1

    out = np.zeros((frames, next_gid), dtype=np.float32)
    out[:, : p1.shape[1]] = p1
    centers = np.array([item["center"] for item in items])
    p1_max = p1.max(axis=1)
    for t in range(frames):
        owner = items[int(np.argmin(np.abs(centers - t * FRAME)))]
        local = t - owner["first"]
        if not (0 <= local < len(owner["probs"])) or not owner["gid"]:
            continue
        row = owner["probs"][local]
        k = int(row.argmax())
        if k in owner["gid"] and row[k] > THRESHOLD and p1_max[t] < variant.gate:
            out[t, owner["gid"][k]] = row[k]
            out[t, : p1.shape[1]] = np.minimum(out[t, : p1.shape[1]], row[k] * 0.99)
    return out


@dataclass(frozen=True)
class Divisive:
    """Recursive split: re-diarize each onepass track's own audio as one stream.

    Speakers a track absorbed now meet inside a single continuous stream, whose
    speaker cache spans the whole concatenation -- identity stays consistent by
    construction, and no cross-window linking is ever needed. A sub-track is
    split off only if it is big enough (a voice-style split of one person is
    the failure this guards against: kaguya60 came out as two tracks).
    """

    min_share: float = 0.15
    min_seconds: float = 5.0
    gap: float = 0.0  # silence inserted between concatenated pieces


DIVISIVE = {
    "div": Divisive(),
    "div-loose": Divisive(min_share=0.08, min_seconds=3.0),
    "div-strict": Divisive(min_share=0.25, min_seconds=10.0),
    "div-gap": Divisive(gap=0.3),
}
SAMPLES_PER_FRAME = int(16000 * FRAME)


def divisive(runner, name, waveform, p1, config):
    owner = np.where((p1 > THRESHOLD).any(axis=1), p1.argmax(axis=1), -1)
    out = np.zeros((len(p1), p1.shape[1] * 4), dtype=np.float32)
    out[:, : p1.shape[1]] = p1
    next_id = p1.shape[1]
    pad = np.zeros(int(config.gap * 16000), dtype=np.float32)
    for track in range(p1.shape[1]):
        frames = np.flatnonzero(owner == track)
        if frames.size * FRAME < 2 * config.min_seconds:
            continue
        runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
        pieces, layout, cursor = [], [], 0
        for run in runs:
            piece = waveform[run[0] * SAMPLES_PER_FRAME : (run[-1] + 1) * SAMPLES_PER_FRAME]
            pieces += [piece, pad]
            layout.append((run, cursor))
            cursor += (len(piece) + len(pad)) // SAMPLES_PER_FRAME
        key = f"div-{track}-gap{config.gap:g}"
        sub = cached(runner, name, key, lambda: {"probs": over4.run(runner, np.concatenate(pieces))})["probs"]
        assign = np.full(len(p1), -1)
        for run, offset in layout:
            local = sub[offset : offset + len(run)]
            if len(local) < len(run):
                local = np.pad(local, ((0, len(run) - len(local)), (0, 0)))
            assign[run] = np.where((local > THRESHOLD).any(axis=1), local.argmax(axis=1), -1)
        counts = np.bincount(assign[assign >= 0], minlength=sub.shape[1])
        if counts.sum() == 0:
            continue
        keep = int(counts.argmax())  # the biggest sub-track keeps the onepass identity
        for k in range(len(counts)):
            if k == keep or counts[k] < config.min_share * counts.sum() or counts[k] * FRAME < config.min_seconds:
                continue
            chosen = assign == k
            out[chosen, next_id] = 1.0
            out[chosen, : p1.shape[1]] = np.minimum(out[chosen, : p1.shape[1]], 0.99)
            next_id += 1
    return out[:, :next_id]


def evaluate(runner, group_rows, variants):
    import soundfile

    totals = {name: {"frames": 0, "matched": 0.0} for name in ["onepass", *variants]}
    per_file = []
    for name, speakers, _ in group_rows:
        waveform, rate = soundfile.read(str(over4.audio(name)), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        p1 = cached(runner, name, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
        grid = over4.truth(name, len(p1))
        scores = {"onepass": over4.attribution(p1, grid)}
        for label in variants:
            if label in DIVISIVE:
                scores[label] = over4.attribution(divisive(runner, name, waveform, p1, DIVISIVE[label]), grid)
                continue
            variant = VARIANTS[label]
            scores[label] = over4.attribution(anchored(p1, windows_for(runner, name, waveform, rate, p1, variant), variant), grid)
        for label, score in scores.items():
            totals[label]["frames"] += score["frames"]
            totals[label]["matched"] += score["matched"]
        per_file.append((name, speakers, {k: round(v["matched"] / max(v["frames"], 1), 3) for k, v in scores.items()}))
    return {k: v["matched"] / max(v["frames"], 1) for k, v in totals.items()}, per_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--variant", nargs="*", default=[*VARIANTS, *DIVISIVE])
    parser.add_argument("--test", metavar="VARIANT")
    parser.add_argument("--list-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.list_test:
        many, few = test_split()
        print("many", [(n, s) for n, s, _ in many])
        print("few", [(n, s) for n, s, _ in few])
        return
    runner = over4.make_runner()
    if arguments.dev:
        many, few = over4.select()
        variants = arguments.variant
    else:
        many, few = test_split()
        variants = [arguments.test]
        print("TEST with frozen variant", arguments.test, asdict({**VARIANTS, **DIVISIVE}[arguments.test]))
    for label, rows in (("many", many), ("few", few)):
        pooled, per_file = evaluate(runner, rows, variants)
        print(f"\n{label}: " + "  ".join(f"{k} {v:.3f} ({100 * (v - pooled['onepass']):+.1f})" for k, v in pooled.items()))
        for name, speakers, scores in per_file:
            print(f"   {name} {speakers:2d}  " + "  ".join(f"{k} {v:.3f}" for k, v in scores.items()))


if __name__ == "__main__":
    main()
