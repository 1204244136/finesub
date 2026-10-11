"""Plan 5n, development set only: can a pairwise check veto the split's own mistakes?

For every unit the split would carve off, and every existing identity (the host's own
`keep` sub-track, and each other onepass track), interleave short pieces of the two
(g, u, g, u, ... -- Sortformer separates people on alternating contrast, not on
monologues back to back) and run one stream. The pair is "same" when most of the
unit's frames land in the slot most of the identity's frames land in.

First question (before any veto is used): is the verifier itself any good? The
development files have truth, so each pair is known same or different.
  --measure      verifier accuracy on truly-same / truly-different pairs
  --effect       split vs split-with-veto on the dev lines (raw and ogg inputs)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import over4
import over4_anchor as anchor
import over4_v2 as v2

FRAME, SPF = over4.FRAME, v2.SAMPLES_PER_FRAME
PIECES, PIECE_SECONDS = 6, 2.5


def pieces_of(frames, count=PIECES, seconds=PIECE_SECONDS):
    """Up to `count` contiguous pieces of about `seconds`, spread over the frames."""
    runs = [r for r in np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1) if len(r) * FRAME >= 0.5]
    if not runs:
        return []
    chosen = sorted({round(i * (len(runs) - 1) / max(1, count - 1)) for i in range(min(count, len(runs)))})
    width = int(seconds / FRAME)
    return [runs[i][:width] for i in chosen]


def same_person(runner, key, waveform, identity_frames, unit_frames):
    g, u = pieces_of(identity_frames), pieces_of(unit_frames)
    if not g or not u:
        return None
    audio, spans = [], []
    cursor = 0
    for i in range(max(len(g), len(u))):
        for label, source in (("g", g), ("u", u)):
            if i < len(source):
                piece = waveform[source[i][0] * SPF: (source[i][-1] + 1) * SPF]
                audio.append(piece)
                spans.append((label, cursor, cursor + len(piece) // SPF))
                cursor += len(piece) // SPF
    probs = anchor.cached(runner, key, "verify", lambda: {"probs": over4.run(runner, np.concatenate(audio))})["probs"]
    slots = {"g": [], "u": []}
    for label, a, b in spans:
        local = probs[a:b]
        active = (local > over4.THRESHOLD).any(axis=1)
        slots[label].extend(local[active].argmax(axis=1).tolist())
    if not slots["g"] or not slots["u"]:
        return None
    g_slot = int(np.bincount(slots["g"]).argmax())
    return float(np.mean(np.array(slots["u"]) == g_slot)) >= 0.5


def truth_owner(frames, grid):
    frames = frames[frames < len(grid)]
    rows = grid[frames]
    single = rows.sum(axis=1) == 1
    if not single.any():
        return None
    return int(np.bincount(rows[single].argmax(axis=1), minlength=grid.shape[1]).argmax())


def load(name, kind):
    import soundfile

    if kind == "raw":
        path = over4.audio(name)
    else:
        path = over4.WORK / "vox-sep" / f"{name}-vocal.ogg"
    signal, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    assert rate == 16000
    return signal.mean(axis=1)


def units_and_identities(runner, name, kind):
    waveform = load(name, kind)
    key = f"{name}-{kind}"
    p1 = anchor.cached(runner, key, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
    stats = {}
    v2.split(runner, key, waveform, p1, v2.CONFIGS["split-trig5"], stats)
    owner = np.where((p1 > over4.THRESHOLD).any(axis=1), p1.argmax(axis=1), -1)
    return waveform, key, p1, stats, owner


def measure():
    runner = over4.make_runner()
    many, few = over4.select()
    table = {"same": [0, 0], "different": [0, 0]}  # [verifier said same, pairs]
    for kind in ("raw", "ogg"):
        for name, _, _ in many + few:
            waveform, key, p1, stats, owner = units_and_identities(runner, name, kind)
            if not stats.get("unit_list"):
                continue
            grid = over4.truth(name, len(p1))
            for index, unit in enumerate(stats["unit_list"]):
                who = truth_owner(unit["frames"], grid)
                identities = {f"keep{unit['host']}": stats["keep_frames"][unit["host"]]}
                identities.update({f"track{g}": np.flatnonzero(owner == g) for g in range(p1.shape[1]) if g != unit["host"]})
                for label, frames in identities.items():
                    if frames.size * FRAME < 5:
                        continue
                    other = truth_owner(frames, grid)
                    verdict = same_person(runner, f"{key}-u{index}-{label}", waveform, frames, unit["frames"])
                    if verdict is None or who is None or other is None:
                        continue
                    bucket = "same" if who == other else "different"
                    table[bucket][0] += verdict
                    table[bucket][1] += 1
            print(kind, name, "units", len(stats["unit_list"]), json.dumps(table), flush=True)
    same, diff = table["same"], table["different"]
    print(f"\ntruly SAME pairs: verifier said same {same[0]}/{same[1]} ({same[0] / max(same[1], 1):.0%}) -- this is what blocks false splits")
    print(f"truly DIFFERENT pairs: verifier said same {diff[0]}/{diff[1]} ({diff[0] / max(diff[1], 1):.0%}) -- this would veto real orphans")


def effect():
    """split vs split-with-veto on the development lines, raw and ogg inputs."""
    from over4_breakdown import breakdown
    from calibrate import parse_rttm

    runner = over4.make_runner()
    many, few = over4.select()
    groups = {"many": many, "few": few}
    for kind in ("raw", "ogg"):
        for group, rows in groups.items():
            totals = {arm: {"lines": 0, "right": 0, "identity": 0, "identity_wrong": 0, "vetoed": 0} for arm in ("onepass", "split", "veto")}
            for name, _, _ in rows:
                waveform, key, p1, _, owner = units_and_identities(runner, name, kind)

                def veto(index, unit, stats):
                    identities = {f"keep{unit['host']}": stats["keep_frames"][unit["host"]]}
                    identities.update({f"track{g}": np.flatnonzero(owner == g) for g in range(p1.shape[1]) if g != unit["host"]})
                    return any(
                        same_person(runner, f"{key}-u{index}-{label}", waveform, frames, unit["frames"])
                        for label, frames in identities.items() if frames.size * FRAME >= 5
                    )

                stats_v = {}
                arms = {"onepass": p1,
                        "split": v2.split(runner, key, waveform, p1, v2.CONFIGS["split-trig5"]),
                        "veto": v2.split(runner, key, waveform, p1, v2.CONFIGS["split-trig5"], stats_v, veto)}
                reference = parse_rttm(over4.VOX / "voxconverse-master" / "dev" / f"{name}.rttm")
                grid = over4.truth(name, len(p1))
                for arm, probs in arms.items():
                    mapped = v2.mapping(probs, grid)
                    lines = v2.line_scores(probs, reference, mapped)
                    counts = breakdown(probs, reference, grid)
                    totals[arm]["lines"] += len(lines)
                    totals[arm]["right"] += sum(r for r, _ in lines)
                    totals[arm]["identity"] += counts["identity"]
                    totals[arm]["identity_wrong"] += counts["identity_wrong"]
                totals["veto"]["vetoed"] += stats_v.get("vetoed", 0)
            summary = {arm: f"lines {t['right']}/{t['lines']} ({t['right'] / max(t['lines'], 1):.3f}), identity wrong {t['identity_wrong']}/{t['identity']}"
                       for arm, t in totals.items()}
            print(f"{kind} {group}: " + " | ".join(f"{a}: {s}" for a, s in summary.items()) + f" | units vetoed {totals['veto']['vetoed']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--effect", action="store_true")
    arguments = parser.parse_args()
    if arguments.measure:
        measure()
    if arguments.effect:
        effect()


if __name__ == "__main__":
    main()
