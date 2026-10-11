"""Plan 5k: >4 speakers, second try -- split v2, production-path metric, noise band.

Answers the audit (`docs/report/diarization-branch-audit-2026-09-10.md`):
  2.2  every arm also runs under x(1+1e-7) and x(1-1e-7): a noise band, not one draw
  2.4  scored on the production path -- segments_from_activity -> assign_subtitle on
       the reference grid, "is this line's label set right" -- plus frame AA with and
       without overlap frames, per file and pooled, with a bootstrap interval
  2.6  per-file results + git HEAD + start time written next to the numbers
  4.2  third pass: split-off pieces from different hosts are re-diarized together,
       so one person split out of two host tracks becomes one identity
  4.4  real probabilities, not 1.0 / 0.99 -- the matrix can go through hysteresis
  4.5  optional trigger: split only when onepass filled all four slots

  --dev [--config NAME ...]   the 5i development files
  --final CONFIG              the untouched remainder of both pools, once
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np

import over4
import over4_anchor as anchor
from calibrate import parse_rttm, true_speakers_over

FRAME, THRESHOLD = over4.FRAME, over4.THRESHOLD
SAMPLES_PER_FRAME = int(16000 * FRAME)
PERTURBATIONS = {"plain": 1.0, "up": 1.0 + 1e-7, "down": 1.0 - 1e-7}


@dataclass(frozen=True)
class Split:
    min_share: float = 0.15
    min_seconds: float = 5.0
    merge_pass: bool = True  # audit 4.2
    trigger_full: bool = False  # audit 4.5: only when four tracks are each busy
    trigger_seconds: float = 20.0


CONFIGS = {
    "div": Split(merge_pass=False),  # 5i.4's frozen variant, rebuilt with real probabilities
    "split": Split(),
    "split-trig": Split(trigger_full=True),
    "split-trig10": Split(trigger_full=True, trigger_seconds=10.0),
    # afjiv (5 people, 2.4 min) never reaches 20 s on its fourth track, yet split
    # takes it from 0.607 to 1.000 on lines: short files need a lower bar.
    "split-trig5": Split(trigger_full=True, trigger_seconds=5.0),
}


# --- the algorithm ------------------------------------------------------------


def slots_full(p1, seconds):
    return int(((p1 > THRESHOLD).sum(axis=0) * FRAME >= seconds).sum()) >= p1.shape[1]


def _concat(waveform, frames):
    """Audio of these frames, contiguous runs back to back; returns (audio, runs)."""
    runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1) if frames.size else []
    pieces = [waveform[run[0] * SAMPLES_PER_FRAME : (run[-1] + 1) * SAMPLES_PER_FRAME] for run in runs]
    return (np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)), runs


def _back(sub, runs):
    """Sub-pass probabilities, re-laid onto the original frame order of `runs`."""
    rows, cursor = [], 0
    for run in runs:
        local = sub[cursor : cursor + len(run)]
        if len(local) < len(run):
            local = np.pad(local, ((0, len(run) - len(local)), (0, 0)))
        rows.append(local)
        cursor += len(run)
    return np.concatenate(rows) if rows else np.zeros((0, sub.shape[1]))


def split(runner, key, waveform, p1, config, stats=None, veto=None):
    """`stats`, when given, receives what happened: triggered / units / groups.

    `veto(index, unit, stats)`, when given, may refuse a unit before the merge pass
    (plan 5n); the merge pass is then cached under its own key, since its input differs.
    """
    stats = stats if stats is not None else {}
    stats.update(triggered=False, units=0, groups=0)
    if config.trigger_full and not slots_full(p1, config.trigger_seconds):
        return p1
    stats["triggered"] = True
    owner = np.where((p1 > THRESHOLD).any(axis=1), p1.argmax(axis=1), -1)
    units = []
    for host in range(p1.shape[1]):
        frames = np.flatnonzero(owner == host)
        if frames.size * FRAME < 2 * config.min_seconds:
            continue
        audio, runs = _concat(waveform, frames)
        sub = anchor.cached(runner, key, f"split-{host}", lambda: {"probs": over4.run(runner, audio)})["probs"]
        subp = _back(sub, runs)
        assign = np.where((subp > THRESHOLD).any(axis=1), subp.argmax(axis=1), -1)
        counts = np.bincount(assign[assign >= 0], minlength=subp.shape[1])
        if not counts.sum():
            continue
        keep = int(counts.argmax())
        stats.setdefault("keep_frames", {})[host] = frames[assign == keep]
        for k in range(len(counts)):
            if k == keep or counts[k] < config.min_share * counts.sum() or counts[k] * FRAME < config.min_seconds:
                continue
            chosen = assign == k
            units.append({"host": host, "frames": frames[chosen], "p": subp[chosen, k], "keep_p": subp[chosen, keep]})
    if veto is not None:
        kept = [u for index, u in enumerate(units) if not veto(index, u, stats)]
        stats["vetoed"] = len(units) - len(kept)
        units = kept
    if not units:
        return p1
    group = list(range(len(units)))
    if config.merge_pass and len(units) > 1:
        frames = np.sort(np.concatenate([u["frames"] for u in units]))
        audio, runs = _concat(waveform, frames)
        merge_key = "split-merge-veto" if veto is not None else "split-merge"
        third = anchor.cached(runner, key, merge_key, lambda: {"probs": over4.run(runner, audio)})["probs"]
        thirdp = _back(third, runs)
        position = {int(f): i for i, f in enumerate(frames)}
        votes = []
        for unit in units:
            rows = thirdp[[position[int(f)] for f in unit["frames"]]]
            votes.append(int(np.bincount(rows.argmax(axis=1), minlength=rows.shape[1]).argmax()))
        first_of = {}
        group = [first_of.setdefault(vote, index) for index, vote in enumerate(votes)]
    columns = sorted(set(group))
    stats.update(units=len(units), groups=len(columns), unit_list=units)
    out = np.concatenate([p1, np.zeros((len(p1), len(columns)), dtype=np.float32)], axis=1)
    for unit, g in zip(units, group):
        column = p1.shape[1] + columns.index(g)
        out[unit["frames"], unit["host"]] = unit["keep_p"]
        out[unit["frames"], column] = np.maximum(out[unit["frames"], column], unit["p"])
    return out


# --- scoring --------------------------------------------------------------------


def mapping(probs, grid):
    from scipy.optimize import linear_sum_assignment

    frames = min(len(probs), len(grid))
    active, ref = probs[:frames] > THRESHOLD, grid[:frames]
    table = active.T.astype(np.float64) @ ref.astype(np.float64)
    rows, cols = linear_sum_assignment(-table)
    return {int(r): int(c) for r, c in zip(rows, cols) if table[r, c] > 0}


def frame_scores(probs, grid, mapped):
    frames = min(len(probs), len(grid))
    probs, grid = probs[:frames], grid[:frames]
    active = probs > THRESHOLD
    single = (grid.sum(axis=1) == 1) & active.any(axis=1)
    loud = probs.argmax(axis=1)
    single_right = np.array([mapped.get(int(o)) == int(r) for o, r in zip(loud[single], grid[single].argmax(axis=1))])
    both = grid.any(axis=1) & active.any(axis=1)
    set_right = 0
    for t in np.flatnonzero(both):
        ours = {mapped.get(int(k), -1 - int(k)) for k in np.flatnonzero(active[t])}
        set_right += ours == set(np.flatnonzero(grid[t]).tolist())
    return {"single_frames": int(single.sum()), "single_right": int(single_right.sum()),
            "set_frames": int(both.sum()), "set_right": int(set_right)}


def line_scores(probs, reference, mapped):
    """Production path: hysteresis -> assign_subtitle on the reference grid."""
    from finesub.speech.diarization.assign import assign_subtitle
    from finesub.speech.diarization.segments import segments_from_activity

    speakers = sorted({s.speaker for s in reference})
    timeline = segments_from_activity(probs.tolist())
    p_max = probs.max(axis=1)
    rows = []
    for segment in reference:
        truth = {speakers.index(s) for s in true_speakers_over(segment, reference)}
        tracks = assign_subtitle(segment.start, segment.end, timeline).extras.get("tracks", [])
        ours = {mapped.get(int(k), -1 - int(k)) for k in tracks}
        first, last = int(segment.start / FRAME), max(int(segment.start / FRAME) + 1, int(segment.end / FRAME))
        rows.append((ours == truth, float(p_max[first:last].mean()) if last <= len(p_max) else 0.0))
    return rows


def score_file(runner, name, config, perturbation):
    import soundfile

    waveform, rate = soundfile.read(str(over4.audio(name)), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1) * np.float32(PERTURBATIONS[perturbation])
    key = f"{name}-{perturbation}"
    p1 = anchor.cached(runner, key, "onepass", lambda: {"probs": over4.run(runner, waveform)})["probs"]
    arms = {"onepass": p1, "split": split(runner, key, waveform, p1, config)}
    reference = parse_rttm(over4.VOX / "voxconverse-master" / "dev" / f"{name}.rttm")
    grid = over4.truth(name, len(p1))
    out = {"slots_full": slots_full(p1, 20.0)}
    for arm, probs in arms.items():
        mapped = mapping(probs, grid)
        lines = line_scores(probs, reference, mapped)
        out[arm] = {**frame_scores(probs, grid, mapped), "lines": len(lines), "lines_right": sum(r for r, _ in lines),
                    "line_pmax": [round(p, 4) for _, p in lines], "line_right": [bool(r) for r, _ in lines]}
    return out


def pooled(results, group, arm, field):
    rows = [r for r in results if r["group"] == group]
    right = {"line": "lines_right", "single": "single_right", "set": "set_right"}[field]
    total = {"line": "lines", "single": "single_frames", "set": "set_frames"}[field]
    return sum(r[arm][right] for r in rows) / max(sum(r[arm][total] for r in rows), 1)


def bootstrap(results, group, field, draws=2000, seed=7):
    rows = [r for r in results if r["group"] == group and r["perturbation"] == "plain"]
    right = {"line": "lines_right", "single": "single_right", "set": "set_right"}[field]
    total = {"line": "lines", "single": "single_frames", "set": "set_frames"}[field]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        a = sum(r["onepass"][right] for r in sample) / max(sum(r["onepass"][total] for r in sample), 1)
        b = sum(r["split"][right] for r in sample) / max(sum(r["split"][total] for r in sample), 1)
        deltas.append(b - a)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run(groups, config_name, record):
    runner = over4.make_runner()
    config = CONFIGS[config_name]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    started = time.time()
    results = []
    for group, rows in groups.items():
        for name, speakers, _ in rows:
            for perturbation in PERTURBATIONS:
                results.append({"group": group, "name": name, "speakers": speakers, "perturbation": perturbation,
                                **score_file(runner, name, config, perturbation)})
    summary = {"head": head, "started": started, "finished": time.time(), "geometry": over4.GEOMETRY,
               "config": config_name, "config_fields": asdict(config)}
    for group in groups:
        for field in ("line", "single", "set"):
            per = {}
            for p in PERTURBATIONS:
                sub = [r for r in results if r["group"] == group and r["perturbation"] == p]
                per[p] = (pooled(sub, group, "onepass", field), pooled(sub, group, "split", field))
            one = [v[0] for v in per.values()]
            lo, hi = bootstrap(results, group, field)
            summary[f"{group}/{field}"] = {
                "onepass": [round(min(one), 4), round(max(one), 4)],
                "split": [round(min(v[1] for v in per.values()), 4), round(max(v[1] for v in per.values()), 4)],
                "delta_by_perturbation": {p: round(v[1] - v[0], 4) for p, v in per.items()},
                "onepass_noise": round(max(one) - min(one), 4),
                "relative_error_reduction": round(
                    (per["plain"][1] - per["plain"][0]) / max(1 - per["plain"][0], 1e-9), 3),
                "bootstrap95_delta": [round(lo, 4), round(hi, 4)],
            }
        rows = [r for r in results if r["group"] == group and r["perturbation"] == "plain"]
        summary[f"{group}/slots_full"] = f"{sum(r['slots_full'] for r in rows)}/{len(rows)}"
        # The labelling proposal's `review` flag, measured on lines (not a criterion).
        for arm in ("onepass", "split"):
            right = np.array([x for r in rows for x in r[arm]["line_right"]])
            pmax = np.array([x for r in rows for x in r[arm]["line_pmax"]])
            table = {}
            for tau in (0.7, 0.8, 0.9):
                flagged = pmax < tau
                wrong = ~right
                table[str(tau)] = {
                    "flagged": round(float(flagged.mean()), 3),
                    "caught": round(float((flagged & wrong).sum() / max(wrong.sum(), 1)), 3),
                    "precision": round(float((flagged & wrong).sum() / max(flagged.sum(), 1)), 3),
                }
            summary[f"{group}/review/{arm}"] = {"wrong_lines": round(float((~right).mean()), 3), **table}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if record:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"summary": summary, "files": results}, ensure_ascii=False), encoding="utf-8")
        print("record ->", record)


def remainder():
    """Both pools minus the development and test files: never run before 5k."""
    many_dev, few_dev = over4.select()
    many_test, few_test = anchor.test_split()
    used = {n for n, _, _ in many_dev + few_dev + many_test + few_test}
    rows = []
    for rttm in sorted((over4.VOX / "voxconverse-master" / "dev").glob("*.rttm")):
        reference = parse_rttm(rttm)
        if rttm.stem not in used and reference:
            rows.append((rttm.stem, len({s.speaker for s in reference}), max(s.end for s in reference) / 60))
    return ([r for r in rows if r[1] >= 5 and r[2] <= 12], [r for r in rows if 2 <= r[1] <= 4 and r[2] <= 12])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--config", nargs="*", default=["div", "split", "split-trig"])
    parser.add_argument("--final", metavar="CONFIG")
    parser.add_argument("--list-final", action="store_true")
    arguments = parser.parse_args()
    if arguments.list_final:
        many, few = remainder()
        print(f"remainder: many {len(many)}, few {len(few)}")
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if arguments.dev:
        many, few = over4.select()
        for name in arguments.config:
            run({"many": many, "few": few}, name, over4.WORK / f"over4_v2-dev-{over4.GEOMETRY}-{name}-{stamp}.json")
    if arguments.final:
        many, few = remainder()
        record = over4.DATA / "results" / f"over4_v2-final-{over4.GEOMETRY}-{arguments.final}-{stamp}.json"
        run({"many": many, "few": few}, arguments.final, record)


if __name__ == "__main__":
    main()
