"""Scoring-only per-speaker recall under the same global one-to-one mapping."""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from accuracy_dz3 import OUT
from accuracy_overlap import timeline


def speaker_recall(reference, hypothesis, duration):
    refs, hyps = timeline(reference, duration), timeline(hypothesis, duration)
    ref_ids = sorted({s for _, _, s in reference})
    hyp_ids = sorted({s for _, _, s in hypothesis})
    ri, hi = {s: i for i, s in enumerate(ref_ids)}, {s: i for i, s in enumerate(hyp_ids)}
    matrix, seconds = np.zeros((len(ri), len(hi))), np.zeros(len(ri))
    for a, b, speakers in refs:
        for speaker in speakers:
            seconds[ri[speaker]] += b-a
    i = j = 0
    while i < len(refs) and j < len(hyps):
        a, b, r = refs[i]
        c, d, h = hyps[j]
        width = min(b, d)-max(a, c)
        if width > 0:
            for x in r:
                for y in h:
                    matrix[ri[x], hi[y]] += width
        if b <= d:
            i += 1
        if d <= b:
            j += 1
    rows, cols = linear_sum_assignment(-matrix)
    mapping = dict(zip(rows, cols))
    return [dict(reference=s, seconds=float(seconds[index]),
                 matched_hypothesis=hyp_ids[mapping[index]]
                 if index in mapping and matrix[index, mapping[index]] > 0 else None,
                 recall=float(matrix[index, mapping[index]]/seconds[index])
                 if index in mapping and seconds[index] > 0 else 0.,
                 # Diagnostic only: unlike the actual metric, this may reuse a
                 # majority speaker's track. Temporal coverage is not identity recovery.
                 best_temporal_overlap_track=hyp_ids[int(matrix[index].argmax())]
                 if len(hyp_ids) and matrix[index].max() > 0 else None,
                 best_temporal_overlap_recall=float(matrix[index].max()/seconds[index])
                 if len(hyp_ids) and seconds[index] > 0 else 0.) for index, s in enumerate(ref_ids)]


def collect():
    from accuracy_dz3_meetings import MEETINGS, VOX
    results = {}
    for ident in MEETINGS + VOX:
        results[ident] = {}
        paths = list((OUT / ident).glob("w*.json")) + list((OUT / "overlap" / ident).glob("*.json"))
        for path in paths:
            record = json.loads(path.read_text())
            if "hypothesis" in record:
                results[ident][path.stem] = speaker_recall(record["reference"], record["hypothesis"], record["duration"])
    return results


def rare_local_diagnostics():
    """Post-hoc reference-guided probe; never an inference or routing algorithm.

    For reference speakers with <3 seconds total speech, choose the local slot
    with the most temporal overlap in each window. Report its activity and the
    official clean-frame eligibility, not a new global diarization score.
    """
    from scipy.ndimage import median_filter
    from accuracy_dz3_meetings import MEETINGS, VOX
    results = {}
    for ident in MEETINGS + VOX:
        baseline = OUT / ident / "w16_default.json"
        if not baseline.exists():
            continue
        record = json.loads(baseline.read_text())
        reference = timeline(record["reference"], record["duration"])
        people = speaker_recall(record["reference"], record["hypothesis"], record["duration"])
        rare = [s["reference"] for s in people if 0 < s["seconds"] < 3]
        if not rare:
            continue
        results[ident] = {}
        for duration in (16, 8):
            cache = OUT / ident / f"w{duration}"
            with np.load(cache / "diarizen-local.npz") as saved:
                local = median_filter(saved["data"], size=(1, 11, 1), mode="reflect")
                sw, rf = saved["window"], saved["receptive_field"]
            embeddings = np.load(cache / "embeddings.npy")
            offsets = rf[0]+rf[1]/2+np.arange(local.shape[1])*rf[2]
            clean = local * (local.sum(axis=2, keepdims=True) == 1)
            clean_counts = clean.sum(axis=1)
            required = round(.1*local.shape[1])
            valid = ~np.isnan(embeddings).any(axis=2)
            eligible = (clean_counts >= required) & valid & (local.sum(axis=1) > 0)
            fallback = int(eligible.sum()) < 2
            if fallback:
                eligible = valid & (local.sum(axis=1) > 0)
            rows = {}
            for speaker in rare:
                windows = []
                for index, activity in enumerate(local):
                    start = sw[0]+index*sw[2]
                    times = start+offsets
                    target = np.zeros(len(times), dtype=bool)
                    for a, b, active in reference:
                        if speaker in active:
                            target |= (times >= a) & (times < b)
                    if not target.any():
                        continue
                    overlaps = activity[target].sum(axis=0)
                    slot = int(overlaps.argmax())
                    count = float(activity[:, slot].sum())
                    windows.append(dict(start=float(start), slot=slot,
                        target_seconds=float(target.sum()*rf[2]),
                        local_temporal_recall=float(overlaps[slot]/target.sum()),
                        slot_activity_seconds=float(count*rf[2]),
                        target_fraction_of_slot=float(overlaps[slot]/count) if count else 0.,
                        clean_seconds=float(clean_counts[index, slot]*rf[2]),
                        eligible_for_cluster_training=bool(eligible[index, slot])))
                rows[speaker] = windows
            results[ident][str(duration)] = dict(required_clean_frames=required,
                required_clean_seconds=float(required*rf[2]), fallback=fallback, speakers=rows)
    return results


if __name__ == "__main__":
    results = collect()
    (OUT / "speaker-recall-diagnostics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for ident, arms in results.items():
        for arm, speakers in arms.items():
            rare = [s for s in speakers if s["seconds"] < 3]
            if rare:
                print(ident, arm, rare)
