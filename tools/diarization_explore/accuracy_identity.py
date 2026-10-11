"""B-cubed identity score: unlike per-track oracle naming, fragmentation is penalized."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import accuracy_lab as lab


def bcubed(hypothesis, reference, weights):
    cells, hyp, ref = defaultdict(float), defaultdict(float), defaultdict(float)
    for h, r, w in zip(hypothesis, reference, weights):
        cells[h, r] += w
        hyp[h] += w
        ref[r] += w
    total = sum(weights)
    if not total:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = sum(v*v/hyp[h] for (h, _), v in cells.items())/total
    recall = sum(v*v/ref[r] for (_, r), v in cells.items())/total
    return {"precision": precision, "recall": recall,
            "f1": 2*precision*recall/max(precision+recall, 1e-12)}


def identity_score(probs, stable, grid, frame_seconds=None):
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity
    # `frame_seconds` is the grid of `probs`; the truth grid stays at `lab.FRAME` (A3).
    step = lab.FRAME if frame_seconds is None else frame_seconds
    assignments = assign_all(stable, segments_from_activity(probs.tolist(), frame_seconds=step))
    hyp, ref, weights = [], [], []
    labelled = 0.0
    for i, (s, assignment) in enumerate(zip(stable, assignments)):
        cover = grid[int(s["start"]/lab.FRAME):int(np.ceil(s["end"]/lab.FRAME))].sum(axis=0)
        if not cover.any():
            continue
        h = assignment.track
        w = s["end"]-s["start"]
        # An abstention is never equated to another abstention as an identity.
        hyp.append(h if h is not None else -i-1)
        ref.append(int(cover.argmax()))
        weights.append(w)
        labelled += w*(h is not None)
    return bcubed(hyp, ref, weights) | {"coverage": labelled/max(sum(weights), 1e-12)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=lab.DEV)
    p.add_argument("--arms", nargs="+", default=["base", "ema", "refresh", "split", "community", "wavlm_0.15"])
    args = p.parse_args()
    totals = defaultdict(list)
    for ident in args.ids:
        stable, _, grid, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        for arm in args.arms:
            path = lab.OUT / ident / f"{arm}.npz"
            if not path.exists():
                continue
            value = identity_score(np.load(path)["probs"], stable, grid)
            record = path.with_suffix(".json")
            payload = json.loads(record.read_text(encoding="utf-8"))
            payload["bcubed"] = value
            payload["bcubed_script"] = lab.digest(__file__)
            record.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            totals[arm].append(value)
    for arm, values in totals.items():
        print(arm, len(values), {k: round(float(np.mean([v[k] for v in values])), 4) for k in values[0]})


if __name__ == "__main__":
    main()
