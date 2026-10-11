"""Post-hoc descriptive readings for P2 step 1 (not gates): every number in the report's step-1 readings that
`p2_features.py evaluate`, `p2_nonlinear.py`, `p2_ablation.py`, `p2_frames.py evaluate` and `p2_signals2.py` do not print.

    python tools/diarization_explore/p2_readings.py

Needs rows.pkl, frames.pkl (`p2_frames.py features`) and slot_identity.pkl (`p2_slot_identity.py`). Truth = sets
(`p2_features` docstring): a label is right when its owner is any of the tied people.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf  # noqa: E402

SLOTS = pickle.loads((pf.CACHE / "slot_identity.pkl").read_bytes())
SPLITS = (("dev", "dev"), ("held-out", "heldout"))


def others(r):
    return [c["track"] for c in sorted(r["candidates"], key=lambda c: (-c["cover"], -c["prob"])) if c["track"] != r["leader"]]


def truth_comparison():
    """Step-1 readings under each truth (owners derived per truth), and how the two truths differ."""
    from sklearn.metrics import roc_auc_score
    by_truth = {t: pf.load_rows(t)[0] for t in ("onset", "covered")}
    for name, split in SPLITS:
        print(f"== {name} ==")
        for truth, rows in by_truth.items():
            rs = [r for f in pf.SPLIT[split] for r in rows[f] if r["state"] == "ordinary"]
            w = np.array([r["wrong"] for r in rs])
            sent = np.array([r["margin"] <= 0.5 for r in rs])
            high = np.array([r["lead_share"] >= 0.9 for r in rs])
            budget = np.argsort([r["margin"] for r in rs], kind="stable")[: int(round(pf.FLAG * len(rs)))]
            sent_wrong = [r for r, wi, si in zip(rs, w, sent) if wi and si]
            ceiling = lambda n: np.mean([any(pf.right(r, t) for t in others(r)[:n]) for r in sent_wrong])  # noqa: E731
            tied = np.mean([len(r["truth"]) > 1 for r in rs])
            print(f"truth {truth:8s}: {len(rs)} ordinary rows ({tied:.1%} with a tie), errors {w.sum()} ({w.mean():.1%}), "
                  f"{w[high].sum() / w.sum():.0%} at lead share >= 0.9; margin <= 0.5 sends {sent.mean():.1%}, error rate there {w[sent].mean():.1%}, "
                  f"catches {w[sent].sum() / w.sum():.1%}; 15% by margin catches {w[budget].sum() / w.sum():.1%}; AUC margin {roc_auc_score(w, [-r['margin'] for r in rs]):.3f}; "
                  f"ceiling on sent wrong rows: 2-way {ceiling(1):.1%}, 3-way {ceiling(2):.1%}, all {ceiling(99):.1%}")
        # rows wrong under onset but right under covered (same rows, each truth's own owners)
        onset = {(r["file"], r["index"]): r for f in pf.SPLIT[split] for r in by_truth["onset"][f] if r["state"] == "ordinary"}
        cover = {(r["file"], r["index"]): r for f in pf.SPLIT[split] for r in by_truth["covered"][f] if r["state"] == "ordinary"}
        both = [k for k in onset if k in cover]
        ow = [k for k in both if onset[k]["wrong"]]
        flips = [k for k in ow if not cover[k]["wrong"]]
        rw = [k for k in both if not onset[k]["wrong"]]
        print(f"wrong under onset: {len(ow)}; right under covered {len(flips) / len(ow):.1%} "
              f"(sent {np.mean([not cover[k]['wrong'] for k in ow if onset[k]['margin'] <= 0.5]):.1%}, "
              f"not sent {np.mean([not cover[k]['wrong'] for k in ow if onset[k]['margin'] > 0.5]):.1%}); "
              f"right under onset but wrong under covered: {np.mean([cover[k]['wrong'] for k in rw]):.1%}")


def main():
    from sklearn.metrics import roc_auc_score
    rows, _ = pf.load_rows("covered")
    dev_rows = [r for f in pf.SPLIT["dev"] for r in rows[f] if r["state"] == "ordinary"]
    model = pf.fit_candidates(dev_rows)
    for name, split in SPLITS:
        rs = [r for f in pf.SPLIT[split] for r in rows[f] if r["state"] == "ordinary"]
        sent = [r for r in rs if r["margin"] <= 0.5]
        wrong, right = [r for r in sent if r["wrong"]], [r for r in sent if not r["wrong"]]
        print(f"== {name} (covered truth): {len(rs)} ordinary rows, {sum(r['wrong'] for r in rs)} wrong; margin <= 0.5: {len(sent)} rows, {len(wrong)} wrong ==")
        for q in (0.0, 0.17):
            kept = [sum(p >= q for _, p in pf.s_k(model, r)) for r in sent]
            hit = lambda part: np.mean([any(p >= q and pf.right(r, k) for k, p in pf.s_k(model, r)) for r in part])  # noqa: E731
            print(f"q={q}: candidates per row mean {np.mean(kept):.2f}, counts {np.bincount(kept).tolist()}; "
                  f"recall all {hit(sent):.1%}, on wrong rows {hit(wrong):.1%}, on right rows {hit(right):.1%}")
        top = lambda r: max(pf.s_k(model, r), key=lambda x: x[1])[0]  # noqa: E731
        print(f"top s_k right: on wrong rows {np.mean([pf.right(r, top(r)) for r in wrong]):.1%}, on right rows {np.mean([pf.right(r, top(r)) for r in right]):.1%}; "
              f"switching every sent row to top s_k: net {sum(pf.right(r, top(r)) - (not r['wrong']) for r in sent):+d} rows")
        print(f"runner-up by coverage right on wrong sent rows: {np.mean([bool(others(r)) and pf.right(r, others(r)[0]) for r in wrong]):.1%}")
        y = np.array([r["wrong"] for r in rs])
        print(f"AUC time margin {roc_auc_score(y, [-r['margin'] for r in rs]):.3f}, competitor mean frame prob {roc_auc_score(y, [r['p_comp'] for r in rs]):.3f}")
        for label, part in (("all ordinary", rs), ("lead share >= 0.9", [r for r in rs if r["lead_share"] >= 0.9])):
            print(f"median leader frame prob ({label}): wrong {np.median([r['p_lead'] for r in part if r['wrong']]):.2f}, "
                  f"right {np.median([r['p_lead'] for r in part if not r['wrong']]):.2f}")
        slot = lambda r: SLOTS[(r["file"], r["index"])]["slot_track"]  # noqa: E731
        apart = [r for r in rs if slot(r) != r["leader"]]
        distinct = [r for r in apart if r["owner"].get(slot(r)) != r["owner"].get(r["leader"])]
        print(f"slot track != leader: {len(apart) / len(rs):.1%} of rows, error rate there {np.mean([r['wrong'] for r in apart]):.1%}, "
              f"{sum(r['wrong'] for r in apart) / max(1, y.sum()):.1%} of errors; slot right {np.mean([pf.right(r, slot(r)) for r in apart]):.1%}, "
              f"production right {np.mean([not r['wrong'] for r in apart]):.1%}; excluding same-owner pairs ({len(distinct)}): slot right "
              f"{np.mean([pf.right(r, slot(r)) for r in distinct]):.1%}, production right {np.mean([not r['wrong'] for r in distinct]):.1%}")
        missed = [r for f in pf.SPLIT[split] for r in rows[f] if r["state"] == "missed"]
        print(f"missed rows {len(missed)}: slot track right {np.mean([pf.right(r, SLOTS[(r['file'], r['index'])]['slot_track']) for r in missed]):.1%}, "
              f"top s_k right {np.mean([bool(pf.s_k(model, r)) and pf.right(r, top(r)) for r in missed]):.1%}, "
              f"nearest-labelled baseline {np.mean([r['nearest'] is not None and pf.right(r, r['nearest']) for r in missed]):.1%}")
    truth_comparison()


if __name__ == "__main__":
    main()
