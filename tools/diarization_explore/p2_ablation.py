"""Post-hoc ablation for P2 step 1 (not a gate): what Nemotron frame probabilities and VBx soft assignment each add,
to the trigger (which rows are wrong) and to the candidate scores (which track is right)."""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

d, _ = pf.load_rows()
dev = [f for f in pf.SPLIT["dev"] if f in d]
held = [f for f in pf.SPLIT["heldout"] if f in d]
ordinary = lambda fs: [r for f in fs for r in d[f] if r["state"] == "ordinary"]  # noqa: E731
dev_rows, held_rows = ordinary(dev), ordinary(held)
NAMES = ["lead_share", "margin", "p_lead", "p_comp", "none_frac", "soft_margin", "seconds", "chars", "same_prev", "same_next", "track_share"]
COVER, FRAME, SOFT, ROW = ["lead_share", "margin"], ["p_lead", "p_comp", "none_frac"], ["soft_margin"], ["seconds", "chars", "same_prev", "same_next", "track_share"]


def catch_rate(score, rows):
    order = np.argsort(-score)
    k = int(round(pf.FLAG * len(rows)))
    wrong = np.array([r["wrong"] for r in rows])
    return wrong[order[:k]].sum() / wrong.sum()


print("== trigger: share of held-out errors caught when 15% of ordinary rows are flagged ==")
print(f"{'time margin alone (production)':40s} {catch_rate(-np.array([r['margin'] for r in held_rows]), held_rows):.1%}")
for label, cols in (("coverage", COVER), ("coverage + frame probs", COVER + FRAME), ("coverage + VBx soft", COVER + SOFT),
                    ("coverage + frame + soft", COVER + FRAME + SOFT), ("frame probs alone", FRAME), ("VBx soft alone", SOFT),
                    ("all (pre-registered)", NAMES)):
    idx = [NAMES.index(c) for c in cols]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(pf.matrix(dev_rows)[:, idx], [r["wrong"] for r in dev_rows])
    print(f"{label:40s} {catch_rate(model.predict_proba(pf.matrix(held_rows)[:, idx])[:, 1], held_rows):.1%}")

print("\n== candidates, held-out rows with time margin <= 0.5 ==")
flagged = [r for r in held_rows if r["margin"] <= 0.5 and r["candidates"]]
original = pf.candidate_x
for label, keep in (("coverage only", [0]), ("frame prob only", [1]), ("VBx soft only", [2]), ("coverage + frame", [0, 1]),
                    ("coverage + soft", [0, 2]), ("all three", [0, 1, 2])):
    pf.candidate_x = lambda r, keep=keep: original(r) * np.isin(np.arange(3), keep)
    model = pf.fit_candidates(dev_rows)
    top1, top2, top1_wrong, wrong_n = 0, 0, 0, 0
    for r in flagged:
        ranked = [k for k, _ in sorted(pf.s_k(model, r), key=lambda x: -x[1])]
        ok = [pf.right(r, k) for k in ranked]
        top1 += ok[0]
        top2 += any(ok[:2])
        if r["wrong"]:
            wrong_n += 1
            top1_wrong += ok[0]
    n = len(flagged)
    print(f"{label:20s} top-1 right {top1 / n:.1%}   truth in top-2 {top2 / n:.1%}   top-1 right on rows production got wrong {top1_wrong / wrong_n:.1%}")
pf.candidate_x = original
print(f"(production label right on these rows: {np.mean([not r['wrong'] for r in flagged]):.1%}; {len(flagged)} rows, {wrong_n} wrong)")
