"""Post-hoc: nonlinear / piecewise combinations of the same row features (not a gate)."""
import pickle, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
d, _ = pf.load_rows()
dev = [f for f in pf.SPLIT["dev"] if f in d]; held = [f for f in pf.SPLIT["heldout"] if f in d]
ordi = lambda fs: [r for f in fs for r in d[f] if r["state"] == "ordinary"]
dev_rows, held_rows = ordi(dev), ordi(held)
Xd, yd = pf.matrix(dev_rows), np.array([r["wrong"] for r in dev_rows], int)
Xh = pf.matrix(held_rows)
models = {
    "linear (pre-registered)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "splines (smooth piecewise)": lambda: make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(max_iter=4000, C=0.3)),
    "boosted trees (piecewise + interactions)": lambda: HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=300, min_samples_leaf=100, random_state=0),
}
def catch(score, rows, files):
    order = np.argsort(-score); k = int(round(pf.FLAG * len(rows)))
    flag = set(order[:k]); per = []
    for f in files:
        idx = [i for i, r in enumerate(rows) if r["file"] == f]
        per.append((sum(rows[i]["wrong"] for i in idx if i in flag), sum(rows[i]["wrong"] for i in idx)))
    return per
margin_score = -np.array([r["margin"] for r in held_rows])
base = catch(margin_score, held_rows, held)
def compare(score, label):
    per = catch(score, held_rows, held)
    arr = [(a, b, e) for (a, e), (b, _) in zip(per, base)]
    stat, lo, hi = pf.bootstrap_diff(arr)
    tot = sum(e for *_, e in arr)
    print(f"{label:45s} catches {sum(a for a,_,_ in arr)/tot:.1%} vs margin {sum(b for _,b,_ in arr)/tot:.1%}: {stat*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]")
for name, make in models.items():
    m = make().fit(Xd, yd)
    compare(m.predict_proba(Xh)[:, 1], name)
# hand-written piecewise rules, thresholds picked on dev only (grid), applied to held-out
p_comp = lambda rs: np.array([r["p_comp"] for r in rs]); mg = lambda rs: np.array([r["margin"] for r in rs])
sec = lambda rs: np.array([r["seconds"] for r in rs])
best = None
for t in np.arange(0.05, 0.95, 0.05):
    s = (p_comp(dev_rows) > t) * 1.0 + (1 - mg(dev_rows))  # rule: competitor prob above t jumps ahead, then by margin
    order = np.argsort(-s); k = int(round(pf.FLAG * len(dev_rows)))
    c = yd[order[:k]].sum() / yd.sum()
    if best is None or c > best[1]: best = (t, c)
t = best[0]
compare((p_comp(held_rows) > t) * 1.0 + (1 - mg(held_rows)), f"rule: p_comp>{t:.2f} first, then margin")
best = None
for t in np.arange(0.2, 3.0, 0.2):
    s = (sec(dev_rows) < t) * 1.0 + (1 - mg(dev_rows))
    order = np.argsort(-s); k = int(round(pf.FLAG * len(dev_rows)))
    c = yd[order[:k]].sum() / yd.sum()
    if best is None or c > best[1]: best = (t, c)
t = best[0]
compare((sec(held_rows) < t) * 1.0 + (1 - mg(held_rows)), f"rule: rows < {t:.1f}s first, then margin")
