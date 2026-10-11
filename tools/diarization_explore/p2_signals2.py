"""Post-hoc (not a gate): pre-screening signals under the covered truth (owner 2026-10-05: main truth = covered).

New per-row signals, all from the cached pass (no model run):
  - raw slot competition, identity-free: per frame the second-highest Nemotron slot probability (mean, q90, share of
    frames > 0.5) -- "at the same instant, how sure is the model that someone else is also speaking";
  - squeeze (Codex): share of the row's active (window, slot) activity whose `hard` track is not the slot's best soft
    centroid -- the one-to-one window assignment pushed it elsewhere;
  - embedding support (Codex): for the leader's (window, slot) pairs, the share of the embedding mask (overlap removed,
    as `embed` extracts) that falls inside the row; and the soft margin weighted by it.
Evaluated with the existing row means and per-frame shape: held-out share of covered-truth errors caught at fixed send
budgets, models fit on dev only.

    python tools/diarization_explore/p2_signals2.py [features|evaluate|compact]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf  # noqa: E402
import p2_frames  # noqa: E402

OUT = pf.CACHE / "signals2.pkl"
NAMES = ("second_mean", "second_q90", "second_hi", "squeeze", "squeeze_leader", "support", "support_soft_margin")
BUDGETS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def file_signals(ident, records):
    from finesub.speech.diarization.diarizen_embed import chunk_masks
    stable = pf.reference(ident)[0]
    data = np.load(pf.CACHE / f"{ident}.npz")
    frames = data["native"].shape[0]
    _, _, starts, _ = pf.slot_maps(data, frames)
    prob = pf.resample(data["probs"], float(data["frame_seconds"]), frames)
    local, hard, soft, cot = data["local"], data["hard"], data["soft"], data["centroid_of_track"]
    masks = chunk_masks(local)
    per_window = local.shape[1]
    valid = np.flatnonzero(cot >= 0)
    out = {}
    for r in records:
        row = stable[r["index"]]
        lo = int(float(row["start"]) / pf.STEP)
        hi = min(max(lo + 1, int(np.ceil(float(row["end"]) / pf.STEP))), frames)
        second = np.sort(prob[lo:hi], axis=1)[:, -2]
        act_all = squeezed = act_lead = squeezed_lead = 0.0
        support_w = support_sum = margin_sum = 0.0
        for c in np.flatnonzero((starts < hi) & (starts + per_window > lo)):
            a, b = max(lo, starts[c]) - starts[c], min(hi, starts[c] + per_window) - starts[c]
            for j in range(local.shape[2]):
                act = float(local[c, a:b, j].sum())
                if act <= 0 or np.isnan(soft[c, j]).all():
                    continue
                k = int(hard[c, j])
                scores = soft[c, j, cot[valid]]
                best = int(valid[int(np.argmax(scores))])
                act_all += act
                squeezed += act * (best != k)
                if k == r["leader"]:
                    act_lead += act
                    squeezed_lead += act * (best != k)
                    total = float(masks[c, :, j].sum())
                    support = float(masks[c, a:b, j].sum()) / total if total else 0.0
                    own = soft[c, j, cot[k]] if cot[k] >= 0 else np.nan
                    rival = np.nanmax(np.delete(scores, np.flatnonzero(valid == k))) if len(valid) > 1 else 0.0
                    support_sum += act * support
                    margin_sum += act * support * (own - rival)
                    support_w += act
        out[r["index"]] = dict(
            second_mean=float(second.mean()), second_q90=float(np.quantile(second, 0.9)), second_hi=float((second > 0.5).mean()),
            squeeze=squeezed / act_all if act_all else 0.0, squeeze_leader=squeezed_lead / act_lead if act_lead else 0.0,
            support=support_sum / support_w if support_w else 0.0,
            support_soft_margin=margin_sum / support_sum if support_sum else 0.0)
    return out


def build():
    rows = pickle.loads((pf.CACHE / "rows.pkl").read_bytes())["rows"]
    feats = {}
    for ident in sorted(rows, key=lambda i: i.startswith("onetonton")):  # ⚠ hina first
        feats[ident] = file_signals(ident, [r for r in rows[ident] if r["leader"] is not None])
        print(ident, flush=True)
    OUT.write_bytes(pickle.dumps(feats))


def evaluate():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows, _ = pf.load_rows("covered")
    frames = pickle.loads(p2_frames.OUT.read_bytes())
    new = pickle.loads(OUT.read_bytes())
    pick = lambda fs: [r for f in fs for r in rows[f] if r["state"] == "ordinary"]  # noqa: E731
    dev, held = pick(pf.SPLIT["dev"]), pick(pf.SPLIT["heldout"])
    wrong = lambda rs: np.array([r["wrong"] for r in rs])  # noqa: E731
    yd, yh = wrong(dev), wrong(held)
    fx = lambda rs: np.array([[frames[r["file"]][r["index"]][k] for k in p2_frames.FRAME_NAMES] for r in rs])  # noqa: E731
    # ⚠ support_soft_margin is NaN on the 14 rows whose leader is a sentinel track (no centroid): no soft evidence -> 0.
    nx = lambda rs: np.nan_to_num(np.array([[new[r["file"]][r["index"]][k] for k in NAMES] for r in rs]))  # noqa: E731

    def curve(score):
        order = np.argsort(-np.asarray(score), kind="stable")
        return " ".join(f"{b:.0%}:{yh[order[: int(round(b * len(held)))]].sum() / yh.sum():.1%}" for b in BUDGETS)

    print(f"held-out: {len(held)} ordinary rows, {yh.sum()} covered-truth errors ({yh.mean():.1%}); share of errors caught at each send budget")
    print(f"{'time margin (production)':34s} AUC {roc_auc_score(yh, [-r['margin'] for r in held]):.3f}  {curve([-r['margin'] for r in held])}")
    singles = {"competitor mean frame prob": [r["p_comp"] for r in held],
               "competitor wins (share of frames)": [frames[r["file"]][r["index"]]["comp_wins"] for r in held],
               "overlap frames (2 tracks > 0.5)": [frames[r["file"]][r["index"]]["overlap"] for r in held]}
    singles.update({f"{k}": list(nx(held)[:, i]) for i, k in enumerate(NAMES)})
    for label, s in singles.items():
        sign = -1 if label in ("support", "support_soft_margin") else 1
        s = sign * np.asarray(s)
        print(f"{label:34s} AUC {roc_auc_score(yh, s):.3f}  {curve(s)}")
    sets = {"row means (11)": (pf.matrix(dev), pf.matrix(held)),
            "11 + per-frame (15)": (np.c_[pf.matrix(dev), fx(dev)], np.c_[pf.matrix(held), fx(held)]),
            "11 + per-frame + new (7)": (np.c_[pf.matrix(dev), fx(dev), nx(dev)], np.c_[pf.matrix(held), fx(held), nx(held)])}
    for label, (xd, xh) in sets.items():
        for kind, model in (("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))),
                            ("trees", HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=300,
                                                                    min_samples_leaf=100, random_state=0))):
            s = model.fit(xd, yd).predict_proba(xh)[:, 1]
            print(f"{label + ' ' + kind:34s} AUC {roc_auc_score(yh, s):.3f}  {curve(s)}")
    print(f"(margin <= 0.5 sends {np.mean([r['margin'] <= 0.5 for r in held]):.1%} and catches "
          f"{yh[[r['margin'] <= 0.5 for r in held]].sum() / yh.sum():.1%}; rows with lead share >= 0.9 hold "
          f"{yh[[r['lead_share'] >= 0.9 for r in held]].sum() / yh.sum():.0%} of the errors)")




def compact():
    """Small linear models under the covered truth: which few features carry it, with file-paired bootstrap vs margin."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows, _ = pf.load_rows("covered")
    pick = lambda fs: [r for f in fs for r in rows[f] if r["state"] == "ordinary"]  # noqa: E731
    dev, held = pick(pf.SPLIT["dev"]), pick(pf.SPLIT["heldout"])
    wrong = lambda rs: np.array([r["wrong"] for r in rs])  # noqa: E731
    yd, yh = wrong(dev), wrong(held)
    margin_score = -np.array([r["margin"] for r in held])
    files = sorted({r["file"] for r in held})

    def versus(score, budget):
        k = int(round(budget * len(held)))
        flag_s = set(np.argsort(-score, kind="stable")[:k])
        flag_m = set(np.argsort(-margin_score, kind="stable")[:k])
        per = []
        for f in files:
            idx = [i for i, r in enumerate(held) if r["file"] == f]
            per.append((sum(yh[i] for i in idx if i in flag_s), sum(yh[i] for i in idx if i in flag_m), sum(yh[i] for i in idx)))
        stat, lo, hi = pf.bootstrap_diff(per)
        return f"{stat * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"

    for label, cols in (("all 11", list(pf.ROW_FEATURES)), ("margin + lead_share + p_comp", ["margin", "lead_share", "p_comp"]),
                        ("+ p_lead + seconds + chars", ["margin", "lead_share", "p_comp", "p_lead", "seconds", "chars"]),
                        ("coverage + row only (no frame prob / soft)", ["lead_share", "margin", "seconds", "chars", "same_prev", "same_next", "track_share"])):
        idx = [pf.ROW_FEATURES.index(c) for c in cols]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)).fit(pf.matrix(dev)[:, idx], yd)
        s = model.predict_proba(pf.matrix(held)[:, idx])[:, 1]
        coef = dict(zip(cols, model[-1].coef_[0].round(2)))
        print(f"{label}: vs margin at 15% {versus(s, 0.15)}, at 25% {versus(s, 0.25)}, at 30% {versus(s, 0.30)}")
        print(f"   coefficients (standardised): {coef}")


if __name__ == "__main__":
    {"features": build, "evaluate": evaluate, "compact": compact}[(sys.argv[1:] or ["evaluate"])[0]]()
