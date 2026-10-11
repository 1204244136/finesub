"""Does production's pre-screen see the 魔裁 errors? (owner 2026-10-06; criteria: report section 5.12, written first.)

Rows: production-labelled rows with nameplate truth (`manosaba_truth.py`), features from `p2_features.row_records` over
the GPU recompute (`p2_recompute.py --only manosaba-a manosaba-b manosaba-c`, A/A identical). A row is wrong when its
track's majority person is not its truth. Arms: time margin (ascending); P(错), the logistic model fitted on hina dev
ordinary rows (unchanged); secondary, the same 11 features refitted leave-one-clip-out on 魔裁. Ties at the cut are
taken in expectation (the time margin is 1.0 on most rows).

    python tools/diarization_explore/manosaba_signals.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manosaba_truth as mt  # noqa: E402
import p2_features as pf  # noqa: E402

BUDGETS = (0.15, 0.25, 0.30)
GATE = (0.25, 0.40)  # budget, recall the transferred P(错) must reach (and its lower bound must beat the budget)
ROWS = pf.CACHE / "manosaba-rows.pkl"


def rows():
    """{clip: derived records}, truth = the nameplate person (both truth kinds)."""
    if not ROWS.exists():
        table, out = mt.names(), {}
        for clip in mt.CLIPS:
            truth = {r["index"]: r["truth"] for r in mt.rows_with_truth(clip, table)}
            stable, _ = mt.production(clip)
            by_row = {id(row): i for i, row in enumerate(stable)}

            def truths(row, truth=truth, by_row=by_row):
                who = truth.get(by_row[id(row)])
                return ((who,), (who,)) if who else ((), ())
            out[clip], _ = pf.row_records(clip, stable, truths)
        ROWS.write_bytes(pickle.dumps(out))
    return {clip: pf.derive(rs, "covered") for clip, rs in pickle.loads(ROWS.read_bytes()).items()}


def caught(scores, wrong, budget):
    """Expected share of the errors among the top `budget` share of rows by score (ties in random order)."""
    scores, wrong = np.asarray(scores, float), np.asarray(wrong, float)
    k = budget * len(scores)
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    above = s > s[int(np.ceil(k)) - 1] if k >= 1 else np.zeros(len(s), bool)
    tie = s == s[int(np.ceil(k)) - 1] if k >= 1 else np.zeros(len(s), bool)
    take_tie = (k - above.sum()) / max(1, tie.sum())
    w = wrong[order]
    return (w[above].sum() + take_tie * w[tie].sum()) / max(1.0, wrong.sum())


def auc(score, wrong):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(wrong, score)


def hina_model():
    """P(错) as fitted in P2 step 1: hina dev, ordinary rows, the covered truth."""
    hina, _ = pf.load_rows("covered")
    return pf.fit_error([r for f in pf.SPLIT["dev"] if f in hina for r in hina[f] if r["state"] == "ordinary"])


def main():
    material = rows()
    labelled = [r for clip in mt.CLIPS for r in material[clip] if "wrong" in r]
    wrong = np.array([r["wrong"] for r in labelled], float)
    clips = np.array([r["file"] for r in labelled])
    print(f"labelled rows with truth {len(labelled)}, wrong {int(wrong.sum())} ({wrong.mean():.1%}); "
          f"rows whose time margin is 1.0: {np.mean([r['margin'] >= 1.0 for r in labelled]):.0%}")

    model = hina_model()
    x = pf.matrix(labelled)
    arms = {"time margin": -np.array([r["margin"] for r in labelled]),
            "P(错), hina model": model.predict_proba(x)[:, 1]}
    loco = np.zeros(len(labelled))
    for clip in mt.CLIPS:
        held = clips == clip
        fit = pf.fit_error([r for r, h in zip(labelled, held) if not h])
        loco[held] = fit.predict_proba(x[held])[:, 1]
    arms["P(错), 魔裁 leave-one-clip-out"] = loco

    rng = np.random.default_rng(20261006)
    draws = [rng.integers(0, len(labelled), len(labelled)) for _ in range(2000)]
    for name, score in arms.items():
        cells = []
        for b in BUDGETS:
            point = caught(score, wrong, b)
            sims = [caught(score[d], wrong[d], b) for d in draws]
            cells.append(f"{b:.0%}: {point:.1%} [{np.percentile(sims, 2.5):.1%}, {np.percentile(sims, 97.5):.1%}]")
        print(f"{name:32s} AUC {auc(score, wrong):.3f}; errors caught at " + "; ".join(cells))
    budget, need = GATE
    point = caught(arms["P(错), hina model"], wrong, budget)
    low = np.percentile([caught(arms["P(错), hina model"][d], wrong[d], budget) for d in draws], 2.5)
    print(f"verdict (5.12): P(错) at {budget:.0%} catches {point:.1%}, lower bound {low:.1%} -> "
          f"{'SEES the errors' if point >= need and low > budget else 'does not see them'} (gate >= {need:.0%} and lower bound > {budget:.0%})")
    print("per-feature AUC (descriptive; > 0.5 = higher value on wrong rows):",
          {f: round(auc(x[:, i], wrong), 3) for i, f in enumerate(pf.ROW_FEATURES)})
    for clip in mt.CLIPS:
        held = clips == clip
        print(f"  {clip}: rows {held.sum()}, wrong {int(wrong[held].sum())}; at 25%: margin {caught(arms['time margin'][held], wrong[held], 0.25):.1%}, "
              f"hina model {caught(arms['P(错), hina model'][held], wrong[held], 0.25):.1%}, loco {caught(loco[held], wrong[held], 0.25):.1%}")


if __name__ == "__main__":
    main()
