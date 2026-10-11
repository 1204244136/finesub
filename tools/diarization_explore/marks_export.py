"""Print the constants `finesub.speech.diarization.marks` carries, from step 1's fit (owner 2026-10-07).

The same fit `marks_step1.py score` reads: P(wrong) and the candidate model on the P2 dev hina files (ordinary rows,
back-channel rows in), p = the dev quantile that flags 25% of ordinary rows without back-channel rows, q by `choose_q`.
Paste the output over the constant block in `src/finesub/speech/diarization/marks.py` when refitting.

    python tools/diarization_explore/marks_export.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marks_step1 as ms  # noqa: E402
import p2_features as pf  # noqa: E402


def fitted():
    """(error model, candidate model, p, q) exactly as `marks_step1.score` derives them."""
    groups_, _ = ms.groups()
    with_bc = [r for rs in ms.ordinary(groups_["dev"], True).values() for r in rs]
    model = pf.fit_error(with_bc)
    cmodel = pf.fit_candidates(with_bc)
    dev_rows = [r for v in ms.ordinary(groups_["dev"]).values() for r in v]
    scores = model.predict_proba(pf.matrix(dev_rows))[:, 1]
    p = float(np.quantile(scores, 1 - ms.BUDGET))
    q = ms.choose_q([r for r, s in zip(dev_rows, scores) if s >= p], cmodel)
    return model, cmodel, p, q


def vector(values):
    return "(" + ", ".join(repr(float(v)) for v in values) + ("," if len(values) == 1 else "") + ")"


def main():
    model, cmodel, p, q = fitted()
    scaler, regression = model[0], model[-1]
    assert tuple(pf.ROW_FEATURES) == ("lead_share", "margin", "p_lead", "p_comp", "none_frac", "soft_margin", "seconds",
                                      "chars", "same_prev", "same_next", "track_share")
    print(f"ERROR_MEAN = {vector(scaler.mean_)}")
    print(f"ERROR_SCALE = {vector(scaler.scale_)}")
    print(f"ERROR_COEF = {vector(regression.coef_[0])}")
    print(f"ERROR_INTERCEPT = {float(regression.intercept_[0])!r}")
    print(f"CANDIDATE_MEAN = {vector(cmodel['mean'])}")
    print(f"CANDIDATE_STD = {vector(cmodel['std'])}")
    print(f"CANDIDATE_WEIGHTS = {vector(cmodel['w'])}")
    print(f"P_WRONG = {p!r}")
    print(f"ALTERNATIVE = {float(q)!r}")


if __name__ == "__main__":
    main()
