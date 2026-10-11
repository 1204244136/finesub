"""Post-hoc (not a gate): use Nemotron frame probabilities frame by frame instead of as row means.

`p2_features` reduces a row's frames to three numbers (leader mean, strongest competitor mean, share of frames where no
track > 0.5). This adds the shape of the per-frame evidence: per-frame leader - competitor margin (min, quantiles, share
of frames the competitor wins), overlap frames, leader dips, competitor peaks, edges vs middle, and the raw (unmapped)
slot picture: how many slots fire, how often the winning slot changes, how often the winning slot maps to the leader.
Same dev fit / held-out read, same 15% budget as the step-1 trigger gate.

    python tools/diarization_explore/p2_frames.py [features|evaluate]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf  # noqa: E402

OUT = pf.CACHE / "frames.pkl"
FRAME_NAMES = ("fm_min", "fm_q10", "fm_q50", "fm_mean", "comp_wins", "overlap", "lead_min", "lead_std", "comp_max",
               "comp_edge", "comp_mid", "slots_firing", "slot_switches", "argmax_is_leader", "vote_share")


def file_features(ident, records):
    stable = pf.reference(ident)[0]
    data = np.load(pf.CACHE / f"{ident}.npz")
    frames, tracks_n = data["native"].shape
    maps, _, _, _ = pf.slot_maps(data, frames)
    prob = pf.resample(data["probs"], float(data["frame_seconds"]), frames)
    trackprob = np.zeros((frames, tracks_n), dtype=np.float32)
    for j in range(maps.shape[0]):
        ok = maps[j] >= 0
        np.maximum.at(trackprob, (np.flatnonzero(ok), maps[j][ok]), prob[ok, j])
    out = {}
    for r in records:
        row = stable[r["index"]]
        lo = int(float(row["start"]) / pf.STEP)
        hi = min(max(lo + 1, int(np.ceil(float(row["end"]) / pf.STEP))), frames)
        leader = r["leader"]
        tp, sp = trackprob[lo:hi], prob[lo:hi]
        others = np.delete(tp, leader, axis=1) if leader is not None else tp
        lead = tp[:, leader] if leader is not None else np.zeros(len(tp))
        comp = others.max(axis=1) if others.shape[1] else np.zeros(len(tp))
        fm = lead - comp
        n = len(tp)
        edge = np.r_[0:max(1, n // 4), max(0, n - n // 4):n]
        mid = np.arange(n // 4, max(n // 4 + 1, n - n // 4))
        # ⚠ Only frames where something is speaking: on silence argmax is 0 and would bias votes and switches.
        speaking = tp.max(axis=1) > 0.5
        win = sp.argmax(axis=1)
        mapped = maps[win, np.arange(lo, hi)]
        votes = np.bincount(np.argmax(tp[speaking], axis=1), minlength=tracks_n)
        out[r["index"]] = dict(
            fm_min=float(fm.min()), fm_q10=float(np.quantile(fm, 0.1)), fm_q50=float(np.median(fm)), fm_mean=float(fm.mean()),
            comp_wins=float((comp > lead).mean()), overlap=float(((tp > 0.5).sum(axis=1) >= 2).mean()),
            lead_min=float(lead.min()), lead_std=float(lead.std()), comp_max=float(comp.max()),
            comp_edge=float(comp[edge].mean()), comp_mid=float(comp[mid[mid < n]].mean()) if (mid < n).any() else float(comp.mean()),
            slots_firing=float((sp.max(axis=0) > 0.5).sum()), slot_switches=float((np.diff(win[speaking]) != 0).sum()) / max(1, int(speaking.sum()) - 1),
            argmax_is_leader=float((mapped == leader).mean()) if leader is not None else 0.0,
            vote_share=float(votes.max() / max(1, votes.sum())), vote_track=int(votes.argmax()))
    return out


def build():
    rows = pickle.loads((pf.CACHE / "rows.pkl").read_bytes())["rows"]
    order = sorted(rows, key=lambda i: i.startswith("onetonton"))  # ⚠ hina first, as p2_features.build_rows
    feats = {}
    for ident in order:
        feats[ident] = file_features(ident, [r for r in rows[ident] if r["leader"] is not None])
        print(ident, len(feats[ident]), flush=True)
    OUT.write_bytes(pickle.dumps(feats))


def evaluate():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows, _ = pf.load_rows()
    feats = pickle.loads(OUT.read_bytes())
    dev = [f for f in pf.SPLIT["dev"] if f in rows]
    held = [f for f in pf.SPLIT["heldout"] if f in rows]
    ordinary = lambda fs: [r for f in fs for r in rows[f] if r["state"] == "ordinary"]  # noqa: E731
    dev_rows, held_rows = ordinary(dev), ordinary(held)
    frame_x = lambda rs: np.array([[feats[r["file"]][r["index"]][k] for k in FRAME_NAMES] for r in rs])  # noqa: E731
    yd = np.array([r["wrong"] for r in dev_rows])
    wrong = np.array([r["wrong"] for r in held_rows])
    margin_flag = np.argsort(np.array([r["margin"] for r in held_rows]))[: int(round(pf.FLAG * len(held_rows)))]

    def catch(score):
        flag = np.argsort(-score)[: int(round(pf.FLAG * len(held_rows)))]
        per = []
        for f in held:
            idx = np.array([i for i, r in enumerate(held_rows) if r["file"] == f])
            per.append((wrong[np.intersect1d(flag, idx)].sum(), wrong[np.intersect1d(margin_flag, idx)].sum(), wrong[idx].sum()))
        stat, lo, hi = pf.bootstrap_diff(per)
        return wrong[flag].sum() / wrong.sum(), stat, lo, hi

    print(f"time margin alone: {wrong[margin_flag].sum() / wrong.sum():.1%} of {wrong.sum()} held-out errors at 15% flagged")
    sets = {"row means (pre-registered 11)": (pf.matrix(dev_rows), pf.matrix(held_rows)),
            "per-frame shape alone (15)": (frame_x(dev_rows), frame_x(held_rows)),
            "coverage + per-frame shape": (np.c_[pf.matrix(dev_rows)[:, :2], frame_x(dev_rows)], np.c_[pf.matrix(held_rows)[:, :2], frame_x(held_rows)]),
            "all 11 + per-frame shape": (np.c_[pf.matrix(dev_rows), frame_x(dev_rows)], np.c_[pf.matrix(held_rows), frame_x(held_rows)])}
    for label, (xd, xh) in sets.items():
        for kind, model in (("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))),
                            ("trees", HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=300,
                                                                    min_samples_leaf=100, random_state=0))):
            rate, stat, lo, hi = catch(model.fit(xd, yd).predict_proba(xh)[:, 1])
            print(f"{label:32s} {kind:6s} {rate:.1%}  vs margin {stat * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]")
    flagged = [r for r in held_rows if r["margin"] <= 0.5]
    bad = [r for r in flagged if r["wrong"]]
    vote = lambda r: pf.right(r, feats[r["file"]][r["index"]]["vote_track"])  # noqa: E731
    print(f"per-frame majority track (argmax each frame, then vote) on margin <= 0.5 rows: right {np.mean([vote(r) for r in flagged]):.1%} "
          f"(production {np.mean([not r['wrong'] for r in flagged]):.1%}); on rows production got wrong {np.mean([vote(r) for r in bad]):.1%}")


if __name__ == "__main__":
    build() if (sys.argv[1:] or ["evaluate"])[0] == "features" else evaluate()
