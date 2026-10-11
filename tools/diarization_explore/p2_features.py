"""P2 step 1, CPU part: row features, the two scores, and the pre-registered readings (owner 2026-10-05).

Design: `docs/reports/2026-10-04-llm-speaker-prompts.md`, section "P2 设计". Reads `p2_recompute.py`'s cache
(tmp/diarization-p2/<file>.npz, A/A verdicts in aa.json); calls no model.

    python tools/diarization_explore/p2_features.py rows        # build per-row tables (dev + held-out)
    python tools/diarization_explore/p2_features.py evaluate [--truth onset|covered]   # fit on dev, read on held-out

Truth (owner 2026-10-05): every row carries two truths as sets of people -- `covered` (everyone tied for the most
reference coverage of the row; the main truth since 2026-10-05) and `onset` (the one person whose reference line starts
in the row, else `covered`; the pre-registered truth). A label is right when its track's owner is IN the set: when two or
three reference lines cover the whole row, any of them counts (before 2026-10-05 the tie went to the first name in
sorting order). `load_rows(truth)` derives owners, `wrong` and the row states for the chosen truth.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))

CACHE = n3.MAIN / "tmp" / "diarization-p2"
SPLIT = json.loads((HERE / "p2_split.json").read_text(encoding="utf-8"))
STEP = 0.02            # the 20 ms grid of `native`
PURITY = 0.80          # slot majority purity needed for the file-level fallback
CANDIDATE_PROB = 0.05  # missed-row / candidate floor on the mean mapped frame probability
MAIN_TRUTH = "covered"  # owner 2026-10-05
MERGED = 0.80
WINDOW = 150.0
FLAG = 0.15
RECALL_DEV, RECALL_HELD = 0.90, 0.85
ROW_FEATURES = ("lead_share", "margin", "p_lead", "p_comp", "none_frac", "soft_margin", "seconds", "chars",
                "same_prev", "same_next", "track_share")
CAND_FEATURES = ("cover", "prob", "soft")


# --- per-file material ---------------------------------------------------------------------------

_CONFIGURED = set()


def _configure(corpus):
    """`corpus_keys` once per corpus: onetonton's `configure()` nests lab.OUT and rebinds sources each call."""
    import nemotron3_cost as nc
    if corpus not in _CONFIGURED:
        nc.corpus_keys(corpus)
        _CONFIGURED.add(corpus)


def reference(ident):
    """(stable, grid, people, starts) from the refined references."""
    import backend_cost as bc
    import refined_multi as rm
    import relabel_eval as ev
    if ident.startswith("onetonton-"):
        _configure("onetonton")
        stable, _, grid, people, _ = ev.episode(int(ident[-2:]))
        return stable, grid, people, bc.refined_starts(int(ident[-2:]))
    assert "onetonton" not in _CONFIGURED, "hina must load before onetonton (configure rebinds refined_multi.sources)"
    _configure("hina")
    loaded = rm.episode(ident, rm.sources()[ident])
    if loaded is None:
        return None
    stable, _, grid, people, _, lines = loaded
    return stable, grid, people, sorted((a, w) for a, _, w in lines)


def slot_maps(data, frames):
    """Per slot j: track id per 20 ms frame (-1 = unattributed), and the slot's majority purity."""
    from finesub.speech.diarization import diarizen_assemble as da
    from finesub.speech.diarization.nemotron3_backend import STEP_SECONDS
    local, hard, centroid_of_track = data["local"], data["hard"], data["centroid_of_track"]
    chunks, per_window, slots = local.shape
    tracks = len(centroid_of_track)
    valid = np.array([centroid_of_track[k] >= 0 for k in range(tracks)])
    starts = np.array([da.closest_frame(c * STEP_SECONDS + 0.5 * da.FRAME_DURATION) for c in range(chunks)])
    activity = local.sum(axis=1)  # (chunks, slots)
    maps, purity = np.full((slots, frames), -1, dtype=np.int32), np.zeros(slots)
    for j in range(slots):
        weight = np.zeros((frames, tracks), dtype=np.float32)
        for c in np.flatnonzero(activity[:, j] > 0):  # ⚠ mask by activity, as `assemble` does -- not by `hard`
            k = int(hard[c, j])
            if not (0 <= k < tracks) or not valid[k]:
                continue
            a, b = max(0, starts[c]), min(frames, starts[c] + per_window)
            if b > a:
                weight[a:b, k] += activity[c, j]
        totals = weight.sum(axis=0)
        purity[j] = totals.max() / totals.sum() if totals.sum() else 0.0
        covered = weight.sum(axis=1) > 0
        maps[j, covered] = weight[covered].argmax(axis=1)
        if purity[j] >= PURITY and totals.sum():
            maps[j, ~covered] = int(totals.argmax())  # file-level fallback for sub-threshold activity
    return maps, purity, starts, activity


def resample(probs, frame_seconds, frames):
    """Model-rate probabilities -> the 20 ms grid, overlap-weighted mean."""
    out = np.zeros((frames, probs.shape[1]), dtype=np.float32)
    ratio = STEP / frame_seconds
    for t in range(frames):
        lo, hi = t * ratio, (t + 1) * ratio
        i0, i1 = int(np.floor(lo)), int(np.ceil(hi))
        idx = np.arange(i0, min(i1, len(probs)))
        if not len(idx):
            continue
        w = np.minimum(idx + 1, hi) - np.maximum(idx, lo)
        out[t] = (probs[idx] * w[:, None]).sum(axis=0) / w.sum()
    return out


def file_rows(ident):
    """One record per row with a refined-reference truth: production label, features, candidates with features."""
    import relabel_eval as ev
    material = reference(ident)
    if material is None:
        return None, None
    stable, grid, people, starts_ref = material

    def truths(row):
        starters = {w for a, w in starts_ref if float(row["start"]) - 0.3 <= a < float(row["end"])}
        cover, _ = ev.span_cover(row, grid)
        # ⚠ A set, not argmax: extended reference end times often let two or three lines cover the whole row, and
        # argmax would hand the tie to whichever name sorts first.
        covered = tuple(sorted(people[i] for i in np.flatnonzero(cover == cover.max()))) if cover.any() else ()
        onset = (next(iter(starters)),) if len(starters) == 1 else covered
        return onset, covered
    return row_records(ident, stable, truths)


def row_records(ident, stable, truths):
    """Records for the rows `truths(row) -> (onset, covered)` gives a truth, from `CACHE/<ident>.npz`."""
    from finesub.speech.diarization.assign import assign_all, track_coverage
    from finesub.speech.diarization.segments import TrackSegment, segments_from_activity

    data = np.load(CACHE / f"{ident}.npz")
    native, soft, hard = data["native"], data["soft"], data["hard"]
    centroid_of_track = data["centroid_of_track"]
    frames, tracks_n = native.shape
    maps, purity, chunk_starts, activity = slot_maps(data, frames)
    prob = resample(data["probs"], float(data["frame_seconds"]), frames)
    trackprob = np.zeros((frames, tracks_n), dtype=np.float32)
    for j in range(maps.shape[0]):
        ok = maps[j] >= 0
        np.maximum.at(trackprob, (np.flatnonzero(ok), maps[j][ok]), prob[ok, j])
    tracks = segments_from_activity(native.tolist(), frame_seconds=STEP)
    speakers = assign_all(stable, [TrackSegment(track=t.track, start=t.start, end=t.end) for t in tracks])
    per_window = data["local"].shape[1]
    local = data["local"]
    total_time = Counter()
    for t in tracks:
        total_time[t.track] += t.end - t.start
    all_time = sum(total_time.values()) or 1.0
    segments = [TrackSegment(track=t.track, start=t.start, end=t.end) for t in tracks]
    # ⚠ Everything production could know is computed on ALL stable rows, before the truth filter below:
    # neighbours and the nearest-labelled baseline (P1's: nearest labelled row by centre time) must not
    # skip rows the reference happens to leave empty.
    leaders = [(speaker.to_dict().get("tracks") or [None])[0] for speaker in speakers]
    centres_all = np.array([(float(r["start"]) + float(r["end"])) / 2 for r in stable])
    labelled_idx = np.array([i for i, k in enumerate(leaders) if k is not None])
    prev_of, next_of, nearest_of = {}, {}, {}
    last = None
    for i, k in enumerate(leaders):
        prev_of[i] = last
        if k is not None:
            last = k
    last = None
    for i in range(len(leaders) - 1, -1, -1):
        next_of[i] = last
        if leaders[i] is not None:
            last = leaders[i]
    for i in range(len(leaders)):
        pool = labelled_idx[labelled_idx != i]
        nearest_of[i] = leaders[int(pool[np.argmin(np.abs(centres_all[pool] - centres_all[i]))])] if len(pool) else None

    records = []
    for index, (row, speaker) in enumerate(zip(stable, speakers)):
        onset, covered = truths(row)
        if not onset and not covered:
            continue
        d = speaker.to_dict()
        ids = d.get("tracks") or []
        # ⚠ Not `d["coverage"]`: `assign_subtitle` drops it on rows no track reaches 0.2, i.e. on every missed row.
        coverage = {int(k): float(v) for k, v in track_coverage(float(row["start"]), float(row["end"]), segments).items()}
        lo = int(float(row["start"]) / STEP)
        hi = max(lo + 1, int(np.ceil(float(row["end"]) / STEP)))
        hi = min(hi, frames)
        if hi <= lo:
            continue
        rows_prob = trackprob[lo:hi].mean(axis=0)
        none_frac = float((trackprob[lo:hi].max(axis=1) <= 0.5).mean())
        # soft per track: activity-weighted mean over the (chunk, slot) pairs active inside the row
        soft_sum, soft_w, present = np.zeros(tracks_n), 0.0, set()
        own_margin_sum = own_margin_w = 0.0
        leader = int(ids[0]) if ids else None
        for c in np.flatnonzero((chunk_starts < hi) & (chunk_starts + per_window > lo)):
            a, b = max(lo, chunk_starts[c]) - chunk_starts[c], min(hi, chunk_starts[c] + per_window) - chunk_starts[c]
            for j in range(local.shape[2]):
                act = float(local[c, a:b, j].sum())
                # ⚠ A slot active for <= min_num_frames gets no embedding (`embed(only_active=True)`): its soft row is
                # all NaN and production's `constrained_argmax` fills NaN with the minimum, so its `hard` is an
                # arbitrary pick, not soft evidence. Skip it here (it still counts in `slot_maps`, as in `assemble`).
                if act <= 0 or np.isnan(soft[c, j]).all():
                    continue
                k = int(hard[c, j])
                if 0 <= k < tracks_n and centroid_of_track[k] >= 0:
                    present.add(k)
                scores = np.array([soft[c, j, centroid_of_track[t]] if centroid_of_track[t] >= 0 else np.nan
                                   for t in range(tracks_n)])
                soft_sum += np.nan_to_num(scores) * act
                soft_w += act
                if leader is not None and k == leader and centroid_of_track[leader] >= 0:
                    own = scores[leader]
                    others = np.delete(scores, leader)
                    others = others[~np.isnan(others)]
                    own_margin_sum += act * (own - (others.max() if others.size else 0.0))
                    own_margin_w += act
        soft_k = soft_sum / soft_w if soft_w else np.zeros(tracks_n)
        candidates = sorted({k for k in range(tracks_n) if centroid_of_track[k] >= 0 and
                             (coverage.get(k, 0.0) > 0 or rows_prob[k] > CANDIDATE_PROB or k in present)})
        shares = sorted(coverage.values(), reverse=True)
        record = dict(file=ident, index=index, truths={"onset": onset, "covered": covered}, leader=leader,
                      centre=float(centres_all[index]),
                      seconds=float(row["end"]) - float(row["start"]),
                      chars=len(re.sub(r"[\W_]", "", str(row.get("text", "")))),
                      lead_share=shares[0] if shares else 0.0,
                      margin=(shares[0] - (shares[1] if len(shares) > 1 else 0.0)) / shares[0] if shares and shares[0] else 0.0,
                      p_lead=float(rows_prob[leader]) if leader is not None else 0.0,
                      p_comp=float(max([rows_prob[k] for k in range(tracks_n) if k != leader] or [0.0])),
                      none_frac=none_frac,
                      soft_margin=own_margin_sum / own_margin_w if own_margin_w else 0.0,
                      track_share=total_time[leader] / all_time if leader is not None else 0.0,
                      candidates=[dict(track=k, cover=coverage.get(k, 0.0), prob=float(rows_prob[k]), soft=float(soft_k[k]))
                                  for k in candidates],
                      same_prev=float(prev_of[index] == leader), same_next=float(next_of[index] == leader),
                      nearest=nearest_of[index], sentinel_leader=leader is not None and centroid_of_track[leader] < 0)
        records.append(record)
    return records, {"purity": purity.tolist()}


def derive(records, truth):
    """One file's records under `truth`: owners, `wrong`, states. Row-level, time-weighted, labelled rows only;
    a row whose truth is several tied people gives each of them an equal share of its seconds."""
    records = [dict(r, truth=r["truths"][truth]) for r in records]
    labelled = [r for r in records if r["leader"] is not None and r["truth"]]
    weight = defaultdict(Counter)
    for r in labelled:
        for person in r["truth"]:
            weight[r["leader"]][person] += r["seconds"] / len(r["truth"])
    owner = {int(k): c.most_common(1)[0][0] for k, c in weight.items()}
    track_purity = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in weight.items()}
    centres = np.array([r["centre"] for r in labelled]) if labelled else np.zeros(0)
    for r in records:
        r["owner"] = owner
        if not r["truth"]:
            r["state"] = "no_truth"
        elif r["leader"] is None:
            r["state"] = "missed"
        elif r["leader"] not in owner:
            r["state"] = "no_truth"
        else:
            near = [labelled[j] for j in np.flatnonzero(np.abs(centres - r["centre"]) <= WINDOW) if labelled[j]["leader"] == r["leader"]]
            local_owner = Counter()
            for n in near:
                for person in n["truth"]:
                    local_owner[person] += n["seconds"] / len(n["truth"])
            r["wrong"] = owner[r["leader"]] not in r["truth"]
            r["state"] = ("merged" if track_purity[r["leader"]] < MERGED else
                          "switched" if local_owner and local_owner.most_common(1)[0][0] != owner[r["leader"]] else "ordinary")
    return records


def load_rows(truth=MAIN_TRUTH):
    """({file: records}, meta) from rows.pkl, derived under `truth` ("covered" or "onset")."""
    data = pickle.loads((CACHE / "rows.pkl").read_bytes())
    return {f: derive(rs, truth) for f, rs in data["rows"].items()}, data["meta"]


def right(r, track):
    """Is `track`'s owner one of the row's truth people?"""
    return r["owner"].get(track) in r["truth"]


def build_rows():
    rows, meta = {}, {}
    order = sorted(SPLIT["dev"] + SPLIT["heldout"], key=lambda i: i.startswith("onetonton"))  # ⚠ hina first
    for ident in order:
        if not (CACHE / f"{ident}.npz").exists():
            print("skip", ident, "(no recompute)")
            continue
        records, info = file_rows(ident)
        if records is None:
            print("skip", ident, "(no reference)")
            continue
        rows[ident], meta[ident] = records, info
        print(ident, len(records), "rows", flush=True)
    (CACHE / "rows.pkl").write_bytes(pickle.dumps({"rows": rows, "meta": meta}))


# --- models ------------------------------------------------------------------------------------

def matrix(rows):
    return np.array([[r["lead_share"], r["margin"], r["p_lead"], r["p_comp"], r["none_frac"], r["soft_margin"],
                      r["seconds"], r["chars"], r["same_prev"], r["same_next"], r["track_share"]] for r in rows], dtype=float)


def fit_error(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(matrix(rows), np.array([r["wrong"] for r in rows], dtype=int))
    return model


def candidate_x(r):
    return np.array([[c["cover"], c["prob"], c["soft"]] for c in r["candidates"]], dtype=float)


def fit_candidates(rows):
    """Conditional logit: softmax over a row's candidates; target = any candidate owned by the truth."""
    from scipy.optimize import minimize
    usable = [r for r in rows if r["candidates"] and any(right(r, c["track"]) for c in r["candidates"])]
    dropped = 1 - len(usable) / max(1, len(rows))
    stack = np.vstack([candidate_x(r) for r in usable])
    mean, std = stack.mean(axis=0), stack.std(axis=0) + 1e-9
    width = max(len(r["candidates"]) for r in usable)
    X = np.zeros((len(usable), width, 3))
    mask = np.zeros((len(usable), width), dtype=bool)
    hit = np.zeros((len(usable), width), dtype=bool)
    for i, r in enumerate(usable):
        n = len(r["candidates"])
        X[i, :n] = (candidate_x(r) - mean) / std
        mask[i, :n] = True
        hit[i, :n] = [right(r, c["track"]) for c in r["candidates"]]

    def nll(w):
        z = np.where(mask, X @ w, -np.inf)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        ph = np.where(hit, p, 0.0).sum(axis=1)
        # gradient of -log sum_{hit} p:  sum_k (p_k - q_k) x_k, q = p restricted to hits and renormalised
        qn = np.where(hit, p, 0.0) / (ph[:, None] + 1e-12)
        grad = ((p - qn)[:, :, None] * X).sum(axis=(0, 1)) + 2e-3 * w
        return -np.log(ph + 1e-12).sum() + 1e-3 * (w @ w), grad
    w = minimize(nll, np.zeros(3), jac=True, method="L-BFGS-B").x
    return {"w": w, "mean": mean, "std": std, "dropped": dropped}


def s_k(model, r):
    if not r["candidates"]:
        return []
    z = ((candidate_x(r) - model["mean"]) / model["std"]) @ model["w"]
    z -= z.max()
    p = np.exp(z) / np.exp(z).sum()
    return list(zip([c["track"] for c in r["candidates"]], p))


def bootstrap_diff(per_file, seed=20261005, draws=5000):
    """per_file: [(caught_a, caught_b, errors)]; statistic = (sum a - sum b) / sum errors, files resampled."""
    rng = np.random.default_rng(seed)
    arr = np.array(per_file, dtype=float)
    stat = (arr[:, 0].sum() - arr[:, 1].sum()) / arr[:, 2].sum()
    sims = []
    for _ in range(draws):
        pick = arr[rng.integers(0, len(arr), len(arr))]
        sims.append((pick[:, 0].sum() - pick[:, 1].sum()) / max(1.0, pick[:, 2].sum()))
    return stat, np.percentile(sims, 2.5), np.percentile(sims, 97.5)


def recall_at(rows, model, q):
    hits = 0
    for r in rows:
        if any(p >= q and right(r, k) for k, p in s_k(model, r)):
            hits += 1
    return hits / max(1, len(rows))


def tier_b_share(rows, model, q):
    return sum(any(p >= q for _, p in s_k(model, r)) for r in rows) / max(1, len(rows))


def evaluate(truth=MAIN_TRUTH):
    rows, meta = load_rows(truth)
    print(f"truth: {truth}" + ("" if truth == "onset" else "  -- POST-HOC: the pre-registered gates were written for the onset truth; "
                                                          "PASS/adopted below is a reading, not a verdict"))
    aa = json.loads((CACHE / "aa.json").read_text(encoding="utf-8"))
    dev = [f for f in SPLIT["dev"] if f in rows]
    held = [f for f in SPLIT["heldout"] if f in rows]
    assert not set(dev) & set(held)
    print(f"files with readings: dev {len(dev)}/{len(SPLIT['dev'])}, held-out {len(held)}/{len(SPLIT['heldout'])}")
    bad = [f for f in dev + held if not aa.get(f, {}).get("identical")]
    print(f"A/A: {len(dev) + len(held) - len(bad)} identical, failed: {bad}")
    dev, held = [f for f in dev if f not in bad], [f for f in held if f not in bad]
    purities = np.concatenate([np.array(meta[f]["purity"]) for f in dev + held])
    active_p = purities[purities > 0]
    print(f"slot majority purity: median {np.median(active_p):.2f}, share >= {PURITY}: {(active_p >= PURITY).mean():.0%} of {len(active_p)} active slots")

    sentinel = sum(r.get("sentinel_leader", False) for f in dev + held for r in rows[f])
    print(f"rows whose production leader is a sentinel track (no centroid): {sentinel}")
    ordinary = lambda files: [r for f in files for r in rows[f] if r["state"] == "ordinary"]  # noqa: E731
    dev_rows, held_rows = ordinary(dev), ordinary(held)
    # P(错): leave-one-file-out on dev, full-dev model on held-out
    dev_score = {}
    for f in dev:
        model = fit_error([r for g in dev if g != f for r in rows[g] if r["state"] == "ordinary"])
        part = [r for r in rows[f] if r["state"] == "ordinary"]
        if part:
            for r, s in zip(part, model.predict_proba(matrix(part))[:, 1]):
                dev_score[(f, r["index"])] = s
    full = fit_error(dev_rows)
    held_score = {(r["file"], r["index"]): s for r, s in zip(held_rows, full.predict_proba(matrix(held_rows))[:, 1])}
    coef = dict(zip(ROW_FEATURES, full[-1].coef_[0].round(2)))
    print("P(错) coefficients (standardised):", coef)
    p = float(np.quantile([dev_score[(r["file"], r["index"])] for r in dev_rows], 1 - FLAG))
    print(f"p (dev, >= {FLAG:.0%} flagged) = {p:.3f}; on held-out it flags {np.mean([held_score[(r['file'], r['index'])] >= p for r in held_rows]):.1%}")

    # step-1 trigger gate: equal 15% budgets on held-out, by P(错) vs by time margin
    def flagged(rs, key):
        order = sorted(rs, key=key)
        return {(r["file"], r["index"]) for r in order[: int(round(FLAG * len(rs)))]}
    by_p = flagged(held_rows, lambda r: -held_score[(r["file"], r["index"])])
    by_t = flagged(held_rows, lambda r: r["margin"])
    per_file = []
    for f in held:
        rs = [r for r in held_rows if r["file"] == f]
        errors = sum(r["wrong"] for r in rs)
        per_file.append((sum(r["wrong"] for r in rs if (f, r["index"]) in by_p),
                         sum(r["wrong"] for r in rs if (f, r["index"]) in by_t), errors))
    total_err = sum(e for *_, e in per_file)
    stat, lo, hi = bootstrap_diff(per_file)
    caught_p = sum(a for a, _, _ in per_file) / total_err
    caught_t = sum(b for _, b, _ in per_file) / total_err
    passed_trigger = stat >= 0.05 and lo > 0
    adopted = "P(错)" if passed_trigger else "time margin <= 0.5"
    print(f"held-out: {len(held_rows)} ordinary labelled rows, {total_err} wrong. At {FLAG:.0%} flagged: "
          f"P(错) catches {caught_p:.1%}, time margin {caught_t:.1%}; diff {stat * 100:+.1f} pts [{lo * 100:+.1f}, {hi * 100:+.1f}] "
          f"-> trigger gate {'PASS' if passed_trigger else 'fail'}")

    # candidate scores, q per trigger, tiering gate
    cand_dev = {}
    for f in dev:
        m = fit_candidates([r for g in dev if g != f for r in rows[g] if r["state"] == "ordinary"])
        cand_dev[f] = m
    cand_full = fit_candidates(dev_rows)
    print(f"adopted trigger: {adopted}")
    print(f"s_k weights (cover, prob, soft; standardised): {np.round(cand_full['w'], 2)}; rows without a true candidate dropped "
          f"from the fit: {cand_full['dropped']:.1%}")
    for trigger, dev_flag, held_flag in (
            ("P(错)", lambda r: dev_score[(r["file"], r["index"])] >= p, lambda r: (r["file"], r["index"]) in by_p),
            ("time margin <= 0.5", lambda r: r["margin"] <= 0.5, lambda r: r["margin"] <= 0.5)):
        dev_f = [r for r in dev_rows if dev_flag(r)]
        # q chosen on dev with leave-one-file-out candidate models
        best = None
        for q in np.round(np.arange(0.0, 1.0001, 0.01), 2):
            hits = sum(any(pr >= q and right(r, k) for k, pr in s_k(cand_dev[r["file"]], r)) for r in dev_f)
            if hits / max(1, len(dev_f)) >= RECALL_DEV:
                best = q
        held_f = [r for r in held_rows if held_flag(r)]
        if trigger == "P(错)":  # the operating set is the dev p, not the top-15% budget used by the trigger gate
            at_p = [r for r in held_rows if held_score[(r["file"], r["index"])] >= p]
        curve = []
        for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            share = sum(any(pr >= q for _, pr in s_k(cand_dev[r["file"]], r)) for r in dev_f) / max(1, len(dev_f))
            curve.append(f"{q:.1f}:{share:.0%}")
        print(f"[{trigger}] dev tier-乙 share vs q: " + " ".join(curve))
        if best is None:
            floor = sum(any(right(r, k) for k, _ in s_k(cand_dev[r["file"]], r)) for r in dev_f) / max(1, len(dev_f))
            print(f"[{trigger}] no q reaches {RECALL_DEV:.0%} recall on dev (recall at q=0: {floor:.1%}) -> no tiering")
            continue
        rec = recall_at(held_f, cand_full, best)
        print(f"[{trigger}] q = {best:.2f}; held-out flagged {len(held_f)} rows: candidate recall {rec:.1%} "
              f"(gate >= {RECALL_HELD:.0%}: {'PASS' if rec >= RECALL_HELD else 'fail'}), tier 乙 share {tier_b_share(held_f, cand_full, best):.1%}"
              + ("  <- adopted" if trigger == adopted else ""))
        if trigger == "P(错)":
            print(f"    at the dev p ({len(at_p)} held-out rows): candidate recall {recall_at(at_p, cand_full, best):.1%}, "
                  f"tier 乙 share {tier_b_share(at_p, cand_full, best):.1%}")

    # missed rows: top s_k vs the nearest-labelled-row baseline (descriptive, out-of-distribution for s_k)
    for name, files in (("dev", dev), ("held-out", held)):
        missed = [r for f in files for r in rows[f] if r["state"] == "missed"]
        top = sum(bool(s_k(cand_full, r)) and right(r, max(s_k(cand_full, r), key=lambda x: x[1])[0]) for r in missed)
        has = sum(bool(r["candidates"]) for r in missed)
        base = sum(r["nearest"] is not None and right(r, r["nearest"]) for r in missed)
        print(f"missed rows ({name}): {len(missed)}; with a candidate {has}; top s_k right {top} ({top / max(1, len(missed)):.0%}), "
              f"nearest-labelled baseline right {base} ({base / max(1, len(missed)):.0%})")
    excluded = Counter(r["state"] for f in held for r in rows[f])
    print("held-out row states:", dict(excluded))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("rows", "evaluate"))
    parser.add_argument("--truth", choices=("covered", "onset"), default=MAIN_TRUTH)
    args = parser.parse_args()
    build_rows() if args.command == "rows" else evaluate(args.truth)


if __name__ == "__main__":
    main()
