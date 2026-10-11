"""Confidence marks, step 1 (owner 2026-10-07): make the marks precise without a judge. Criteria written before any run:
`docs/reports/2026-10-07-diar-marks-step1.md`.

    python tools/diarization_explore/marks_step1.py rows       # fresh held-out + 魔裁 rows and slot identities (needs p2_recompute caches)
    python tools/diarization_explore/marks_step1.py callhome   # GPU: production timelines on CALLHOME; per-channel VAD overlap truth
    python tools/diarization_explore/marks_step1.py score      # G1-G3 and G5 (G4 with `callhome`)

Groups: dev = P2 dev hina (fit, thresholds); fresh = 26 valid hina files P2 never used (the verdicts); old held-out, 姐姐咚咚 and 魔裁
are read, not fitted. Truth "covered" (owner 2026-10-05); errors counted on ordinary rows, back-channel rows (<= 4
characters without punctuation) out of the main reading (owner 2026-10-07).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_features as pf  # noqa: E402

#: P2 never used them; no 「男女主」 speaker; on accuracy-lab's validity list (tmp/diarization-accuracy-lab/inventory.json drops
#: hina-12 and hina-58: their references stop long before the audio does)
FRESH = ("hina-02", "hina-05", "hina-07", "hina-11", "hina-16", "hina-17", "hina-22", "hina-24", "hina-28", "hina-31", "hina-38", "hina-39",
         "hina-41", "hina-43", "hina-46", "hina-47", "hina-48", "hina-49", "hina-53", "hina-54", "hina-55", "hina-57",
         "hina-60", "hina-62", "hina-66", "hina-69")
MANOSABA = ("manosaba-a", "manosaba-b", "manosaba-c")
assert not set(FRESH) & set(pf.SPLIT["dev"]) and not set(FRESH) & set(pf.SPLIT["heldout"])
TUNED = ("hina-02", "hina-07", "hina-11")  # accuracy_lab.DEV: production's diarization knobs were tuned on them
BUDGET, BUDGET_ALT = 0.25, 0.15
BACKCHANNEL = 4          # characters without punctuation
TIER_RIGHT = 0.80        # q: smallest value with >= 80% of tier-B wrong rows suggesting the true track (dev)
TIER_MIN = 20            # ... over at least this many wrong tier-B rows
PURITIES = (0.6, 0.7, 0.8, 0.9)
BIG_TRACK = 0.10
SEED = 20261007
GAIN = 0.05             # G1: the combined score needs production to keep frame probabilities and soft assignments
FRESH_ROWS = pf.CACHE / "rows-fresh.pkl"
SLOTS_EXTRA = pf.CACHE / "slot_identity-fresh.pkl"
CALLHOME_OUT = pf.CACHE / "callhome"


# --- material ---------------------------------------------------------------------------------------

def slot_identity(ident, records, stable):
    """{row index: slot, its file-level track, its purity} -- `p2_slot_identity.py` for one file."""
    data = np.load(pf.CACHE / f"{ident}.npz")
    frames = data["native"].shape[0]
    maps, purity, _, _ = pf.slot_maps(data, frames)
    prob = pf.resample(data["probs"], float(data["frame_seconds"]), frames)
    major = {}
    for j in range(maps.shape[0]):
        m = maps[j][(maps[j] >= 0) & (prob[:, j] > 0.5)]
        major[j] = Counter(m.tolist()).most_common(1)[0][0] if len(m) else -1
    out = {}
    for r in records:
        row = stable[r["index"]]
        lo = int(float(row["start"]) / pf.STEP)
        hi = min(max(lo + 1, int(np.ceil(float(row["end"]) / pf.STEP))), frames)
        j = int(prob[lo:hi].sum(axis=0).argmax())
        out[(ident, r["index"])] = {"slot": j, "slot_track": major[j], "slot_purity": float(purity[j])}
    return out


def build():
    import manosaba_signals as ms
    import manosaba_truth as mt
    rows, slots = {}, {}
    for ident in FRESH:
        records, _ = pf.file_rows(ident)
        if records is None:
            print("skip", ident, "(no reference)")
            continue
        rows[ident] = records
        slots.update(slot_identity(ident, records, pf.reference(ident)[0]))
        print(ident, len(records), "rows", flush=True)
    FRESH_ROWS.write_bytes(pickle.dumps(rows))
    for clip, records in ms.rows().items():
        slots.update(slot_identity(clip, records, mt.production(clip)[0]))
    SLOTS_EXTRA.write_bytes(pickle.dumps(slots))


def groups():
    """{group: {file: derived records}} and the slot identities of every row."""
    import manosaba_signals as ms
    old, _ = pf.load_rows()
    fresh = {f: pf.derive(rs, pf.MAIN_TRUTH) for f, rs in pickle.loads(FRESH_ROWS.read_bytes()).items() if f in FRESH}
    slots = pickle.loads((pf.CACHE / "slot_identity.pkl").read_bytes()) | pickle.loads(SLOTS_EXTRA.read_bytes())
    aa = json.loads((pf.CACHE / "aa.json").read_text(encoding="utf-8"))
    failed = sorted(f for f in list(old) + list(fresh) + list(MANOSABA) if not aa.get(f, {}).get("identical"))
    print(f"left out, recomputed timeline not identical to production (or no A/A): {failed or 'none'}")
    old = {f: v for f, v in old.items() if f not in failed}
    fresh = {f: v for f, v in fresh.items() if f not in failed}
    pick = lambda names: {f: old[f] for f in names if f in old}  # noqa: E731
    return {"dev": pick(f for f in pf.SPLIT["dev"] if f.startswith("hina-")), "fresh": fresh,
            "fresh without the tuned files": {f: v for f, v in fresh.items() if f not in TUNED},
            "old held-out": pick(pf.SPLIT["heldout"]), "姐姐咚咚": pick(f for f in pf.SPLIT["dev"] if f.startswith("onetonton-")),
            "魔裁": {f: v for f, v in ms.rows().items() if f not in failed}}, slots


def ordinary(files, backchannel=False, states=("ordinary",)):
    return {f: [r for r in rs if r["state"] in states and (backchannel or r["chars"] > BACKCHANNEL)] for f, rs in files.items()}


LABELLED = ("ordinary", "merged", "switched")


# --- statistics -------------------------------------------------------------------------------------

def caught_weights(scores, budget):
    """Per row: 1 above the cut, the tie share at it, else 0 -- the top `budget` share by score, ties in expectation."""
    scores = np.asarray(scores, float)
    k = budget * len(scores)
    if k < 1:
        return np.zeros(len(scores))
    cut = np.sort(scores)[::-1][int(np.ceil(k)) - 1]
    above, tie = scores > cut, scores == cut
    return above + tie * (k - above.sum()) / max(1, tie.sum())


def file_bootstrap(per_file, stat, draws=5000):
    """(value, lo, hi) of `stat(list of per-file tuples)`, files resampled."""
    rng = np.random.default_rng(SEED)
    keys = list(per_file)
    value = stat([per_file[k] for k in keys])
    sims = [stat([per_file[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(draws)]
    sims = [s for s in sims if s is not None]
    return value, np.percentile(sims, 2.5), np.percentile(sims, 97.5)


def fmt(v, pp=False):
    value, lo, hi = v
    if value is None:
        return "—"
    return f"{100 * value:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] pp" if pp else f"{100 * value:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def ratio(parts):
    num, den = sum(p[0] for p in parts), sum(p[1] for p in parts)
    return num / den if den else None


# --- G1-G3 ------------------------------------------------------------------------------------------

def suspicion(model):
    """{name: rows -> scores (higher = more likely wrong)}: the combined score, the time margin, production's `?`."""
    return {"combined": lambda rs: model.predict_proba(pf.matrix(rs))[:, 1] if rs else np.zeros(0),
            "margin": lambda rs: -np.array([r["margin"] for r in rs], float),
            # production's `?` (assign.py): the runner-up also reaches min_share 0.20 and is within 15% of the leader
            "review": lambda rs: np.array([float(r["margin"] <= 0.15 and r["lead_share"] * (1 - r["margin"]) >= 0.20) for r in rs])}


def g1(files, score, budget):
    """{file: (caught by a, caught by b, errors)} at `budget` over the pooled rows -- a, b are scoring names."""
    keys = [(f, i) for f, rs in files.items() for i in range(len(rs))]
    flat = [files[f][i] for f, i in keys]
    wrong = np.array([r["wrong"] for r in flat], float)
    caught = {name: caught_weights(fn(flat), budget) * wrong for name, fn in score.items()}
    out = defaultdict(lambda: np.zeros(len(score) + 1))
    for j, (f, _) in enumerate(keys):
        out[f] += [caught[name][j] for name in score] + [wrong[j]]
    return {f: tuple(v) for f, v in out.items()}


def alternative(cmodel, r):
    """(track, score) of the best candidate other than the leader, or (None, 0)."""
    alts = [(k, p) for k, p in pf.s_k(cmodel, r) if k != r["leader"]]
    return max(alts, key=lambda kp: kp[1]) if alts else (None, 0.0)


def choose_q(rows, cmodel):
    """Smallest q with >= TIER_RIGHT of the wrong tier-B rows suggesting a true track (dev)."""
    pairs = [(alternative(cmodel, r), r) for r in rows if r["wrong"]]
    for q in sorted({round(p, 3) for (_, p), _ in pairs}):
        hit = [pf.right(r, k) for (k, p), r in pairs if k is not None and p >= q]
        if len(hit) >= TIER_MIN and np.mean(hit) >= TIER_RIGHT:
            return q
    return None


def score():
    groups_, slots = groups()
    dev = ordinary(groups_["dev"])
    model = pf.fit_error([r for rs in ordinary(groups_["dev"], True).values() for r in rs])
    cmodel = pf.fit_candidates([r for rs in ordinary(groups_["dev"], True).values() for r in rs])
    score_fns = suspicion(model)
    print(f"dev hina files {len(dev)}; fresh {len(groups_['fresh'])}; candidate model dropped {cmodel['dropped']:.1%} of dev rows")
    labelled = [r for rs in groups_["fresh"].values() for r in rs if r["leader"] is not None]
    two = [r["lead_share"] * (1 - r["margin"]) >= 0.20 for r in labelled]  # assign.py: runner-up >= min_share (and >= 0.2 of the leader)
    print(f"fresh: labelled rows with two tracks (s0+s1) {np.mean(two):.1%} of {len(labelled)}")

    # G1 on every group, main reading on fresh (ordinary rows, no back-channel rows)
    diff = lambda parts: ((sum(p[0] for p in parts) - sum(p[1] for p in parts)) / sum(p[2] for p in parts)  # noqa: E731
                          if sum(p[2] for p in parts) else None)
    caught = lambda parts, j: sum(p[j] for p in parts) / max(1e-9, sum(p[-1] for p in parts))  # noqa: E731
    chosen = None
    for name, files in groups_.items():
        excluded = Counter(r["state"] for rs in files.values() for r in rs if r["state"] in ("merged", "switched"))
        print(f"G1 {name}: rows left out of the main reading by state {dict(excluded)}")
        for label, rows in (("", ordinary(files)), (" (with back-channel rows)", ordinary(files, True)),
                            (" (every labelled row with truth, back-channel rows out)", ordinary(files, False, LABELLED))):
            n = sum(len(v) for v in rows.values())
            errs = sum(r["wrong"] for v in rows.values() for r in v)
            if not n:
                continue
            per = g1(rows, {"combined": score_fns["combined"], "margin": score_fns["margin"]}, BUDGET)
            alt = g1(rows, {"combined": score_fns["combined"], "margin": score_fns["margin"]}, BUDGET_ALT)
            share_review = np.mean([score_fns["review"]([r])[0] for v in rows.values() for r in v])
            rev = g1(rows, {"combined": score_fns["combined"], "review": score_fns["review"]}, share_review)
            main = file_bootstrap(per, diff)
            print(f"  {name}{label}: {n} rows, {errs} wrong ({errs / n:.1%}); at {BUDGET:.0%} combined {caught(per.values(), 0):.1%} "
                  f"vs margin {caught(per.values(), 1):.1%}: {fmt(main, pp=True)}; at {BUDGET_ALT:.0%} "
                  f"{caught(alt.values(), 0):.1%} vs {caught(alt.values(), 1):.1%}: {fmt(file_bootstrap(alt, diff), pp=True)}; "
                  f"production `?` flags {share_review:.1%}: combined {caught(rev.values(), 0):.1%} vs `?` {caught(rev.values(), 1):.1%}: "
                  f"{fmt(file_bootstrap(rev, diff), pp=True)}")
            if name == "fresh" and not label:
                ok = main[0] >= GAIN and main[1] > 0
                chosen = "combined" if ok else "margin"
                print(f"  VERDICT G1: {'combined score' if ok else 'time margin'} ({100 * main[0]:+.1f} pp, lower bound {100 * main[1]:+.1f})")

    # G2: tiers under the chosen score; p and q on dev
    dev_rows = [r for v in dev.values() for r in v]
    dev_scores = score_fns[chosen](dev_rows)
    p = np.quantile(dev_scores, 1 - BUDGET)
    q = choose_q([r for r, s in zip(dev_rows, dev_scores) if s >= p], cmodel)
    fresh_rows = [r for v in ordinary(groups_["fresh"]).values() for r in v]
    for name in ("combined", "margin"):
        s = score_fns[name](dev_rows)
        cut = np.quantile(s, 1 - BUDGET)
        print(f"dev p for {name} (dev {np.mean(s >= cut):.1%} flagged) flags {np.mean(score_fns[name](fresh_rows) >= cut):.1%} of fresh rows")
    print(f"G2 ({chosen}): p = {p:.4f} (dev {np.mean(dev_scores >= p):.1%} flagged, fresh {np.mean(score_fns[chosen](fresh_rows) >= p):.1%}); "
          f"q = {q}{' -- no q reaches the bar on dev: one tier' if q is None else ''}")
    for name, files in groups_.items():
        rows = ordinary(files)
        per = {}
        for f, rs in rows.items():
            sc = score_fns[chosen](rs)
            flagged = [r for r, s in zip(rs, sc) if s >= p]
            tier_b = [(r, alternative(cmodel, r)) for r in flagged]
            tier_b = [(r, k) for r, (k, s) in tier_b if q is not None and k is not None and s >= q]
            per[f] = (sum(pf.right(r, k) for r, k in tier_b if r["wrong"]), sum(r["wrong"] for r, _ in tier_b), len(tier_b), len(flagged), len(rs),
                      sum(pf.right(r, k) for r, k in tier_b))
        flagged = sum(v[3] for v in per.values())
        if not flagged:
            continue
        right = file_bootstrap({f: v[:2] for f, v in per.items()}, ratio)
        print(f"G2 {name}: flagged {flagged}/{sum(v[4] for v in per.values())} ({flagged / sum(v[4] for v in per.values()):.1%}); "
              f"tier B {sum(v[2] for v in per.values())} ({sum(v[2] for v in per.values()) / flagged:.1%} of flagged); "
              f"wrong tier-B rows whose suggestion is right {fmt(right)}; production wrong on {sum(v[1] for v in per.values())} tier-B rows; "
              f"suggestion right on all tier-B rows {sum(v[5] for v in per.values()) / max(1, sum(v[2] for v in per.values())):.1%}")
        if name == "fresh":
            share = sum(v[2] for v in per.values()) / flagged
            ok = q is not None and right[0] is not None and right[0] >= 0.70 and right[1] >= 0.60 and share >= 0.20
            print(f"  VERDICT G2: {'two tiers' if ok else 'one tier, no suggestion'}")

    # G3: missed rows, mixed rule (slot when its purity reaches the threshold, else the nearest labelled row) vs nearest
    def missed(files, backchannel=False):
        return {f: [r for r in rs if r["state"] == "missed" and (backchannel or r["chars"] > BACKCHANNEL)] for f, rs in files.items()}

    def fills(files, t, backchannel=False):
        """{file: (mixed right, nearest right, missed rows, rows filled by slot)}."""
        out = {}
        for f, rs in missed(files, backchannel).items():
            mixed = nearest = by_slot = 0
            for r in rs:
                s = slots.get((f, r["index"]))
                base = r["nearest"] is not None and pf.right(r, r["nearest"])
                nearest += base
                if s and s["slot_track"] >= 0 and s["slot_purity"] >= t:
                    mixed += pf.right(r, s["slot_track"])
                    by_slot += 1
                else:
                    mixed += base
            out[f] = (mixed, nearest, len(rs), by_slot)
        return out
    totals = {t: sum(v[0] for v in fills(groups_["dev"], t).values()) for t in PURITIES}
    best = max(PURITIES, key=lambda t: (totals[t], t))
    print(f"G3 dev: mixed-rule right by purity threshold {totals}; nearest {sum(v[1] for v in fills(groups_['dev'], 1.1).values())}; threshold {best}")
    for name, files in groups_.items():
        for label, bc in (("", False), (" (with back-channel rows)", True)):
            per = {f: v for f, v in fills(files, best, bc).items() if v[2]}
            n = sum(v[2] for v in per.values())
            if not n:
                continue
            d = file_bootstrap(per, lambda parts: sum(p[0] - p[1] for p in parts) / sum(p[2] for p in parts))
            print(f"G3 {name}{label}: missed rows {n}, filled by slot {sum(v[3] for v in per.values())}; mixed right "
                  f"{sum(v[0] for v in per.values()) / n:.1%}, nearest right {sum(v[1] for v in per.values()) / n:.1%}; mixed - nearest {fmt(d, pp=True)}")
            if name == "fresh" and not label:
                print(f"  VERDICT G3: {'mixed rule' if d[1] > 0 else 'nearest-row baseline'}")


# --- G5: track level --------------------------------------------------------------------------------

def tracks():
    """Per big track: purity (truth), share of speech, mean cosine distance of its embeddings to their mean."""
    from sklearn.metrics import roc_auc_score
    groups_, _ = groups()
    table = []
    for name, files in groups_.items():
        if name == "fresh without the tuned files":
            continue
        for f, rs in files.items():
            labelled = [r for r in rs if r["leader"] is not None and r["truth"]]
            seconds = Counter()
            weight = defaultdict(Counter)
            for r in labelled:
                seconds[r["leader"]] += r["seconds"]
                for person in r["truth"]:
                    weight[r["leader"]][person] += r["seconds"] / len(r["truth"])
            total = sum(seconds.values())
            data = np.load(pf.CACHE / f"{f}.npz")
            emb, hard = data["embeddings"], data["hard"]
            active = data["local"].sum(axis=1) > 0
            for k, s in seconds.items():
                if s < BIG_TRACK * total:
                    continue
                vec = emb[(hard == k) & active]
                vec = vec[np.isfinite(vec).all(axis=1) & (np.linalg.norm(vec, axis=1) > 0)]
                if len(vec) < 2:
                    continue
                vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
                centre = vec.mean(axis=0)
                centre /= np.linalg.norm(centre)
                purity = weight[k].most_common(1)[0][1] / sum(weight[k].values())
                table.append((name, f, k, purity, s / total, float(np.mean(1 - vec @ centre))))
    y = np.array([t[3] < pf.MERGED for t in table])
    print(f"G5: {len(table)} big tracks, {y.sum()} merged (purity < {pf.MERGED}): " + ", ".join(f"{t[1]} s{t[2]} {t[3]:.2f}" for t in table if t[3] < pf.MERGED))
    for j, what in ((4, "share of speech"), (5, "embedding dispersion")):
        print(f"  {what}: AUC {roc_auc_score(y, [t[j] for t in table]):.3f}" if 0 < y.sum() < len(y) else "  no positives")


# --- G4: CALLHOME -----------------------------------------------------------------------------------

def callhome():
    import librosa
    import soundfile
    import torch
    from silero_vad import load_silero_vad

    import edit_judge_truth as et
    import p2_recompute as pr
    import s4_callhome as sc
    from finesub.speech.diarization.assign import TrackSegment, assign_all
    from finesub.speech.diarization.stage import run_diarization, write_timeline

    if not pr.machine_free():
        raise SystemExit("hw status: busy -- run later")
    CALLHOME_OUT.mkdir(parents=True, exist_ok=True)
    vad = load_silero_vad()
    per = {}
    for ident in et.CALLHOME:
        call = ident.split("-", 1)[1]
        run = et.CALLHOME_RUNS / call
        path = CALLHOME_OUT / f"{call}-diar.json"
        if not path.exists():
            write_timeline(run_diarization(run / f"{call}-vocal.ogg", device="cuda"), path)
        timeline = json.loads(path.read_text(encoding="utf-8"))["tracks"]
        stable = json.loads((run / f"{call}-stable.json").read_text(encoding="utf-8"))["segments"]
        speakers = assign_all(stable, [TrackSegment(track=int(t["track"]), start=float(t["start"]), end=float(t["end"])) for t in timeline])
        # per-channel truth on the same cut as the run (s4_callhome --prepare)
        begin = max(0.0, min(s for _, s, _, _ in sc.utterances(et.CALLHOME_CHA / f"{call}.cha")) - sc.EDGE)
        signal, rate = soundfile.read(str(et.CALLHOME_CHA.parent / "mp3" / f"{call}.mp3"), dtype="float32", always_2d=True)
        assert signal.shape[1] == 2, (call, signal.shape)
        chans = [librosa.resample(signal[int(begin * rate):, ch], orig_sr=rate, target_sr=16000) for ch in range(2)]
        mono, mono_rate = soundfile.read(str(et.CALLHOME_CHA.parent / "audio" / f"{call}.wav"), dtype="float32")
        assert mono_rate == 16000
        a, b = (chans[0] + chans[1])[:30 * 16000], mono[:30 * 16000]
        lags = range(-80, 81)
        best = max(lags, key=lambda k: float(np.dot(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) - max(0, k)])))
        assert abs(best) <= 2, (call, "channels and the pipeline's mono are not aligned", best)
        n = int(min(len(c) for c in chans) / 16000 / pf.STEP)
        voiced, level = [], []
        for c in chans:
            x = torch.from_numpy(c[:len(c) // 512 * 512])
            probs = vad.audio_forward(x, 16000).squeeze(0).numpy()  # one value per 32 ms, no padding
            vad.reset_states()
            at = (np.arange(n) * pf.STEP / 0.032).astype(int).clip(0, len(probs) - 1)
            voiced.append(probs[at] >= 0.5)
            hop = int(pf.STEP * 16000)
            frames = c[:n * hop].reshape(n, hop)
            level.append(10 * np.log10((frames ** 2).mean(axis=1) + 1e-10))
        both_raw = voiced[0] & voiced[1]
        both = both_raw & (np.abs(level[0] - level[1]) <= 20.0)
        rows = []
        for seg, spk in zip(stable, speakers):
            lo, hi = int(float(seg["start"]) / pf.STEP), min(n, int(np.ceil(float(seg["end"]) / pf.STEP)))
            if hi <= lo:
                continue
            length = (hi - lo) * pf.STEP
            marked = len(spk.to_dict().get("tracks") or []) >= 2
            flags = []
            for grid in (both, both_raw):
                seconds = grid[lo:hi].sum() * pf.STEP
                flags += [seconds >= 0.3 and seconds >= 0.2 * length, seconds >= 0.3 or seconds >= 0.2 * length]
            # s0+s1's own meaning (assign.py): both people speak in the row, one after the other counts -- each channel
            # voiced on >= 20% of the row and the weaker >= 0.2 of the stronger (peer, 2026-10-07; post hoc)
            share = [v[lo:hi].mean() for v in voiced]
            flags.append(min(share) >= 0.2 and min(share) >= 0.2 * max(share))
            rows.append((flags, marked))
        per[call] = [(sum(f[v] and m for f, m in rows), sum(m for _, m in rows), sum(f[v] for f, _ in rows), len(rows)) for v in range(5)]
        print(call, per[call][0], flush=True)
    names = ("main: >= 0.3 s and >= 20%, energy guard", ">= 0.3 s or >= 20%, energy guard", ">= 0.3 s and >= 20%, no guard", ">= 0.3 s or >= 20%, no guard",
             "post hoc: both people in the row (s0+s1's own meaning)")
    for v, name in enumerate(names):
        sub = {c: p[v] for c, p in per.items()}
        prec = file_bootstrap(sub, lambda parts: sum(p[0] for p in parts) / max(1, sum(p[1] for p in parts)))
        rec = file_bootstrap(sub, lambda parts: sum(p[0] for p in parts) / max(1, sum(p[2] for p in parts)))
        rows = sum(p[3] for p in sub.values())
        print(f"G4 CALLHOME [{name}]: {rows} rows, {sum(p[2] for p in sub.values())} overlapped ({sum(p[2] for p in sub.values()) / rows:.1%}), "
              f"{sum(p[1] for p in sub.values())} marked s0+s1; precision {fmt(prec)}, recall {fmt(rec)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("rows", "callhome", "score", "tracks"))
    args = parser.parse_args()
    {"rows": build, "callhome": callhome, "score": score, "tracks": tracks}[args.command]()


if __name__ == "__main__":
    main()
