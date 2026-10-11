"""Gemini 3.5 Transcribe and MOSS Pro on round 3's six windows, next to production and the judges (owner 2026-10-06).

Each model is run on the audio span the audio judge heard (the window's rows and context, ±0.5 s; `run` calls
`tools.ts_diar_explore.episode --span`): 35t in 300 s windows with 60 s overlap (`g35t-w300`) and, added after the
first reading, in 540 s windows (`g35t`, the size that did best on the whole 魔裁 clips) -- labels linked across the
overlap -- and MOSS Pro in one request per span. Its speaker segments label production's ASR rows through the same
`assign_all`, and each speaker is given a person inside the window:

  many-to-one   the speaker's majority person by rows (the rule a judge's new track gets; generous: never a split)
  one-to-one    Hungarian over rows: one speaker per person, the rest name no one (the main reading)

Rows scored are round 3's: production labelled them, they have truth, and no profile took them as a reference; a row
the model left without a speaker counts wrong. Criteria: report 5.18.

    python tools/diarization_explore/window_external.py run [r4]      # calls the models (cached per window)
    python tools/diarization_explore/window_external.py score [r4]

`r4` runs on round 4's four hina windows instead (owner 2026-10-06), with 35t at 540 s only (the day's call plan had
room for one window size; 300 s did no better on round 3).
"""
from __future__ import annotations

import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_judge_r3 as r3  # noqa: E402

sys.path.insert(0, str(HERE.parents[1]))
from tools.ts_diar_explore import arms as A  # noqa: E402

MODELS = (("35t 300 s", "g35t-w300"), ("35t 540 s", "g35t"), ("MOSS Pro", "mosspro"))
RUN = {"r3": MODELS, "r4": MODELS[1:]}
JUDGES = ("text-closed", "text-open_gen", "audio-open_gen")
SEED = 20261010
MAX_WORD = 10.0


def span(m, a, b):
    shown = r3.shown_rows(m, a, b)
    return max(0.0, float(m.stable[shown[0]]["start"]) - 0.5), float(m.stable[shown[-1]]["end"]) + 0.5


def run(rnd):
    """Model by model; a span a daily quota cannot take is skipped and left for a later run (outputs are cached)."""
    spans = [(ident, *span(r3.load(ident), a, b)) for ident, a, b in r3.windows()]
    for _, name in RUN[rnd]:
        for ident, lo, hi in spans:
            if A.path(ident, name, span=(lo, hi)).exists():
                continue
            done = subprocess.run([sys.executable, "-m", "tools.ts_diar_explore.episode", ident, name, "--span", f"{lo:.3f}", f"{hi:.3f}"],
                                  cwd=str(HERE.parents[1]))
            if done.returncode:  # a daily quota (35t's call plan, MOSS Pro's key hours) or a failed call: rerun later
                print(f"skipped: {name} on {ident} ({hi - lo:.0f} s) -- see the error above", flush=True)


def leaders(m, ident, name, lo, hi):
    """Row index -> the model's speaker label (None when it gave the row no speaker)."""
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import TrackSegment
    arm = A.load(ident, name, span=(lo, hi))
    # word-timed output (35t) is used word by word: a word with a broken end time (one ran 512 s) would otherwise turn
    # into a segment covering the whole window; segment-only output (MOSS Pro) keeps its sentence segments
    units = [w for w in arm["words"] if float(w["end"]) - float(w["start"]) <= MAX_WORD] if arm.get("words") else arm.get("segments", [])
    segments = [s for s in units if s.get("speaker") and float(s["end"]) > float(s["start"])]
    ids = {label: k for k, label in enumerate(sorted({s["speaker"] for s in segments}))}
    speakers = assign_all(m.stable, [TrackSegment(track=ids[s["speaker"]], start=float(s["start"]), end=float(s["end"])) for s in segments])
    return [(s.to_dict().get("tracks") or [None])[0] for s in speakers]


def mapping(rows, lead, truth):
    """(many-to-one, one-to-one): speaker -> person, from the window's rows with truth (a mixed row splits its weight)."""
    from scipy.optimize import linear_sum_assignment
    weight = defaultdict(Counter)
    for i in rows:
        if lead[i] is not None:
            for p in truth[i]:
                weight[lead[i]][p] += 1 / len(truth[i])
    many = {s: max(c.items(), key=lambda kv: (kv[1], kv[0]))[0] for s, c in weight.items()}
    speakers, people = sorted(weight), sorted({p for c in weight.values() for p in c})
    cost = np.array([[-weight[s][p] for p in people] for s in speakers]) if speakers else np.zeros((0, 0))
    one = {speakers[r]: people[c] for r, c in zip(*linear_sum_assignment(cost)) if weight[speakers[r]][people[c]] > 0}
    return many, one


def score(rnd):
    rng = np.random.default_rng(SEED)
    base = r3.outcome({})
    judged = {arm: r3.outcome(r3.answers(arm)) for arm in JUDGES}
    keys = [k for k, v in base.items() if v["labelled"]]
    cols = {"production": np.array([base[k]["before"] in base[k]["truth"] for k in keys], float)}
    for arm in JUDGES:
        cols[arm] = np.array([judged[arm][k]["anonymous"] in judged[arm][k]["truth"] for k in keys], float)
    for label, name in MODELS:
        many_ok, one_ok, missing = {}, {}, []
        for ident, a, b in r3.windows():
            m = r3.load(ident)
            lo, hi = span(m, a, b)
            if not A.path(ident, name, span=(lo, hi)).exists():
                missing.append(ident)
                continue
            lead = leaders(m, ident, name, lo, hi)
            rows = [i for i in range(a, b) if i in m.truth]
            many, one = mapping(rows, lead, m.truth)
            for i in rows:
                many_ok[(ident, a, i)] = many.get(lead[i]) in m.truth[i]
                one_ok[(ident, a, i)] = one.get(lead[i]) in m.truth[i]
            print(f"  {label} {ident}: {len(set(x for x in lead[a:b] if x is not None))} speakers in the window, "
                  f"{sum(lead[i] is None for i in rows)} rows with truth left without a speaker")
        if missing:
            print(f"{label}: no output yet for {missing}")
        cols[f"{label} many-to-one"] = np.array([many_ok.get(k, np.nan) for k in keys], float)
        cols[f"{label} one-to-one"] = np.array([one_ok.get(k, np.nan) for k in keys], float)

    kinds = np.array([base[k]["kind"] for k in keys])
    clusters = [base[k]["cluster"] for k in keys]

    def interval(d, mask):
        groups = defaultdict(list)
        for j in np.flatnonzero(mask):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [np.nanmean(d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])]) for _ in range(5000)]
        return f"{np.nanmean(d[mask]):+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]"

    for kind in ("manosaba", "hina", "onetonton"):
        mask = kinds == kind
        print(f"{kind}: {mask.sum()} labelled rows with truth; right " + ", ".join(f"{c} {np.nanmean(v[mask]):.1%}" for c, v in cols.items()))
        for label, _ in MODELS:
            ext = cols[f"{label} one-to-one"]
            if np.isnan(ext[mask]).all():
                continue
            for other in ("audio-open_gen", "text-open_gen", "production"):
                print(f"  MAIN {label} one-to-one - {other}: {interval(100 * (ext - cols[other]), mask & ~np.isnan(ext))}")
        long, short = cols["35t 540 s one-to-one"], cols["35t 300 s one-to-one"]
        both = mask & ~np.isnan(long) & ~np.isnan(short)
        if both.any():
            print(f"  35t 540 s - 300 s (one-to-one): {interval(100 * (long - short), both)}")


def main():
    rnd = sys.argv[2] if len(sys.argv) > 2 else "r3"
    if rnd == "r4":
        import window_judge_r4  # noqa: F401 -- patches round 3's windows / load / OUT to round 4's
    {"run": run, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"](rnd)


if __name__ == "__main__":
    main()
