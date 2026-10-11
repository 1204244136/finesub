"""Judge rounds 3-5 rescored as partitions (owner 2026-10-06, post hoc): who keeps a merged track is the judge's call.

Rounds 3-5 score a row by the file-level owner of the track it ends on (the majority person by seconds over the whole
file). On hina-71 the judges split the merged track s5 (女开拓者 / 昔涟 / 大黑塔) almost right, but left a different person
on s5 than its file-level owner (昔涟): every row of that person counts wrong although the grouping is right. The owner's
review clips 2-4 are such rows. Here every labelling -- production, each arm, and the external models of 5.18 -- is
scored as a partition of the window's rows, two ways (the rules `window_external.py` gives 35t and MOSS Pro):

  one-to-one    labels (track ids, new-track labels) matched to people by rows (Hungarian, inside the window): a person
                split over two labels loses one of them -- including production's own splits (hina-06's 咔库库)
  many-to-one   each label gets its majority person in the window: splits cost nothing, mixing does

    python tools/diarization_explore/window_partition_score.py r3   # one process per round (P2 loads hina first)
    python tools/diarization_explore/window_partition_score.py r4
    python tools/diarization_explore/window_partition_score.py r5
    python tools/diarization_explore/window_partition_score.py rows r3 audio-open_gen   # rows the arm and production disagree on
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED = 20261012


def _round(rnd):
    if rnd == "r5":
        import window_judge_r5 as r5
        return r5.r3, r5.ARMS, {arm: r5.earlier(arm) for arm in r5.ARMS}
    if rnd == "r4":
        import window_judge_r4 as r4
        return r4.r3, r4.ARMS, {}
    import window_judge_r3 as r3
    return r3, r3.ARMS, {}


def final_labels(r3, m, ident, a, b, picks):
    """Row -> the label it ends on: a picked track, a new-track label, or production's track (keep / abstain / unjudged)."""
    out = {}
    for i in range(a, b):
        lead = m.leader(i)
        pick = picks.get((ident, a, i), "")
        if re.fullmatch(r"s\d+", pick):
            out[i] = f"s{int(pick[1:])}"
        elif pick.startswith("@"):
            out[i] = pick
        else:
            out[i] = f"s{lead}" if lead is not None else None
    return out


def match(rows, labels, truth, one_to_one=True):
    """{label: person}: labels matched to people by rows inside the window (a mixed row splits its weight)."""
    from scipy.optimize import linear_sum_assignment
    weight = defaultdict(Counter)
    for i in rows:
        if labels.get(i) is not None:
            for p in truth[i]:
                weight[labels[i]][p] += 1 / len(truth[i])
    if not one_to_one:
        return {k: max(c.items(), key=lambda kv: (kv[1], kv[0]))[0] for k, c in weight.items()}
    keys, people = sorted(weight), sorted({p for c in weight.values() for p in c})
    if not keys:
        return {}
    cost = np.array([[-weight[k][p] for p in people] for k in keys])
    return {keys[r]: people[c] for r, c in zip(*linear_sum_assignment(cost)) if weight[keys[r]][people[c]] > 0}


def partition_right(rows, labels, truth, one_to_one=True):
    """{row: right?} under `match`."""
    found = match(rows, labels, truth, one_to_one)
    return {i: found.get(labels[i]) in truth[i] for i in rows}


def main(rnd):
    r3, arms, earlier = _round(rnd)
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    clusters = [base[k]["cluster"] for k in keys]
    rng = np.random.default_rng(SEED)
    sets = {"production": {}}
    sets.update({arm: r3.answers(arm) for arm in arms})
    sets.update({f"{arm} (before the fixes)": got for arm, got in earlier.items()})
    cols, many = {}, {}
    for name, picks in sets.items():
        right, right_many = {}, {}
        for ident, a, b in r3.windows():
            m = r3.load(ident)
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            labels = final_labels(r3, m, ident, a, b, picks)
            right.update({(ident, a, i): ok for i, ok in partition_right(rows, labels, m.truth).items()})
            right_many.update({(ident, a, i): ok for i, ok in partition_right(rows, labels, m.truth, one_to_one=False).items()})
        cols[name] = np.array([right[k] for k in keys], float)
        many[name] = np.array([right_many[k] for k in keys], float)
    file_level = {arm: r3.outcome(sets[arm]) for arm in arms}

    def interval(d, mask):
        groups = defaultdict(list)
        for j in np.flatnonzero(mask):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return f"{d[mask].mean():+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]"

    for ident, a, _ in r3.windows():
        mask = np.array([k[:2] == (ident, a) for k in keys])
        line = ", ".join(f"{n} {v[mask].mean():.1%}" for n, v in cols.items())
        line += "; file-level owner: " + ", ".join(
            f"{arm} {np.mean([file_level[arm][k]['anonymous'] in file_level[arm][k]['truth'] for k, x in zip(keys, mask) if x]):.1%}" for arm in arms)
        print(f"{ident} ({mask.sum()} rows): partition {line}")
    groups = {"魔裁": [k[0].startswith("manosaba-") for k in keys], "hina-71": [k[0] == "hina-71" for k in keys],
              "其余 hina": [k[0].startswith("hina-") and k[0] != "hina-71" for k in keys], "姐姐咚咚": [k[0].startswith("onetonton-") for k in keys]}
    for g, mask in groups.items():
        mask = np.array(mask)
        if not mask.any():
            continue
        print(f"{g}: one-to-one " + ", ".join(f"{n} {v[mask].mean():.1%}" for n, v in cols.items()))
        print(f"{'':{len(g)}}  many-to-one " + ", ".join(f"{n} {v[mask].mean():.1%}" for n, v in many.items()))
        for arm in arms:
            print(f"  {arm} - production: one-to-one {interval(100 * (cols[arm] - cols['production']), mask)}, "
                  f"many-to-one {interval(100 * (many[arm] - many['production']), mask)}")
            if f"{arm} (before the fixes)" in cols:
                before = f"{arm} (before the fixes)"
                print(f"  {arm} now - before the fixes: one-to-one {interval(100 * (cols[arm] - cols[before]), mask)}, "
                      f"many-to-one {interval(100 * (many[arm] - many[before]), mask)}")


def rows(rnd, arm):
    """Per window outside 魔裁 and hina-71: the rows where `arm` and production differ (one-to-one), with the judge's reason."""
    import json
    r3, _, _ = _round(rnd)
    picks, why = r3.answers(arm), {}
    for path in (r3.OUT / arm).glob("*.json"):
        if not path.name.endswith(".failed.json"):
            ident, a = path.stem.rsplit("-", 1)
            why.update({(ident, int(a), int(k[1:])): (v or {}).get("why", "")
                        for k, v in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items()})
    base = r3.outcome({})
    for ident, a, b in r3.windows():
        if ident.startswith("manosaba-") or ident == "hina-71":
            continue
        m = r3.load(ident)
        scored = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
        before, after = final_labels(r3, m, ident, a, b, {}), final_labels(r3, m, ident, a, b, picks)
        found_before, found_after = match(scored, before, m.truth), match(scored, after, m.truth)
        lines, fixed, broke = [], 0, 0
        for i in scored:
            was, now = found_before.get(before[i]) in m.truth[i], found_after.get(after[i]) in m.truth[i]
            if was != now:
                fixed, broke = fixed + now, broke + was
                s = m.stable[i]
                lines.append(f"  {'fixed' if now else 'broke'} q{i} {float(s['start']):.1f}-{float(s['end']):.1f} {before[i]} -> {after[i]} "
                             f"truth {'/'.join(sorted(m.truth[i]))} | {str(s.get('text', '')).strip()} | {why.get((ident, a, i), '')}")
        print(f"{ident}-{a}: rows {len(scored)}, fixed {fixed}, broke {broke}")
        print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
