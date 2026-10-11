"""Speaker judge round 7 (owner 2026-10-07): round 6 with the remaining semantic edits reverted, run twice on 魔裁.

Round 6 (report 5.19) was still about 10 points below round 3 on the two 魔裁 windows. Here three judge-prompt edits
of round 5 go back to round 3's sentences -- the "two acoustic errors" warning (back to "轨号相同不等于同一个人，
轨号不同也不等于不同的人"), principle 1 ("同一个人连着说下去，通常是同一个人"), and `keep` ("现在的轨号是对的",
with a bracket for `??` and `s5+s4` rows) -- and the reply gains a `<tracks>` block naming who each label stands for
(owner 2026-10-07: compared with the most favourable window partition, not tested on its own). Round 5's cast blocks
are reused, as in round 6. Run A and its same-config repeat B measure the run-to-run spread. Criteria: report 5.19.

    python tools/diarization_explore/window_judge_r7.py build
    python tools/diarization_explore/window_judge_r7.py run --arm text-open_gen --tag a     # and --tag b
    python tools/diarization_explore/window_judge_r7.py score
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_judge_r6 as r6  # noqa: E402  (patches round 3 to the 魔裁 windows; OUT re-patched below)
import window_partition_score as ps  # noqa: E402

r3 = r6.r3
ROOT = r3.lr.OUT / "r7"
ARMS = r6.ARMS
TAGS = ("a", "b")
SEED = 20261014


def use(tag):
    """Point round 3's OUT (replies, cast blocks, audio) at run `tag`; round 5's cast blocks copied in."""
    r3.OUT = r6.OUT = ROOT / tag
    r6._cast_blocks()


def build():
    use("a")
    r6.build()


def score():
    rng = np.random.default_rng(SEED)
    use("a")
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    clusters = [base[k]["cluster"] for k in keys]
    every = np.ones(len(keys), bool)

    def interval(d):
        groups = defaultdict(list)
        for j in range(len(keys)):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return f"{d.mean():+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]"

    def partition(picks):
        right = {}
        for ident, a, b in r3.windows():
            m = r3.load(ident)
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            right.update({(ident, a, i): ok for i, ok in ps.partition_right(rows, ps.final_labels(r3, m, ident, a, b, picks), m.truth).items()})
        return np.array([right[k] for k in keys], float)

    def by_declared(picks, names):
        """Right by the judge's own `<tracks>` block (no truth used to name labels); and how often its name for a label
        agrees with the most favourable partition's person for that label, over the rows."""
        right, agree, named = {}, 0, 0
        for ident, a, b in r3.windows():
            m = r3.load(ident)
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            labels = ps.final_labels(r3, m, ident, a, b, picks)
            best = ps.match(rows, labels, m.truth)
            table = names.get((ident, a), {})
            for i in rows:
                person = m.person_of(table[labels[i]]) if labels[i] in table and table[labels[i]] != "不明" else None
                right[(ident, a, i)] = person in m.truth[i]
                if person is not None:
                    named += 1
                    agree += person == best.get(labels[i])
        return np.array([right[k] for k in keys], float), agree, named

    prod = partition({})
    print(f"labelled rows with truth {len(keys)}; production (partition) {prod.mean():.1%}")
    for arm in ARMS:
        cols = {"r3": partition(r3.answers(arm, r6.ROUND3 / arm)), "r6": partition(r3.answers(arm, r3.lr.OUT / "r6" / arm))}
        declared = {}
        for tag in TAGS:
            got = r3.answers(arm, ROOT / tag / arm)
            if got:
                cols[f"r7{tag}"] = partition(got)
                declared[tag] = by_declared(got, r3.declared(arm, ROOT / tag / arm))
        print(f"{arm}: " + ", ".join(f"{n} {v.mean():.1%}" for n, v in cols.items()))
        for tag in declared:
            col = cols[f"r7{tag}"]
            print(f"  MAIN r7{tag} - r3: {interval(100 * (col - cols['r3']))}; r7{tag} - r6: {interval(100 * (col - cols['r6']))}")
            right, agree, named = declared[tag]
            print(f"  r7{tag} by its own <tracks> names: {right.mean():.1%}; its name agrees with the most favourable partition on "
                  f"{agree}/{named} named rows")
        if "r7a" in cols and "r7b" in cols:
            print(f"  run-to-run r7a - r7b: {interval(100 * (cols['r7a'] - cols['r7b']))}; rows right in one run and wrong in the other "
                  f"(a lower bound on rows that changed person) {int((cols['r7a'] != cols['r7b']).sum())}/{len(keys)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "run", "score"))
    parser.add_argument("--arm", choices=ARMS, default="text-open_gen")
    parser.add_argument("--tag", choices=TAGS, default="a")
    args = parser.parse_args()
    if args.command == "run":
        use(args.tag)
        r3.run(args.arm, 0, 1)
    else:
        {"build": build, "score": score}[args.command]()


if __name__ == "__main__":
    main()
