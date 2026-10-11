"""Speaker judge round 6 (owner 2026-10-07): today's prompts on round 3's two 魔裁 windows only.

Round 5 (`window_judge_r5.py`, report 5.19) lost on 魔裁 after the cold read's bundle of changes; two of them were then
withdrawn and the owner added a caution line to the audio note. The reviewer pointed out that kept changes (the `keep`
wording "保留现在的主轨", the dropped "轨号相同不等于同一个人") may carry the loss too. This round reruns the two 魔裁
windows with the prompts as they are now, reusing round 5's cast blocks (the cast-block prompt has not changed since),
so round 6 - round 5 isolates the judge-prompt edits and round 6 - round 3 the bundle that stays. Criteria: report 5.19.

    python tools/diarization_explore/window_judge_r6.py build | run --arm text-open_gen | score
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_judge_r5 as r5  # noqa: E402  (patches round 3's load / windows / OUT; re-patched below)
import window_partition_score as ps  # noqa: E402

r3 = r5.r3
OUT = r3.lr.OUT / "r6"
ROUND3, ROUND5 = r5.ROUND3, r5.OUT
ARMS = r5.ARMS
SEED = 20261013


def windows():
    return [w for w in r5.windows() if w[0].startswith("manosaba-")]


r3.OUT = OUT
r3.windows = windows


def _cast_blocks():
    (OUT / "cast").mkdir(parents=True, exist_ok=True)
    for ident, a, _ in windows():
        src = ROUND5 / "cast" / f"{ident}-{a}.json"
        dst = OUT / "cast" / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)


def build():
    from finesub.llm.token_budget import default_token_counter
    _cast_blocks()
    counter = default_token_counter()
    total = 0
    for arm in ARMS:
        cs = r3.calls(arm)
        text = sum(counter.count_text(c[1]) + counter.count_text(c[2]) for c in cs)
        audio = sum(c[5] for c in cs) * 32
        out_tokens = sum(len(c[3]) for c in cs) * 35
        total += text + audio + out_tokens
        print(f"{arm}: {len(cs)} calls, text input {text}, audio {audio:.0f}, output ~{out_tokens}")
        (OUT / f"prompt-{arm}-example.md").write_text(cs[0][1] + "\n\n=====\n\n" + cs[0][2], encoding="utf-8")
    print(f"all ~{total:.0f} tokens (a floor: thinking and retries not counted; cast blocks reused); counted by {counter.source}")


def score():
    rng = np.random.default_rng(SEED)
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    clusters = [base[k]["cluster"] for k in keys]

    def interval(d, mask):
        from collections import defaultdict
        groups = defaultdict(list)
        for j in np.flatnonzero(mask):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return f"{d[mask].mean():+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]"

    def partition(picks):
        right = {}
        for ident, a, b in windows():
            m = r3.load(ident)
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            right.update({(ident, a, i): ok for i, ok in ps.partition_right(rows, ps.final_labels(r3, m, ident, a, b, picks), m.truth).items()})
        return np.array([right[k] for k in keys], float)

    every = np.ones(len(keys), bool)
    per_window = {ident: np.array([k[0] == ident for k in keys]) for ident, _, _ in windows()}
    prod = partition({})
    print(f"labelled rows with truth {len(keys)}; production (partition) {prod.mean():.1%}")
    for arm in ARMS:
        now = r3.answers(arm)
        if not now:
            continue
        cols = {"r6": partition(now), "r5": partition(r3.answers(arm, ROUND5 / arm)), "r3": partition(r3.answers(arm, ROUND3 / arm))}
        print(f"{arm}: " + ", ".join(f"{n} {v.mean():.1%}" for n, v in cols.items()) + "; per window "
              + "; ".join(f"{i} " + "/".join(f"{v[mask].mean():.1%}" for v in cols.values()) for i, mask in per_window.items()))
        print(f"  MAIN r6 - r3: {interval(100 * (cols['r6'] - cols['r3']), every)}; r6 - r5: {interval(100 * (cols['r6'] - cols['r5']), every)}; "
              f"r6 - production: {interval(100 * (cols['r6'] - prod), every)}")
        res = r3.outcome(now)
        print(r3.new_track_report(res))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "run", "score"))
    parser.add_argument("--arm", choices=ARMS, default="text-open_gen")
    args = parser.parse_args()
    if args.command == "run":
        _cast_blocks()
        r3.run(args.arm, 0, 1)
    else:
        {"build": build, "score": score}[args.command]()


if __name__ == "__main__":
    main()
