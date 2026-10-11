"""Speaker judge round 4 (owner 2026-10-06): round 3's best arm on talk material with more people and more errors.

Round 3 (`window_judge_r3.py`, report 5.14) found "the model narrows the cast, then may open new tracks" best on 魔裁,
but hina / 姐姐咚咚 had few errors and every person owned a track. Here: four hina files with many people and a high
production error rate, one 200-row window each, judged with round 3's code, prompts and scoring. Criteria: report 5.15.

  hina-71    the one file of the P2 corpora whose wrong rows include people without a track of their own (115 rows);
             its window is the 200 rows holding the most such rows
  hina-06, hina-08, hina-14   6 people, production error about 12%; window placed at random (fixed seed)

The cast list (what a knowledge base would give, and the cast-block writer's candidates) is the file's people plus
DISTRACTORS names drawn from other hina files, so narrowing it to the window is real work, as 魔裁's roster was.
Arms: text-closed (the baseline), text-open_gen, audio-open_gen.

    python tools/diarization_explore/window_judge_r4.py build | cast | run --arm text-closed | score
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_judge_r3 as r3  # noqa: E402

OUT = r3.lr.OUT / "r4"
SEED = 20261008
FILES = ("hina-71", "hina-06", "hina-08", "hina-14")
WINDOW = 200
DISTRACTORS = 10
ARMS = ("text-closed", "text-open_gen", "audio-open_gen")
#: hina's reference labels some speakers by role, not name; a role label as a distractor can be the window's real speaker
ROLE_LABELS = {"主持人", "嘉宾", "男主", "女主", "教师"}
#: names of one person (a game's player character, a title); a distractor in a group with a real speaker is no distractor
ALIASES = ({"女开拓者", "男开拓者", "穹", "星", "开拓者"}, {"空", "荧", "旅行者"}, {"火神", "玛薇卡"})

r3.OUT = OUT  # round 3's functions read OUT and windows() from their module: this round's go there instead
_load = r3.load


def same_person(a, b):
    return a == b or a in b or b in a or any(a in g and b in g for g in ALIASES)


def cast_list(ident):
    """The file's people plus DISTRACTORS other hina people, sorted, frozen on disk (`cast.json`). A distractor is never a
    role label, nor a name of one of the file's people (松泽 / 松泽千晶, the player character's names)."""
    path = OUT / "cast.json"
    if not path.exists():
        rng = np.random.default_rng(SEED + 1)
        rows = r3._rows()
        others = sorted({p for f, rs in rows.items() if f.startswith("hina-") and f not in FILES for r in rs for p in r["truth"]} - ROLE_LABELS)
        cast = {}
        for name in FILES:
            people = _load(name).people
            pool = [p for p in others if not any(same_person(p, q) for q in people)]
            cast[name] = sorted(set(people) | {pool[i] for i in rng.choice(len(pool), DISTRACTORS, replace=False)})
        path.write_text(json.dumps(cast, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))[ident]


def load(ident):
    m = _load(ident)
    if not getattr(m, "_r4", False):
        m.people = cast_list(ident)
        m.roster = tuple(m.people)
        m._r4 = True
    return m


def windows():
    path = OUT / "windows.json"
    if not path.exists():
        rng = np.random.default_rng(SEED)
        out = []
        for ident in FILES:
            m = _load(ident)
            n = len(m.stable)
            owned = set(m.owner.values())
            if ident == "hina-71":
                # the window holding the most wrong rows whose person has no track (the only place a new track can help)
                unfixable = np.array([i in m.truth and m.leader(i) is not None and not (m.truth[i] & owned) for i in range(n)], int)
                starts = range(r3.CONTEXT, n - WINDOW - r3.CONTEXT + 1)
                a = max(starts, key=lambda s: (unfixable[s:s + WINDOW].sum(), -s))
            else:
                a = int(rng.integers(r3.CONTEXT, n - WINDOW - r3.CONTEXT + 1))
            out.append((ident, a, a + WINDOW))
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"windows": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    return [tuple(w) for w in json.loads(path.read_text(encoding="utf-8"))["windows"]]


r3.load = load
r3.windows = windows


def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    count = counter.count_text
    total = 0
    for ident, a, b in windows():
        m = load(ident)
        owned = set(m.owner.values())
        wrong = [i for i in range(a, b) if i in m.truth and m.leader(i) is not None and m.owner.get(m.leader(i)) not in m.truth[i]]
        print(f"  {ident} rows {a}-{b}: wrong {len(wrong)}, of them the person has no track {sum(not (m.truth[i] & owned) for i in wrong)}; "
              f"cast list {len(m.people)} names, true people of the file {len({p for s in m.truth.values() for p in s})}")
    for ident, a, b in windows():
        total += sum(count(x) for x in r3.cast_prompt(load(ident), a, b))
    for arm in ARMS:
        if arm.endswith("open_gen") and r3.generated_cast(*windows()[0][:2]) is None:
            arm_for_count = arm.replace("open_gen", "open")  # same size; the generated blocks do not exist yet
        else:
            arm_for_count = arm
        cs = r3.calls(arm_for_count)
        text = sum(count(c[1]) + count(c[2]) for c in cs)
        audio = sum(c[5] for c in cs) * 32
        out_tokens = sum(len(c[3]) for c in cs) * 40
        total += text + audio + out_tokens
        print(f"{arm}: {len(cs)} calls, text input {text}, audio {audio:.0f}, output ~{out_tokens}")
        (OUT / f"prompt-{arm}-example.md").write_text(cs[0][1] + "\n\n=====\n\n" + cs[0][2], encoding="utf-8")
    print(f"all ~{total:.0f} tokens (a floor); counted by {counter.source}")


def score():
    wins = windows()
    rng = np.random.default_rng(SEED)
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    before = np.array([base[k]["before"] in base[k]["truth"] for k in keys], float)
    h71 = np.array([k[0] == "hina-71" for k in keys])
    cache = {}

    def interval(x, mask):
        if mask.tobytes() not in cache:
            groups = defaultdict(list)
            for j in np.flatnonzero(mask):
                groups[base[keys[j]]["cluster"]].append(j)
            groups = list(groups.values())
            cache[mask.tobytes()] = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(5000)]
        sims = [x[d].mean() for d in cache[mask.tobytes()]]
        return f"{x[mask].mean():+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]"

    every = np.ones(len(keys), bool)
    per_window = lambda x: ", ".join(f"{i} {x[[k[:2] == (i, a) for k in keys]].mean():+.1f}" for i, a, _ in wins)  # noqa: E731
    print(f"labelled rows with truth {len(keys)} (reference rows inside a window left out); production right {before.mean():.1%}; "
          + ", ".join(f"{i} {before[[k[0] == i for k in keys]].mean():.1%}" for i, _, _ in wins))
    for ident, a, b in wins:
        got = r3.generated_cast(ident, a)
        if got is not None:
            m = load(ident)
            truth = {p for i in r3.shown_rows(m, a, b) if i in m.truth for p in m.truth[i]}
            named = {m.person_of(n) for n in got} - {None}
            print(f"  cast block {ident}: {len(named & truth)}/{len(named)} names right, {len(named & truth)}/{len(truth)} speakers found "
                  f"(out of a list of {len(m.people)})")
    right = {}
    for arm in ARMS:
        got = r3.answers(arm)
        if not got:
            continue
        missing = [f"{i}-{a}" for i, a, _ in wins if not any(k[:2] == (i, a) for k in got)]
        res = r3.outcome(got)
        right[arm] = np.array([res[k]["anonymous"] in res[k]["truth"] for k in keys], float)
        named = np.array([res[k]["named"] in res[k]["truth"] for k in keys], float)
        change = 100 * (right[arm] - before)
        new = [k for k, v in res.items() if v["new"]]
        print(f"{arm:14s}: windows without a reply {missing or 'none'}; strict anonymous all {interval(change, every)}, hina-71 "
              f"{interval(change, h71)}, the other three {interval(change, ~h71)}; per window {per_window(change)}; strict named all "
              f"{interval(100 * (named - before), every)}; fixed {int(((right[arm] == 1) & (before == 0)).sum())}, broken "
              f"{int(((right[arm] == 0) & (before == 1)).sum())}; new-track rows {len(new)}")
        print(r3.new_track_report(res) + "\n" + r3.unlabelled_report(res))
    for arm in ("text-open_gen", "audio-open_gen"):
        if arm in right and "text-closed" in right:
            d = 100 * (right[arm] - right["text-closed"])
            print(f"MAIN {arm} - text-closed: hina-71 (merged speakers) {interval(d, h71)}; the other three (rows on the wrong "
                  f"existing track) {interval(d, ~h71)}; all four {interval(d, every)}; per window {per_window(d)}")
    if "audio-open_gen" in right and "text-open_gen" in right:
        d = 100 * (right["audio-open_gen"] - right["text-open_gen"])
        print(f"paired audio-open_gen - text-open_gen: all {interval(d, every)}; per window {per_window(d)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "cast", "run", "score"))
    parser.add_argument("--arm", choices=ARMS, default="text-closed")
    args = parser.parse_args()
    {"build": build, "cast": r3.cast, "score": score}.get(args.command, lambda: r3.run(args.arm, 0, 1))()


if __name__ == "__main__":
    main()
