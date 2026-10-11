"""Speaker judge round 5 (owner 2026-10-06): the cold read's fixes, rerun on the windows of rounds 3-4 that test them.

The cold read of the round 3-4 prompts (report 5.17) found defects and four owner decisions, applied in round 3's
templates and code: names shown with how the material writes them (hina: Chinese name, Japanese in brackets); a
game PV's lines belong to the character in it; no `conf`; one person's speech split over two tracks named as an
acoustic error; `keep` defined for `??` and `s5+s4` rows; tracks without example lines or voice samples said so; the
cast block's "unknown" kept out of the judge's cast. These are experiment prompts, not production's. Two more changes
ran first and were withdrawn after their reading (report 5.19: fewer new tracks on 魔裁): "a run of rows on one track
is mostly one person" and a new-track condition "no track's example lines are like the row's person".

Windows: round 4's four hina windows and round 3's two 魔裁 windows. Arms: text-open_gen, audio-open_gen, each
compared row by row with its own earlier answers on the same windows (round 3 / 4 folders). Criteria: report 5.19.

    python tools/diarization_explore/window_judge_r5.py build | cast | run --arm text-open_gen | score
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_judge_r4 as r4  # noqa: E402  (patches round 3's OUT / load / windows; this module re-patches them)

r3 = r4.r3
ROUND3, ROUND4 = r3.lr.OUT / "r3", r3.lr.OUT / "r4"
OUT = r3.lr.OUT / "r5"
SEED = 20261011
ARMS = ("text-open_gen", "audio-open_gen")
#: how the material writes a name, from memory of the games' Japanese releases; a name without a sure Japanese form is
#: left bare (people's nicknames, role labels, characters I could not place). 魔裁's roster is Japanese already.
ALIASES = {
    "多莉": "ドリー", "大黑塔": "大ヘルタ／ヘルタ", "女开拓者": "開拓者／星", "男开拓者": "開拓者／穹", "姬子": "姫子", "昔涟": "キュレネ",
    "星期日": "サンデー", "欧洛伦": "オロルン", "火神": "マーヴィカ", "猫头鹰先生": "ミスターフクロウ", "玛拉妮": "ムアラニ",
    "赛飞儿": "サフェル", "风堇": "ヒアンシー", "伊法": "イファ", "灵砂": "霊砂", "爱可菲": "エスコフィエ", "翡翠": "ジェイド",
    "莱欧斯利": "リオセスリ", "那维莱特": "ヌヴィレット", "银狼": "銀狼", "飞霄": "飛霄", "Saber": "セイバー", "三月七": "三月なのか",
    "丹恒": "丹恒", "松泽": "松澤", "派蒙": "パイモン", "瑶瑶": "ヨォーヨ", "白厄": "ファイノン", "空": "空／旅人", "钟离": "鍾離",
    "乱破": "乱破", "托马": "トーマ", "林尼": "リネ", "温迪": "ウェンティ", "砂金": "アベンチュリン", "达达利亚": "タルタリヤ",
}

r3.OUT = OUT


def windows():
    hina = [tuple(w) for w in json.loads((ROUND4 / "windows.json").read_text(encoding="utf-8"))["windows"]]
    manosaba = [tuple(w) for w in json.loads((ROUND3 / "windows.json").read_text(encoding="utf-8")) if w[0].startswith("manosaba-")]
    return hina + manosaba


def load(ident):
    m = r4._load(ident) if ident.startswith("manosaba-") else r4.load(ident)  # hina: round 4's cast list with distractors
    m.aliases = {n: a for n, a in ALIASES.items() if n in m.roster}
    return m


r3.load = load
r3.windows = windows


def earlier(arm):
    """The same arm's answers from rounds 3 (魔裁) and 4 (hina) on these windows."""
    got = {**r3.answers(arm, ROUND3 / arm), **r3.answers(arm, ROUND4 / arm)}
    wins = {w[:2] for w in windows()}
    return {k: v for k, v in got.items() if k[:2] in wins}


def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    count = counter.count_text
    total = sum(count(x) for i, a, b in windows() for x in r3.cast_prompt(load(i), a, b))
    print(f"cast blocks: {len(windows())} calls, input {total}")
    for arm in ARMS:
        cs = r3.calls(arm.replace("open_gen", "open")) if r3.generated_cast(*windows()[0][:2]) is None else r3.calls(arm)
        text = sum(count(c[1]) + count(c[2]) for c in cs)
        audio = sum(c[5] for c in cs) * 32
        out_tokens = sum(len(c[3]) for c in cs) * 35
        total += text + audio + out_tokens
        print(f"{arm}: {len(cs)} calls, text input {text}, audio {audio:.0f}, output ~{out_tokens}")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"prompt-{arm}-example.md").write_text(cs[0][1] + "\n\n=====\n\n" + cs[0][2], encoding="utf-8")
    print(f"all ~{total:.0f} tokens (a floor: thinking and retries not counted); counted by {counter.source}")


def score():
    rng = np.random.default_rng(SEED)
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    before = np.array([base[k]["before"] in base[k]["truth"] for k in keys], float)
    group = np.array(["魔裁" if k[0].startswith("manosaba-") else "hina-71" if k[0] == "hina-71" else "hina 其余三窗" for k in keys])
    clusters = [base[k]["cluster"] for k in keys]

    def interval(d, mask):
        groups = defaultdict(list)
        for j in np.flatnonzero(mask):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return f"{d[mask].mean():+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]"

    print(f"labelled rows with truth {len(keys)}; production right " + ", ".join(f"{g} {before[group == g].mean():.1%}" for g in dict.fromkeys(group)))
    for ident, a, b in windows():
        got = r3.generated_cast(ident, a)
        if got is not None:
            m = load(ident)
            truth = {p for i in r3.shown_rows(m, a, b) if i in m.truth for p in m.truth[i]}
            named = {m.person_of(n) for n in got} - {None}
            print(f"  cast block {ident}: {len(named & truth)}/{len(named)} names right, {len(named & truth)}/{len(truth)} speakers found")
    for arm in ARMS:
        now, then = r3.answers(arm), earlier(arm)
        if not now:
            continue
        res_now, res_then = r3.outcome(now), r3.outcome(then)
        right_now = np.array([res_now[k]["anonymous"] in res_now[k]["truth"] for k in keys], float)
        right_then = np.array([res_then[k]["anonymous"] in res_then[k]["truth"] for k in keys], float)
        for g in dict.fromkeys(group):
            mask = group == g
            print(f"MAIN {arm} {g}: now {right_now[mask].mean():.1%}, before the fixes {right_then[mask].mean():.1%}; "
                  f"now - before {interval(100 * (right_now - right_then), mask)}; now - production {interval(100 * (right_now - before), mask)}")
        moved = lambda res: {k for k, v in res.items() if v["labelled"] and v["pick"] not in ("", "keep", "abstain")}  # noqa: E731
        print(f"  {arm}: rows changed now {len(moved(res_now))}, before {len(moved(res_then))}; fixed now "
              f"{int(((right_now == 1) & (before == 0)).sum())} / before {int(((right_then == 1) & (before == 0)).sum())}, broken now "
              f"{int(((right_now == 0) & (before == 1)).sum())} / before {int(((right_then == 0) & (before == 1)).sum())}")
        print(r3.new_track_report(res_now) + "\n" + r3.unlabelled_report(res_now))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "cast", "run", "score"))
    parser.add_argument("--arm", choices=ARMS, default="text-open_gen")
    args = parser.parse_args()
    {"build": build, "cast": r3.cast, "score": score}.get(args.command, lambda: r3.run(args.arm, 0, 1))()


if __name__ == "__main__":
    main()
