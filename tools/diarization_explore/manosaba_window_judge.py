"""魔裁 judge round 2 (owner 2026-10-06): contiguous windows, every labelled row judged, and a new-track option.

Round 1 (`manosaba_judge.py`, report 5.11) asked about scattered rows with the hina options and could not measure the
judge. This round follows the owner's direction: the judge reads a contiguous window (as a production correction window
would) and answers for every labelled row in it; the question is how much opening one more operation -- "this row is a
person of the cast who has no track: open a new track for them" -- adds. Criteria: report section 5.13, written first.

  closed     keep / any track of the clip / abstain; cast block = the window's true speakers (names only)
  open       + `@name` (a new track for that person; one name = one track within the window); the same true cast block
  open_gen   `open` with the cast block an agy call wrote from the window's text and the game's main cast (the block a
             correction window could produce), instead of the true one

Windows: WINDOWS per clip, each WINDOW_ROWS consecutive ASR rows plus CONTEXT rows either side as context, placed at
random (fixed seed), not overlapping. References (example lines per track) are the rows production is most sure of
(VBx soft margin), outside every window. agy (Gemini 3.7 Flash), one call per window and arm, text only.

Scoring (truth used only here): a row's person before is its track's majority person over the clip (as in 5.11);
after, `sK` -> that track's majority person, `@name` -> that name, keep / abstain -> unchanged. The reading is the
change in the share of labelled rows whose person is right, per 100 labelled rows, bootstrapped over (window, track).
Strict (the main reading): `@name` for a person who already owns a track of the clip counts wrong -- in production it
splits that person over two tracks. Lenient: it counts right. Both are also read by stratum: rows whose true person
owns no track of the clip (only `@name` can fix them) and the rest.

    python tools/diarization_explore/manosaba_window_judge.py build      # windows, cast blocks' prompts, token counts
    python tools/diarization_explore/manosaba_window_judge.py cast       # agy writes the cast blocks (for open_gen)
    python tools/diarization_explore/manosaba_window_judge.py run --arm closed|open|open_gen [--missed]
    python tools/diarization_explore/manosaba_window_judge.py score

`--missed` (owner 2026-10-06, after the main run had started): the same windows, prompts and arms, but the rows to judge
are the ones production left unlabelled (`??`); a track or `@name` labels them, keep / abstain leaves them empty. Read
against the zero-cost baseline "the track of the nearest labelled row" on the same rows.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import llm_retrack_judge as lj  # noqa: E402
import manosaba_judge as mj  # noqa: E402
import manosaba_truth as mt  # noqa: E402

OUT = mj.OUT / "r2"
SEED = 20261006
WINDOWS = 3          # per clip
WINDOW_ROWS = 30     # consecutive ASR rows judged per window
CONTEXT = 5          # rows either side shown as context
REF_LINES = 3        # example lines per track
ARMS = ("closed", "open", "open_gen")
TITLE = "一部日本视觉小说游戏的剧情录像（无解说，标题略）"
#: the game's main cast, as a knowledge base would list it (display names, `CharacterNames.bytes`); the cast-block
#: writer gets it, the judge gets only the window's cast block
ROSTER = ("桜羽エマ", "二階堂ヒロ", "夏目アンアン", "城ケ崎ノア", "蓮見レイア", "佐伯ミリア", "宝生マーゴ", "黒部ナノカ",
          "紫藤アリサ", "橘シェリー", "遠野ハンナ", "沢渡ココ", "氷上メルル", "月代ユキ", "ゴクチョー")
EXAMPLE = {
    "closed": ['{"id":"q3","track":"keep","conf":"high","why":"延续了自己上一句的话题"}',
               '{"id":"q4","track":"s2","conf":"median","why":"在回答上一句对她的称呼，自称与 s2 的例句一致"}',
               '{"id":"q5","track":"abstain","conf":"low","why":"自称和现在这条轨的例句不像，但说不出是谁"}'],
}
# a name from outside the game, so the example cannot point at anyone in a cast block
EXAMPLE["open"] = EXAMPLE["closed"][:2] + ['{"id":"q5","track":"@山田花子","conf":"median","why":"被叫作花子的人在回答，画像里没有一条轨是她"}',
                                           '{"id":"q6","track":"abstain","conf":"low","why":"只有一声惊叫，说不出是谁"}']
_CONF = {}


def display_names():
    """{character id: display name}, the first display name of each id."""
    out = {}
    for jp, ident in mt.names().items():
        out.setdefault(ident, jp)
    return out


def person_of_name(name):
    """A display name (as the model wrote it) -> character id, or None."""
    name = name.strip().lstrip("@").strip()
    return mt.speaker_of(name, mt.names()) if name else None


def windows():
    """{clip: [(first row, last row + 1)]}, frozen on disk."""
    path = OUT / "windows.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    out = {}
    for clip in mt.CLIPS:
        n = len(mj.material(clip)[0])
        spans = []
        while len(spans) < WINDOWS:
            a = int(rng.integers(CONTEXT, n - WINDOW_ROWS - CONTEXT))
            if all(a + WINDOW_ROWS + CONTEXT <= s or a >= e + CONTEXT for s, e in spans):
                spans.append((a, a + WINDOW_ROWS))
        out[clip] = sorted(spans)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def confidence(clip):
    """{row index: VBx soft margin of its leading track} over every ASR row (production-side, no truth)."""
    if clip not in _CONF:
        import p2_features as pf
        cache = pf.CACHE / f"{clip}-allrows.pkl"
        if not cache.exists():
            stable, _ = mt.production(clip)
            records, _ = pf.row_records(clip, stable, lambda row: (("-",), ("-",)))
            cache.write_bytes(pickle.dumps(records))
        _CONF[clip] = {r["index"]: r["soft_margin"] for r in pickle.loads(cache.read_bytes())}
    return _CONF[clip]


def profiles(clip):
    """Per track: share of speech and REF_LINES example lines production is most sure of, none inside a window."""
    stable, speakers, _, _ = mj.material(clip)
    inside = {i for a, b in windows()[clip] for i in range(a, b)}
    conf = confidence(clip)
    single = {i: (s.to_dict().get("tracks") or []) for i, s in enumerate(speakers)}
    share = Counter()
    for i, tracks in single.items():
        if tracks:
            share[tracks[0]] += float(stable[i]["end"]) - float(stable[i]["start"])
    total = sum(share.values()) or 1.0
    lines = []
    for track in sorted(share, key=lambda k: -share[k]):
        picks = sorted((i for i, tr in single.items() if tr == [track] and i not in inside
                        and len(re.sub(r"[\W_]", "", str(stable[i].get("text", "")))) >= 8),
                       key=lambda i: -conf.get(i, -np.inf))[:REF_LINES]
        if not picks:
            print(f"warning: {clip}: s{track} has no example line", file=sys.stderr)
        lines.append(f"- s{track}：占语音 {share[track] / total * 100:.0f}%。例句：" +
                     " / ".join(f"「{str(stable[i].get('text', '')).strip()}」" for i in sorted(picks)))
    return "\n".join(lines)


def window_rows(clip, span, missed=False):
    """(judged row indices, the rows shown): the labelled rows of the window, or with `missed` its unlabelled ones."""
    stable, speakers, _, _ = mj.material(clip)
    a, b = span
    shown = range(max(0, a - CONTEXT), min(len(stable), b + CONTEXT))
    judged = [i for i in range(a, b) if bool(speakers[i].to_dict().get("tracks")) != missed]
    return judged, shown


def true_cast(clip, span):
    """The true speakers of every row the judge sees (window and context), in roster order: the order they speak in
    would tell who opens the window."""
    _, _, rows, _ = mj.material(clip)
    names = display_names()
    _, shown = window_rows(clip, span)
    people = {names[r["truth"]] for r in rows if r["index"] in shown}
    return sorted(people, key=lambda n: (ROSTER.index(n) if n in ROSTER else len(ROSTER), n))


def cast_prompt(clip, span):
    stable = mj.material(clip)[0]
    _, shown = window_rows(clip, span)
    rows = "\n".join(f"{float(stable[i]['start']):.1f}|{str(stable[i].get('text', '')).strip()}" for i in shown)
    system = (HERE / "cast_block_system.md").read_text(encoding="utf-8")
    user = lr.template("cast_block_user.md").substitute(title=TITLE, roster="、".join(ROSTER), rows=rows)
    return system, user


def generated_cast(clip, span):
    path = OUT / "cast" / f"{clip}-{span[0]}.json"
    return json.loads(path.read_text(encoding="utf-8"))["cast"] if path.exists() else None


def calls(arm, missed=False):
    """[(key, system, user, ids)]."""
    out = []
    options = "open" if arm.startswith("open") else "closed"
    system = string.Template((HERE / "window_judge_system.md").read_text(encoding="utf-8")).substitute(
        options=(HERE / f"window_judge_options_{options}.md").read_text(encoding="utf-8"), example="\n".join(EXAMPLE[options]))
    for clip, spans in windows().items():
        stable, speakers, _, _ = mj.material(clip)
        cells = [lr.cell(s) for s in speakers]
        profile = profiles(clip)
        for span in spans:
            judged, shown = window_rows(clip, span, missed)
            if not judged:
                continue
            cast = generated_cast(clip, span) if arm == "open_gen" else true_cast(clip, span)
            if cast is None:
                raise SystemExit(f"{clip} {span}: no generated cast block yet -- run `cast` first")
            rows = "\n".join((f"▶ q{i} " if i in judged else "") + lr.line(stable[i], cells[i]) for i in shown)
            user = lr.template("window_judge_user_missed.md" if missed else "window_judge_user.md").substitute(title=TITLE, cast="、".join(cast) or "（无）", profiles=profile,
                                                                  rows=rows, count=len(judged))
            out.append((f"{clip}-{span[0]}", system, user, [f"q{i}" for i in judged]))
    return out


def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    count = counter.count_text
    spans = windows()
    print("windows:", {c: s for c, s in spans.items()})
    cast_tokens = sum(count(s) + count(u) for c, ss in spans.items() for s, u in (cast_prompt(c, span) for span in ss))
    print(f"cast blocks: {sum(map(len, spans.values()))} calls, {cast_tokens} input tokens")
    total = cast_tokens
    for arm in ("closed", "open"):
        cs = calls(arm)
        tokens = sum(count(c[1]) + count(c[2]) for c in cs)
        rows = sum(len(c[3]) for c in cs)
        # output: about 40 tokens per JSON line
        print(f"{arm}: {len(cs)} calls, {rows} judged rows, input {tokens} tokens, output ~{rows * 40}")
        total += tokens + rows * 40
        (OUT / f"prompt-{arm}-example.md").write_text(cs[0][1] + "\n\n=====\n\n" + cs[0][2], encoding="utf-8")
    print(f"open_gen ~ open; all arms ~{total + tokens + rows * 40:.0f} tokens (input + output; a floor: thinking, the agent's "
          f"own wrapper, cast blocks' output and retries not counted); counted by {counter.source}")


def _complete(complete, system, user, ids, block):
    for attempt in range(4):
        try:
            text = complete(system, user)
        except SystemExit:
            raise
        except Exception as error:  # noqa: BLE001 - a transient provider error
            print(f"{type(error).__name__} {str(error)[:80]}; retry", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        answer = lr.parse(text, "judge", "id", ids) if block == "judge" else _cast(text)
        if answer is not None:
            return answer, text
        print("malformed reply; retry", flush=True)
    return None, text


def _cast(text):
    match = re.search(r"<cast>(.*?)</cast>", text or "", re.S)
    return [line.strip() for line in match.group(1).splitlines() if line.strip()] if match else None


def cast():
    complete = lr.client_for(lj.TARGET)
    (OUT / "cast").mkdir(parents=True, exist_ok=True)
    for clip, spans in windows().items():
        for span in spans:
            path = OUT / "cast" / f"{clip}-{span[0]}.json"
            if path.exists():
                continue
            answer, raw = _complete(complete, *cast_prompt(clip, span), None, "cast")
            path.write_text(json.dumps({"cast": answer, "raw": raw}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{clip}-{span[0]}: {answer}", flush=True)


def run(arm, missed=False):
    complete = lr.client_for(lj.TARGET)
    folder = OUT / (f"{arm}-missed" if missed else arm)
    folder.mkdir(parents=True, exist_ok=True)
    for key, system, user, ids in calls(arm, missed):
        path = folder / f"{key}.json"
        if path.exists():
            continue
        answer, raw = _complete(complete, system, user, ids, "judge")
        path.write_text(json.dumps({"answer": answer, "raw": raw}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{arm} {key}: {'ok' if answer else 'GAVE UP'}", flush=True)


def score():
    names = display_names()
    rng = np.random.default_rng(SEED)
    rows, cluster = [], []  # (key, truth, person before, own track exists)
    for clip, spans in windows().items():
        _, speakers, truth_rows, owner = mj.material(clip)
        truth = {r["index"]: r["truth"] for r in truth_rows}
        owned = set(owner.values())
        for span in spans:
            judged, _ = window_rows(clip, span)
            for i in judged:
                if i not in truth:
                    continue
                leader = speakers[i].to_dict()["tracks"][0]
                rows.append(((clip, span[0], i), truth[i], owner.get(leader), truth[i] in owned))
                cluster.append((clip, span[0], leader))
    groups = defaultdict(list)
    for j, c in enumerate(cluster):
        groups[c].append(j)
    groups = list(groups.values())
    draws = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(5000)]
    before = np.array([p == t for _, t, p, _ in rows], float)
    print(f"{len(rows)} judged rows with truth in {sum(map(len, windows().values()))} windows ({len(groups)} window x track clusters); "
          f"production right {before.mean():.1%}; wrong rows whose person has no track of the clip {sum((p != t) and not o for _, t, p, o in rows)}")

    gen = {f"{c}-{s[0]}": generated_cast(c, s) for c, ss in windows().items() for s in ss}
    if all(v is not None for v in gen.values()):
        hit = sum(len({person_of_name(n) for n in gen[f"{c}-{s[0]}"]} & {person_of_name(n) for n in true_cast(c, s)})
                  for c, ss in windows().items() for s in ss)
        said = sum(len({person_of_name(n) for n in v} - {None}) for v in gen.values())
        real = sum(len(true_cast(c, s)) for c, ss in windows().items() for s in ss)
        print(f"generated cast blocks: {hit}/{said} names right (precision), {hit}/{real} true speakers found (recall)")

    after = {}
    unfixable = np.array([not o for *_, o in rows])
    for arm in ARMS:
        answers = {}
        for path in (OUT / arm).glob("*.json"):
            clip, start = path.stem.rsplit("-", 1)
            for k, v in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items():
                answers[(clip, int(start), int(k[1:]))] = str((v or {}).get("track", "")).strip()
        if not answers:
            continue
        right, strict, tally, new_rows = [], [], Counter(), []
        for (key, t, p, o) in rows:
            clip = key[0]
            owner = mj.material(clip)[3]
            pick = answers.get(key)
            if pick is None:
                person, kind = p, "missing"
            elif re.fullmatch(r"s\d+", pick) and int(pick[1:]) != mj.material(clip)[1][key[2]].to_dict()["tracks"][0]:
                person, kind = owner.get(int(pick[1:])), "track"
            elif pick.startswith("@"):
                person, kind = person_of_name(pick), "new"
                new_rows.append((key, pick, t, person in set(owner.values())))
            else:
                person, kind = p, "keep" if pick in ("keep",) or re.fullmatch(r"s\d+", pick) else "abstain"
            right.append(person == t)
            strict.append(person == t and not (kind == "new" and person in set(owner.values())))
            was = p == t
            tally[(kind, "fixed" if (not was and person == t) else "broken" if (was and person != t) else
                   "still wrong" if not was else "still right")] += 1
        after[arm] = {"strict": np.array(strict, float), "lenient": np.array(right, float)}
        print(f"{arm:9s}: {dict(sorted(tally.items()))}")
        for rule in ("strict", "lenient"):
            change = 100 * (after[arm][rule] - before)
            sims = [change[d].mean() for d in draws]
            print(f"           {rule:7s}: right {after[arm][rule].mean():.1%}; change {change.mean():+.1f} "
                  f"[{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}] per 100 labelled rows; "
                  f"on rows whose person has no track {change[unfixable].sum() / 100:+.0f} rows, on the rest {change[~unfixable].sum() / 100:+.0f}")
        if new_rows:
            split = sum(1 for _, _, _, has in new_rows if has)
            ok = sum(1 for _, pick, t, _ in new_rows if person_of_name(pick) == t)
            by_track = defaultdict(Counter)
            for (clip, start, _), pick, t, _ in new_rows:
                by_track[(clip, start, pick)][t] += 1
            purity = sum(c.most_common(1)[0][1] for c in by_track.values()) / len(new_rows)
            print(f"           new tracks: {len(by_track)} opened over {len(new_rows)} rows; name right on {ok}; "
                  f"{split} rows named a person who has a track of the clip (a split); row purity within each new track {purity:.0%}")
    for a, b in (("open", "closed"), ("open_gen", "open"), ("open_gen", "closed")):
        if a in after and b in after:
            for rule in ("strict", "lenient"):
                d = 100 * (after[a][rule] - after[b][rule])
                sims = [d[x].mean() for x in draws]
                print(f"paired {a} - {b}, {rule}: {d.mean():+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}] "
                      f"per 100 labelled rows (rows whose person has no track {d[unfixable].sum() / 100:+.0f}, the rest {d[~unfixable].sum() / 100:+.0f})")


def score_missed():
    """The unlabelled rows: how many each arm labels, how many of those right, vs the nearest labelled row's track."""
    rows = []  # (key, truth, baseline person, person owns a track)
    for clip, spans in windows().items():
        stable, speakers, truth_rows, owner = mj.material(clip)
        truth = {r["index"]: r["truth"] for r in truth_rows}
        labelled = [i for i, s in enumerate(speakers) if s.to_dict().get("tracks")]
        centre = lambda i: (float(stable[i]["start"]) + float(stable[i]["end"])) / 2  # noqa: E731
        for span in spans:
            for i in window_rows(clip, span, missed=True)[0]:
                if i in truth:
                    near = min(labelled, key=lambda j: abs(centre(j) - centre(i)))
                    rows.append(((clip, span[0], i), truth[i], owner.get(speakers[near].to_dict()["tracks"][0]), truth[i] in set(owner.values())))
    if not rows:
        return
    print(f"unlabelled rows with truth in the windows: {len(rows)}; nearest labelled row's track right on {sum(b == t for _, t, b, _ in rows)}")
    for arm in ARMS:
        folder = OUT / f"{arm}-missed"
        answers = {}
        for path in folder.glob("*.json"):
            clip, start = path.stem.rsplit("-", 1)
            for k, v in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items():
                answers[(clip, int(start), int(k[1:]))] = str((v or {}).get("track", "")).strip()
        if not answers:
            continue
        tally = Counter()
        for key, t, _, _ in rows:
            owner = mj.material(key[0])[3]
            pick = answers.get(key, "")
            if re.fullmatch(r"s\d+", pick):
                person, kind = owner.get(int(pick[1:])), "track"
            elif pick.startswith("@"):
                person, kind = person_of_name(pick), "new"
            else:
                tally["left empty"] += 1
                continue
            split = kind == "new" and person in set(owner.values())
            tally[f"{kind} {'right' if person == t and not split else 'right only if splits count' if person == t else 'wrong'}"] += 1
        print(f"  {arm:9s} on unlabelled rows: {dict(sorted(tally.items()))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "cast", "run", "score"))
    parser.add_argument("--arm", choices=ARMS, default="closed")
    parser.add_argument("--missed", action="store_true", help="judge the windows' unlabelled rows instead")
    args = parser.parse_args()
    {"build": build, "cast": cast, "score": lambda: (score(), score_missed())}.get(args.command, lambda: run(args.arm, args.missed))()


if __name__ == "__main__":
    main()
