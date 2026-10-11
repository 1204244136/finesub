"""Speaker judge round 8 (owner 2026-10-07, marks plan step 2): do step 1's marks help the window judge?

Step 1 (`marks_step1.py`, report `docs/reports/2026-10-07-diar-marks-step1.md`) settled the marks production can give
without a judge: the combined score flags the least sure quarter of the labelled rows, in two tiers (tier B carries a
spare track), and missed rows are filled by the mixed slot rule. Here the round-7 judge (`text-open_gen`,
`audio-open_gen`, prompts unchanged) runs on fresh windows twice without the marks (`plain-a`, `plain-b`) and twice with
them (`marks-a`, `marks-b`): the cell `s3` becomes `s3（低）` / `s3（低，备选 s5）`, a missed row shows its filled track
`s4（补）`, and the system prompt gains `window_judge_r8_marks_note.md`. Criteria: report
`docs/reports/2026-10-07-diar-marks-step2.md`, written first.

Windows (`plan()`): the two merged windows earlier rounds used (魔裁 b, hina-71; no fresh file has merged tracks), four
fresh ordinary hina windows of 150 rows, 魔裁 c (fresh, mostly pure); text arms judge all seven. Audio arms judged the
first three, then (owner 2026-10-07, report section 5) the other three ordinary windows too (`fill`), and four more
ordinary windows drawn by `extend` for the audio arm only.

    python tools/diarization_explore/window_judge_r8.py build      # windows, marks, cast lists; token count
    python tools/diarization_explore/window_judge_r8.py cast       # the generated cast blocks, shared by every run
    python tools/diarization_explore/window_judge_r8.py run --arm text-open_gen --tag plain-a [--shard 0 --shards 2]
    python tools/diarization_explore/window_judge_r8.py score
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marks_step1 as ms1  # noqa: E402
import window_judge_r4 as r4  # noqa: E402  (patches round 3's load / windows / OUT; re-patched below)
import window_partition_score as ps  # noqa: E402

r3 = r4.r3
pf = ms1.pf
ROOT = r3.lr.OUT / "r8"
SEED = 20261009
WINDOW = 200
ORDINARY = 150  # rows per ordinary hina window (200 put the four runs over the ~0.7M input the owner allowed)
ARMS = ("text-open_gen", "audio-open_gen")
TAGS = ("plain-a", "plain-b", "marks-a", "marks-b")
AUDIO_WINDOWS = 3  # the two merged windows and the first ordinary one (four put the floor over the allowed input)
PURITY = 0.7  # step 1's slot-purity threshold (G3, chosen on dev)
NOTE = (HERE / "window_judge_r8_marks_note.md").read_text(encoding="utf-8")
ANCHOR = "一行里有时混着两个人的话"
_load = r4._load  # round 3's own loader
_ORIG_CALLS = r3.calls
MARKS = False


def _fresh_rows():
    rows = r3._rows()
    if not set(ms1.FRESH) <= set(rows):
        rows.update({f: pf.derive(rs, pf.MAIN_TRUTH) for f, rs in pickle.loads(ms1.FRESH_ROWS.read_bytes()).items()})
    return rows


def cast_list(ident, files):
    """Round 4's rule: the file's people plus DISTRACTORS other hina people (no role labels, no alias of a real one)."""
    path = ROOT / "cast-list.json"
    if not path.exists():
        rng = np.random.default_rng(SEED + 1)
        rows = _fresh_rows()
        others = sorted({p for f, rs in rows.items() if f.startswith("hina-") and f not in files for r in rs for p in r["truth"]} - r4.ROLE_LABELS)
        cast = {}
        for name in files:
            people = _load(name).people
            pool = [p for p in others if not any(r4.same_person(p, q) for q in people)]
            cast[name] = sorted(set(people) | {pool[i] for i in rng.choice(len(pool), r4.DISTRACTORS, replace=False)})
        ROOT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cast, ensure_ascii=False, indent=1), encoding="utf-8")
    cast = json.loads(path.read_text(encoding="utf-8"))
    if ident not in cast:  # a window added later (`extend`): same rule, its own seed, the frozen entries untouched
        rng = np.random.default_rng([SEED + 1, int(ident.rsplit("-", 1)[1])])
        rows = _fresh_rows()
        others = sorted({p for f, rs in rows.items() if f.startswith("hina-") and f not in files for r in rs for p in r["truth"]} - r4.ROLE_LABELS)
        people = _load(ident).people
        pool = [p for p in others if not any(r4.same_person(p, q) for q in people)]
        cast[ident] = sorted(set(people) | {pool[i] for i in rng.choice(len(pool), r4.DISTRACTORS, replace=False)})
        path.write_text(json.dumps(cast, ensure_ascii=False, indent=1), encoding="utf-8")
    return cast[ident]


def load(ident):
    _fresh_rows()
    m = _load(ident)
    if ident.startswith("hina-") and not getattr(m, "_r8", False):
        m.people = cast_list(ident, [w["ident"] for w in plan() if w["ident"].startswith("hina-")])
        m.roster = tuple(m.people)
        m._r8 = True
    return m


def plan():
    """[{ident, a, b, group, audio}], frozen on disk.

    No fresh file has merged tracks (step 1: none of the 26 has a wrong row whose person lacks a track; 魔裁 c has 4 such
    rows in 212), so the merged group reuses the two merged windows earlier rounds tuned the prompt on -- 魔裁 b (round 3)
    and hina-71 (round 4). In-sample for the prompt, but both inputs run the same prompt, so the marks - plain difference
    is what is read there. Fresh: 魔裁 c (mostly pure) and four ordinary hina windows."""
    path = ROOT / "windows.json"
    if not path.exists():
        rng = np.random.default_rng(SEED)
        rows = _fresh_rows()
        out = [dict(ident="manosaba-b", a=18, b=218, group="并轨（样本内）"), dict(ident="hina-71", a=27, b=227, group="并轨（样本内）")]
        rates = {f: float(np.mean([r["wrong"] for r in rows[f] if "wrong" in r])) for f in ms1.FRESH
                 if len(_load(f).stable) >= ORDINARY + 2 * r3.CONTEXT}
        median = float(np.median(list(rates.values())))
        eligible = sorted(f for f, v in rates.items() if v >= median)
        for f in [eligible[i] for i in rng.choice(len(eligible), 4, replace=False)]:
            n = len(_load(f).stable)
            a = int(rng.integers(r3.CONTEXT, n - ORDINARY - r3.CONTEXT + 1))
            out.append(dict(ident=f, a=a, b=a + ORDINARY, group="hina 普通", error=round(rates[f], 3)))
        m = _load("manosaba-c")
        a = int(rng.integers(r3.CONTEXT, len(m.stable) - WINDOW - r3.CONTEXT + 1))
        out.append(dict(ident="manosaba-c", a=a, b=a + WINDOW, group="魔裁 c"))
        for j, w in enumerate(out):
            w["audio"] = j < AUDIO_WINDOWS or w["group"] == "hina 普通"  # the fill-in made every ordinary window an audio one
            w["fill"] = j >= AUDIO_WINDOWS and w["group"] == "hina 普通"
        ROOT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def extend():
    """Owner 2026-10-07: if the audio arm's fresh reading is still not enough after the fill-in (report section 5), four more
    fresh ordinary hina windows for the audio arm only, drawn like the first four from the files not yet used."""
    wins = plan()
    if any(w.get("extra") for w in wins):
        return wins
    rng = np.random.default_rng(SEED + 2)
    rows = _fresh_rows()
    used = {w["ident"] for w in wins}
    rates = {f: float(np.mean([r["wrong"] for r in rows[f] if "wrong" in r])) for f in ms1.FRESH
             if len(_load(f).stable) >= ORDINARY + 2 * r3.CONTEXT}
    median = float(np.median(list(rates.values())))
    eligible = sorted(f for f, v in rates.items() if v >= median and f not in used)
    for f in [eligible[i] for i in rng.choice(len(eligible), 4, replace=False)]:
        n = len(_load(f).stable)
        a = int(rng.integers(r3.CONTEXT, n - ORDINARY - r3.CONTEXT + 1))
        wins.append(dict(ident=f, a=a, b=a + ORDINARY, group="hina 普通", error=round(rates[f], 3), audio=True, text=False, extra=True))
    (ROOT / "windows.json").write_text(json.dumps(wins, ensure_ascii=False, indent=1), encoding="utf-8")
    return wins


def windows():
    return [(w["ident"], w["a"], w["b"]) for w in plan()]


r3.load = load
r3.windows = windows


# --- marks ------------------------------------------------------------------------------------------

def _models():
    """(combined-score model, candidate model, p, q) as step 1 fitted them on dev hina, cached."""
    path = ROOT / "marks-model.pkl"
    if not path.exists():
        groups, _ = ms1.groups()
        dev_all = [r for rs in ms1.ordinary(groups["dev"], True).values() for r in rs]
        model, cmodel = pf.fit_error(dev_all), pf.fit_candidates(dev_all)
        dev = [r for rs in ms1.ordinary(groups["dev"]).values() for r in rs]
        scores = model.predict_proba(pf.matrix(dev))[:, 1]
        p = float(np.quantile(scores, 1 - ms1.BUDGET))
        q = ms1.choose_q([r for r, s in zip(dev, scores) if s >= p], cmodel)
        ROOT.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps((model, cmodel, p, q)))
    return pickle.loads(path.read_bytes())


_MARKS = {}


def marks(ident):
    """({row: marked cell}, {missed row: filled track}) for one file, from step 1's rules on every row."""
    if ident not in _MARKS:
        model, cmodel, p, q = _models()
        m = load(ident)
        r3._confidence(ident, m.stable)
        records = pickle.loads((pf.CACHE / f"{ident}-allrows.pkl").read_bytes())
        slots = ms1.slot_identity(ident, records, m.stable)
        cells, fills = {}, {}
        for r in records:
            i = r["index"]
            base = r3.lr.cell(m.speakers[i])
            if r["leader"] is None:
                s = slots.get((ident, i))
                fill = s["slot_track"] if s and s["slot_track"] >= 0 and s["slot_purity"] >= PURITY else r["nearest"]
                if fill is not None:
                    fills[i] = int(fill)
                    cells[i] = f"s{int(fill)}（补）"
            elif model.predict_proba(pf.matrix([r]))[0, 1] >= p:
                k, s = ms1.alternative(cmodel, r)
                cells[i] = base + (f"（低，备选 s{k}）" if q is not None and k is not None and s >= q else "（低）")
        _MARKS[ident] = (cells, fills)
    return _MARKS[ident]


def calls(arm):
    """Round 3's calls on this round's windows; with MARKS, the marked cells and the note. Audio arms: the audio windows."""
    keep = {f"{w['ident']}-{w['a']}" for w in plan() if (w["audio"] if arm.startswith("audio") else w.get("text", True))}
    if MARKS:
        by_speaker = {}
        for ident, _, _ in windows():
            m = load(ident)
            by_speaker.update({id(m.speakers[i]): cell for i, cell in marks(ident)[0].items()})
        plain_cell = r3.lr.cell
        r3.lr.cell = lambda s: by_speaker.get(id(s)) or plain_cell(s)
        try:
            out = _ORIG_CALLS(arm)
        finally:
            r3.lr.cell = plain_cell
        assert all(c[1].count(ANCHOR) == 1 for c in out)
        out = [(k, s.replace(ANCHOR, NOTE + ANCHOR), u, ids, wav, sec) for k, s, u, ids, wav, sec in out]
    else:
        out = _ORIG_CALLS(arm)
    return [c for c in out if c[0] in keep]


r3.calls = calls


def use(tag):
    """Point round 3's OUT at run `tag` (its replies and audio) and copy the shared cast blocks in."""
    global MARKS
    MARKS = tag.startswith("marks")
    r3.OUT = ROOT / tag
    (r3.OUT / "cast").mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "shared" / "cast").glob("*.json"):
        shutil.copyfile(src, r3.OUT / "cast" / src.name)


# --- commands ---------------------------------------------------------------------------------------

def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    count = counter.count_text
    total = 0
    r3.OUT = ROOT / "build"  # the audio files the counts need go here, not into an earlier round's folder
    for w in plan():
        m = load(w["ident"])
        a, b = w["a"], w["b"]
        owned = set(m.owner.values())
        wrong = [i for i in range(a, b) if i in m.truth and m.leader(i) is not None and m.owner.get(m.leader(i)) not in m.truth[i]]
        cells, fills = marks(w["ident"])
        print(f"  {w['group']} {w['ident']} rows {a}-{b} (audio {w['audio']}): wrong {len(wrong)}, person without a track "
              f"{sum(not (m.truth[i] & owned) for i in wrong)}; marked {sum(a <= i < b and i not in fills for i in cells)}, "
              f"filled {sum(a <= i < b for i in fills)}; cast list {len(m.people)}")
    for ident, a, b in windows():
        total += sum(count(x) for x in r3.cast_prompt(load(ident), a, b))
    print(f"cast blocks: input {total}")
    global MARKS
    for marked in (False, True):
        MARKS = marked
        for arm in ARMS:
            cs = calls(arm.replace("open_gen", "open"))  # same size as open_gen; the generated blocks may not exist yet
            text = sum(count(c[1]) + count(c[2]) for c in cs)
            audio = sum(c[5] for c in cs) * 32
            out_tokens = sum(len(c[3]) for c in cs) * 40
            total += 2 * (text + audio)  # two runs; input only, as the budget counts
            print(f"{'marks' if marked else 'plain'} {arm}: {len(cs)} calls, text input {text}, audio {audio:.0f}, output ~{out_tokens} (per run)")
            ROOT.mkdir(parents=True, exist_ok=True)
            (ROOT / f"prompt-{'marks' if marked else 'plain'}-{arm}-example.md").write_text(cs[-1][1] + "\n\n=====\n\n" + cs[-1][2], encoding="utf-8")
    MARKS = False
    print(f"input over every run ~{total:.0f} tokens (a floor: thinking, retries not counted); counted by {counter.source}")


def cast():
    r3.OUT = ROOT / "shared"
    r3.cast()


def score():
    rng = np.random.default_rng(SEED)
    use("plain-a")
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    clusters = [base[k]["cluster"] for k in keys]
    group_of = {(w["ident"], w["a"]): w["group"] for w in plan()}
    audio_of = {(w["ident"], w["a"]): w["audio"] for w in plan()}
    text_of = {(w["ident"], w["a"]): w.get("text", True) for w in plan()}

    def labels(picks, marked):
        out, missed = {}, {}
        for ident, a, b in windows():
            m = load(ident)
            final = ps.final_labels(r3, m, ident, a, b, picks)
            if marked:
                fills = marks(ident)[1]
                for i in range(a, b):
                    if final[i] is None and i in fills:
                        final[i] = f"s{fills[i]}"
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            found = ps.match(rows, final, m.truth)
            out.update({(ident, a, i): found.get(final[i]) in m.truth[i] for i in rows})
            missed.update({(ident, a, i): final[i] is not None and found.get(final[i]) in m.truth[i]
                           for i in range(a, b) if i in m.truth and m.leader(i) is None})
        return out, missed

    def interval(d, mask):
        groups = defaultdict(list)
        for j in np.flatnonzero(mask):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return d[mask].mean(), np.percentile(sims, 2.5), np.percentile(sims, 97.5)

    prod, _ = labels({}, False)
    prod = np.array([prod[k] for k in keys], float)
    print(f"labelled rows with truth {len(keys)}; production {prod.mean():.1%}")
    for arm in ARMS:
        in_arm = np.array([text_of[k[:2]] if arm.startswith("text") else audio_of[k[:2]] for k in keys])
        cols, missed = {}, {}
        for tag in TAGS:
            got = r3.answers(arm, ROOT / tag / arm)
            if got:
                right, miss = labels(got, tag.startswith("marks"))
                cols[tag] = np.array([right[k] for k in keys], float)
                missed[tag] = miss
        if len(cols) < 4:
            print(f"{arm}: runs so far {sorted(cols)}")
            continue
        d_a, d_b = 100 * (cols["marks-a"] - cols["plain-a"]), 100 * (cols["marks-b"] - cols["plain-b"])
        d = (d_a + d_b) / 2
        print(f"{arm}: production {prod[in_arm].mean():.1%}; " + ", ".join(f"{t} {v[in_arm].mean():.1%}" for t, v in cols.items()))
        run_gap = max(abs(100 * (cols["plain-a"] - cols["plain-b"])[in_arm].mean()), abs(100 * (cols["marks-a"] - cols["marks-b"])[in_arm].mean()))
        fresh_groups = ("hina 普通", "魔裁 c")
        rows_of = lambda gs: in_arm & np.array([group_of[k[:2]] in gs for k in keys])  # noqa: E731
        for name, mask in [("fresh windows (main)", rows_of(fresh_groups))] + [(g, rows_of((g,))) for g in ("并轨（样本内）",) + fresh_groups]:
            if not mask.any():
                continue
            v, lo, hi = interval(d, mask)
            pair_a, pair_b = d_a[mask].mean(), d_b[mask].mean()
            gap = max(abs(100 * (cols["plain-a"] - cols["plain-b"])[mask].mean()), abs(100 * (cols["marks-a"] - cols["marks-b"])[mask].mean()))
            print(f"  {name} ({mask.sum()} rows): marks - plain {v:+.1f} [{lo:+.1f}, {hi:+.1f}] (run a {pair_a:+.1f}, run b {pair_b:+.1f}); "
                  f"plain a - b {100 * (cols['plain-a'] - cols['plain-b'])[mask].mean():+.1f}, marks a - b {100 * (cols['marks-a'] - cols['marks-b'])[mask].mean():+.1f}")
            if name.startswith("fresh"):
                verdict = ("helps" if lo > 0 and pair_a > 0 and pair_b > 0 and v > gap else
                           "hurts" if hi < 0 and pair_a < 0 and pair_b < 0 and -v > gap else "no measurable difference")
                print(f"  VERDICT {arm} (fresh windows): {verdict} (the same input's two runs differ by up to {gap:.1f})")
            if name == "并轨（样本内）":
                harm = pair_a <= -5 and pair_b <= -5
                print(f"  HARM GATE {arm} (merged): {'harmful on merged material' if harm else 'not triggered'}")
        print(f"  (all windows of the arm: same-input run gap up to {run_gap:.1f})")
        for tag in TAGS:
            got = r3.answers(arm, ROOT / tag / arm)
            if got:
                print(f"  {tag}:\n" + r3.new_track_report(r3.outcome(got)))
        ordinary = in_arm & np.array([group_of[k[:2]] == "hina 普通" for k in keys])
        for tag in cols:
            picks = r3.answers(arm, ROOT / tag / arm)
            new = {g: sum(v.startswith("@") for k, v in picks.items() if group_of[k[:2]] == g) for g in ("并轨（样本内）", "hina 普通", "魔裁 c")}
            print(f"  {tag}: rows given a new track {new}")
        marked = np.array([k[2] in marks(k[0])[0] for k in keys])
        for tag, col in cols.items():
            picks = r3.answers(arm, ROOT / tag / arm)
            changed = np.array([(lambda p, lead: p.startswith("@") or (p[:1] == "s" and p[1:].isdigit() and int(p[1:]) != lead))(
                picks.get(k, ""), load(k[0]).leader(k[2])) for k in keys])
            print(f"  {tag}: ordinary windows, rows changed on marked / unmarked rows {int((changed & marked & ordinary).sum())} / "
                  f"{int((changed & ~marked & ordinary).sum())} (of {int((marked & ordinary).sum())} / {int((~marked & ordinary).sum())}); "
                  f"broke on unmarked {int(((prod == 1) & (col == 0) & ~marked & ordinary).sum())}")
        for tag, col in cols.items():
            broke = int(((prod == 1) & (col == 0) & ordinary).sum())
            fixed = int(((prod == 0) & (col == 1) & ordinary).sum())
            miss = [v for k, v in missed[tag].items() if (text_of[k[:2]] if arm.startswith("text") else audio_of[k[:2]])]
            print(f"  {tag}: ordinary windows fixed {fixed}, broke {broke}; missed rows with truth {len(miss)}, right {np.mean(miss) if miss else 0:.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "cast", "run", "score", "extend"))
    parser.add_argument("--arm", choices=ARMS, default="text-open_gen")
    parser.add_argument("--tag", choices=TAGS, default="plain-a")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "run":
        use(args.tag)
        r3.run(args.arm, args.shard, args.shards)
    else:
        {"build": build, "cast": cast, "score": score, "extend": extend}[args.command]()


if __name__ == "__main__":
    main()
