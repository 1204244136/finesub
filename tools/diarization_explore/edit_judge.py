"""Edit judge (owner 2026-10-07): how well can the audio speaker judge split rows, insert missed lines and fix words,
given every help -- the ceiling of each action, against one arm that uses no truth.

Plan, criteria and budget: <MAIN>/tmp/diarization-retrack/edit/plan.md (written before any run). Truth: Japanese text
(`edit_judge_truth.py`): hina's bilingual fansub twins, 魔裁's OCR text boxes, CALLHOME Japanese transcripts.

Windows (`windows.json`, frozen on first build): WIN consecutive ASR rows to act on (▶) and CONTEXT rows either side,
placed over the truth's events (split rows, lexical missing lines) so few calls see many of them; a window choice that
uses the truth, in every arm (stated in the plan). Control windows (no event) are added for the plain arm only.
One task per prompt (split / insert / edit), two arms each:
  plain    no truth: the KB-level cast (hina: the file's people + DISTRACTORS other hina names; 魔裁: the game's roster;
           CALLHOME: no names), production's tracks and profiles, audio, the whole window to search
  oracle   truth (oracle): the window's true speakers, and the places to look at -- every true case in the window plus as
           many decoys (`hints`), the model told only that some of them are cases
Both arms carry audio (each track's voice samples, then the window's recording), as round 3's audio arms do.

    python tools/diarization_explore/edit_judge.py build                       # windows, hints, prompts, token count
    python tools/diarization_explore/edit_judge.py run --task split --arm oracle [--shard 0 --shards 4] [--repeat 2]
    python tools/diarization_explore/edit_judge.py score
    python tools/diarization_explore/edit_judge.py consistency                  # run 1 vs the rerun windows' run 2
    python tools/diarization_explore/edit_judge.py readings                     # per material, oracle - plain, reruns
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edit_judge_truth as et  # noqa: E402
import llm_retrack as lr  # noqa: E402
import llm_retrack_judge as lj  # noqa: E402
import window_judge_r3 as r3  # noqa: E402

OUT = et.OUT
SEED = 20261013
WIN, CONTEXT = 20, 5
TASKS = ("split", "insert", "edit")
ARMS = ("plain", "oracle")
N_CALLHOME_EXTRA = 12  # CALLHOME windows besides those with a content split: random ones with >= 2 lexical missing lines
N_CONTROL = 4        # hina windows without any event, plain arm only
N_EDIT = {"hina": 6, "manosaba": 2, "callhome": 4}   # edit-task windows, drawn from the event windows
HINT_CAP = 8         # at most this many true cases hinted per window (edit), as many decoys
DISTRACTORS = 10
LATIN = re.compile(r"[A-Za-z]{3,}")
MATERIALS = {**r3.MATERIALS,
             "callhome": dict(title="日语电话录音（家人或朋友之间的闲聊，标题略）",
                              scene="素材是两个人之间的日语电话录音（家人或朋友闲聊）：附和、抢话、两个人同时说话都很多。", short="日语电话闲聊")}
#: round 5's table (`window_judge_r5.ALIASES`, copied: importing that module rebinds round 3's loaders) plus the hina
#: 60-74 people I can place; names without a sure Japanese form stay bare
ALIASES = {
    "多莉": "ドリー", "大黑塔": "大ヘルタ／ヘルタ", "女开拓者": "開拓者／星", "男开拓者": "開拓者／穹", "姬子": "姫子", "昔涟": "キュレネ",
    "星期日": "サンデー", "欧洛伦": "オロルン", "火神": "マーヴィカ", "猫头鹰先生": "ミスターフクロウ", "玛拉妮": "ムアラニ",
    "赛飞儿": "サフェル", "风堇": "ヒアンシー", "伊法": "イファ", "灵砂": "霊砂", "爱可菲": "エスコフィエ", "翡翠": "ジェイド",
    "莱欧斯利": "リオセスリ", "那维莱特": "ヌヴィレット", "银狼": "銀狼", "飞霄": "飛霄", "Saber": "セイバー", "三月七": "三月なのか",
    "丹恒": "丹恒", "松泽": "松澤", "派蒙": "パイモン", "瑶瑶": "ヨォーヨ", "白厄": "ファイノン", "空": "空／旅人", "钟离": "鍾離",
    "乱破": "乱破", "托马": "トーマ", "林尼": "リネ", "温迪": "ウェンティ", "砂金": "アベンチュリン", "达达利亚": "タルタリヤ",
    "流萤": "ホタル", "知更鸟": "ロビン", "黑天鹅": "ブラックスワン", "景元": "景元", "云璃": "雲璃", "驭空": "御空", "波提欧": "ブートヒル",
    "缇宝": "トリビー", "阿格莱雅": "アグライア", "万敌": "モーディス", "遐蝶": "キャストリス", "刻律德菈": "ケリュドラ", "火花": "花火",
    "那刻夏": "アナイクス", "螺丝咕姆": "スクリューラム", "Archer": "アーチャー",
}
#: from round 4: role labels and names of one person, kept out of the distractors
ROLE_LABELS = {"主持人", "嘉宾", "男主", "女主", "教师"}
SAME = ({"女开拓者", "男开拓者", "穹", "星", "开拓者"}, {"空", "荧", "旅行者"}, {"火神", "玛薇卡"})


#: files whose「主持人」is Mr. Owl (his own lines: 「司会者 ミスターフク郎」, 「フクロウ大だ」); there 猫头鹰先生 is no distractor
HOST_IS_OWL = ("hina-61", "hina-63", "hina-64")


def same_person(a, b):
    return a == b or a in b or b in a or any(a in g and b in g for g in SAME)


def group(ident):
    return ident.split("-")[0]


# --- material ------------------------------------------------------------------------------------

_MATERIAL = {}


def material(ident):
    """Round 3's Material (CALLHOME: built here), with the plain arm's cast in `people` / `roster`."""
    if ident in _MATERIAL:
        return _MATERIAL[ident]
    t = et.load(ident)
    if group(ident) == "callhome":
        stable, speakers = et.production(ident)
        call = ident.split("-", 1)[1]
        m = r3.Material(ident, "callhome", stable, speakers, {}, {}, [], (), et.CALLHOME_RUNS / call / f"{call}-vocal.ogg", {})
    else:
        m = r3.load(ident)
        spoken = sorted({w for _, _, w, _ in t.lines if w is not None})
        if group(ident) == "hina":
            rng = np.random.default_rng(SEED + sum(map(ord, ident)))
            pool = sorted({p for f, rs in r3._rows().items() if f.startswith("hina-") and f != ident for r in rs for p in r["truth"]}
                          - ROLE_LABELS)
            real = set(spoken) | ({"猫头鹰先生"} if ident in HOST_IS_OWL else set())
            pool = [p for p in pool if not any(same_person(p, q) for q in real)]
            m.people = sorted(set(spoken) | {pool[i] for i in rng.choice(len(pool), DISTRACTORS, replace=False)})
        else:
            m.people = sorted(set(m.roster) | set(spoken))
        m.roster = tuple(m.people)
        m.aliases = {n: a for n, a in ALIASES.items() if n in m.people}
        if ident in HOST_IS_OWL:
            m.aliases["主持人"] = ALIASES["猫头鹰先生"]
    _MATERIAL[ident] = m
    return m


def person_of_label(m, label):
    """A label as the model wrote it -> a person: `@name` by name (None for codes such as @甲); tracks are mapped by
    `track_people`."""
    if not label.startswith("@") or group(m.ident) == "callhome":
        return None
    return m.person_of(label)


def track_people(t, a, b):
    """{track: person} for a window: the majority truth person of its `single` rows on each track, inside the window
    (rows a-CONTEXT..b+CONTEXT), falling back to the file."""
    m = material(t.ident)

    def majority(rows):
        votes = defaultdict(Counter)
        for i in rows:
            lead, person = m.leader(i), t.row_person(i)
            if lead is not None and person is not None:
                votes[lead][person] += 1
        return {k: c.most_common(1)[0][0] for k, c in votes.items()}
    return {**majority(range(len(t.stable))), **majority(range(max(0, a - CONTEXT), min(len(t.stable), b + CONTEXT)))}


def label_person(t, a, b, label):
    label = str(label or "").strip()
    if re.fullmatch(r"s\d+", label):
        return track_people(t, a, b).get(int(label[1:]))
    return person_of_label(material(t.ident), label)


# --- windows and hints ---------------------------------------------------------------------------

def missing_lexical(t):
    return [x for x in t.missing if x["lexical"] and not LATIN.search(x["text"])]


def split_class(t, i):
    k = t.rows[i]["split"][0]
    words = t.words(i)
    first, second = ("".join(w["word"] for w in words[:k]), "".join(w["word"] for w in words[k:]))
    return "content" if et.is_lexical(first) and et.is_lexical(second) else "backchannel"


def events(t):
    """[(first row, last row, kind)]: split rows (`split-content` / `split-backchannel`) and lexical missing lines."""
    out = [(i, i, f"split-{split_class(t, i)}") for i, v in t.rows.items() if v["kind"] == "split"]
    out += [(min(x["anchors"]), max(x["anchors"]), "insert") for x in missing_lexical(t)]
    return sorted(out)


def place(t):
    """Non-overlapping windows over a file's events: [(a, b, [events])]."""
    n, out = len(t.stable), []
    for lo, hi, kind in events(t):
        if out and lo >= out[-1][0] and hi < out[-1][1]:
            out[-1][2].append((lo, hi, kind))
            continue
        a = min(max(CONTEXT, lo - WIN // 3), n - WIN - CONTEXT)
        if out and a < out[-1][1]:
            a = out[-1][1]
        if a + WIN > n - CONTEXT or hi >= a + WIN or lo < a:
            continue  # cannot hold it whole without overlapping the previous window
        out.append((a, a + WIN, [(lo, hi, kind)]))
    return out


def windows():
    """{"event": [[ident, a, b]], "control": [...], "edit": [...]}, frozen on disk."""
    path = OUT / "windows.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    chosen, callhome = [], []
    for ident in et.FILES:
        t = et.load(ident)
        for a, b, evs in place(t):
            kinds = Counter(k for *_, k in evs)
            tag = {"split": kinds["split-content"] + kinds["split-backchannel"], "insert": kinds["insert"]}
            if group(ident) != "callhome":
                if kinds["split-content"] or kinds["insert"]:
                    chosen.append([ident, a, b, tag])
            elif kinds["split-content"]:
                callhome.append((0, [ident, a, b, tag]))
            elif kinds["insert"] >= 2:
                callhome.append((1, [ident, a, b, tag]))
    first = [w for p, w in callhome if p == 0]
    rest = [w for p, w in callhome if p == 1]
    chosen += first + [rest[i] for i in sorted(rng.choice(len(rest), min(len(rest), N_CALLHOME_EXTRA), replace=False))]
    control = []
    for ident in rng.permutation([f for f in et.FILES if group(f) == "hina"]):
        t = et.load(str(ident))
        busy = {r for lo, hi, _ in events(t) for r in range(lo - CONTEXT, hi + CONTEXT + 1)}
        busy |= {r for x in t.missing for r in x["anchors"]} | {i for i, v in t.rows.items() if v["kind"] == "multi"}
        n = len(t.stable)
        for _ in range(50):
            a = int(rng.integers(CONTEXT, n - WIN - CONTEXT))
            if not busy & set(range(a - CONTEXT, a + WIN + CONTEXT)):
                control.append([str(ident), a, a + WIN])
                break
        if len(control) == N_CONTROL:
            break
    edit = []
    for g, k in N_EDIT.items():
        pool = [w for w in chosen if group(w[0]) == g]
        edit += [pool[i] for i in sorted(rng.choice(len(pool), min(k, len(pool)), replace=False))]
    out = {"event": chosen, "control": control, "edit": edit}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def task_windows(task, arm):
    """[(ident, a, b)]: split / insert -> the event windows holding a case of that task (+ the control windows in the
    plain arm); edit -> the edit windows."""
    w = windows()
    if task == "edit":
        return [tuple(x[:3]) for x in w["edit"]]
    return [tuple(x[:3]) for x in w["event"] if x[3][task]] + ([tuple(x) for x in w["control"]] if arm == "plain" else [])


def cases(task, ident, a, b):
    """The window's true cases for a task: split -> rows; insert -> missing lexical lines; edit -> single rows with a
    reading distance to the reference."""
    t = et.load(ident)
    if task == "split":
        return [i for i in range(a, b) if t.rows[i]["kind"] == "split"]
    if task == "insert":
        return [x for x in missing_lexical(t) if a <= min(x["anchors"]) and max(x["anchors"]) < b]
    return [i for i in range(a, b) if t.target(i) and t.target(i)[0] > 0]


def hints(task, ident, a, b):
    """[(spot text, true?, key)] for the oracle arm: every true case (edit: at most HINT_CAP) and as many decoys."""
    t = et.load(ident)
    rng = np.random.default_rng(SEED + a + sum(map(ord, ident)) + TASKS.index(task))
    true = cases(task, ident, a, b)
    if task == "edit" and len(true) > HINT_CAP:
        true = [true[i] for i in sorted(rng.choice(len(true), HINT_CAP, replace=False))]
    if task == "insert":
        busy = {r for x in t.missing for r in x["anchors"]}
        decoys = [i for i in range(a, b - 1) if not busy & {i, i + 1}]
        where = lambda lo, hi: f"q{lo} 一带" if lo == hi else f"q{lo} 到 q{hi} 一带"  # noqa: E731
        spots = [(where(min(x["anchors"]), max(x["anchors"])), True, x["line"]) for x in true]
        picked = [decoys[i] for i in sorted(rng.choice(len(decoys), min(len(decoys), max(1, len(true))), replace=False))]
        spots += [(where(i, i + 1), False, i) for i in picked]
    else:
        if task == "split":
            pool = [i for i in range(a, b) if t.rows[i]["kind"] == "single" and len(t.words(i)) >= 3 and i not in true]
        else:
            pool = [i for i in range(a, b) if t.target(i) and t.target(i)[0] == 0]
        picked = [pool[i] for i in sorted(rng.choice(len(pool), min(len(pool), max(1, len(true))), replace=False))]
        spots = [(f"q{i}", True, i) for i in true] + [(f"q{i}", False, i) for i in picked]
    return sorted(spots, key=lambda s: int(re.search(r"\d+", s[0]).group(0)))


# --- prompts -------------------------------------------------------------------------------------

def _text(name):
    return (HERE / name).read_text(encoding="utf-8")


def calls(task, arm):
    """[(key, system, user, wav, audio seconds, ident, a, b)]."""
    out = []
    for ident, a, b in task_windows(task, arm):
        t = et.load(ident)
        m = material(ident)
        spec = MATERIALS[m.kind]
        shown = range(max(0, a - CONTEXT), min(len(m.stable), b + CONTEXT))
        if m.kind == "callhome":
            cast_kind, cast = "说话的人（名字未知）", "说话的人：电话两端的两个人（有时一端换人接电话），名字未知。"
        elif arm == "oracle":
            lo, hi = float(m.stable[shown[0]]["start"]) - 1, float(m.stable[shown[-1]]["end"]) + 1
            people = sorted({w for x, y, w, _ in t.lines if w is not None and y > lo and x < hi})
            cast_kind, cast = "这一段里开口的人", "这一段里开口的人（按参考字幕，oracle）：" + "、".join(m.display(p) for p in people)
        else:
            cast_kind, cast = "可能出场的人物", "可能出场的人物（可能不全，也不一定都在这一段出场）：" + "、".join(m.display(p) for p in m.people)
        system = lr.template("edit_judge_system.md").substitute(
            scene=spec["scene"], cast_kind=cast_kind, task=_text(f"edit_judge_task_{task}.md"),
            labels="" if task == "edit" else "\n" + _text("edit_judge_labels.md"), format="\n" + _text(f"edit_judge_format_{task}.md"))
        profile, voice, _ = r3.profiles(m, a, b)
        heard = {k for i in shown for k in (m.speakers[i].to_dict().get("tracks") or [])}
        voice = {k: v for k, v in voice.items() if k in heard}  # the other tracks are not on any shown row
        import soundfile as sf
        wave = r3._wave(m)
        pieces, layout, cursor = [], [], 0.0
        gap = np.zeros(int(lj.GAP * lj.SR), dtype=np.float32)
        for track in sorted(voice):
            for i in voice[track]:
                piece = wave[int(float(m.stable[i]["start"]) * lj.SR):int(float(m.stable[i]["end"]) * lj.SR)]
                layout.append(f"| s{track} 的声音样本 | {cursor:.1f}–{cursor + len(piece) / lj.SR:.1f} 秒 |")
                pieces.extend([piece, gap])
                cursor += len(piece) / lj.SR + lj.GAP
        lo = max(0.0, float(m.stable[shown[0]]["start"]) - 0.5)
        hi = float(m.stable[shown[-1]]["end"]) + 0.5
        at = cursor
        pieces.append(wave[int(lo * lj.SR):int(hi * lj.SR)])
        layout.append(f"| 这一段字幕的录音 | {at:.1f}–{at + hi - lo:.1f} 秒 |")
        folder = OUT / "av"
        folder.mkdir(parents=True, exist_ok=True)
        wav = folder / f"{ident}-{a}.wav"
        if not wav.exists():  # the same audio for every task and arm; shards run at once, so write once, atomically
            part = wav.with_suffix(f".{time.time_ns()}.part.wav")
            sf.write(str(part), np.concatenate(pieces), lj.SR, subtype="PCM_16")
            try:
                part.replace(wav)
            except OSError:
                part.unlink(missing_ok=True)
        silent = [f"s{track}" for track in sorted(voice) if not voice[track]]
        absent = [f"s{track}" for track in sorted({k for k in (r3.profiles(m, a, b)[1])} - set(voice))]
        profile += ("\n\n音频布局：\n| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
                    + (f"\n\n没有声音样本的轨：{'、'.join(silent)}" if silent else "")
                    + (f"\n\n这一段字幕里没有挂到的轨（没附声音样本）：{'、'.join(absent)}" if absent else ""))
        cells = [lr.cell(s) for s in m.speakers]
        rows = "\n".join(f"{'▶ ' if a <= i < b else ''}q{i} {at + float(m.stable[i]['start']) - lo:.1f}|{cells[i]}|"
                         f"{str(m.stable[i].get('text', '')).strip()}" for i in shown)
        hint = ""
        if arm == "oracle":
            hint = lr.template(f"edit_judge_hint_{task}.md").substitute(
                spots="、".join(dict.fromkeys(s for s, _, _ in hints(task, ident, a, b))))  # several lines at one spot: listed once
        user = lr.template("edit_judge_user.md").substitute(title=spec["title"], cast=cast, profiles=profile, rows=rows, hint=hint, block=task)
        out.append((f"{ident}-{a}", system, user, wav, at + hi - lo, ident, a, b))
    return out


# --- run -----------------------------------------------------------------------------------------

FIELDS = {"split": ("id", "second", "track"), "insert": ("after", "text", "track"), "edit": ("id", "from", "to")}


def parse(text, task):
    """[items] from the reply's block (an empty block is a valid answer), or None when malformed."""
    if (text or "").lstrip().lower().startswith("thought"):
        return None  # agy's thought leak
    match = re.search(rf"<{task}>(.*?)</{task}>", text or "", re.S)
    if not match:
        return None
    items = []
    for raw in match.group(1).splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(item, dict) or any(k not in item for k in FIELDS[task]):
            return None
        items.append(item)
    return items


def run(task, arm, shard, shards, repeat):
    import llm_retrack_av as av
    complete = av.client_for(lj.TARGET)
    folder = OUT / f"{task}-{arm}" / f"r{repeat}"
    folder.mkdir(parents=True, exist_ok=True)
    only = repeat_keys(task, arm) if repeat > 1 else None
    for key, system, user, wav, *_ in calls(task, arm)[shard::shards]:
        if only is not None and key not in only:
            continue
        path = folder / f"{key}.json"
        if path.exists():
            continue
        answer, raw, tries = None, "", 0
        for attempt in range(4):  # the first call and at most three retries
            tries = attempt + 1
            try:
                raw = complete(system, user, wav)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{key}: {type(error).__name__} {str(error)[:80]}; retry", flush=True)
                raw = f"{type(error).__name__}: {error}"
                time.sleep(10 * (attempt + 1))
                continue
            answer = parse(raw, task)
            if answer is not None:
                break
            print(f"{key}: malformed reply; retry", flush=True)
        target = path if answer is not None else path.with_suffix(".failed.json")
        target.write_text(json.dumps({"answer": answer, "raw": raw, "tries": tries}, ensure_ascii=False, indent=1), encoding="utf-8")
        if answer is not None:
            path.with_suffix(".failed.json").unlink(missing_ok=True)
        print(f"{task}-{arm} r{repeat} {key}: {'ok, %d items' % len(answer) if answer is not None else 'GAVE UP'} ({tries} tries)", flush=True)


REPEAT_WINDOWS = {"split": 8, "insert": 8, "edit": 4}


def repeat_keys(task, arm):
    """The windows the consistency rerun (repeat 2) covers: a seeded draw, the same for both arms of a task."""
    wins = [w for w in task_windows(task, "oracle")]
    rng = np.random.default_rng(SEED + 7 * TASKS.index(task))
    return {f"{wins[i][0]}-{wins[i][1]}" for i in rng.choice(len(wins), min(len(wins), REPEAT_WINDOWS[task]), replace=False)}


# --- build ---------------------------------------------------------------------------------------

def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    count = counter.count_text
    w = windows()
    for name in ("event", "control", "edit"):
        print(f"{name} windows: {len(w[name])} " + str(dict(Counter(group(x[0]) for x in w[name]))))
    tally = Counter()
    for ident, a, b in [tuple(x[:3]) for x in w["event"]]:
        t = et.load(ident)
        for lo, hi, kind in events(t):
            if a <= lo and hi < b:
                tally[(group(ident), kind)] += 1
        tally[(group(ident), "single rows")] += sum(t.rows[i]["kind"] == "single" for i in range(a, b))
    for ident, a, b in [tuple(x[:3]) for x in w["edit"]]:
        t = et.load(ident)
        d = [t.target(i) for i in range(a, b)]
        tally[(group(ident), "edit rows with a distance")] += sum(1 for x in d if x and x[0] > 0)
        tally[(group(ident), "edit distance (reading chars)")] += sum(x[0] for x in d if x)
    for key in sorted(tally):
        print(f"  {key}: {tally[key]}")
    total_text = total_audio = 0
    for task in TASKS:
        for arm in ARMS:
            cs = calls(task, arm)
            text = sum(count(c[1]) + count(c[2]) for c in cs)
            audio = sum(c[4] for c in cs)
            total_text += text
            total_audio += audio
            print(f"{task}-{arm}: {len(cs)} calls, text input {text} ({text / max(1, len(cs)):.0f} per call), audio {audio:.0f} s ≈ {audio * 32:.0f} tokens")
            (OUT / f"prompt-{task}-{arm}-example.md").write_text(cs[0][1] + "\n\n=====\n\n" + cs[0][2], encoding="utf-8")
            keys = repeat_keys(task, arm)
            again = [c for c in cs if c[0] in keys]
            rtext = sum(count(c[1]) + count(c[2]) for c in again)
            total_text += rtext
            total_audio += sum(c[4] for c in again)
            print(f"  + consistency rerun of {len(again)} windows: text {rtext}")
    print(f"all: text input {total_text}, audio {total_audio:.0f} s ≈ {total_audio * 32:.0f} tokens; counted by {counter.source} "
          "(a floor: retries, thinking and agy's wrapper not counted)")


# --- scoring -------------------------------------------------------------------------------------

def answers(task, arm, repeat=1):
    """{(ident, a): [items]} for the windows with a reply, and the keys that gave up."""
    folder = OUT / f"{task}-{arm}" / f"r{repeat}"
    got, failed = {}, []
    for path in folder.glob("*.json"):
        if path.name.endswith(".failed.json"):
            failed.append(path.name[:-len(".failed.json")])
            continue
        ident, a = path.stem.rsplit("-", 1)
        got[(ident, int(a))] = json.loads(path.read_text(encoding="utf-8"))["answer"]
    return got, failed


def split_point(t, i, second):
    """The word index the model's `second` starts at (snapped to the nearest word start), or None if not in the row."""
    words = t.words(i)
    text = "".join(w["word"] for w in words)
    at = text.find(str(second))
    if at < 0:
        flat = et.surface(text)
        want = et.surface(second)
        at_flat = flat.find(want) if want else -1
        if at_flat < 0:
            return None
        # map a position in the surface form back to the raw text
        count, at = 0, 0
        for at, ch in enumerate(text):
            if count == at_flat:
                break
            count += bool(et.surface(ch))
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w["word"])
    return int(np.argmin([abs(s - at) for s in starts]))


def score_split(arm, repeat=1):
    """Per window: [(unit, outcome)] -- outcomes for true rows and for single rows."""
    got, failed = answers("split", arm, repeat)
    out = {}
    for ident, a, b in task_windows("split", arm):
        if (ident, a) not in got:
            continue
        t = et.load(ident)
        hinted = {k for _, _, k in hints("split", ident, a, b)} if arm == "oracle" else set(range(a, b))
        said = {}
        for item in got[(ident, a)]:
            i = int(re.sub(r"\D", "", str(item.get("id", ""))) or -1)
            if a <= i < b:
                said[i] = item
        units = []
        for i in range(a, b):
            kind = t.rows[i]["kind"]
            if kind not in ("split", "single") or i not in hinted:
                continue
            item = said.get(i)
            if kind == "single":
                units.append({"row": i, "true": False, "split": item is not None})
                continue
            lo, hi = t.rows[i]["split"]
            k = split_point(t, i, item["second"]) if item else None
            person = label_person(t, a, b, item["track"]) if item else None
            units.append({"row": i, "true": True, "class": split_class(t, i), "split": item is not None,
                          "position": k is not None and lo <= k <= hi, "near": k is not None and lo - 1 <= k <= hi + 1,
                          "person": person == t.rows[i]["people"][1], "said": item, "truth": t.rows[i]})
        out[(ident, a)] = units
    return out, failed


def score_insert(arm, repeat=1):
    from rapidfuzz.distance import Levenshtein
    got, failed = answers("insert", arm, repeat)
    out = {}
    for ident, a, b in task_windows("insert", arm):
        if (ident, a) not in got:
            continue
        t = et.load(ident)
        spots = hints("insert", ident, a, b) if arm == "oracle" else None
        lines = [x for x in t.missing if any(a <= r < b for r in x["anchors"])]
        items = []
        for item in got[(ident, a)]:
            r = int(re.sub(r"\D", "", str(item.get("after", ""))) or -1)
            text = et.strip_tokens(et.reading(item.get("text", "")), et.BACKCHANNEL)
            best = None
            for x in lines:
                if r not in x["anchors"]:
                    continue
                truth = et.strip_tokens(et.reading(x["text"]), et.BACKCHANNEL)
                cer = Levenshtein.distance(text, truth) / max(1, len(truth))
                if best is None or cer < best[0]:
                    best = (cer, x)
            near = "".join(et.reading(t.stable[j].get("text", "")) for j in range(max(0, r - 1), min(len(t.stable), r + 2)))
            duplicate = bool(text) and len(et.matched(text, near)) / len(text) >= 0.7
            items.append({"after": r, "text": item.get("text", ""), "cer": best[0] if best else None,
                          "line": best[1]["line"] if best else None, "lexical": best[1]["lexical"] if best else None,
                          "person": best is not None and label_person(t, a, b, item.get("track")) == best[1]["person"],
                          "duplicate": duplicate, "in_window": a <= r < b})
        # one item per truth line: the closest
        taken = {}
        for k, it in sorted(enumerate(items), key=lambda kv: (kv[1]["cer"] is None, kv[1]["cer"] or 0)):
            if it["line"] is not None and it["line"] not in taken:
                taken[it["line"]] = k
            else:
                it["line"] = None
        truth_lines = [x for x in missing_lexical(t) if a <= min(x["anchors"]) and max(x["anchors"]) < b]
        found = {x["line"]: items[taken[x["line"]]] if x["line"] in taken else None for x in truth_lines}
        decoys = [k for _, true, k in spots if not true] if spots else []
        decoy_hits = [k for k in decoys if any(it["after"] in (k, k + 1) for it in items)]
        out[(ident, a)] = {"items": items, "found": found, "decoys": decoys, "decoy_hits": decoy_hits}
    return out, failed


def score_edit(arm, repeat=1):
    got, failed = answers("edit", arm, repeat)
    out = {}
    for ident, a, b in task_windows("edit", arm):
        if (ident, a) not in got:
            continue
        t = et.load(ident)
        spots = {k: true for _, true, k in hints("edit", ident, a, b)} if arm == "oracle" else None
        by_row = defaultdict(list)
        for item in got[(ident, a)]:
            i = int(re.sub(r"\D", "", str(item.get("id", ""))) or -1)
            by_row[i].append(item)
        # rows where people overlap or the turn changes: a split row +-1, or an anchor of any missing line (back-channels too)
        hard = {j for i, v in t.rows.items() if v["kind"] in ("split", "multi") for j in (i - 1, i, i + 1)}
        hard |= {j for x in t.missing for j in x["anchors"]}
        rows = {}
        for i in range(a, b):
            base = t.target(i)
            if base is None or (spots is not None and i not in spots):
                continue
            text = str(t.stable[i].get("text", ""))
            edits = []
            for item in by_row.get(i, []):
                src, dst = str(item["from"]), str(item["to"])
                if not src or src not in text:
                    edits.append({"item": item, "delta": None})
                    continue
                edits.append({"item": item, "delta": t.target(i, text.replace(src, dst, 1))[0] - base[0]})
            new = text
            for e in edits:
                if e["delta"] is not None:
                    new = new.replace(str(e["item"]["from"]), str(e["item"]["to"]), 1)
            rows[i] = {"d0": base[0], "d1": t.target(i, new)[0], "edits": edits, "hinted_true": spots.get(i) if spots else None,
                       "hard": i in hard}
        stray = [it for i, items in by_row.items() if i not in rows for it in items]
        out[(ident, a)] = {"rows": rows, "stray": stray}
    return out, failed


def bootstrap(per_window, stat, draws=5000):
    """(value, lo, hi): `stat` over the windows' units, 95% percentile interval over windows resampled."""
    keys = list(per_window)
    rng = np.random.default_rng(SEED)
    value = stat([u for k in keys for u in per_window[k]])
    sims = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        v = stat([u for j in pick for u in per_window[keys[j]]])
        if v is not None:
            sims.append(v)
    return value, (np.percentile(sims, 2.5) if sims else None), (np.percentile(sims, 97.5) if sims else None)


def fmt(v):
    value, lo, hi = v
    return "—" if value is None else f"{100 * value:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def rate(pred, base):
    return lambda units: (sum(pred(u) for u in units if base(u)) / n) if (n := sum(1 for u in units if base(u))) else None


def score():
    for arm in ARMS:
        res, failed = score_split(arm)
        if res:
            print(f"== split {arm}: windows {len(res)}, gave up {failed or 'none'}")
            for cls in ("content", "backchannel", None):
                base = (lambda u, c=cls: u["true"] and (c is None or u["class"] == c))
                print(f"  true rows ({cls or 'all'}, {sum(base(u) for us in res.values() for u in us)}): "
                      f"full {fmt(bootstrap(res, rate(lambda u: u['split'] and u['near'] and u['person'], base)))} "
                      f"(strict position {fmt(bootstrap(res, rate(lambda u: u['split'] and u['position'] and u['person'], base)))}), "
                      f"split {fmt(bootstrap(res, rate(lambda u: u['split'], base)))}, "
                      f"position | split {fmt(bootstrap(res, rate(lambda u: u['near'], lambda u, b=base: b(u) and u['split'])))}, "
                      f"person | split {fmt(bootstrap(res, rate(lambda u: u['person'], lambda u, b=base: b(u) and u['split'])))}")
            # the plan's main reading counts the content rows only, so its precision is over splits on content and single rows
            full = lambda u: u["true"] and u["class"] == "content" and u["split"] and u["near"] and u["person"]  # noqa: E731
            scored = lambda u: u["split"] and (not u["true"] or u["class"] == "content")  # noqa: E731
            print(f"  single rows ({sum(not u['true'] for us in res.values() for u in us)}): false split "
                  f"{fmt(bootstrap(res, rate(lambda u: u['split'], lambda u: not u['true'])))}; precision (full / splits on content and "
                  f"single rows) {fmt(bootstrap(res, rate(full, scored)))}; over every split "
                  f"{fmt(bootstrap(res, rate(lambda u: u['true'] and u['near'] and u['person'], lambda u: u['split'])))}")
        res, failed = score_insert(arm)
        if res:
            per = {k: [{"kind": "truth", **({"hit": v is not None and v["cer"] <= 0.5, "person": v is not None and v["person"]})}
                       for v in r["found"].values()] + [{"kind": "item", **it} for it in r["items"] if it["in_window"]]
                   for k, r in res.items()}
            print(f"== insert {arm}: windows {len(res)}, gave up {failed or 'none'}")
            for g in ("callhome", "fansub", None):
                sub = {k: v for k, v in per.items() if g is None or (group(k[0]) == "callhome") == (g == "callhome")}
                if not sub:
                    continue
                print(f"  {g or 'all'}: lexical lines {sum(u['kind'] == 'truth' for us in sub.values() for u in us)}: content right "
                      f"{fmt(bootstrap(sub, rate(lambda u: u['hit'], lambda u: u['kind'] == 'truth')))}, + person "
                      f"{fmt(bootstrap(sub, rate(lambda u: u['hit'] and u['person'], lambda u: u['kind'] == 'truth')))}; items "
                      f"{sum(u['kind'] == 'item' for us in sub.values() for u in us)}: precision (matches a missing line, CER <= 0.5) "
                      f"{fmt(bootstrap(sub, rate(lambda u: u['line'] is not None and u['cer'] <= 0.5, lambda u: u['kind'] == 'item')))}, "
                      f"duplicates of ASR text {sum(u['kind'] == 'item' and u['duplicate'] for us in sub.values() for u in us)}")
            if arm == "oracle":
                hits = sum(len(r["decoy_hits"]) for r in res.values())
                print(f"  decoy spots with an insertion: {hits}/{sum(len(r['decoys']) for r in res.values())}")
        res, failed = score_edit(arm)
        if res:
            per = {k: list(r["rows"].values()) for k, r in res.items()}
            edits = {k: [e for row in r["rows"].values() for e in row["edits"]] for k, r in res.items()}
            print(f"== edit {arm}: windows {len(res)}, gave up {failed or 'none'}; rows {sum(len(v) for v in per.values())}, "
                  f"edits {sum(len(v) for v in edits.values())} (not found in the row "
                  f"{sum(e['delta'] is None for v in edits.values() for e in v)}), on rows out of scope {sum(len(r['stray']) for r in res.values())}")
            removed = bootstrap(per, lambda rows: (sum(r["d0"] - r["d1"] for r in rows) / s) if (s := sum(r["d0"] for r in rows)) else None)
            net = bootstrap(per, lambda rows: sum(r["d0"] - r["d1"] for r in rows) / len(rows) if rows else None)
            prec = bootstrap(edits, rate(lambda e: e["delta"] < 0, lambda e: e["delta"] is not None and e["delta"] != 0))
            for name, want in (("rows in overlap / turn changes", True), ("other rows", False)):
                sub = {k: [r for r in v if r["hard"] == want] for k, v in per.items()}
                sub = {k: v for k, v in sub.items() if v}
                if sub:
                    print(f"  {name} ({sum(len(v) for v in sub.values())}, distance {sum(r['d0'] for v in sub.values() for r in v)}): share removed "
                          + fmt(bootstrap(sub, lambda rows: (sum(r["d0"] - r["d1"] for r in rows) / s) if (s := sum(r["d0"] for r in rows)) else None)))
            print(f"  share of the reading distance removed {fmt(removed)}; net characters fixed per row "
                  f"{net[0]:+.3f} [{net[1]:+.3f}, {net[2]:+.3f}]; precision (improve / changed) {fmt(prec)}; "
                  f"improve {sum(e['delta'] is not None and e['delta'] < 0 for v in edits.values() for e in v)}, "
                  f"harm {sum(e['delta'] is not None and e['delta'] > 0 for v in edits.values() for e in v)}, "
                  f"neutral {sum(e['delta'] == 0 for v in edits.values() for e in v)}")


def consistency():
    """Run 1 vs run 2 on the rerun windows: the same decisions? (split: rows split and where; insert: truth lines found
    and the number of items; edit: (row, from, to) triples), and the main reading on those windows in each run."""
    for task in TASKS:
        for arm in ARMS:
            two, _ = answers(task, arm, 2)
            one, _ = answers(task, arm, 1)
            keys = [k for k in two if k in one]
            if not keys:
                continue
            same = total = 0
            for k in keys:
                if task == "split":
                    sets = [{(int(re.sub(r"\D", "", str(x.get("id", ""))) or -1), str(x.get("second", ""))[:2]) for x in got[k]} for got in (one, two)]
                elif task == "insert":
                    sets = [{(str(x.get("after", "")), et.reading(x.get("text", ""))[:4]) for x in got[k]} for got in (one, two)]
                else:
                    sets = [{(str(x.get("id", "")), str(x.get("from", "")), str(x.get("to", ""))) for x in got[k]} for got in (one, two)]
                same += len(sets[0] & sets[1])
                total += len(sets[0] | sets[1])
            print(f"{task}-{arm}: {len(keys)} windows run twice; decisions in both runs {same}/{total} "
                  f"({same / max(1, total):.0%} of the decisions either run made)")


def readings():
    """The written-in-advance side readings: per material, oracle - plain paired by window, the control windows, and the
    main reading on the rerun windows in run 1 and run 2."""
    def split_units(arm, repeat=1):
        res, _ = score_split(arm, repeat)
        return {k: [u for u in us if u["true"] and u["class"] == "content"] for k, us in res.items()}, res
    full = lambda u: u["split"] and u["near"] and u["person"]  # noqa: E731

    # per material
    for arm in ARMS:
        _, res = split_units(arm)
        for g in ("hina", "manosaba", "callhome"):
            us = [u for k, v in res.items() if group(k[0]) == g for u in v]
            pos = [u for u in us if u["true"] and u["class"] == "content"]
            neg = [u for u in us if not u["true"]]
            print(f"split {arm} {g}: content rows {len(pos)}, full {sum(map(full, pos))}; single rows {len(neg)}, split {sum(u['split'] for u in neg)}")
        if arm == "plain":
            control = {tuple(x[:2]) for x in windows()["control"]}
            neg = [u for k, v in res.items() if k in control for u in v if not u["true"]]
            print(f"split plain control windows: single rows {len(neg)}, split {sum(u['split'] for u in neg)}")
        ins, _ = score_insert(arm)
        for g in ("hina", "manosaba", "callhome"):
            rs = [r for k, r in ins.items() if group(k[0]) == g]
            lines = [v for r in rs for v in r["found"].values()]
            items = [it for r in rs for it in r["items"] if it["in_window"]]
            print(f"insert {arm} {g}: lexical lines {len(lines)}, content right {sum(v is not None and v['cer'] <= 0.5 for v in lines)}; items {len(items)}, "
                  f"matched (CER <= 0.5) {sum(it['line'] is not None and it['cer'] <= 0.5 for it in items)}")
        if arm == "plain":
            control = {tuple(x[:2]) for x in windows()["control"]}
            print(f"insert plain control windows: items {sum(len(r['items']) for k, r in ins.items() if k in control)}")
        ed, _ = score_edit(arm)
        for g in ("hina", "manosaba", "callhome"):
            rows = [r for k, v in ed.items() if group(k[0]) == g for r in v["rows"].values()]
            edits = [e for r in rows for e in r["edits"] if e["delta"] is not None]
            print(f"edit {arm} {g}: rows {len(rows)}, distance {sum(r['d0'] for r in rows)}, removed {sum(r['d0'] - r['d1'] for r in rows)}; "
                  f"improve {sum(e['delta'] < 0 for e in edits)}, harm {sum(e['delta'] > 0 for e in edits)}, neutral {sum(e['delta'] == 0 for e in edits)}")

    # oracle - plain, paired by window (windows both arms ran)
    o, _ = split_units("oracle")
    pl, _ = split_units("plain")
    keys = [k for k in o if k in pl]
    d = {k: [float(full(a)) - float(full(b)) for a, b in zip(o[k], pl[k])] for k in keys if o[k]}
    print("split oracle - plain, content rows (full), paired: " + fmt(bootstrap(d, lambda us: sum(us) / len(us) if us else None)).replace("%", " pp", 1))
    io, _ = score_insert("oracle")
    ip, _ = score_insert("plain")
    hit = lambda v: float(v is not None and v["cer"] <= 0.5)  # noqa: E731
    d = {k: [hit(io[k]["found"][x]) - hit(ip[k]["found"][x]) for x in io[k]["found"]] for k in io if k in ip and io[k]["found"]}
    print("insert oracle - plain, lexical lines (content right), paired: " + fmt(bootstrap(d, lambda us: sum(us) / len(us) if us else None)).replace("%", " pp", 1))
    eo, _ = score_edit("oracle")
    ep, _ = score_edit("plain")
    d = {k: [(r["d0"] - r["d1"]) - (ep[k]["rows"][i]["d0"] - ep[k]["rows"][i]["d1"]) for i, r in eo[k]["rows"].items() if i in ep[k]["rows"]]
         for k in eo if k in ep}
    tot = {k: [ep[k]["rows"][i]["d0"] for i in eo[k]["rows"] if i in ep[k]["rows"]] for k in eo if k in ep}
    print(f"edit oracle - plain on the oracle's hinted rows: characters removed {sum(map(sum, d.values())):+.0f} of a distance {sum(map(sum, tot.values()))}")

    # main readings on the rerun windows, run 1 and run 2
    for task in TASKS:
        keys = repeat_keys(task, "oracle")
        for arm in ARMS:
            line = []
            for rep in (1, 2):
                if task == "split":
                    res, _ = score_split(arm, rep)
                    us = [u for k, v in res.items() if f"{k[0]}-{k[1]}" in keys for u in v]
                    pos = [u for u in us if u["true"]]
                    line.append(f"r{rep}: true rows {len(pos)} full {sum(map(full, pos))}, false splits {sum(u['split'] for u in us if not u['true'])}")
                elif task == "insert":
                    res, _ = score_insert(arm, rep)
                    rs = [r for k, r in res.items() if f"{k[0]}-{k[1]}" in keys]
                    line.append(f"r{rep}: lines {sum(len(r['found']) for r in rs)} content right {sum(hit(v) for r in rs for v in r['found'].values()):.0f}, "
                                f"items {sum(len(r['items']) for r in rs)}")
                else:
                    res, _ = score_edit(arm, rep)
                    rows = [r for k, v in res.items() if f"{k[0]}-{k[1]}" in keys for r in v["rows"].values()]
                    edits = [e for r in rows for e in r["edits"] if e["delta"] is not None]
                    line.append(f"r{rep}: removed {sum(r['d0'] - r['d1'] for r in rows)}/{sum(r['d0'] for r in rows)}, improve "
                                f"{sum(e['delta'] < 0 for e in edits)} harm {sum(e['delta'] > 0 for e in edits)}")
            print(f"rerun windows {task}-{arm}: " + "; ".join(line))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "run", "score", "consistency", "readings"))
    parser.add_argument("--task", choices=TASKS, default="split")
    parser.add_argument("--arm", choices=ARMS, default="oracle")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "score":
        score()
    elif args.command == "consistency":
        consistency()
    elif args.command == "readings":
        readings()
    else:
        run(args.task, args.arm, args.shard, args.shards, args.repeat)


if __name__ == "__main__":
    main()
