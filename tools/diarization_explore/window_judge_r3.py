"""Speaker judge round 3 (owner 2026-10-06): long contiguous windows, three materials, text and audio, closed / open / open_gen.

Round 2 (`manosaba_window_judge.py`, report 5.13) judged 30-row windows of 魔裁 only, text only, with the window's own
true speakers as the cast block. Owner's direction for this round: longer windows (about 100 and about 200 rows), the
魔裁 clips plus hina and 姐姐咚咚 (onetonton), both modalities, all three arms. Criteria: report section 5.14, written first.

Materials (truth used only for scoring and for the true cast block):
  manosaba   the 魔裁 story clips, nameplate truth (`manosaba_truth.py`), production run in out/manosaba/
  hina       refined-reference truth, "covered" (`p2_features.load_rows`), production timeline from the P2 recompute
  onetonton  the same for 姐姐咚咚
A row's truth is a set of people (hina / onetonton: everyone tied for the most reference coverage; 魔裁: one person).
Two windows per material, in two different files, one of WINDOW_ROWS[0] and one of WINDOW_ROWS[1] consecutive ASR rows,
plus CONTEXT rows either side; files drawn from those whose production error rate is at least the material's median.
Every row in the window is judged, labelled or not.

Arms (agy, Gemini 3.7 Flash, one call per window and arm; modality text or audio):
  closed     keep / any track of the file / abstain
  open       + `@label`: open a new track; the label is a cast name or a placeholder (`@甲`); one label = one track per window
  open_gen   `open`, with the cast block an agy call wrote from the window's text and a candidate list, instead of the true one
The true cast block: 魔裁, the game's main cast (what a knowledge base gives; every 魔裁 window here holds nearly all the
clip's speakers, so the clip's own list would tell the window's speakers); hina / onetonton, every person with truth in
the file. References per track: speech share, REF_LINES example lines and (audio) 2 voice samples, the single-track rows
production is most sure of (VBx soft margin) -- outside the window first, inside it when the window holds (nearly) the
whole file; a reference row inside the window is not scored.

Scoring: a labelled row's person before is its track's majority person over the file. After: `sK` -> that track's person;
`@label` -> (anonymous, the main reading) the majority true person among the window's rows given that label, or (named)
the person the label names. Strict (main): a new track whose person already owns a track of the file counts wrong (in
production it splits that person). The reading is the change in the share of labelled rows whose person is right, per
100 labelled rows, bootstrapped over (window, production track). The main verdict is on the 魔裁 windows: in the hina /
onetonton files every person owns a track, so a new track can never be right there and open − closed only measures the
side effect of offering it. Unlabelled rows are read apart: labelled right / wrong vs the zero-cost "track of the nearest
labelled row". A call that gives up is cached under `.failed.json` and reported, never read as "keep everything".

    python tools/diarization_explore/window_judge_r3.py build
    python tools/diarization_explore/window_judge_r3.py cast
    python tools/diarization_explore/window_judge_r3.py run --arm text-closed [--shard 0 --shards 2]
    python tools/diarization_explore/window_judge_r3.py score
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import llm_retrack_judge as lj  # noqa: E402
import manosaba_judge as mj  # noqa: E402
import manosaba_truth as mt  # noqa: E402
import manosaba_window_judge as mw  # noqa: E402
import p2_features as pf  # noqa: E402

OUT = lr.OUT / "r3"
SEED = 20261007
WINDOW_ROWS = (200, 100)
CONTEXT = 5
REF_LINES = 3
MODALITIES = ("text", "audio")
KINDS = ("closed", "open", "open_gen")
ARMS = tuple(f"{m}-{k}" for m in MODALITIES for k in KINDS)
MATERIALS = {
    "manosaba": dict(title="一部日本视觉小说游戏的剧情录像（无解说，标题略）",
                     scene="素材是一部日本视觉小说游戏的剧情录像（没有解说）：每行字幕是一个游戏角色的配音台词，旁白不配音、不在字幕里。",
                     short="日本视觉小说游戏的剧情录像，每行是角色配音"),
    "hina": dict(title="一档游戏相关的日语节目（声优对谈等，标题略）",
                 scene="素材是一档游戏相关的日语节目（声优对谈、直播等），出场的人可能用角色名称呼。中间可能插播游戏的宣传片（PV）等角色配音片段，"
                       "也可能是别的语言（例如中文配音）：这些行的说话人记片中的角色，不记节目里的人。",
                 short="游戏相关的日语节目，声优对谈等"),
    "onetonton": dict(title="一档日语多人对谈节目（标题略）",
                      scene="素材是一档日语多人对谈节目。",
                      short="日语多人对谈节目"),
}
EXAMPLE = {
    "closed": ['{"id":"q3","track":"keep","why":"延续了自己上一句的话题"}',
               '{"id":"q4","track":"s2","why":"在回答上一句对她的称呼，自称与 s2 的例句一致"}',
               '{"id":"q5","track":"abstain","why":"只有一声附和，说不出是谁"}'],
}
# names from outside every material, so the example cannot point at anyone in a cast block
EXAMPLE["open"] = EXAMPLE["closed"][:2] + ['{"id":"q5","track":"@山田花子","why":"被叫作花子的人在回答，画像里没有一条轨是她"}',
                                           '{"id":"q6","track":"@甲","why":"和 q5 对话的另一个人，画像里没有他的轨，认不出是谁"}']
#: the `<tracks>` example to go with EXAMPLE: who each label the example rows end on stands for (round 7 on)
TRACKS = {"closed": "s2 = 佐藤一郎\ns3 = 不明", "open": "s2 = 佐藤一郎\n@山田花子 = 山田花子\n@甲 = 不明"}
#: what a cast block writes for "someone spoke but I cannot tell who" -- never passed on to the judge as a name
UNKNOWN = "不明"


@dataclass
class Material:
    ident: str
    kind: str
    stable: list
    speakers: list
    truth: dict        # row index -> frozenset of people
    owner: dict        # track -> person
    people: list       # every person with truth in the file (the true cast block)
    roster: tuple      # candidate list for the cast-block writer
    vocal: Path
    conf: dict         # row index -> VBx soft margin of its leading track
    aliases: dict = field(default_factory=dict)  # name -> how the material writes it (Japanese), shown in brackets

    def leader(self, i):
        tracks = self.speakers[i].to_dict().get("tracks") or []
        return tracks[0] if tracks else None

    def display(self, name):
        return f"{name}（{self.aliases[name]}）" if name in self.aliases else name

    def person_of(self, label):
        """A name as the model wrote it (a bracketed alias copied along is dropped) -> a person of this material, or None."""
        name = re.sub(r"[（(].*$", "", label.strip().lstrip("@")).strip()
        table = {n: n for n in dict.fromkeys(self.people + list(self.roster))}
        return mt.speaker_of(name, table) if name and table else None


_LOADED, _ROWS = {}, {}


def _rows():
    """P2's derived rows for hina / onetonton under the covered truth, loaded once."""
    if not _ROWS:
        _ROWS.update(pf.load_rows("covered")[0])
    return _ROWS


def _confidence(ident, stable):
    cache = pf.CACHE / f"{ident}-allrows.pkl"
    if not cache.exists():
        records, _ = pf.row_records(ident, stable, lambda row: (("-",), ("-",)))
        cache.write_bytes(pickle.dumps(records))
    return {r["index"]: r["soft_margin"] for r in pickle.loads(cache.read_bytes())}


def load(ident):
    if ident in _LOADED:
        return _LOADED[ident]
    import p2_recompute as pr
    if ident.startswith("manosaba-"):
        names = mw.display_names()
        stable, speakers, rows, owner = mj.material(ident)
        truth = {r["index"]: frozenset({names[r["truth"]]}) for r in rows}
        m = Material(ident, "manosaba", stable, speakers, truth, {k: names[v] for k, v in owner.items()}, list(mw.ROSTER), mw.ROSTER,
                     pr.vocal(ident), _confidence(ident, stable))
    else:
        from finesub.speech.diarization.assign import assign_all
        from finesub.speech.diarization.segments import TrackSegment, segments_from_activity
        stable = pf.reference(ident)[0]
        native = np.load(pf.CACHE / f"{ident}.npz")["native"]
        segments = [TrackSegment(track=t.track, start=t.start, end=t.end) for t in segments_from_activity(native.tolist(), frame_seconds=pf.STEP)]
        speakers = assign_all(stable, segments)
        records = _rows()[ident]
        truth = {r["index"]: frozenset(r["truth"]) for r in records if r["truth"]}
        people = sorted({p for s in truth.values() for p in s})
        kind = "hina" if ident.startswith("hina-") else "onetonton"
        m = Material(ident, kind, stable, speakers, truth, dict(records[0]["owner"]), people, tuple(people), pr.vocal(ident),
                     _confidence(ident, stable))
    _LOADED[ident] = m
    return m


def error_rate(m):
    labelled = [i for i in m.truth if m.leader(i) is not None and m.leader(i) in m.owner]
    return np.mean([m.owner[m.leader(i)] not in m.truth[i] for i in labelled]) if labelled else 0.0


def windows():
    """[(ident, first row, last row + 1)], frozen on disk."""
    path = OUT / "windows.json"
    if path.exists():
        return [tuple(w) for w in json.loads(path.read_text(encoding="utf-8"))]
    rng = np.random.default_rng(SEED)
    rows_all = _rows()
    have = lambda f: (pf.CACHE / f"{f}.npz").exists()  # noqa: E731
    pools = {"manosaba": list(mt.CLIPS),
             "hina": sorted(f for f in rows_all if f.startswith("hina-") and have(f)),  # ⚠ hina loads before onetonton
             "onetonton": sorted(f for f in rows_all if f.startswith("onetonton-") and have(f))}
    out = []
    for kind, files in pools.items():
        if kind == "manosaba":
            rates = {f: error_rate(load(f)) for f in files}
        else:  # P2's rows carry the same rule (the track's majority person by seconds, ties shared)
            rates = {f: float(np.mean([r["wrong"] for r in rows_all[f] if "wrong" in r])) for f in files}
        median = float(np.median(list(rates.values())))
        eligible = [f for f in files if rates[f] >= median]
        picks = [eligible[i] for i in rng.choice(len(eligible), 2, replace=False)]
        for ident, size in zip(picks, WINDOW_ROWS):
            n = len(load(ident).stable)
            size = min(size, n - 2 * CONTEXT)
            a = int(rng.integers(CONTEXT, n - size - CONTEXT + 1))
            out.append((ident, a, a + size))
        print(f"{kind}: error rate median {median:.1%}; picked {[(f, round(rates[f], 3)) for f in picks]}")
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def shown_rows(m, a, b):
    return range(max(0, a - CONTEXT), min(len(m.stable), b + CONTEXT))


_WARNED = set()


def profiles(m, a, b):
    """(profile text, {track: voice-sample rows}, reference rows inside the window): the single-track rows production is
    most sure of, outside the window and its context first."""
    seconds = lambda i: float(m.stable[i]["end"]) - float(m.stable[i]["start"])  # noqa: E731
    single = {i: (s.to_dict().get("tracks") or []) for i, s in enumerate(m.speakers)}
    near = lambda i: a - CONTEXT <= i < b + CONTEXT  # noqa: E731
    rank = lambda i: (near(i), -m.conf.get(i, -np.inf))  # noqa: E731
    share = Counter()
    for i, tracks in single.items():
        if tracks:
            share[tracks[0]] += seconds(i)
    total = sum(share.values()) or 1.0
    lines, voice, used = [], {}, set()
    for track in sorted(share, key=lambda k: -share[k]):
        picks = sorted((i for i, tr in single.items() if tr == [track]
                        and len(re.sub(r"[\W_]", "", str(m.stable[i].get("text", "")))) >= 8), key=rank)[:REF_LINES]
        voice[track] = sorted(sorted((i for i, tr in single.items() if tr == [track] and lj.SAMPLE[0] <= seconds(i) <= lj.SAMPLE[1]),
                                     key=rank)[:2])
        for missing, what in ((not picks, "example line"), (not voice[track], "voice sample")):
            if missing and (m.ident, track, what) not in _WARNED:
                _WARNED.add((m.ident, track, what))
                print(f"warning: {m.ident}: s{track} has no {what}", file=sys.stderr)
        used.update(picks + voice[track])
        lines.append(f"- s{track}：占语音 {share[track] / total * 100:.0f}%。例句：" +
                     (" / ".join(f"「{str(m.stable[i].get('text', '')).strip()}」" for i in sorted(picks)) or "（没有可靠的例句）"))
    return "\n".join(lines), voice, {i for i in used if a <= i < b}


def cast_prompt(m, a, b):
    spec = MATERIALS[m.kind]
    rows = "\n".join(f"{float(m.stable[i]['start']):.1f}|{str(m.stable[i].get('text', '')).strip()}" for i in shown_rows(m, a, b))
    system = lr.template("cast_block_r3_system.md").substitute(scene_short=spec["short"])
    user = lr.template("cast_block_r3_user.md").substitute(title=spec["title"], roster="、".join(m.display(n) for n in m.roster), rows=rows)
    return system, user


def generated_cast(ident, a):
    """The cast block's names, a copied alias dropped; its "unknown" line is not a name and is left out."""
    path = OUT / "cast" / f"{ident}-{a}.json"
    if not path.exists():
        return None
    names = (re.sub(r"[（(].*$", "", n).strip() for n in json.loads(path.read_text(encoding="utf-8"))["cast"])
    return [n for n in dict.fromkeys(names) if n and n != UNKNOWN]


_WAVES = {}


def _wave(m):
    if m.ident not in _WAVES:
        import soundfile as sf
        wave, rate = sf.read(str(m.vocal), dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        if rate != lj.SR:
            import librosa
            wave = librosa.resample(wave, orig_sr=rate, target_sr=lj.SR)
        _WAVES[m.ident] = wave
    return _WAVES[m.ident]


def calls(arm):
    """[(key, system, user, ids, wav or None, audio seconds)]."""
    modality, kind = arm.split("-", 1)
    options = "open" if kind.startswith("open") else "closed"
    out = []
    for ident, a, b in windows():
        m = load(ident)
        spec = MATERIALS[m.kind]
        system = lr.template("window_judge_r3_system.md").substitute(
            scene=spec["scene"], options=(HERE / f"window_judge_r3_options_{options}.md").read_text(encoding="utf-8"),
            audio=(HERE / "window_judge_r3_audio_note.md").read_text(encoding="utf-8") if modality == "audio" else "",
            example="\n".join(EXAMPLE[options]), tracks=TRACKS[options])
        cast = generated_cast(ident, a) if kind == "open_gen" else m.people
        if cast is None:
            raise SystemExit(f"{ident} {a}: no generated cast block yet -- run `cast` first")
        profile, voice, _ = profiles(m, a, b)
        cells = [lr.cell(s) for s in m.speakers]
        shown = shown_rows(m, a, b)
        judged = list(range(a, b))
        wav, seconds = None, 0.0
        if modality == "audio":
            import soundfile as sf
            wave = _wave(m)
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
            seconds = at + hi - lo
            folder = OUT / "av"
            folder.mkdir(parents=True, exist_ok=True)
            wav = folder / f"{ident}-{a}.wav"
            sf.write(str(wav), np.concatenate(pieces), lj.SR, subtype="PCM_16")  # rewritten every time: the layout follows the references
            silent = [f"s{track}" for track in sorted(voice) if not voice[track]]
            profile += ("\n\n音频布局：\n| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
                        + (f"\n\n没有声音样本的轨：{'、'.join(silent)}" if silent else ""))
            when = lambda i: f"{at + float(m.stable[i]['start']) - lo:.1f}"  # noqa: E731
        else:
            when = lambda i: f"{float(m.stable[i]['start']):.1f}"  # noqa: E731
        rows = "\n".join((f"▶ q{i} " if i in judged else "") + f"{when(i)}|{cells[i]}|{str(m.stable[i].get('text', '')).strip()}"
                         for i in shown)
        user = lr.template("window_judge_r3_user.md").substitute(title=spec["title"], cast="、".join(m.display(n) for n in cast) or "（无）",
                                                                 profiles=profile, rows=rows, count=len(judged))
        out.append((f"{ident}-{a}", system, user, [f"q{i}" for i in judged], wav, seconds))
    return out


def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    count = counter.count_text
    wins = windows()
    for ident, a, b in wins:
        m = load(ident)
        labelled = [i for i in range(a, b) if m.leader(i) is not None]
        print(f"  {ident} rows {a}-{b} ({b - a}): labelled {len(labelled)}, unlabelled {b - a - len(labelled)}, with truth {sum(i in m.truth for i in range(a, b))}")
    cast_in = sum(count(s) + count(u) for s, u in (cast_prompt(load(i), a, b) for i, a, b in wins))
    total = cast_in
    print(f"cast blocks: {len(wins)} calls, input {cast_in}")
    for arm in ("text-closed", "text-open", "audio-closed", "audio-open"):
        cs = calls(arm)
        text = sum(count(c[1]) + count(c[2]) for c in cs)
        audio = sum(c[5] for c in cs) * 32
        out_tokens = sum(len(c[3]) for c in cs) * 40
        total += text + audio + out_tokens
        print(f"{arm}: {len(cs)} calls, text input {text}, audio {audio:.0f} ({sum(c[5] for c in cs):.0f} s), output ~{out_tokens}")
        (OUT / f"prompt-{arm}-example.md").write_text(cs[0][1] + "\n\n=====\n\n" + cs[0][2], encoding="utf-8")
        if arm.endswith("open"):
            total += text + audio + out_tokens  # its open_gen twin, about the same size
    print(f"all six arms + cast blocks ~{total:.0f} tokens (open_gen counted as its open twin; a floor: thinking, the "
          f"agent's wrapper, cast blocks' output and retries not counted); counted by {counter.source}")


def _complete(complete, args, ids, block):
    text = ""
    for attempt in range(4):
        try:
            text = complete(*args)
        except SystemExit:
            raise
        except Exception as error:  # noqa: BLE001 - a transient provider error
            print(f"{type(error).__name__} {str(error)[:80]}; retry", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        answer = lr.parse(text, "judge", "id", ids) if block == "judge" else mw._cast(text)
        if answer is not None:
            return answer, text
        print("malformed reply; retry", flush=True)
    return None, text


def cast():
    complete = lr.client_for(lj.TARGET)
    (OUT / "cast").mkdir(parents=True, exist_ok=True)
    for ident, a, b in windows():
        path = OUT / "cast" / f"{ident}-{a}.json"
        if path.exists():
            continue
        answer, raw = _complete(complete, cast_prompt(load(ident), a, b), None, "cast")
        path.write_text(json.dumps({"cast": answer, "raw": raw}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"cast {ident}-{a}: {answer}", flush=True)


def run(arm, shard, shards):
    import llm_retrack_av as av
    complete = av.client_for(lj.TARGET) if arm.startswith("audio") else lr.client_for(lj.TARGET)
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    for key, system, user, ids, wav, _ in calls(arm)[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        args = (system, user, wav) if wav is not None else (system, user)
        answer, raw = _complete(complete, args, ids, "judge")
        # a call that gave up is kept apart, so a rerun tries again and scoring never reads it as "keep everything"
        target = path if answer else path.with_suffix(".failed.json")
        target.write_text(json.dumps({"answer": answer, "raw": raw}, ensure_ascii=False, indent=1), encoding="utf-8")
        if answer:
            path.with_suffix(".failed.json").unlink(missing_ok=True)
        print(f"{arm} {key}: {'ok' if answer else 'GAVE UP'}", flush=True)


# --- scoring ----------------------------------------------------------------------------------------

def declared(arm, folder=None):
    """{(ident, window start): {label: name}} from the `<tracks>` block of each reply (round 7 on; older replies have none)."""
    out = {}
    for path in (folder or OUT / arm).glob("*.json"):
        if path.name.endswith(".failed.json"):
            continue
        ident, a = path.stem.rsplit("-", 1)
        match = re.search(r"<tracks>(.*?)</tracks>", json.loads(path.read_text(encoding="utf-8")).get("raw") or "", re.S)
        if match:
            pairs = (line.split("=", 1) for line in match.group(1).splitlines() if "=" in line)
            out[(ident, int(a))] = {k.strip(): v.strip() for k, v in pairs}
    return out


def answers(arm, folder=None):
    """{(ident, window start, row): pick} from `folder` (default: this round's `OUT / arm`)."""
    out = {}
    for path in (folder or OUT / arm).glob("*.json"):
        if path.name.endswith(".failed.json"):
            continue
        ident, a = path.stem.rsplit("-", 1)
        for k, v in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items():
            out[(ident, int(a), int(k[1:]))] = str((v or {}).get("track", "")).strip()
    return out


def outcome(arm_answers):
    """{(ident, a, row): {"named": person or None, "anonymous": person or None, "new": bool}} for every judged row with truth."""
    out = {}
    for ident, a, b in windows():
        m = load(ident)
        owned = set(m.owner.values())
        references = profiles(m, a, b)[2]
        labels = defaultdict(Counter)
        for i in range(a, b):
            pick = arm_answers.get((ident, a, i), "")
            if pick.startswith("@") and i in m.truth:
                for p in m.truth[i]:
                    labels[pick][p] += 1 / len(m.truth[i])
        # a tie goes to a person without a track of the file (the one a new track can be right for); counts are rows,
        # where the owners are by seconds -- a small difference, stated in 5.14
        majority = {lab: sorted(c.items(), key=lambda kv: (-kv[1], kv[0] in owned, kv[0]))[0][0] for lab, c in labels.items()}
        for i in range(a, b):
            if i not in m.truth or i in references:
                continue
            lead = m.leader(i)
            before = m.owner.get(lead) if lead is not None else None
            pick = arm_answers.get((ident, a, i), "")
            if re.fullmatch(r"s\d+", pick) and int(pick[1:]) != lead:
                named = anonymous = m.owner.get(int(pick[1:]))
                new = False
            elif pick.startswith("@"):
                named, anonymous, new = m.person_of(pick), majority.get(pick), True
                # strict: a new track for someone who owns a track of the file splits that person
                named = None if named in owned else named
                anonymous = None if anonymous in owned else anonymous
            else:
                named = anonymous = before
                new = False
            out[(ident, a, i)] = {"named": named, "anonymous": anonymous, "new": new, "before": before, "pick": pick,
                                  "filled": bool(re.fullmatch(r"s\d+", pick) and int(pick[1:]) != lead) or pick.startswith("@"),
                                  "truth": m.truth[i], "labelled": lead is not None, "cluster": (ident, a, lead),
                                  "kind": m.kind, "label_named": m.person_of(pick) if pick.startswith("@") else None}
    return out


def new_track_report(res):
    new = [k for k, v in res.items() if v["new"]]
    if not new:
        return f"{'':16s}new tracks: none"
    groups = defaultdict(list)
    for k in new:
        groups[(k[0], k[1], res[k]["pick"])].append(k)
    purity = sum(max(Counter(p for k in g for p in res[k]["truth"]).values()) for g in groups.values()) / len(new)
    named_ok = sum(res[k]["label_named"] in res[k]["truth"] for k in new)
    return (f"{'':16s}new tracks: {len(groups)} labels over {len(new)} rows ({len(new) / len(groups):.1f} rows each), row purity "
            f"{purity:.0%}, label names the true person on {named_ok} rows")


def unlabelled_report(res):
    miss = [k for k, v in res.items() if not v["labelled"]]
    filled = [k for k in miss if res[k]["filled"]]
    return (f"{'':16s}unlabelled rows {len(miss)}: labelled {len(filled)}, right (strict anonymous) "
            f"{sum(res[k]['anonymous'] in res[k]['truth'] for k in filled)}")


def score():
    wins = windows()
    rng = np.random.default_rng(SEED)
    base = outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    before = np.array([base[k]["before"] in base[k]["truth"] for k in keys], float)
    kind_of = np.array([base[k]["kind"] for k in keys])
    window_of = [k[:2] for k in keys]
    draws_cache = {}

    def interval(x, mask=None):
        mask = np.ones(len(keys), bool) if mask is None else mask
        if mask.tobytes() not in draws_cache:
            groups = defaultdict(list)
            for j in np.flatnonzero(mask):
                groups[base[keys[j]]["cluster"]].append(j)
            groups = list(groups.values())
            draws_cache[mask.tobytes()] = [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(5000)]
        sims = [x[d].mean() for d in draws_cache[mask.tobytes()]]
        return f"{x[mask].mean():+.1f} [{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]"

    per_window = lambda x: ", ".join(f"{i}-{a} {x[[w == (i, a) for w in window_of]].mean():+.1f}" for i, a, _ in wins)  # noqa: E731
    mano = kind_of == "manosaba"
    print(f"labelled rows with truth {len(keys)} in {len(wins)} windows (reference rows inside a window left out); production right "
          f"{before.mean():.1%}; " + "; ".join(f"{kd} {np.mean(before[kind_of == kd]):.1%} of {(kind_of == kd).sum()}" for kd in MATERIALS))

    gen = {(i, a): generated_cast(i, a) for i, a, _ in wins}
    if all(v is not None for v in gen.values()):
        for kd in MATERIALS:
            hit = said = real = 0
            for ident, a, b in wins:
                m = load(ident)
                if m.kind != kd:
                    continue
                truth = {p for i in shown_rows(m, a, b) if i in m.truth for p in m.truth[i]}
                named = {m.person_of(n) for n in gen[(ident, a)]} - {None}
                hit, said, real = hit + len(named & truth), said + len(named), real + len(truth)
            print(f"generated cast blocks, {kd}: {hit}/{said} names right, {hit}/{real} true speakers of the windows found")

    right = {}
    for arm in ARMS:
        got = answers(arm)
        failed = sorted(p.stem for p in (OUT / arm).glob("*.failed.json"))
        if not got:
            continue
        missing = [f"{i}-{a}" for i, a, _ in wins if not any(k[:2] == (i, a) for k in got)]
        res = outcome(got)
        right[arm] = {rule: np.array([res[k][rule] in res[k]["truth"] for k in keys], float) for rule in ("anonymous", "named")}
        change = 100 * (right[arm]["anonymous"] - before)
        fixed = int(((right[arm]["anonymous"] == 1) & (before == 0)).sum())
        broken = int(((right[arm]["anonymous"] == 0) & (before == 1)).sum())
        print(f"{arm:14s}: windows without a reply {missing or 'none'} (gave up: {failed or 'none'})")
        print(f"{'':16s}strict anonymous: 魔裁 {interval(change, mano)}, hina+onetonton {interval(change, ~mano)}, all {interval(change)}; "
              f"strict named, 魔裁 {interval(100 * (right[arm]['named'] - before), mano)}; fixed {fixed}, broken {broken}")
        print(new_track_report(res) + "\n" + unlabelled_report(res))
    for modality in MODALITIES:
        a, b = f"{modality}-open", f"{modality}-closed"
        if a in right and b in right:
            d = 100 * (right[a]["anonymous"] - right[b]["anonymous"])
            print(f"MAIN {a} - {b}, strict anonymous, 魔裁: {interval(d, mano)}; per window {per_window(d)}; "
                  f"hina+onetonton (the option's side effect) {interval(d, ~mano)}; named, 魔裁 "
                  f"{interval(100 * (right[a]['named'] - right[b]['named']), mano)}")
        a, b = f"{modality}-open_gen", f"{modality}-open"
        if a in right and b in right:
            d = 100 * (right[a]["anonymous"] - right[b]["anonymous"])
            print(f"paired {a} - {b}: 魔裁 {interval(d, mano)}, hina+onetonton {interval(d, ~mano)}")
    for kind in KINDS:
        a, b = f"audio-{kind}", f"text-{kind}"
        if a in right and b in right:
            d = 100 * (right[a]["anonymous"] - right[b]["anonymous"])
            print(f"paired {a} - {b}: all {interval(d)}; 魔裁 {interval(d, mano)}; per window {per_window(d)}")
    near_right = total = 0
    for ident, a, b in wins:
        m = load(ident)
        labelled = [i for i in range(len(m.stable)) if m.leader(i) is not None]
        centre = lambda i: (float(m.stable[i]["start"]) + float(m.stable[i]["end"])) / 2  # noqa: E731
        for i in range(a, b):
            if m.leader(i) is None and i in m.truth:
                near = min(labelled, key=lambda j: abs(centre(j) - centre(i)))
                near_right += m.owner.get(m.leader(near)) in m.truth[i]
                total += 1
    print(f"unlabelled rows: the nearest labelled row's track is right on {near_right}/{total}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "cast", "run", "score"))
    parser.add_argument("--arm", choices=ARMS, default="text-closed")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    {"build": build, "cast": cast, "score": score}.get(args.command, lambda: run(args.arm, args.shard, args.shards))()


if __name__ == "__main__":
    main()
