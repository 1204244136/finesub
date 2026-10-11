"""Speaker judge round 10 (owner 2026-10-08): correction and the speaker judge in one call, the third line beside
round 9's two (the judge on raw ASR, the judge on the corrected text).

One call per production correction window does both. The base is production's correction prompt for that window as
round 9's correction sent it (production's dry run on round 9's slices, with the research blocks the dry run leaves out
spliced back from round 9's exchange). Owner 2026-10-08: the speaker part is two more columns of `<translated>`, not a
second answer block -- `spk` (who says the row: the input's cell copied, another track, or `@new`) and `spk_note` (a
reason, only on rows whose `spk` differs from the input). Besides that: each source's acoustic cell as a `spk` column of
the CSV input, a section on judging speakers in the system prompt (round 9's judge wording), the cast block and track
profiles in the user message, the voice samples appended to the window clip, and a small `<tracks>` block naming them.
Criteria: the marks plan, 「第三条线：纠错与认人合在一次调用里」, written first.

Both modalities run on all eleven windows (twelve correction windows: hina-71's slice is two). Calls go to round 9's
pinned agy target. The request round production offers (`<requested_entries>` / `<search_queries>`) is not served:
a reply that asks gets no fill and is retried, counted.

    python tools/diarization_explore/window_judge_r10.py render      # production's dry run on copies of round 9's slices
    python tools/diarization_explore/window_judge_r10.py build       # joint prompts and clips, token counts, examples
    python tools/diarization_explore/window_judge_r10.py run --modality audio --pass 1 [--only hina-46] [--shard 0 --shards 2]
    python tools/diarization_explore/window_judge_r10.py score
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_judge_r9 as r9  # noqa: E402

r3, lr, lj = r9.r3, r9.lr, r9.lj
ROOT = lr.OUT / "r10"
MODALITIES = r9.MODALITIES
HEADER = "local_id|start|duration|gap|text"
JOINT_HEADER = "local_id|start|duration|gap|spk|text"
OUT_HEADER = "type|position|duration|gap|corrected_text|translation|conf|char_count|note"
JOINT_OUT_HEADER = "type|position|spk|spk_note|duration|gap|corrected_text|translation|conf|char_count|note"

#: the cells the production oneshot's CSV examples get, keyed by each example's first row; rows not listed are s0.
#: The main example is a streamer watching a PV: narration (1-7) and the PV's characters (28-42) on other tracks, the
#: two teasing children (36 and 38, 37 -- its note says "another person") on two more, the streamer's aside 41 back on
#: s0; 18 is the hallucinated run, no track. The cross-speaker example's answerer is a second track. Every cell agrees
#: with what the example's notes say, except one acoustic error the output fixes (ONESHOT_FIX): the streamer's aside 41
#: lies under the PV and is on the PV's track; every other output row copies its cell.
ONESHOT_CELLS = {
    "かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた": {
        **{i: "s1" for i in (*range(1, 8), *range(28, 43))}, 36: "s2", 37: "s3", 38: "s2", 18: "??"},
    "世界に溶け込もうと": {},
    "なんかあのイベント報酬がさ": {},
    "だから結局その仕様変更は": {},
    "うちらが知ってるのは": {},
    "世界に溶けこもうとその子は": {},
    "これ絶対課金するわ": {2: "s1"},
    "啊": {},
}
ONESHOT_FIX = {("かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた", 41):
               ("s0", "主播在 PV 中间插话、评论 PV 的口吻，与 19–25 是同一个人")}




def template(name):
    return lr.template(f"window_judge_r10_{name}.md")


# --- production's correction prompts ------------------------------------------------------------------

def render_dir(w):
    return ROOT / "render" / r9.slice_dir(w).name


def render():
    """Production's dry run for both media on a copy of each round-9 slice (round 9's artifacts stay untouched). The
    default preset renders the request round for text as well; agy's text cell would render tool use instead."""
    for w in r9.plan():
        source, folder = r9.slice_dir(w), render_dir(w)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / f"{source.name}-stable.json", folder / f"{source.name}-stable.json")
        if not (folder / "artifacts").exists():
            shutil.copytree(source / "artifacts", folder / "artifacts", ignore=shutil.ignore_patterns("exchanges", "clips"))
        for media in MODALITIES:
            command = [sys.executable, "-m", "finesub.llm.correction_translation", str(folder / f"{source.name}-stable.json"),
                       "--audio", str(r9.source_audio(w["ident"])), "--media", media,
                       "-o", str(folder / f"{source.name}.srt"), "--task-artifact-dir", str(folder / "artifacts"),
                       "--prompt-dir", str(folder / f"prompts-{media}"), "--task-id", r9.source_audio(w["ident"]).stem]
            if w["ident"] not in r9.WHOLE:
                command += ["--fast", "off"]
            with open(folder / "render.log", "a", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{folder.name}: {', '.join(p.name for p in sorted((folder / 'prompts-audio').glob('correction-*.txt')))}")


#: the user-message span the dry run leaves empty (background research, window summary, pre-injected knowledge entries)
RESEARCH = ("通用背景知识和术语（", "此前所有窗口的累积建议（")


def sent(w, chunk):
    """(system, user) round 9's correction sent for this window, from its first attempt's exchange record."""
    path, = sorted((r9.slice_dir(w) / "artifacts" / "exchanges").glob(f"*-correction-{chunk}-attempt0.md"))
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n## 请求（system）\n\n", 1)[1]
    system, user = text.split("\n## 请求（user）\n\n", 1)
    return system.rstrip("\n"), user.split("\n## 模型响应", 1)[0].rstrip("\n")


def chunks(w, media):
    """[(chunk id, system, user)]: production's rendered correction prompts for this slice with round 9's research spliced
    in. With audio this is what round 9's correction sent, checked (hina-71's second window differs: round 9 ran it
    after the first and its advice ledger was not empty)."""
    out = []
    a, b = RESEARCH
    for path in sorted((render_dir(w) / f"prompts-{media}").glob("correction-*.txt")):
        if not re.fullmatch(r"correction-\d+", path.stem):
            continue  # the basic-tier variant is not what the quality cell sends
        chunk = path.stem.split("-")[1]
        system, user = path.read_text(encoding="utf-8").replace("\r\n", "\n").removeprefix("[system]\n").split("\n[user]\n", 1)
        system_sent, user_sent = sent(w, chunk)
        assert system.rstrip("\n") == system_sent or media == "text", (w["ident"], chunk)
        user = user[:user.index(a)] + user_sent[user_sent.index(a):user_sent.index(b)] + user[user.index(b):]
        if media == "audio" and user.rstrip("\n") != user_sent:
            print(f"note: {w['ident']} {chunk}: the audio prompt differs from what round 9 sent outside the research blocks")
        out.append((chunk, system, user))
    return out


# --- the joint prompt ---------------------------------------------------------------------------------

def _rows(block):
    """[(local id, start, text)] of a rendered CSV block's data rows."""
    out = []
    for line in block.splitlines():
        if line in (HEADER, JOINT_HEADER) or not line.strip():
            continue
        cols = line.split("|", 4)
        out.append((int(cols[0]), float(cols[1]), cols[4]))
    return out


def _with_cells(block, cell_of):
    lines = []
    for line in block.split("\n"):
        if line == HEADER:
            lines.append(JOINT_HEADER)
        elif line.strip():
            cols = line.split("|", 4)
            lines.append("|".join(cols[:4] + [cell_of(int(cols[0])), cols[4]]))
        else:
            lines.append(line)
    return "\n".join(lines)


def _blocks(text, tag):
    return list(re.finditer(rf"(?m)^<{tag}>\n(.*?)^</{tag}>$", text, re.S))


def _replace_once(text, old, new):
    assert text.count(old) == 1, (old[:60], text.count(old))
    return text.replace(old, new)


def _unjoint(text):
    """The CSV columns this round adds, taken back out: what `_check` compares with production's text."""
    lines = []
    for line in text.split("\n"):
        if line == JOINT_HEADER:
            line = HEADER
        elif re.match(r"-?\d+\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|", line):
            cols = line.split("|", 5)
            line = "|".join(cols[:4] + cols[5:])
        elif line.startswith("sub|"):
            cols = line.split("|", 4)
            line = "|".join(cols[:2] + cols[4:])
        lines.append(line)
    return "\n".join(lines).replace(JOINT_OUT_HEADER, OUT_HEADER)


def _check(joint, original, edits):
    """Production's text stays byte-for-byte: undo the phrase edits and the added columns, compare."""
    undone = joint
    for old, new in edits:
        undone = undone.replace(new, old) if old else undone.replace(new, "", 1)
    assert _unjoint(undone) == original, "the joint prompt changed production's text beyond the listed edits"


def _apply(text, edits):
    for old, new in edits:
        if old:
            text = text.replace(old, new) if old.endswith("列") else _replace_once(text, old, new)
    return text


def _oneshot_columns(system):
    """The oneshot's CSV examples with the input `spk` column, and their output rows with `spk` / `spk_note`: an output row
    copies its first source's cell, except the one acoustic error ONESHOT_FIX corrects."""
    out, at, cells, example = [], 0, {}, None
    for match in sorted(_blocks(system, "asr_result") + _blocks(system, "preceding_context"), key=lambda m: m.start()):
        out.append(_sub_rows(system[at:match.start(1)], cells, example))
        if match.group(0).startswith("<asr_result>"):
            example = _rows(match.group(1))[0][2]
            assert example in ONESHOT_CELLS, f"unknown oneshot example starting {example!r}"
            cells = ONESHOT_CELLS[example]
            out.append(_with_cells(match.group(1), lambda i: cells.get(i, "s0")))
        else:
            out.append(_with_cells(match.group(1), lambda i: "s0"))
        at = match.end(1)
    fixed = _sub_rows(system[at:], cells, example)
    assert all(f"|{spk}|{note}|" in "".join(out) + fixed for spk, note in ONESHOT_FIX.values()), "an ONESHOT_FIX row was not found"
    return "".join(out) + fixed


def _sub_rows(text, cells, example):
    lines = []
    for line in text.split("\n"):
        if line == OUT_HEADER:
            line = JOINT_OUT_HEADER
        elif line.startswith("sub|"):
            cols = line.split("|", 2)
            first = int(cols[1].split(",")[0])
            spk, note = ONESHOT_FIX.get((example, first), (cells.get(first, "s0"), ""))
            line = "|".join(cols[:2] + [spk, note, cols[2]])
        lines.append(line)
    return "\n".join(lines)


def joint_system(system, modality, m):
    spec = r3.MATERIALS[m.kind]
    section = template("speaker").substitute(
        scene=spec["scene"], audio=template("audio_note").template if modality == "audio" else "",
        audio_inventory="；带音频时，音频末尾还接着每条轨的声音样本" if modality == "audio" else "").rstrip("\n")
    edits = [(f"`{HEADER}`", f"`{JOINT_HEADER}`"),
             ("片段尾部离下一段话的gap|文本`", "片段尾部离下一段话的gap|声学层给这一段挂的说话人轨号（见「说话人复核」）|文本`"),
             (f"`{OUT_HEADER}`", f"`{JOINT_OUT_HEADER}`"),
             ("9 列", "11 列"),
             ("   - `duration`：本条字幕跨度秒数",
              "   - `spk`、`spk_note`：这一行是谁说的，改了轨号时写理由；写法见「说话人复核」。`spk` 不得为空，`spk_note` 照抄时留空。\n"
              "   - `duration`：本条字幕跨度秒数"),
             ("\n完整 oneshot。", "\n" + section + "\n\n完整 oneshot。"),
             ("corrected_text、translation、note 中的 `|` 改用全角", "corrected_text、translation、spk_note、note 中的 `|` 改用全角"),
             ("写各自列，理由只写 note。", "写各自列，理由只写 note（改轨号的理由写 `spk_note`）。"),
             ("局部逐行推理一律由 translated 内的 `#` 注释承载", "局部逐行推理一律由 translated 内的 `#` 注释承载（改轨号的理由写 `spk_note`）"),
             ("type/conf/char_count/note/`#` 注释均不进入 SRT", "type/spk/spk_note/conf/char_count/note/`#` 注释均不进入 SRT")]
    if modality == "audio":
        edits.append(("padding 区间内可能有语音，但它不属于本窗口的输入范围，不要为其输出字幕。",
                      "padding 区间内可能有语音，但它不属于本窗口的输入范围，不要为其输出字幕。本任务的音频在剪辑之后还接着各轨的声音样本，见「说话人复核」。"))
    joint = _apply(_oneshot_columns(system), edits)
    _check(joint, system, edits)
    return joint


_norm = lambda s: re.sub(r"\s+", "", str(s).replace("|", "｜"))  # noqa: E731


def window_rows(w, m, user):
    """[file row of local id 1..n]: where the window's CSV sits in the file, found by its text (a rendered plan can
    differ from the window records of round 9's run, so the prompt itself is the reference)."""
    texts = [_norm(text) for _, _, text in _rows(_blocks(user, "asr_result")[0].group(1))]
    lo, hi = r9.slice_bounds(w)
    hi = len(m.stable) if hi is None else min(hi, len(m.stable))
    found = [i for i in range(lo, hi - len(texts) + 1)
             if all(_norm(m.stable[i + k].get("text", "")) == t for k, t in enumerate(texts))]
    assert len(found) == 1, (w["ident"], found)
    return list(range(found[0], found[0] + len(texts)))


def clip_range(w, m, rows):
    """Production's clip for a window (`compute_clip_range`; the slice is the whole file production saw)."""
    from finesub.media.clips import CLIP_EDGE_PAD_SECONDS, CLIP_PAD_SECONDS
    from finesub.media.ffmpeg import probe_media_duration
    lo, hi = r9.slice_bounds(w)
    last = (len(m.stable) if hi is None else min(hi, len(m.stable))) - 1
    start = max(0.0, float(m.stable[rows[0]]["start"]) - (CLIP_EDGE_PAD_SECONDS if rows[0] == lo else CLIP_PAD_SECONDS))
    end = float(m.stable[rows[-1]]["end"]) + (CLIP_EDGE_PAD_SECONDS if rows[-1] == last else CLIP_PAD_SECONDS)
    return start, max(min(end, probe_media_duration(r9.source_audio(w["ident"]))), float(m.stable[rows[-1]]["end"]))


def _cells_in(block, rows, m):
    """`block` with each row's acoustic cell; a context row (id <= 0) is the row that many places before the window."""
    row_of = lambda local: rows[local - 1] if local >= 1 else rows[0] + local - 1  # noqa: E731
    for local, _, text in _rows(block):
        assert _norm(m.stable[row_of(local)].get("text", "")) == _norm(text), (m.ident, local, text)
    return _with_cells(block, lambda local: lr.cell(m.speakers[row_of(local)]))


def joint_user(user, w, rows, m, profile):
    (asr,), (before,) = _blocks(user, "asr_result"), _blocks(user, "preceding_context")
    assert len(_rows(asr.group(1))) == len(rows)
    columns = (user[:before.start(1)] + _cells_in(before.group(1), rows, m) + user[before.end(1):asr.start(1)]
               + _cells_in(asr.group(1), rows, m) + user[asr.end(1):])
    cast = r9.truth_cast(m, w["a"], w["b"])
    block = template("user").substitute(title=r3.MATERIALS[m.kind]["title"], cast="、".join(m.display(n) for n in cast) or "（无）",
                                        profiles=profile)
    edits = [(f"`{OUT_HEADER}`", f"`{JOINT_OUT_HEADER}`"),
             ("9 列", "11 列"),
             ("\n只读前文 ASR（", "\n" + block + "只读前文 ASR（"),
             ("5. `</translated>` 之后必须输出有且仅有一个 `<next_advice>",
              "5. `</translated>` 之后先输出 `<tracks>`（见 system「说话人复核」），再输出有且仅有一个 `<next_advice>"),
             ("其后 `<next_advice>`、`<keep_entries>`", "其后 `<tracks>`、`<next_advice>`、`<keep_entries>`")]
    joint = _apply(columns, edits)
    _check(joint, user, edits)
    return joint


def _decode(path, start, seconds):
    from finesub.media.ffmpeg import resolve_ffmpeg
    raw = subprocess.run([resolve_ffmpeg(), "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{seconds:.3f}",
                          "-vn", "-ac", "1", "-ar", str(lj.SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def joint_audio(w, m, rows, voice, target):
    """Production's window clip with the voice samples appended, all cut from the source production hears (round 9's
    judge heard the vocal stem): (aac path, layout text, seconds). Appended, so the CSV's clip times stay production's."""
    import soundfile as sf
    from finesub.media.ffmpeg import extract_audio_clip
    source = r9.source_audio(w["ident"])
    meta = target.with_suffix(".json")
    if meta.exists():
        cached = json.loads(meta.read_text(encoding="utf-8"))
        return target, cached["layout"], cached["seconds"]
    start, end = clip_range(w, m, rows)
    pieces = [_decode(source, start, end - start)]
    cursor = len(pieces[0]) / lj.SR
    layout = [f"| 本窗口的剪辑 | 0.0–{cursor:.1f} 秒 |"]
    gap = np.zeros(int(lj.GAP * lj.SR), dtype=np.float32)
    for track in sorted(voice):
        for i in voice[track]:
            piece = _decode(source, float(m.stable[i]["start"]), float(m.stable[i]["end"]) - float(m.stable[i]["start"]))
            pieces += [gap, piece]
            cursor += lj.GAP
            layout.append(f"| s{track} 的声音样本 | {cursor:.1f}–{cursor + len(piece) / lj.SR:.1f} 秒 |")
            cursor += len(piece) / lj.SR
    target.parent.mkdir(parents=True, exist_ok=True)
    wav = target.with_suffix(".wav")
    sf.write(str(wav), np.concatenate(pieces), lj.SR, subtype="PCM_16")
    extract_audio_clip(wav, 0.0, cursor, target)
    wav.unlink()
    silent = [f"s{track}" for track in sorted(voice) if not voice[track]]
    text = ("\n\n音频布局：\n| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
            + (f"\n\n没有声音样本的轨：{'、'.join(silent)}" if silent else ""))
    meta.write_text(json.dumps({"layout": text, "seconds": cursor, "clip": [start, end]}, ensure_ascii=False), encoding="utf-8")
    return target, text, cursor


def calls(modality, only=None):
    """[(key, system, user, rows, audio path or None, audio seconds)]: one per correction window."""
    out = []
    for w in r9.plan():
        if only and w["ident"] not in only:
            continue
        m = r9.load(w["ident"])
        profile, voice, _ = r3.profiles(m, w["a"], w["b"])
        for chunk, system, user in chunks(w, modality):
            rows = window_rows(w, m, user)
            key = f"{w['ident']}-{w['a']}-{chunk}"
            audio, seconds, text = None, 0.0, profile
            if modality == "audio":
                start = float(_rows(_blocks(user, "asr_result")[0].group(1))[0][1])
                assert abs(float(m.stable[rows[0]]["start"]) - clip_range(w, m, rows)[0] - start) <= 0.051, key
                audio, layout, seconds = joint_audio(w, m, rows, voice, ROOT / "build" / f"{key}.aac")
                text = profile + layout
            out.append((key, joint_system(system, modality, m), joint_user(user, w, rows, m, text), rows, audio, seconds))
    return out


def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    total = 0
    for modality in MODALITIES:
        cs = calls(modality)
        text = sum(counter.count_text(c[1]) + counter.count_text(c[2]) for c in cs)
        audio = sum(c[5] for c in cs) * 32
        shell = 35000 * len(cs)  # agy's agent shell per call (round 9's measured usage)
        total += text + audio + shell
        print(f"{modality}: {len(cs)} calls, text {text}, audio {audio:.0f} ({sum(c[5] for c in cs):.0f} s), agy shell ~{shell}, "
              f"rows {sum(len(c[3]) for c in cs)}")
        key, system, user, *_ = next(c for c in cs if c[0].startswith("hina-46"))
        (ROOT / f"prompt-{modality}-example.md").write_text(f"[system]\n{system}\n[user]\n{user}", encoding="utf-8")
    print(f"one pass, input ~{total:.0f} tokens (a floor: retries not counted); counted by {counter.source}")


# --- running and parsing ------------------------------------------------------------------------------

ASKED = re.compile(r"<(requested_entries|search_queries)>\s*\S", re.S)


def parse(text, rows, m):
    """((rows, discarded file rows, tracks text), None) or (None, error). `<translated>` loses its two speaker columns and
    goes through production's own validator; each subtitle row keeps its `spk` / `spk_note`. A row that left the speaker
    columns out keeps its acoustic labels (counted), not a retry."""
    from finesub.llm.chunking import SubtitleSegment
    from finesub.llm.output_protocol import extract_top_level_tagged_blocks, validate_translated_csv_text
    blocks = extract_top_level_tagged_blocks(text or "", "translated")
    if len(blocks) != 1:
        return None, f"{len(blocks)} <translated> blocks"
    lines, speaker = [], {}
    for line in blocks[0].split("\n"):
        if line.strip() == JOINT_OUT_HEADER:
            line = OUT_HEADER
        elif line.strip() == OUT_HEADER:
            return None, "header without the speaker columns"
        elif line.strip().lower().startswith("sub|"):
            cols = line.split("|")
            # duration and gap are the first two numbers after position; what lies between is spk (and spk_note, which
            # may hold a stray `|`); production's own split keeps a `|` inside note
            at = next((j for j in range(2, len(cols) - 1) if NUMBER.fullmatch(cols[j].strip()) and NUMBER.fullmatch(cols[j + 1].strip())), None)
            if at is not None and at >= 3:
                if VOID not in line:
                    speaker[",".join(c.strip() for c in cols[1].split(","))] = (cols[2].strip(), "｜".join(cols[3:at]).strip())
                line = "|".join(cols[:2] + cols[at:])
        lines.append(line)
    result = validate_translated_csv_text(text.replace(blocks[0], "\n".join(lines), 1), [
        SubtitleSegment(id=str(k + 1), start=float(m.stable[i]["start"]), end=float(m.stable[i]["end"]), text=str(m.stable[i].get("text", "")))
        for k, i in enumerate(rows)], require_singles=False, require_headers=True, forbid_start_column=True)
    if not result.ok:
        return None, "translated: " + "; ".join(result.errors[:3])
    to_row = lambda local_id: rows[int(local_id) - 1]  # noqa: E731
    out = [(tuple(to_row(s) for s in seg.source_ids), seg.corrected_text, *speaker.get(",".join(seg.source_ids), ("", "")),
            seg.translation, seg.note) for seg in result.segments]
    tracks = extract_top_level_tagged_blocks(text, "tracks")
    return (out, sorted(to_row(s) for s in result.discarded_ids), tracks[0].strip() if tracks else ""), None


VOID = "<void>"
NUMBER = re.compile(r"\d+(?:\.\d+)?")
SPK = re.compile(r"\?\?|s\d+(?:\+s\d+)*|@\S+")


def pass_dir(number, modality):
    return ROOT / f"pass{number}" / modality


def run(modality, number, shard, shards, only=None):
    complete = r9.client_for(modality)
    folder = pass_dir(number, modality)
    folder.mkdir(parents=True, exist_ok=True)
    for key, system, user, rows, audio, _ in calls(modality, only)[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        m = r9.load(key.rsplit("-", 2)[0])
        got, text, attempts, notes = None, "", [], []
        for attempt in range(4):
            try:
                reply = complete(system, user, audio)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            text = reply.content
            attempts.append(dict(reply.raw_response.get("usage") or {}, finish_reason=str(getattr(reply, "finish_reason", "") or "")))
            got, error = parse(text, rows, m)
            if got is not None:
                break
            # the request round is not served: a reply that asks is retried like a malformed one, and counted
            notes.append("asked for inputs" if ASKED.search(text or "") and "<translated>" not in (text or "") else error)
            print(f"{key}: {notes[-1]}; retry", flush=True)
        payload = {"rows": rows, "raw": text, "usage": attempts, "retries": notes}
        if got is not None:
            segments, discarded, tracks = got
            payload.update(segments=[{"sources": list(s), "text": t, "spk": a, "spk_note": y, "translation": tr, "note": n}
                                     for s, t, a, y, tr, n in segments],
                           discarded=discarded, tracks=tracks)
        target = path if got is not None else path.with_suffix(".failed.json")
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        if got is not None:
            path.with_suffix(".failed.json").unlink(missing_ok=True)
        print(f"{modality} pass {number} {key}: {'ok' if got is not None else 'GAVE UP'} after {len(attempts)} replies", flush=True)


# --- scoring ------------------------------------------------------------------------------------------

def final_rows(modality, number, w):
    """([(file rows, corrected text, spk, spk_note, chunk)] the slice ends on, the file rows the correction dropped):
    production's newest-wins merge across the slice's windows. (None, None) when a window has no accepted reply."""
    from finesub.llm.output_protocol import TranslatedCsvSegment, merge_translated_csv_windows
    m = r9.load(w["ident"])
    merged, meta, dropped, folder = [], {}, set(), pass_dir(number, modality)
    if any(folder.glob(f"{w['ident']}-{w['a']}-*.failed.json")) or not any(folder.glob(f"{w['ident']}-{w['a']}-*.json")):
        return None, None
    for path in sorted(folder.glob(f"{w['ident']}-{w['a']}-*.json")):
        reply = json.loads(path.read_text(encoding="utf-8"))
        chunk = path.stem.rsplit("-", 1)[1]
        new = []
        for seg in reply["segments"]:
            s = seg["sources"]
            row = TranslatedCsvSegment(source_ids=tuple(str(i) for i in s), start=float(m.stable[s[0]]["start"]),
                                       end=float(m.stable[s[-1]]["end"]), corrected_text=seg["text"], translation=seg["text"])
            meta[id(row)] = (s, seg["text"], seg["spk"], seg["spk_note"], chunk)
            new.append(row)
        merged = merge_translated_csv_windows(merged, [str(i) for i in reply["rows"]], new)
        dropped |= set(reply["discarded"])
    out = [meta[id(row)] for row in merged]
    covered = {i for sources, *_ in out for i in sources}
    return out, dropped - covered


def pick_of(spk, chunk):
    """A row's `spk` as a pick: a track or a new label moves its sources; a copied `??` / `s5+s4` or anything else leaves
    them on their acoustic labels. A new label belongs to its correction window (never merged across windows)."""
    if re.fullmatch(r"s\d+", spk):
        return spk
    if spk.startswith("@") and len(spk) > 1:
        return f"@{chunk}:{spk[1:]}"
    return None


def joint_picks(modality, number):
    """{(ident, a, file row): pick}: each subtitle row's `spk` spread over its sources inside the judge window."""
    out = {}
    for w in r9.plan():
        rows, _ = final_rows(modality, number, w)
        for sources, _, spk, _, chunk in rows or []:
            pick = pick_of(spk, chunk)
            if pick:
                out.update({(w["ident"], w["a"], s): pick for s in sources if w["a"] <= s < w["b"]})
    return out


def corrected_prompt_picks(modality, number=1):
    """Round 9's corrected line under the prompt's meaning of `keep` (a merged row's keep = its shown main track)."""
    out = dict(r9.picks_of(f"{modality}-corrected", number))
    answers = r3.answers(f"{modality}-corrected", r9.pass_dir(number, f"{modality}-corrected"))
    for w in r9.plan():
        for q, sources, _, cell, _ in r9.rows_for(w, "corrected"):
            if q and len(sources) > 1 and answers.get((w["ident"], w["a"], int(q[1:]))) == "keep" and cell != "??":
                out.update({(w["ident"], w["a"], s): cell.split("+")[0] for s in sources if w["a"] <= s < w["b"]})
    return out


def score():
    rng = np.random.default_rng(r9.SEED)
    r3.OUT = r9.ROOT / "build"
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    clusters = [base[k]["cluster"] for k in keys]
    group = {(w["ident"], w["a"]): w["group"] for w in r9.plan()}
    fresh = np.array([group[k[:2]] in r9.FRESH for k in keys])
    short = np.array([r9.backchannel(r9.load(k[0]), k[2]) for k in keys])
    main = fresh & ~short

    def labelled_right(picks):
        right = {}
        for ident, a, b in r9.windows():
            m = r9.load(ident)
            final = r9.ps.final_labels(r3, m, ident, a, b, picks)
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            found = r9.ps.match(rows, final, m.truth)
            right.update({(ident, a, i): found.get(final[i]) in m.truth[i] for i in rows})
        return np.array([right[k] for k in keys], float)

    def interval(d, mask):
        groups = {}
        for j in np.flatnonzero(mask):
            groups.setdefault(clusters[j], []).append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return d[mask].mean(), np.percentile(sims, 2.5), np.percentile(sims, 97.5)

    print(f"labelled rows with truth {len(keys)} (main readout: fresh, not back-channel {main.sum()}); "
          f"production right {labelled_right({})[main].mean():.1%}")
    passes = sorted(int(p.name[4:]) for p in ROOT.glob("pass*") if p.name[4:].isdigit())
    for modality in MODALITIES:
        for number in passes:
            if not pass_dir(number, modality).exists():
                continue
            missing = [w["ident"] for w in r9.plan() if final_rows(modality, number, w)[0] is None]
            if missing:
                print(f"{modality} pass {number}: windows without an accepted reply {missing}; not scored")
                continue
            joint = labelled_right(joint_picks(modality, number))
            others = {"raw": labelled_right(r9.picks_of(f"{modality}-raw", 1)),
                      "corrected (keep = shown main)": labelled_right(corrected_prompt_picks(modality))}
            print(f"{modality} joint pass {number}: right {joint[main].mean():.1%} on the main readout "
                  + ", ".join(f"({name} {col[main].mean():.1%})" for name, col in others.items()))
            for name, col in others.items():
                d = 100 * (joint - col)
                for label, mask in (("MAIN fresh, not back-channel", main), ("fresh, all rows", fresh), ("fresh, back-channel", fresh & short),
                                    ("merged (in sample), not back-channel", ~fresh & ~short)):
                    v, lo, hi = interval(d, mask)
                    print(f"  joint - {name}, {label} ({mask.sum()} rows): {v:+.1f} [{lo:+.1f}, {hi:+.1f}]")
                v, lo, hi = interval(d, main)
                verdict = ("not worse" if lo > -(r9.DELTA - r9.SINGLE_PASS) else
                           "worse" if hi < -(r9.DELTA + r9.SINGLE_PASS) else "uncertain (one pass)")
                print(f"  VERDICT {modality} joint vs {name}: {verdict}")
            describe(modality, number)
            compare_correction(modality, number)
            cost(modality, number)


def describe(modality, number):
    """Mechanism: merges across true speakers, speaker changes, rows that left the speaker columns out."""
    mixed = merged = rows_in = dropped_in = changed = blank = malformed = 0
    for w in r9.plan():
        m = r9.load(w["ident"])
        rows, dropped = final_rows(modality, number, w)
        inside = range(w["a"], w["b"])
        dropped_in += sum(1 for s in dropped if s in inside)
        for sources, _, spk, _, chunk in rows:
            if not any(s in inside for s in sources):
                continue
            rows_in += 1
            blank += not spk
            malformed += bool(spk) and not SPK.fullmatch(spk)
            pick = pick_of(spk, chunk)
            changed += bool(pick) and any(pick != f"s{m.leader(s)}" for s in sources)
            if len(sources) > 1:
                merged += 1
                mixed += len({frozenset(m.truth[s]) for s in sources if s in m.truth}) > 1
    print(f"  {modality} joint, rows touching the judge windows {rows_in}: merged {merged} ({merged / max(rows_in, 1):.1%}), of them merging "
          f"different true speakers {mixed} ({mixed / max(merged, 1):.1%}); speaker changed {changed}; spk left out {blank}, "
          f"malformed {malformed} (both kept acoustic); window sources dropped {dropped_in}")
    group = {(w["ident"], w["a"]): w["group"] for w in r9.plan()}
    for name, picks in (("joint", joint_picks(modality, number)), ("round 9 judge on raw ASR", r9.picks_of(f"{modality}-raw", 1))):
        moved = {"fresh": 0, "merged windows": 0}
        for (ident, a, i), pick in picks.items():
            leader = r9.load(ident).leader(i)
            if (pick.startswith("@") or re.fullmatch(r"s\d+", pick)) and pick != f"s{leader}":
                moved["fresh" if group[(ident, a)] in r9.FRESH else "merged windows"] += 1
        print(f"  {modality} {name}: window sources moved off their acoustic track {moved}")


def compare_correction(modality, number):
    """O2 (owner 2026-10-08, after the run, no threshold): the joint correction's shape against round 9's correction on
    the same slices -- same grouping and same corrected text, merges, dropped sources; whole slices."""
    pairs = {"rows": [0, 0], "merged": [0, 0], "dropped": [0, 0]}
    same_group = same_text = total = 0
    for w in r9.plan():
        rows, dropped = final_rows(modality, number, w)
        theirs = {tuple(s): text for s, text in r9.corrected_rows(w)}
        covered_r9 = {i for s in theirs for i in s}
        pairs["rows"][0] += len(rows)
        pairs["rows"][1] += len(theirs)
        pairs["merged"][0] += sum(1 for s, *_ in rows if len(s) > 1)
        pairs["merged"][1] += sum(1 for s in theirs if len(s) > 1)
        pairs["dropped"][0] += len(dropped)
        slice_rows = {i for s, *_ in rows for i in s} | set(dropped)
        pairs["dropped"][1] += len(slice_rows - covered_r9)
        for sources, text, *_ in rows:
            total += 1
            if tuple(sources) in theirs:
                same_group += 1
                same_text += _norm(theirs[tuple(sources)]) == _norm(text)
    print(f"  {modality} joint correction vs round 9's: subtitle rows {pairs['rows'][0]} / {pairs['rows'][1]}, merged rows "
          f"{pairs['merged'][0]} / {pairs['merged'][1]}, dropped sources {pairs['dropped'][0]} / {pairs['dropped'][1]}; "
          f"rows grouped the same {same_group / max(total, 1):.1%}, and with the same corrected text {same_text / max(total, 1):.1%}")


def _sent_usage(w):
    """[(input, output, thinking)] of the accepted (last) attempt of each window round 9's correction ran on this slice."""
    last = {}
    for path in sorted((r9.slice_dir(w) / "artifacts" / "exchanges").glob("*-correction-*.md")):
        last[re.search(r"correction-(\d+)-attempt", path.name).group(1)] = path
    out = []
    for path in last.values():
        text = path.read_text(encoding="utf-8")
        i = re.search(r"input_tokens: \d+ / \d+ / (\d+)", text)
        o = re.search(r"output_tokens_breakdown: (\d+) / (\d+) / (\d+)", text)
        if i and o:
            out.append((int(i.group(1)), int(o.group(3)), int(o.group(2))))
    return out


def cost(modality, number):
    """Output per source row: the joint call against round 9's correction plus round 9's judge on raw ASR."""
    think = lambda u: (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)  # noqa: E731
    replies = [json.loads(p.read_text(encoding="utf-8")) for p in pass_dir(number, modality).glob("*.json")
               if not p.name.endswith(".failed.json")]
    rows = sum(len(r["rows"]) for r in replies)
    last = [r["usage"][-1] for r in replies]
    every = [u for r in replies for u in r["usage"]]
    out, thinking = sum(u.get("completion_tokens", 0) for u in last), sum(think(u) for u in last)
    sent = [u for w in r9.plan() for u in _sent_usage(w)]
    corr_rows = sum(len(json.loads((r9.slice_dir(w) / f"{r9.slice_dir(w).name}-stable.json").read_text(encoding="utf-8"))["segments"])
                    for w in r9.plan())  # the sources round 9's correction read
    judge = [json.loads(p.read_text(encoding="utf-8")) for p in r9.pass_dir(1, f"{modality}-raw").glob("*.json")
             if not p.name.endswith(".failed.json")]
    judged = sum(len(r["answer"] or {}) for r in judge)
    j_out = sum(r["usage"][-1].get("completion_tokens", 0) for r in judge)
    j_think = sum(think(r["usage"][-1]) for r in judge)
    c_out, c_think = sum(o for _, o, _ in sent), sum(t for *_, t in sent)
    print(f"  {modality} joint usage: {len(replies)} calls, {len(every) - len(last)} retries "
          f"{[n for r in replies for n in r.get('retries') or []]}; input {sum(u.get('prompt_tokens', 0) for u in every)} with retries; "
          f"accepted output {out} (thinking {thinking}), with retries {sum(u.get('completion_tokens', 0) for u in every)}")
    print(f"  output per source row (thinking in brackets): joint {out / rows:.1f} ({thinking / rows:.1f}); round 9 correction "
          f"{c_out / max(corr_rows, 1):.1f} ({c_think / max(corr_rows, 1):.1f}) + judge on raw ASR {j_out / max(judged, 1):.1f} "
          f"({j_think / max(judged, 1):.1f}) = {c_out / max(corr_rows, 1) + j_out / max(judged, 1):.1f}; accepted replies on both sides"
          + ("; round 9's correction heard the audio" if modality == "text" else ""))


# --- audit tables -------------------------------------------------------------------------------------

AUDIT = ("hina-47", "hina-05", "manosaba-c")  # slices round 9 corrected on the same agy model as the joint calls


def _standalone(w):
    """{first file row: (file rows, corrected, translation, note)} of round 9's correction, from its annotated CSV."""
    folder = r9.slice_dir(w)
    lines = (folder / f"{folder.name}-annotated.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].lstrip("# ").split("|")
    out = {}
    for line in lines[1:]:
        if line.strip():
            cols = dict(zip(header, line.split("|", len(header) - 1)))
            rows = [int(x) - 1 for x in cols["position"].split(",")]
            out[rows[0]] = (rows, cols["corrected"].strip(), cols["translation"].strip(), cols["note"].strip())
    return out


def _joint(w, modality, number=1):
    """{first file row: (file rows, corrected, translation, note, spk, spk_note)} of the joint call (one-window slices)."""
    (path,) = [p for p in pass_dir(number, modality).glob(f"{w['ident']}-{w['a']}-*.json") if not p.name.endswith(".failed.json")]
    reply = json.loads(path.read_text(encoding="utf-8"))
    got, error = parse(reply["raw"], reply["rows"], r9.load(w["ident"]))
    assert got is not None, error
    return {s[0]: (list(s), text, translation, note, spk, spk_note) for s, text, spk, spk_note, translation, note in got[0]}


def audit():
    """Side-by-side tables for a sampled audit (owner 2026-10-08): per source row in the judge window, the standalone
    correction (round 9) and the joint call, with the truth and the acoustic cell (`audit/<modality>/<window>.md`)."""
    clean = lambda s: str(s).replace("|", "｜").replace("\n", " ")  # noqa: E731
    for w in r9.plan():
        if w["ident"] not in AUDIT:
            continue
        m = r9.load(w["ident"])
        alone = _standalone(w)
        alone_of = {i: first for first, (rows, *_) in alone.items() for i in rows}
        for modality in MODALITIES:
            joint = _joint(w, modality)
            joint_of = {i: first for first, (rows, *_) in joint.items() for i in rows}
            lines = [f"# {w['ident']}，窗内源段 {w['a']}–{w['b'] - 1}，合并 pass 为{'带音频' if modality == 'audio' else '纯文本'}", "",
                     "每行一个源段。`真值` 是人工标的说话人，`声学轨` 是声学层挂的轨（合并 pass 的输入 `spk`）。",
                     "「单独纠错」是生产纠错窗口单独跑的结果（听过音频，medium 思考）；「合并」是同一窗口带说话人复核的一次调用（high 思考）。",
                     "一个输出行并了几个源段时，内容写在第一个源段那行，后面的源段写「↑ 并入」；被丢弃的写「（丢弃）」。`异` 标出两边分组、纠错文或译文不同的源段。", "",
                     "| 源 | 真值 | 声学轨 | ASR 原文 | 异 | 单独：纠错文 | 单独：译文 | 单独：note | 合并：spk | 合并：spk_note | 合并：纠错文 | 合并：译文 | 合并：note |",
                     "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for i in range(w["a"], w["b"]):
                a_first, j_first = alone_of.get(i), joint_of.get(i)
                if a_first is None:
                    a_cells = ["（丢弃）", "", ""]
                elif a_first != i:
                    a_cells = ["↑ 并入", "", ""]
                else:
                    rows, text, tr, note = alone[i]
                    a_cells = [("[" + ",".join(str(r) for r in rows) + "] " if len(rows) > 1 else "") + text, tr, note]
                if j_first is None:
                    j_cells = ["", "", "（丢弃）", "", ""]
                elif j_first != i:
                    j_cells = ["", "", "↑ 并入", "", ""]
                else:
                    rows, text, tr, note, spk, spk_note = joint[i]
                    j_cells = [spk, spk_note, ("[" + ",".join(str(r) for r in rows) + "] " if len(rows) > 1 else "") + text, tr, note]
                differs = (a_first is None) != (j_first is None) or (a_first is not None and (
                    alone[a_first][0] != joint[j_first][0] or _norm(alone[a_first][1]) != _norm(joint[j_first][1])
                    or _norm(alone[a_first][2]) != _norm(joint[j_first][2])))
                truth = "、".join(sorted(m.truth.get(i, ()))) or "-"
                lines.append("| " + " | ".join(clean(c) for c in [i, truth, lr.cell(m.speakers[i]), str(m.stable[i].get("text", "")).strip(),
                                                                   "异" if differs else ""] + a_cells + j_cells) + " |")
            folder = ROOT / "audit" / modality
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{w['ident']}-{w['a']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{modality} {w['ident']}: {sum(1 for l in lines if '| 异 |' in l)} differing of {w['b'] - w['a']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("render", "build", "run", "score", "audit"))
    parser.add_argument("--modality", choices=MODALITIES, default="text")
    parser.add_argument("--pass", dest="number", type=int, default=1)
    parser.add_argument("--only", nargs="*", help="run: these idents only (a smoke call)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "run":
        run(args.modality, args.number, args.shard, args.shards, args.only)
    else:
        {"render": render, "build": build, "score": score, "audit": audit}[args.command]()


if __name__ == "__main__":
    main()
