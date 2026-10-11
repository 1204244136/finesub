"""A review video of P2 round 2 for the owner (2026-10-05): the judged rows, every arm's answer in its own layer.

Part A -- labelled flagged rows of sample v2 (dev): rows where at least one arm changed the label, arms that disagree
first, spread over files. Part B -- held-out missed rows: agy / Flash-Lite answers next to the two no-model fills
(nearest labelled row, Nemotron slot). Each clip is the row with PAD seconds either side, cut from the vocal track
production used; a short silence separates clips. Layers, top to bottom: header (clip, file, truth, verdicts),
reference lines in the clip (person: Chinese text), ASR rows with production labels (judged row highlighted), then
one line per arm. Under every ASR row: its speaker from production and after each arm (rows an arm did not judge
keep production's label; abstain keeps it too), ✓/✗ against the truth. Track ids are shown with their owner's name
(owners from the covered truth).

    python tools/diarization_explore/p2_review_video.py      # writes <MAIN>/out/p2-review/p2-review.mp4
"""
from __future__ import annotations

import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import llm_retrack_judge as lj  # noqa: E402
import llm_retrack_missed as lm  # noqa: E402
import p2_features as pf  # noqa: E402

OUT = lr.n3.MAIN / "out" / "p2-review"
PAD, GAP, MAX_CLIP, BUDGET = 3.0, 0.8, 15.0, 290.0
SR = 16000
WIDTH, HEIGHT = 1280, 720
#: y position (top anchor) of each layer
#: y of each layer; ASR rows take two lines each (text, then the speaker after production and after each arm)
LAYERS = {"head": 18, "ref": 62, "asr": 186, "text": 470, "audio": 530, "moss": 590, "extra": 650}
COLOURS = {"head": "&H00FFFF&", "ref": "&HFFFFFF&", "asr": "&HC0C0C0&", "text": "&H90EE90&", "audio": "&HFFC080&",
           "moss": "&H80C0FF&", "extra": "&HFF80FF&"}
MARK = {"fixed": "✓改对", "broken": "✗改坏", "wrong_to_wrong": "✗错改错", "same_person": "≈换到并列者", "kept": "＝不改",
        "abstain": "？弃权", "missing": "—无输出"}


def reference_lines(ident):
    """[(start, end, person, text)] -- the refined reference (Chinese lines; bilingual Japanese twins dropped)."""
    import refined_multi as rm
    if ident.startswith("onetonton-"):
        import refined_ass as ra
        return [(a, b, w, t) for a, b, w, t in ra.events(ra.episodes()[int(ident[-2:]) - 1]) if w != ra.NOTE and t]
    out = []
    for line in rm.sources()[ident].read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        style = parts[3].strip()
        if re.search(r"-日文\d*$", style):
            continue
        who = rm.speaker(style)
        text = re.sub(r"\{[^}]*\}", "", parts[9]).replace(r"\N", " ").strip()
        if who and text and not text.startswith("※"):
            out.append((rm.seconds(parts[1]), rm.seconds(parts[2]), who, text))
    return out


def ass_time(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def esc(text):
    return text.replace("{", "（").replace("}", "）").replace("\n", " ")


def name(owner, track):
    return f"s{track}={owner.get(track, '?')}" if track is not None else "（无）"


def answer_text(owner, pick, why=""):
    if not pick or pick in ("abstain",):
        core = "abstain"
    elif pick == "keep":
        core = "keep"
    elif re.fullmatch(r"s\d+", pick):
        core = name(owner, int(pick[1:]))
    else:
        core = pick
    return core + (f"  「{why[:40]}」" if why else "")


def answers(folder, ident):
    out = {}
    for path in folder.glob(f"{ident}*.json"):
        if path.stem == ident or re.fullmatch(rf"{re.escape(ident)}-\d+", path.stem):
            out.update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
    return out


def pick_rows():
    """[(part, ident, index, info)] in display order."""
    rows, _ = pf.load_rows("covered")
    data = lj.sample("v2")
    res = {arm: lj.outcomes("v2", arm, rows) for arm in ("text", "audio", "mosspro")}
    changed = {"fixed", "broken", "wrong_to_wrong", "same_person"}
    cand = []
    for ident, items in data["files"].items():
        for item in items:
            key = (ident, item["index"])
            o = {arm: res[arm][key][0] for arm in res}
            if not any(v in changed for v in o.values()):
                continue
            disagree = len({("good" if v == "fixed" else "bad" if v in ("broken", "wrong_to_wrong") else "none") for v in o.values()}) > 1
            cand.append((not disagree, "broken" not in o.values(), ident, item["index"], o))
    cand.sort(key=lambda c: (c[0], c[1]))
    # round-robin over files inside each priority group, so one file does not fill the video
    ordered, groups = [], {}
    for c in cand:
        groups.setdefault((c[0], c[1]), []).append(c)
    for g in sorted(groups):
        by_file = {}
        for c in groups[g]:
            by_file.setdefault(c[2], []).append(c)
        while any(by_file.values()):
            for f in sorted(by_file):
                if by_file[f]:
                    ordered.append(by_file[f].pop(0))
    a_rows = [("A", c[2], c[3], c[4]) for c in ordered]
    # part B: held-out missed rows where the four fills do not all agree
    slots = pickle.loads((pf.CACHE / "slot_identity.pkl").read_bytes())
    agy = {}
    lite = {}
    for path in (lm.OUT / "agy").glob("*.json"):
        agy.update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
    for path in (lm.OUT / "lite").glob("*.json"):
        lite.update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
    b_rows = []
    for ident, idx in lm.items().items():
        by_index = {r["index"]: r for r in rows[ident]}
        for index in idx:
            r = by_index[index]
            pa = str(agy.get(f"q{index}", {}).get("track", "abstain"))
            ok_agy = pa.startswith("s") and pa[1:].isdigit() and pf.right(r, int(pa[1:]))
            ok_slot = pf.right(r, slots[(ident, index)]["slot_track"])
            if ok_agy != ok_slot:
                b_rows.append(("B", ident, index, {"agy_right": ok_agy, "slot_right": ok_slot}))
    rng = np.random.default_rng(20261005)
    b_rows = [b_rows[i] for i in sorted(rng.choice(len(b_rows), min(8, len(b_rows)), replace=False))]
    return a_rows, b_rows, agy, lite, slots


def main():
    import soundfile as sf
    OUT.mkdir(parents=True, exist_ok=True)
    rows, _ = pf.load_rows("covered")
    a_rows, b_rows, agy, lite, slots = pick_rows()
    files = sorted({x[1] for x in a_rows + b_rows}, key=lambda i: i.startswith("onetonton"))  # ⚠ hina first
    stables, refs = {}, {}
    for f in files:  # ⚠ reference lines read in the same hina-first pass: onetonton's configure rebinds refined_multi.sources
        stables[f] = pf.reference(f)[0]
        refs[f] = reference_lines(f)
    speakers = {f: lj._speakers(f, stables[f]) for f in files}
    waves = {}
    arm_answers = {arm: {f: answers(lj.OUT / f"v2-{arm}", f) for f in files} for arm in ("text", "audio", "mosspro")}
    res = {arm: lj.outcomes("v2", arm, rows) for arm in ("text", "audio", "mosspro")}

    # each arm's speaker for every row it judged: a track id, or None to keep production's label
    decisions = {}
    for arm in ("text", "audio", "mosspro"):
        for f in files:
            for key, reply in arm_answers[arm][f].items():
                pick = str(reply.get("track", "")).strip()
                decisions[(arm, f, int(key[1:]))] = int(pick[1:]) if re.fullmatch(r"s\d+", pick) else None
    for arm, source in (("agy", agy), ("lite", lite)):
        for f, idx in lm.items().items():
            for i in idx:
                pick = str(source.get(f"q{i}", {}).get("track", "abstain")).strip()
                decisions[(arm, f, i)] = int(pick[1:]) if re.fullmatch(r"s\d+", pick) else None
    for (f, i), s in slots.items():
        decisions[("slot", f, i)] = s["slot_track"]

    def decided(arm, f, i, produced):
        """The row's speaker after `arm`: its pick where it judged the row and changed it, else production's."""
        if arm == "slot":
            return decisions.get(("slot", f, i), produced) if produced is None else produced
        pick = decisions.get((arm, f, i))
        return produced if pick is None else pick

    events, audio, cursor = [], [], 0.0
    plan = []
    budget_b = sum(min(MAX_CLIP, float(stables[f][i]["end"]) - float(stables[f][i]["start"]) + 2 * PAD) + GAP for _, f, i, _ in b_rows)
    used = 0.0
    for part, ident, index, info in a_rows:
        row = stables[ident][index]
        length = min(MAX_CLIP, float(row["end"]) - float(row["start"]) + 2 * PAD) + GAP
        if used + length + budget_b > BUDGET:
            continue
        plan.append((part, ident, index, info))
        used += length
    plan += b_rows

    for n, (part, ident, index, info) in enumerate(plan, 1):
        stable, refl = stables[ident], refs[ident]
        row = stable[index]
        r = next(x for x in rows[ident] if x["index"] == index)
        owner = r["owner"]
        s, e = float(row["start"]), float(row["end"])
        a = max(0.0, s - PAD)
        b = min(a + MAX_CLIP, e + PAD)
        if ident not in waves:
            wave, rate = sf.read(str(lj.vocal(ident)), dtype="float32", always_2d=True)
            assert rate == SR, rate
            waves[ident] = wave.mean(axis=1)
        audio += [waves[ident][int(a * SR):int(b * SR)], np.zeros(int(GAP * SR), dtype=np.float32)]
        t0 = cursor
        t1 = cursor + (b - a)
        cursor = t1 + GAP

        def at(x):
            return t0 + (x - a)
        truth = "、".join(r["truth"]) or "（无）"
        if part == "A":
            verdicts = "  ".join(f"{label}{MARK[info[arm]]}" for label, arm in (("文字", "text"), ("音频", "audio"), ("MOSS", "mosspro")))
            head = f"A{n:02d} · {ident} · {s:.1f}s · 真值：{truth} · 生产：{name(owner, r['leader'])} · {verdicts}"
        else:
            head = (f"B{n:02d} · {ident} · {s:.1f}s · 漏检行 · 真值：{truth} · agy {'✓' if info['agy_right'] else '✗'}"
                    f"  按槽位 {'✓' if info['slot_right'] else '✗'}")
        events.append(("head", t0, t1, esc(head), LAYERS["head"]))
        # reference lines overlapping the clip, stacked by order
        shown = [x for x in refl if x[0] < b and x[1] > a][:4]
        for k, (ra_, rb_, who, text) in enumerate(shown):
            events.append(("ref", at(max(a, ra_)), at(min(b, rb_)), esc(f"参考 {who}：{text}"), LAYERS["ref"] + 30 * k))
        # ASR rows in the clip with production labels
        # the judged row always, plus up to three other rows of the clip nearest to it, in time order
        near = [j for j in range(len(stable)) if float(stable[j]["start"]) < b and float(stable[j]["end"]) > a]
        near = sorted(sorted(near, key=lambda j: (j != index, abs(j - index)))[:4])
        by_index = {x["index"]: x for x in rows[ident]}
        arms = (("文字", "text"), ("音频", "audio"), ("MOSS", "mosspro")) if part == "A" else (("agy", "agy"), ("Lite", "lite"), ("槽位", "slot"))
        for k, j in enumerate(near):
            tracks = speakers[ident][j].to_dict().get("tracks") or []
            produced = tracks[0] if tracks else None
            rj = by_index.get(j)

            def tag(track):
                if track is None:
                    return "??"
                ok = "" if rj is None or not rj["truth"] else (" ✓" if pf.right(rj, track) else " ✗")
                return name(owner, track) + ok
            after = "  ".join(f"{label} {tag(decided(arm, ident, j, produced))}" for label, arm in arms)
            mark = "▶ " if j == index else ""
            colour = "{\\c&H00FFFF&}" if j == index else ""
            y = LAYERS["asr"] + 70 * k
            span = (at(max(a, float(stable[j]["start"]))), at(min(b, float(stable[j]["end"]))))
            events.append(("asr", *span, colour + esc(f"{mark}ASR：{str(stable[j].get('text', '')).strip()}"), y))
            events.append(("asr", *span, colour + esc(f"    生产 {tag(produced)}  →  {after}"), y + 30))
        if part == "A":
            for arm, label in (("text", "文字裁判"), ("audio", "音频裁判"), ("mosspro", "MOSS Pro")):
                reply = arm_answers[arm][ident].get(f"q{index}") or {}
                why = "" if arm == "mosspro" else str(reply.get("why", ""))
                txt = f"{label} {MARK[res[arm][(ident, index)][0]]}：{answer_text(owner, str(reply.get('track', '')).strip(), why)}"
                style = "moss" if arm == "mosspro" else arm
                events.append((style, t0, t1, esc(txt), LAYERS[style]))
        else:
            ra_ = agy.get(f"q{index}", {})
            rl_ = lite.get(f"q{index}", {})
            events.append(("text", t0, t1, esc(f"agy：{answer_text(owner, str(ra_.get('track', 'abstain')), str(ra_.get('why', '')))}"), LAYERS["text"]))
            events.append(("audio", t0, t1, esc(f"Flash-Lite：{answer_text(owner, str(rl_.get('track', 'abstain')), str(rl_.get('why', '')))}"), LAYERS["audio"]))
            events.append(("moss", t0, t1, esc(f"按槽位：{name(owner, slots[(ident, index)]['slot_track'])}"), LAYERS["moss"]))
            events.append(("extra", t0, t1, esc(f"最近一条有标签的行：{name(owner, r['nearest'])}"), LAYERS["extra"]))

    wav = OUT / "p2-review.wav"
    sf.write(str(wav), np.concatenate(audio), SR, subtype="PCM_16")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""" + "\n".join(
        f"Style: {k},Microsoft YaHei,{22 if k != 'head' else 24},{COLOURS[k]},&H000000FF,&H00000000,&H00000000,{-1 if k == 'head' else 0},0,0,0,100,100,0,0,1,1.5,0,7,20,20,0,1"
        for k in LAYERS) + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    lines = [f"Dialogue: 0,{ass_time(s0)},{ass_time(s1)},{style},,0,0,0,,{{\\pos(20,{y})}}{text}" for style, s0, s1, text, y in events]
    (OUT / "p2-review.ass").write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")
    total = cursor
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x101010:s={WIDTH}x{HEIGHT}:r=10:d={total:.2f}", "-i", "p2-review.wav",
           "-vf", "subtitles=p2-review.ass:fontsdir=C\\\\:/Windows/Fonts", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-b:a", "96k", "-shortest", "p2-review.mp4"]
    subprocess.run(cmd, cwd=OUT, check=True, capture_output=True)
    print(f"{len(plan)} clips ({sum(p[0] == 'A' for p in plan)} labelled, {sum(p[0] == 'B' for p in plan)} missed), {total:.0f} s -> {OUT / 'p2-review.mp4'}")


if __name__ == "__main__":
    main()
