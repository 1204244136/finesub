"""A review video of judge rounds 3-4 for the owner to label by ear (2026-10-06): does hina's ruler misjudge the judges?

The owner suspects the refined-subtitle ruler on talk material (hina, 姐姐咚咚) is noisy enough to count right changes as
broken. 20 rows are drawn at random (fixed seed) from three talk windows of rounds 3-4 -- few, so a later prompt can be
rerun on all labelled rows for the cost of three windows (owner 2026-10-06): hina-71 (merged speakers), hina-06 (rows on
the wrong existing track), 姐姐咚咚 13 (another show); none was covered by the earlier review video (`p2_review_video.py`).
Stratified by what the ruler says about the three arms the rounds share (text-closed, text-open_gen, audio-open_gen),
every labelling scored as a partition of the window (`window_partition_score`: its labels matched one-to-one to people
inside the window -- so a judge that splits a merged track and leaves another person on it is not counted wrong):

  broken   some arm moved a row the ruler counts right                          QUOTA["broken"]
  fixed    some arm fixed a row the ruler counts wrong, no arm broke it          QUOTA["fixed"]
  missed   the ruler counts production wrong, no arm moved the row               QUOTA["missed"]
  quiet    the ruler counts production right, no arm moved the row (control)     QUOTA["quiet"]

Each clip is the row with PAD seconds either side, cut from the vocal track production used. On screen for the whole clip:
the row and up to four rows around it with their start in clip seconds, each with the label production and every arm
ends on (a track id or a new-track label, ＝ when an arm left production's label) and the person that label stands for in
that labelling's partition of the window, then the refined reference lines (the ruler) in the clip, shown at
the owner's request for reference (2026-10-06); every line turns green while it is spoken. A cast line under the heading numbers the window's people (by rows spoken, the
same numbers in every clip of a window) for the owner to label with. No right / wrong mark is shown. Labels go into `OWNER` below (tracks or names heard in the ▶ row), then `score`.

    python tools/diarization_explore/judge_review_video.py spec r3   # one process per round (P2 loads hina first)
    python tools/diarization_explore/judge_review_video.py spec r4
    python tools/diarization_explore/judge_review_video.py render    # <MAIN>/out/judge-review/judge-review.mp4
    python tools/diarization_explore/judge_review_video.py score
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import p2_review_video as pv  # noqa: E402
import window_partition_score as ps  # noqa: E402

OUT = lr.n3.MAIN / "out" / "judge-review"
SEED = 20261009
ARMS = (("文字封闭", "text-closed"), ("文字开放", "text-open_gen"), ("音频开放", "audio-open_gen"))
#: the three windows the clips come from (the earlier review video covered hina-14/25/27/37 and onetonton-01/05)
WINDOWS = {"r3": ("onetonton-13",), "r4": ("hina-71", "hina-06")}
QUOTA = {"broken": 7, "fixed": 6, "missed": 4, "quiet": 3}
PER_WINDOW, APART = 8, 10.0
PAD, GAP, MAX_CLIP = 3.0, 0.8, 15.0
LAYERS = {"head": 18, "cast": 52, "asr": 92, "ref": 462, "why": 590}
COLOURS = {"head": "&H00FFFF&", "cast": "&HFFFFFF&", "asr": "&HC0C0C0&", "ref": "&HE0B080&", "why": "&H90EE90&"}
#: a line turns this colour while it is being spoken (an overlay on top of its resting copy)
LIVE = "{\\c&H40FF40&}"

#: (file, row) -> what the owner heard in the ▶ row: track ids (int) and / or names (str; a person number from the video's
#: cast line is entered as the name it stands for); None = cannot tell. Keyed by
#: row, not clip number, so labels survive a re-pick (owner 2026-10-06, the first four from the first cut of the video)
OWNER: dict[tuple[str, int], set | None] = {
    # from the first cut of the video (its onetonton-13 row 485 = s0 is the same voice as C16 below)
    ("hina-71", 61): {"大黑塔"},
    ("hina-71", 180): {"大黑塔"},
    ("hina-71", 102): {"女开拓者"},
    # the partition cut, C01-C20 (owner 2026-10-07; person numbers from the cast line, "?" = the owner was unsure)
    ("hina-71", 112): {"女开拓者", "昔涟"},  # C01 前半 2 后半 3
    ("hina-71", 30): {"大黑塔"},             # C02 断句与真值不同（owner：很多地方都是）
    ("onetonton-13", 522): {"樱川惠"},       # C03 ?
    ("hina-71", 76): {"大黑塔"},             # C04
    ("hina-71", 126): {"昔涟"},              # C05
    ("onetonton-13", 565): {"小日向美香"},   # C06 ?
    ("hina-71", 88): {"猫头鹰先生"},         # C07
    ("hina-06", 577): {"咔库库"},            # C08
    ("hina-06", 648): {"爱可菲"},            # C09
    ("hina-71", 122): {"猫头鹰先生"},        # C10
    ("onetonton-13", 415): {"青木佑磨", "小日向美香"},  # C11 同时
    ("hina-06", 544): {"咔库库"},            # C12 owner：真值错
    ("hina-71", 83): {"昔涟"},               # C13
    ("hina-71", 152): {"女开拓者"},          # C14
    ("onetonton-13", 452): {"青木佑磨"},     # C15
    ("onetonton-13", 485): {"樱川惠"},       # C16 (s0 in the first cut: the same voice)
    ("hina-06", 607): {"咔库库"},            # C17
    ("onetonton-13", 448): {"小日向美香"},   # C18
    ("onetonton-13", 596): {"樱川惠"},       # C19
    ("onetonton-13", 530): {"青木佑磨"},     # C20
}
UNSURE = {("onetonton-13", 522), ("onetonton-13", 565)}


def _module(rnd):
    if rnd == "r4":
        import window_judge_r4 as r4
        return r4.r3
    import window_judge_r3 as r3
    return r3


def spec(rnd):
    """Every candidate row of this round's talk windows, with what the video and the scoring need, to spec-<round>.json."""
    r3 = _module(rnd)
    base = r3.outcome({})
    picks = {"production": {}, **{arm: r3.answers(arm) for _, arm in ARMS}}
    out = []
    for ident, a, b in r3.windows():  # windows() order loads hina before onetonton
        if ident not in WINDOWS[rnd]:
            continue
        m = r3.load(ident)
        rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
        labels = {name: ps.final_labels(r3, m, ident, a, b, got) for name, got in picks.items()}
        found = {name: ps.match(rows, labels[name], m.truth) for name in picks}
        # the window's people, numbered by how many of its rows they speak (fixed for every clip of the window), so the
        # owner can label with a number (owner 2026-10-06)
        weight = Counter()
        for i in range(a, b):
            for person in m.truth.get(i, ()):
                weight[person] += 1 / len(m.truth[i])
        people = [person for person, _ in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))]

        def ends(name, j):
            """(label, person it stands for) of row j in a labelling; rows outside the window keep production's track."""
            lead = m.leader(j)
            label = labels[name].get(j, f"s{lead}" if lead is not None else None)
            return label, found[name].get(label) or found["production"].get(label) or m.owner.get(lead)
        for i in rows:
            ok_before = ends("production", i)[1] in m.truth[i]
            moved = {arm: labels[arm][i] != labels["production"][i] for _, arm in ARMS}
            after = {arm: ends(arm, i)[1] in m.truth[i] for _, arm in ARMS}
            if ok_before and not all(after.values()):
                stratum = "broken"
            elif not ok_before and any(after.values()):
                stratum = "fixed"
            elif not any(moved.values()):
                stratum = "missed" if not ok_before else "quiet"
            else:
                continue  # moved to another wrong person, or between two tracks of one person
            s, e = float(m.stable[i]["start"]), float(m.stable[i]["end"])
            lo = max(0.0, s - PAD)
            hi = min(lo + MAX_CLIP, e + PAD)
            near = [j for j in range(len(m.stable)) if float(m.stable[j]["start"]) < hi and float(m.stable[j]["end"]) > lo]
            near = sorted(sorted(near, key=lambda j: (j != i, abs(j - i)))[:5])
            out.append({
                "round": rnd, "file": ident, "window": a, "row": i, "stratum": stratum, "start": s, "end": e, "clip": [lo, hi],
                "vocal": str(m.vocal), "truth": sorted(m.truth[i]), "ruler_after": after, "people": people,
                "track_people": {f"s{k}": v for k, v in m.owner.items()},
                "rows": [{"row": j, "start": float(m.stable[j]["start"]), "end": float(m.stable[j]["end"]),
                          "text": str(m.stable[j].get("text", "")).strip(),
                          "ends": {name: ends(name, j) for name in picks}} for j in near],
                "why": {arm: _why(r3, arm, ident, a, i) for _, arm in ARMS},
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"spec-{rnd}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{rnd}: {len(out)} candidate rows, {dict(Counter(c['stratum'] for c in out))}")


def _why(r3, arm, ident, a, i):
    path = r3.OUT / arm / f"{ident}-{a}.json"
    answer = json.loads(path.read_text(encoding="utf-8"))["answer"] or {}
    return str((answer.get(f"q{i}") or {}).get("why", ""))


def pick():
    """The 20 clips: per stratum a seeded shuffle, taken in turn, at most PER_WINDOW per window, APART seconds between clips
    of one file."""
    cands = [c for rnd in WINDOWS for c in json.loads((OUT / f"spec-{rnd}.json").read_text(encoding="utf-8")) if c["file"] in WINDOWS[rnd]]
    rng = np.random.default_rng(SEED)
    pools = {s: [cands[k] for k in rng.permutation([k for k, c in enumerate(cands) if c["stratum"] == s])] for s in QUOTA}
    chosen, need = [], dict(QUOTA)
    while any(need.values()):
        progressed = False
        for s in QUOTA:
            while need[s] and pools[s]:
                c = pools[s].pop(0)
                same = [x for x in chosen if x["file"] == c["file"]]
                if sum(x["window"] == c["window"] for x in same) >= PER_WINDOW or any(abs(x["start"] - c["start"]) < APART for x in same):
                    continue
                chosen.append(c)
                need[s] -= 1
                progressed = True
                break
        if not progressed:
            raise SystemExit(f"cannot fill the quota: still need {need}")
    order = rng.permutation(len(chosen))  # strata mixed on screen, so the clip order tells nothing
    return [chosen[k] for k in order]


def label_text(label, person, same=False):
    if label is None:
        return "??"
    return f"{'＝' if same else ''}{label}（{person or '?'}）"


def render():
    import soundfile as sf
    import p2_features as pf
    clips = pick()
    (OUT / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    refs = {}
    for f in sorted({c["file"] for c in clips}, key=lambda i: i.startswith("onetonton")):  # ⚠ hina first, as P2 requires
        pf.reference(f)
        refs[f] = pv.reference_lines(f)
    events, audio, cursor, waves = [], [], 0.0, {}
    for n, c in enumerate(clips, 1):
        lo, hi = c["clip"]
        if c["vocal"] not in waves:
            wave, rate = sf.read(c["vocal"], dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if rate != pv.SR:
                import librosa
                wave = librosa.resample(wave, orig_sr=rate, target_sr=pv.SR)
            waves[c["vocal"]] = wave
        audio += [waves[c["vocal"]][int(lo * pv.SR):int(hi * pv.SR)], np.zeros(int(GAP * pv.SR), dtype=np.float32)]
        t0, t1 = cursor, cursor + hi - lo
        cursor = t1 + GAP
        at = lambda x: t0 + (x - lo)  # noqa: E731

        def live(style, start, end, text, y):
            """The line at rest for the whole clip, plus a highlighted copy on top while it is spoken."""
            events.append((0, style, t0, t1, text, y))
            if start < hi and end > lo:
                events.append((1, style, at(max(lo, start)), at(min(hi, end)), LIVE + text.split("}", 1)[-1] if text.startswith("{") else LIVE + text, y))
        events.append((0, "cast", t0, t1, pv.esc("人物：" + "   ".join(f"{k} {p}" for k, p in enumerate(c["people"], 1))), LAYERS["cast"]))
        events.append((0, "head", t0, t1, pv.esc(f"C{n:02d} · {c['file']} · {c['start']:.1f}s · 标 ▶ 那一行是谁在说（人物编号、轨号或名字）"), LAYERS["head"]))
        for k, r in enumerate(c["rows"]):
            mine = r["row"] == c["row"]
            colour = "{\\c&H00FFFF&}" if mine else ""
            prod = label_text(*r["ends"]["production"])
            arms = "  ".join(f"{label} {label_text(*r['ends'][arm], same=r['ends'][arm][0] == r['ends']['production'][0])}"
                             for label, arm in ARMS)
            y = LAYERS["asr"] + 74 * k
            # every row stays up for the whole clip, with its start in clip seconds, so the owner can read ahead and seek
            live("asr", r["start"], r["end"], colour + pv.esc(f"{'▶ ' if mine else ''}{max(0.0, r['start'] - lo):4.1f}s  {r['text']}"), y)
            events.append((0, "asr", t0, t1, colour + pv.esc(f"    生产 {prod}  →  {arms}"), y + 30))
        # the refined reference (the ruler) under the ASR rows, shown at the owner's request (2026-10-06)
        shown = sorted((x for x in refs[c["file"]] if x[0] < hi and x[1] > lo), key=lambda x: x[0])[:4]
        for k, (ra, rb, who, text) in enumerate(shown):
            live("ref", ra, rb, pv.esc(f"{max(0.0, ra - lo):4.1f}s  真值 {who}：{text}"), LAYERS["ref"] + 30 * k)
        for k, (label, arm) in enumerate(ARMS):
            if c["why"][arm]:
                events.append((0, "why", t0, t1, pv.esc(f"{label}：{c['why'][arm][:44]}"), LAYERS["why"] + 32 * k))
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT / "judge-review.wav"), np.concatenate(audio), pv.SR, subtype="PCM_16")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {pv.WIDTH}
PlayResY: {pv.HEIGHT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""" + "\n".join(
        f"Style: {k},Microsoft YaHei,{24 if k == 'head' else 22},{COLOURS[k]},&H000000FF,&H00000000,&H00000000,{-1 if k == 'head' else 0},0,0,0,100,100,0,0,1,1.5,0,7,20,20,0,1"
        for k in LAYERS) + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    lines = [f"Dialogue: {layer},{pv.ass_time(s0)},{pv.ass_time(s1)},{style},,0,0,0,,{{\\pos(20,{y})}}{text}" for layer, style, s0, s1, text, y in events]
    (OUT / "judge-review.ass").write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x101010:s={pv.WIDTH}x{pv.HEIGHT}:r=10:d={cursor:.2f}", "-i", "judge-review.wav",
           "-vf", "subtitles=judge-review.ass:fontsdir=C\\\\:/Windows/Fonts", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-b:a", "96k", "-shortest", "judge-review.mp4"]
    subprocess.run(cmd, cwd=OUT, check=True, capture_output=True)
    print(f"{len(clips)} clips {dict(Counter(c['stratum'] for c in clips))}, {cursor:.0f} s -> {OUT / 'judge-review.mp4'}")


def score():
    """Per clip: the ruler, production and each arm against the owner's ear. A labelling is right when the label it ends on
    is a track the owner named, or the person that label stands for in its partition is a name the owner named.
    "ruler" = the ruler's truth for the row shares a person with what the owner heard."""
    # every labelled row, whether or not it is in the current cut of the video (labels are kept across re-picks)
    clips = [c for rnd in WINDOWS for c in json.loads((OUT / f"spec-{rnd}.json").read_text(encoding="utf-8"))
             if (c["file"], c["row"]) in OWNER]
    tally, sure = Counter(), Counter()
    for n, c in enumerate(clips, 1):
        heard = OWNER.get((c["file"], c["row"]))
        if heard is None:
            continue
        row = next(r for r in c["rows"] if r["row"] == c["row"])
        tracks = {f"s{x}" for x in heard if isinstance(x, int)}
        names = {x for x in heard if isinstance(x, str)}

        def right(ends):
            label, person = ends
            return label in tracks or (label or "").lstrip("@") in names or person in names

        heard_people = names | {c["track_people"].get(x) for x in tracks}  # a track the owner named -> whose voice it is over the file
        verdicts = {"ruler": bool(set(c["truth"]) & heard_people), "production": right(row["ends"]["production"])}
        for _, arm in ARMS:
            verdicts[arm] = right(row["ends"][arm])
        tally.update(k for k, ok in verdicts.items() if ok)
        tally["n"] += 1
        if (c["file"], c["row"]) not in UNSURE:
            sure.update(k for k, ok in verdicts.items() if ok)
            sure["n"] += 1
        ruler_arm = {arm: c["ruler_after"][arm] for _, arm in ARMS}
        print(f"{c['stratum']:6s} {c['file']} row {c['row']}: owner {sorted(map(str, heard))}; truth {c['truth']}; "
              + ", ".join(f"{k} {'✓' if ok else '✗'}" for k, ok in verdicts.items())
              + "; ruler on arms " + ", ".join(f"{arm} {'✓' if ok else '✗'}" for arm, ok in ruler_arm.items()))
    print("all labelled:", {k: f"{v}/{tally['n']}" for k, v in tally.items() if k != "n"})
    print("without the owner's unsure rows:", {k: f"{v}/{sure['n']}" for k, v in sure.items() if k != "n"})
    print("note: rows were drawn by the ruler's verdict on these arms, and an arm's person comes from its window partition, "
          "itself matched with the ruler -- arm and ruler counts are not independent")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "render"
    if command == "spec":
        spec(sys.argv[2])
    else:
        {"render": render, "score": score}[command]()


if __name__ == "__main__":
    main()
