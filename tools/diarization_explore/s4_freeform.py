"""Plan 5o.11 probe: hand agy a ~2 min slice and let it transcribe by speaker on its own.

No candidate text, no track information, not even the number of people -- the opposite end from
§5h.14–5h.18, where the model was given acoustic facts and per-track candidates and only had to
re-split. The question is whether the whole VAD + ASR + diarization + re-split chain is beatable
by asking a multimodal model directly.

**Probe, not an arm**: it reports numbers, it does not judge. Its output carries its own
timestamps and its own speaker ids, so scoring it on the S4 ruler would need a speaker mapping
and a per-window projection first -- built only if this looks worth it.

  DIAR_S4_CORPUS=ali-holdout python s4_freeform.py --ask R8003_M8001 0 1 2   # truth available
  DIAR_S4_CORPUS=ja          python s4_freeform.py --ask nepolabo 4 9        # no truth
  DIAR_S4_CORPUS=ali-holdout python s4_freeform.py --score R8003_M8001       # vs truth + production
  DIAR_S4_CORPUS=ja          python s4_freeform.py --score nepolabo          # vs production only
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import s4_breakdown as b
import s4_llm as llm
import s4_projection as proj
import s4_timecut as tc

PROMPT = (Path(__file__).resolve().parent / "freeform_system.md")
TIMELINE_PROMPT = (Path(__file__).resolve().parent / "freeform_timeline_system.md")
TIMELINE2_PROMPT = (Path(__file__).resolve().parent / "freeform_timeline2_system.md")
REPAIR_PROMPT = (Path(__file__).resolve().parent / "freeform_repair_system.md")
LINE = re.compile(r"^\s*\[?\s*(\d+(?:\.\d+)?)\s*[-–~]\s*(\d+(?:\.\d+)?)\s*\]?\s*([^:：]{1,24})[:：]\s*(.+?)\s*$")


def out_path(name, index, model, tag=None):
    """`tag` names a variant: `n4` was told four people are talking, `tl` was given the empty
    timeline with its per-segment speaker confidence (probes 2 and 3, plan 5o.11)."""
    tail = "" if tag is None else f"-{tag}"
    return proj.ALI / name / f"freeform-{model}-c{index:02d}{tail}.json"


def timeline_text(name, chunk, confidence, tracks):
    """The empty axis for one chunk: who sounds when, how sure, and whether anyone else overlaps.

    None of our text reaches the model -- that is the whole point of probe 3. The confidence is the
    diarization's own sigmoid for that segment's track (`s4_diar_probs`), not something invented.
    """
    begin, end = chunk["begin"], chunk["end"]
    inside = [(seg, c) for seg, c in zip(tracks, confidence) if seg["start"] < end and seg["end"] > begin]
    rows, totals = [], {}
    for seg, conf in inside:
        a, c = max(seg["start"], begin) - begin, min(seg["end"], end) - begin
        totals[seg["track"]] = totals.get(seg["track"], 0.0) + (c - a)
        alone = not any(other["track"] != seg["track"] and other["start"] < seg["end"] and other["end"] > seg["start"]
                        for other, _ in inside)
        rows.append((a, f"  [{a:.1f}-{c:.1f}] s{seg['track']}  置信 {conf:.2f}  {'独占' if alone else '与别人重叠'}"))
    head = ["每条轨在这段里的总时长：" + "，".join(f"s{k} {v:.0f} 秒" for k, v in sorted(totals.items()))]
    diar = json.loads((proj.ALI / name / f"{name}-diar.json").read_text(encoding="utf-8"))
    hints = diar.get("metadata", {}).get("hints", {})
    if hints.get("co_activity"):
        same = set(hints.get("not_same", ()))
        head.append("轨对同时在响的时间占比（已标定的判据：占比高且两条轨都够长 ⇒ 是不同的人）：" + "，".join(
            f"{pair} {share:.0%}{'（判为不同的人）' if pair in same else ''}"
            for pair, share in sorted(hints["co_activity"].items())))
    return "\n".join(head + ["时间轴（只有时间，没有文字）："] + [text for _, text in sorted(rows)])


def timeline2_text(name, chunk, rows):
    """The empty axis built from the **pipeline's own** segments (probe 3b): one line per subtitle
    line production would emit, its diarization labels and each label's confidence, text removed.

    Probe 3's axis was the raw Sortformer runs -- hundreds of sub-second fragments. These are
    utterances, so filling them in is the same shape as producing a better A0.
    """
    begin, end = chunk["begin"], chunk["end"]
    inside = [r for r in rows if (r["start"] + r["end"]) / 2 >= begin and (r["start"] + r["end"]) / 2 <= end]
    totals = {}
    for row in inside:
        for k in row["tracks"]:
            totals[k] = totals.get(k, 0.0) + (row["end"] - row["start"]) / max(len(row["tracks"]), 1)
    lines = []
    for row in inside:
        who = "+".join(f"s{k}" for k in row["tracks"]) or "?"
        conf = "，".join(f"s{k} {c:.2f}" for k, c in zip(row["tracks"], row["confidence"])) or "无标签"
        # Clamped: a segment straddling the chunk edge would otherwise be handed out as `[-0.8-1.8]`,
        # which is both confusing and unparseable when the model echoes it back.
        a, c = max(0.0, row["start"] - begin), min(end - begin, row["end"] - begin)
        lines.append(f"  [{a:.1f}-{c:.1f}] {who}  置信 {conf}")
    head = ["每条轨在这段里的说话时长：" + "，".join(f"s{k} {v:.0f} 秒" for k, v in sorted(totals.items())),
            f"共 {len(inside)} 句，其中 {sum(1 for r in inside if len(r['tracks']) > 1)} 句机器没分开是谁说的。"]
    diar = json.loads((proj.ALI / name / f"{name}-diar.json").read_text(encoding="utf-8"))
    hints = diar.get("metadata", {}).get("hints", {})
    if hints.get("co_activity"):
        same = set(hints.get("not_same", ()))
        head.append("轨对同时在响的时间占比（已标定的判据：占比高且两条轨都够长 ⇒ 是不同的人）：" + "，".join(
            f"{pair} {share:.0%}{'（判为不同的人）' if pair in same else ''}"
            for pair, share in sorted(hints["co_activity"].items())))
    return "\n".join(head + ["分句表（只有时间与说话人，没有文字）："] + lines)


def production_lines(name, chunk, rows, session):
    """Production's finished lines for one chunk: time, track labels, text (probe 4's input 甲)."""
    begin, end = chunk["begin"], chunk["end"]
    by_segment = {}
    for moment, c, seg in session.chars:
        if begin <= moment <= end:
            by_segment.setdefault(seg, ["", moment, moment])
            by_segment[seg][0] += c
            by_segment[seg][2] = moment
    out = []
    for seg, (text, a, c) in sorted(by_segment.items(), key=lambda kv: kv[1][1]):
        who = "+".join(f"s{k}" for k in session.labels[seg]) or "?"
        out.append(f"  [{a - begin:.1f}-{c - begin:.1f}] {who}: {text}")
    return out


def repair_text(name, chunk, session, first):
    """The two transcripts the second pass adjudicates between."""
    begin = chunk["begin"]
    theirs = [f"  [{r['start']:.1f}-{r['end']:.1f}] {r['speaker']}: {r['text']}" for r in first]
    return ("甲 · 管线的成品：" + chr(10)
            + chr(10).join(production_lines(name, chunk, None, session))
            + chr(10) + chr(10) + "乙 · 另一次听写（说话人是它自己编的号）：" + chr(10)
            + chr(10).join(theirs))


def parse(raw):
    """`[12.3-15.0] S1: text` lines -> [{start, end, speaker, text}]; anything else is dropped."""
    rows = []
    for line in (raw or "").splitlines():
        match = LINE.match(line.strip().lstrip("-•*").strip())
        if match:
            start, end, speaker, text = match.groups()
            rows.append({"start": float(start), "end": float(end), "speaker": speaker.strip(), "text": text})
    return rows


def ask(name, indexes, model="agy", count=None, timeline=False):
    from finesub.llm.media_upload import window_media_ref
    from finesub.llm.routing.config import LLMRole

    import s4_resegment as rsg

    client = llm.client_for(model)
    prompt = {0: PROMPT, 1: TIMELINE_PROMPT, 2: TIMELINE2_PROMPT, 3: REPAIR_PROMPT}[int(timeline)]
    system = prompt.read_text(encoding="utf-8")
    chunks = list(llm.chunks(name))
    axis = session = None
    if int(timeline) == 3:
        session = b.Session(name)
    elif timeline:
        import s4_diar_probs

        axis = (s4_diar_probs.segment_confidence(name) if int(timeline) == 1
                else s4_diar_probs.stable_confidence(name))
    for index in indexes:
        tag = {0: None if count is None else f"n{count}", 1: "tl", 2: "tl2", 3: "rp"}[int(timeline)]
        target = out_path(name, index, model, tag)
        if target.exists():
            print(f"{name} c{index:02d} already asked", flush=True)
            continue
        chunk = chunks[index]
        clip = proj.ALI / name / "llm-clips" / f"c{index:02d}.wav"
        if not clip.exists():
            rsg.llm_clip(name, chunk, clip)
        minutes = (chunk["end"] - chunk["begin"]) / 60
        # Probe 2 changes exactly one thing: the sentence that names the number of speakers.
        knowledge = f"这段里有 {count} 个人在说话。" if count else ""
        if int(timeline) == 3:
            first = json.loads(out_path(name, index, model).read_text(encoding="utf-8"))["rows"]
            user = (f"这段录音长 {minutes:.1f} 分钟（{chunk['end'] - chunk['begin']:.0f} 秒）。时间从 0 算起。\n\n"
                    + repair_text(name, chunk, session, first)
                    + "\n\n听着录音，综合这两份写出最终版，按规定格式输出。")
        elif int(timeline) == 2:
            user = (f"这段录音长 {minutes:.1f} 分钟（{chunk['end'] - chunk['begin']:.0f} 秒）。时间从 0 算起。\n\n"
                    + timeline2_text(name, chunk, axis)
                    + "\n\n把每一句说的话填进去，按规定格式输出。")
        elif timeline:
            tracks, confidence = axis
            user = (f"这段录音长 {minutes:.1f} 分钟（{chunk['end'] - chunk['begin']:.0f} 秒）。时间从 0 算起。\n\n"
                    + timeline_text(name, chunk, confidence, tracks)
                    + "\n\n把每一段说的话填进去，按规定格式输出。")
        else:
            user = (f"这段录音长 {minutes:.1f} 分钟（{chunk['end'] - chunk['begin']:.0f} 秒）。{knowledge}"
                    "按说话人把它转写出来，按规定格式输出。")
        media = window_media_ref(clip, execution_settings=client.execution_settings, routes=client.router.routes)
        reply = client.complete(LLMRole.LIGHTWEIGHT_MULTIMODAL,
                                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                file_ref=media, **llm.CALL_ROUTING[model])
        rows = parse(reply.content)
        target.write_text(json.dumps({"chunk": chunk["begin"], "end": chunk["end"], "model": reply.target_id,
                                      "raw": reply.content, "rows": rows}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        speakers = sorted({r["speaker"] for r in rows})
        print(f"{name} c{index:02d}{'' if tag is None else f' [{tag}]'} rows={len(rows)} "
              f"speakers={speakers} chars={sum(len(r['text']) for r in rows)}", flush=True)


# ---- measuring --------------------------------------------------------------------------

def common(a, c):
    """Characters the two strings share in order (Ratcliff/Obershelp matching blocks)."""
    return sum(block.size for block in difflib.SequenceMatcher(None, a, c, autojunk=False).get_matching_blocks())


def best_mapping(left, right):
    """Greedy speaker pairing by shared characters; returns {left key: right key}."""
    pairs = sorted(((common(a, c), la, lc) for la, a in left.items() for lc, c in right.items()), reverse=True)
    taken_l, taken_r, out = set(), set(), {}
    for _, la, lc in pairs:
        if la not in taken_l and lc not in taken_r:
            out[la] = lc
            taken_l.add(la)
            taken_r.add(lc)
    return out


def in_time_order(pieces):
    """One string in time order. ⚠ The per-speaker dicts must NOT be concatenated for this:
    joining them speaker by speaker scrambles the sequence and an order-sensitive matcher then
    reports a fraction of the real overlap (the first cut of this probe did exactly that)."""
    return "".join(text for _, text in sorted(pieces))


def production_by_track(session, begin, end):
    """A0 production's characters inside the slice, grouped by the tracks its own labels give."""
    by_track = {}
    for t, c, seg in session.chars:
        if begin <= t <= end:
            for k in session.labels[seg] or (-1,):
                by_track[k] = by_track.get(k, "") + c
    return by_track


def agy_by_speaker(rows):
    out = {}
    for row in rows:
        out[row["speaker"]] = out.get(row["speaker"], "") + b.normalize(row["text"])
    return out


def score(name, model="agy", tags=()):
    # `Session` reads a TextGrid, so a corpus without one (nepolabo) gets the truth-free stub the
    # E4 page builder already needed -- same characters and labels, no reference rows.
    import s4_headroom as ref

    if (ref.GRIDS / f"{name}.TextGrid").exists():
        session, truth = b.Session(name), True
    else:
        import s4_e4_page

        session, truth = s4_e4_page.Stub(name), False
    chunks = list(llm.chunks(name))
    for index, chunk in enumerate(chunks):
        for count in (None, *tags):
            if out_path(name, index, model, count).exists():
                break
        else:
            continue
        variants = [(count, out_path(name, index, model, count)) for count in (None, *tags)
                    if out_path(name, index, model, count).exists()]
        payload = json.loads(variants[0][1].read_text(encoding="utf-8"))
        begin, end = chunk["begin"], chunk["end"]
        free = agy_by_speaker(payload["rows"])
        prod = production_by_track(session, begin, end)
        free_flat = in_time_order((r["start"], b.normalize(r["text"])) for r in payload["rows"])
        prod_flat = "".join(c for t, c, _ in session.chars if begin <= t <= end)
        print(f"\n== {name} c{index:02d}  {begin:.0f}-{end:.0f} s")
        # ⚠ the per-track sum double-counts a multi-label line (its text belongs to every track it
        # carries), so the flat length is the honest character count.
        print(f"   agy: {len(free)} 个说话人, {len(free_flat)} 字 ({len(payload['rows'])} 行)")
        print(f"   生产: {len(prod)} 条轨, {len(prod_flat)} 字（按轨累加 {sum(map(len, prod.values()))}，多标行重复计入）")

        if truth:
            refs = {}
            for row in session.all_refs:
                if begin <= (row["start"] + row["end"]) / 2 <= end:
                    refs[row["spk"]] = refs.get(row["spk"], "") + row["norm"]
            print(f"   真值: {len(refs)} 个说话人, {sum(map(len, refs.values()))} 字")
            ref_flat = in_time_order(((row["start"], row["norm"]) for row in session.all_refs
                                      if begin <= (row["start"] + row["end"]) / 2 <= end))
            total = sum(map(len, refs.values()))
            told = []
            for count, path in variants[1:]:
                rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
                told.append((f"agy [{count}]", agy_by_speaker(rows),
                             in_time_order((r["start"], b.normalize(r["text"])) for r in rows)))
            for label, hyp, flat in [("agy 自主", free, free_flat), *told, ("生产 A0", prod, prod_flat)]:
                mapping = best_mapping(refs, hyp)
                got = sum(common(text, hyp.get(mapping.get(spk), "")) for spk, text in refs.items())
                loose = common(ref_flat, flat)
                print(f"   {label}: {len(hyp)} 个说话人  按说话人算对的字 {got}/{total} = {got / total:.1%}"
                      f"   不管说话人只看内容 {loose / total:.1%}   映射 {mapping}")
        else:
            mapping = best_mapping(prod, free)
            shared = sum(common(text, free.get(mapping.get(k), "")) for k, text in prod.items())
            p_total, f_total = sum(map(len, prod.values())), sum(map(len, free.values()))
            loose = common(prod_flat, free_flat)
            print(f"   两边都写出来的字（按人配对后）{shared}：占生产 {shared / max(p_total, 1):.1%}，"
                  f"占 agy {shared / max(f_total, 1):.1%}")
            print(f"   不管说话人只比内容：共同 {loose} 字 = 生产的 {loose / max(len(prod_flat), 1):.1%}"
                  f" / agy 的 {loose / max(len(free_flat), 1):.1%}")
            print(f"   映射 {mapping}")


AUDIT_SLACK = 2.0  #: agy's own timestamps are estimates, so a window takes the lines near it too


def audit(name, model="agy", tags=()):
    """Side by side in the HARD windows only: our three arms against agy's free transcript.

    `--score` is diluted by the ~70% of speech that is one person talking; S4 only cares about the
    overlap regions, and these are those regions (plan §5o.11).
    """
    import s4_e4_page as page
    import s4_headroom as ref

    session = b.Session(name) if (ref.GRIDS / f"{name}.TextGrid").exists() else page.Stub(name)
    active = proj.activity(session.tracks, int(max(x["end"] for x in session.tracks) / proj.FRAME) + 2)
    versions, layout = page.arms_for(session, active)
    mine = free = windows = 0
    for index, chunk in enumerate(llm.chunks(name)):
        variants = [(count, out_path(name, index, model, count)) for count in (None, *tags)
                    if out_path(name, index, model, count).exists()]
        if not variants:
            continue
        rows = {count: json.loads(path.read_text(encoding="utf-8"))["rows"] for count, path in variants}
        for span in chunk["spans"]:
            if span["kind"] != "real":
                continue
            key, sid = (span["start"], span["end"]), f"{span['start']:.2f}-{span['end']:.2f}"
            w0, w1 = tc.window(span)
            near = lambda r: (r["start"] + chunk["begin"] < w1 + AUDIT_SLACK  # noqa: E731
                              and r["end"] + chunk["begin"] > w0 - AUDIT_SLACK)
            windows += 1
            print(f"\n## {name} c{index:02d}  {w0:.2f}-{w1:.2f} s   在响的轨 {layout[sid]['tracks']}")
            for arm in ("A0-W", "T4", "T2-Q"):
                for tracks, text in versions[arm][key]:
                    if text:
                        print(f"   {arm:<6} [{'+'.join('s%d' % k for k in tracks) or '-'}] {text}")
            for count, batch in rows.items():
                label = "agy" if count is None else f"agy/{count}"
                for row in batch:
                    if near(row):
                        print(f"   {label:<6} [{row['speaker']}] {b.normalize(row['text'])}")
            mine += sum(1 for _, x in versions["A0-W"][key] if x)
            free += sum(1 for r in rows[None] if near(r))
    # ⚠ Lines, not characters: A0-W's lines are clipped to the window and agy's are whole
    # utterances that merely reach into it, so a character ratio between them measures the two
    # clipping rules, not how much either one wrote (the first cut of this audit printed 377%).
    print(f"\n{windows} 个困难窗口：生产 A0-W {mine} 行（裁到窗口内），agy 自主 {free} 行（整句，够到窗口就算）")


def matched_positions(reference, hypothesis):
    """Which indices of `reference` the hypothesis covers (difflib's matching blocks)."""
    hit = set()
    for block in difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False).get_matching_blocks():
        hit.update(range(block.a, block.a + block.size))
    return hit


def char_times(row):
    """Reference rows carry no per-character times, so spread them evenly across the row.

    Coarse on purpose: this ruler only decides whether a window holds a character, and a row is a
    sentence of a few seconds. The scorer in 5o.4 is the one that aligns properly.
    """
    norm = row["norm"]
    if not norm:
        return []
    step = (row["end"] - row["start"]) / len(norm)
    return [(row["start"] + (i + 0.5) * step, c) for i, c in enumerate(norm)]


def windows_of(session, chunk):
    """The overlap windows inside one chunk, merged where they touch."""
    spans = [tc.window(s) for s in session.spans
             if s["kind"] == "real" and s["start"] >= chunk["begin"] and s["end"] <= chunk["end"]]
    merged = []
    for a, c in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], c)
        else:
            merged.append([a, c])
    return merged


def retention(name, model="agy", tags=(), windows_only=False, slack=0.0):
    """T9's criterion 5, computed without another API call.

    A0-W keeps 100% of its own correct characters by definition; every re-decode arm in 5o.10
    scored 57-90% and the bar is 95%. T9 puts agy's text on our axis, so its retention is the share
    of production's correct characters that agy also writes.
    """
    session = b.Session(name)
    chunks = list(llm.chunks(name))
    kept = total = 0
    for index, chunk in enumerate(chunks):
        variants = [(tag, out_path(name, index, model, tag)) for tag in (None, *tags)
                    if out_path(name, index, model, tag).exists()]
        if not variants:
            continue
        begin, end = chunk["begin"], chunk["end"]
        spans = windows_of(session, chunk) if windows_only else [[begin, end]]
        inside = lambda x: any(a <= x <= c for a, c in spans)  # noqa: E731
        ref_flat = in_time_order((moment, c) for row in session.all_refs
                                 if begin <= (row["start"] + row["end"]) / 2 <= end
                                 for moment, c in char_times(row) if inside(moment))
        # Both hypotheses get the same reach past the window. agy's lines carry no character
        # times, so a line counts when it touches the widened window; production's characters
        # count under the same widening. The first version widened agy only (slack 2 s against
        # production cut to the window): it moved `tl2` from 51.0% to 56.7% and reordered the
        # three agy forms (audit 5q, A). Default slack is 0.
        near = lambda x: any(a - slack <= x <= c + slack for a, c in spans)  # noqa: E731
        prod_flat = "".join(c for t, c, _ in session.chars if begin <= t <= end and near(t))
        right = matched_positions(ref_flat, prod_flat)
        where = f"{len(spans)} 个窗口（slack {slack:g} s）" if windows_only else "整段"
        print(f"\n== {name} c{index:02d}  {where}：生产写对 {len(right)} 字 / 参考 {len(ref_flat)}")
        for tag, path in variants:
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            flat = in_time_order((r["start"], b.normalize(r["text"])) for r in rows
                                 if any(r["start"] + begin < c + slack and r["end"] + begin > a - slack
                                        for a, c in spans))
            mine = matched_positions(ref_flat, flat)
            both = right & mine
            extra = mine - right
            label = "agy 自主" if tag is None else f"agy [{tag}]"
            print(f"   {label:<12} 保留生产写对的 {len(both)}/{len(right)} = {len(both) / max(len(right), 1):.1%}"
                  f"   另外救回生产没写对的 {len(extra)} 字（+{len(extra) / max(len(ref_flat), 1):.1%} 参考）")
            if tag is None:
                kept += len(both)
                total += len(right)
    if total:
        print(f"\n合计（agy 自主）：保留率 {kept / total:.1%}  —— 判据 5 要 >= 95%")
    return kept, total


def variant_tags(arguments):
    return tuple(f"n{n}" for n in arguments.with_counts) + tuple(arguments.with_tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", nargs="+", metavar=("SESSION", "INDEX"))
    parser.add_argument("--score", metavar="SESSION")
    parser.add_argument("--audit", metavar="SESSION", help="side by side in the overlap windows only")
    parser.add_argument("--retention", metavar="SESSION", help="would T9 keep production's correct characters?")
    parser.add_argument("--windows", action="store_true", help="restrict --retention to the overlap windows")
    parser.add_argument("--slack", type=float, default=0.0,
                        help="seconds both hypotheses may reach past a window (--retention --windows); 5o.12 used 2 on agy only")
    parser.add_argument("--model", default="agy")
    parser.add_argument("--count", type=int, help="tell the model how many people are talking (probe 2)")
    parser.add_argument("--with-counts", type=int, nargs="*", default=(), help="told-the-count variants to show beside it")
    parser.add_argument("--timeline", type=int, nargs="?", const=1, default=0,
                        help="1 = raw Sortformer axis (probe 3), 2 = the pipeline's segments (3b), 3 = two-pass repair (4)")
    parser.add_argument("--with-tags", nargs="*", default=(), help="other variant tags to show beside it")
    arguments = parser.parse_args()
    if arguments.ask:
        ask(arguments.ask[0], [int(x) for x in arguments.ask[1:]], arguments.model, arguments.count,
            arguments.timeline)
    if arguments.score:
        score(arguments.score, arguments.model, variant_tags(arguments))
    if arguments.audit:
        audit(arguments.audit, arguments.model, variant_tags(arguments))
    if arguments.retention:
        retention(arguments.retention, arguments.model, variant_tags(arguments), arguments.windows, arguments.slack)
    if not (arguments.ask or arguments.score or arguments.audit or arguments.retention):
        parser.print_help()


if __name__ == "__main__":
    main()
