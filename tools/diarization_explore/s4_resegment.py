"""Plan 5h.14: re-segment overlaps by speaker -- one line per person -- with a model that hears the audio.

Same 40 chunks, 357 real spans and 267 decoys as 5h.4-5h.7 (`s4_llm.chunks`, the byte-identical
`llm-clips`). Per span the model gets the acoustic facts (which tracks sound, when), Qwen's
plain re-transcription and Qwen's per-track anchored ones (de-leaked, 5h.13), and writes one
line per speaker, `s1|s2` only when it cannot tell whose.

  --show           print one chunk's user prompt
  --ask MODEL RUN  lite | agy
  --score          A0 (production lines), A1 (per-track text as lines), A2 (each model run)
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import s4_breakdown as b
import s4_headroom as ref
import s4_llm as llm
import s4_projection as proj

HERE = Path(__file__).resolve().parent
BLOCK = re.compile(r"<resegment>(.*?)</resegment>", re.DOTALL)
PREFIX = "reseg"
SYSTEM = "resegment_system.md"
#: --variant v2 (plan 5h.16): rule 4 inverted -- unless sure whose it is, name every candidate track.
VARIANTS = {"v1": ("reseg", "resegment_system.md"), "v2": ("reseg2", "resegment_v2_system.md"),
            # plan 5h.18 (T5): the v2 prompt, per-track candidates from 5h.17's T2-Q instead of the anchored re-decode
            "v2t": ("reseg2t", "resegment_v2t_system.md"),
            # plan 5o.9 E1: the same two with no audio -- what hearing it contributes
            "v2-text": ("reseg2-text", "resegment_v2_text_system.md"),
            "v2t-text": ("reseg2t-text", "resegment_v2t_text_system.md")}
CANDIDATES = "anchored"
TEXT_ONLY = False
HEARD = ("right", "homophone", "misheard")


def activity_facts(active, span, begin):
    first, last = int(span["start"] / proj.FRAME), int(np.ceil(span["end"] / proj.FRAME))
    facts = []
    for k in range(active.shape[1]):
        on = active[first:last, k]
        if not on.any():
            continue
        runs, start = [], None
        for index, flag in enumerate(np.append(on, False)):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                runs.append(f"{(first + start) * proj.FRAME - begin:.1f}–{(first + index) * proj.FRAME - begin:.1f}")
                start = None
        facts.append(f"s{k} " + "、".join(runs))
    return "；".join(facts)


_TIMECUT: dict[str, tuple[dict, dict]] = {}


def span_inputs(session: b.Session, span):
    """The two kinds of candidates, as scored in 5h.13: plain, and per track de-leaked.

    With CANDIDATES == "timecut" the per-track ones are 5h.17's T2-Q lines (every track
    sounding in the +-1 s window) rather than the anchored re-decode of the span.
    """
    texts = session.span_texts(span)
    plain = texts["qwen-plain"][0][1]
    if CANDIDATES == "timecut":
        import s4_timecut as tc

        if session.name not in _TIMECUT:
            _TIMECUT[session.name] = (
                json.loads((proj.ALI / session.name / "timecut_qwen.json").read_text(encoding="utf-8")),
                json.loads((proj.ALI / session.name / "timecut_layout.json").read_text(encoding="utf-8")),
            )
        results, layout = _TIMECUT[session.name]
        lines = tc.variant_lines(session, results, layout, "T2")[(span["start"], span["end"])]
        return plain, {tracks[0]: text for tracks, text, _ in lines}
    per_track = {k: b.deleak(text, anchor) for k, text, anchor in texts["qwen-anchored"]}
    return plain, per_track


def build_messages(session: b.Session, chunk, active):
    begin = chunk["begin"]
    lines = []
    for index, segment in enumerate(json.loads((proj.ALI / session.name / f"{session.name}-stable.json").read_text(encoding="utf-8"))["segments"]):
        if segment["end"] > begin and segment["start"] < chunk["end"]:
            spk = "+".join(f"s{k}" for k in session.labels[index]) or "（未标）"
            lines.append(f"{index + 1}|{max(0.0, segment['start'] - begin):.1f}|{segment['end'] - begin:.1f}|{spk}|{segment['text'].strip()}")
    blocks = []
    for span in chunk["spans"]:
        plain, per_track = span_inputs(session, span)
        block = [f"{span['id']}|{span['start'] - begin:.1f}|{span['end'] - begin:.1f}",
                 f"  发声：{activity_facts(active, span, begin) or '（无）'}",
                 f"  Q：{plain or '（空）'}"]
        block += [f"  s{k}：{text or '（空）'}" for k, text in sorted(per_track.items())]
        blocks.append("\n".join(block))
    system = (HERE / SYSTEM).read_text(encoding="utf-8")
    template = "resegment_user_text.md" if TEXT_ONLY else "resegment_user.md"
    user = string.Template((HERE / template).read_text(encoding="utf-8")).substitute(
        duration=f"{chunk['end'] - begin:.1f}", lines="\n".join(lines) or "（本段没有现有字幕）", spans="\n\n".join(blocks))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse(text):
    match = BLOCK.search(text or "")
    if not match:
        return None
    rows = []
    for line in match.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "text" in row and "span" in row:
            tracks = {int(t) for t in re.findall(r"s?(\d+)", str(row.get("track", "")))}
            rows.append({"span": row["span"], "tracks": sorted(tracks), "text": str(row["text"]), "conf": row.get("conf")})
    return rows


def llm_clip(name, chunk, clip):
    """Write one chunk's clip exactly as `s4_llm.ask` does: the source mixed to mono."""
    import soundfile

    audio, rate = soundfile.read(str(llm.source(name)), dtype="float32", always_2d=True)
    clip.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(clip), audio.mean(axis=1)[int(chunk["begin"] * rate): int(chunk["end"] * rate)], rate)


def ask(model, run):
    from finesub.llm.media_upload import window_media_ref
    from finesub.llm.routing.config import LLMRole

    client = llm.client_for(model)
    for name in ref.SESSIONS:
        out = proj.ALI / name / f"{PREFIX}-{model}-run{run}.json"
        if out.exists():
            continue
        session = b.Session(name)
        tracks = proj.timeline(name)
        active = proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)
        records, in_a_row = [], 0
        for index, chunk in enumerate(llm.chunks(name)):
            clip = proj.ALI / name / "llm-clips" / f"c{index:02d}.wav"  # the 5h.5 clips
            if not clip.exists() and not TEXT_ONLY:  # a corpus 5h.5 never ran on (5o.9 E3): cut it the same way
                llm_clip(name, chunk, clip)
            record = dict(chunk)
            try:
                media = None if TEXT_ONLY else window_media_ref(clip, execution_settings=client.execution_settings, routes=client.router.routes)
                reply = client.complete(LLMRole.LIGHTWEIGHT_MULTIMODAL, build_messages(session, chunk, active),
                                        file_ref=media, **llm.CALL_ROUTING[model])
                record["raw"], record["model"] = reply.content, reply.target_id
                record["rows"] = parse(record["raw"])
            except Exception as error:  # a failed chunk counts as "nothing written"
                record["raw"], record["rows"], record["error"] = "", None, repr(error)
            records.append(record)
            print(f"{name} c{index:02d} rows={len(record['rows'] or ())} {'PARSE-FAIL' if record['rows'] is None else ''} "
                  f"{record.get('model', '')} {record.get('error', '')[:120]}", flush=True)
            # An outage must not look like a result: agy's quota pool freezing mid-run
            # (2026-09-11) wrote whole sessions of empty records, which score as "the arm
            # produced nothing" rather than "the arm never ran". Stop instead of writing.
            in_a_row = in_a_row + 1 if record["rows"] is None else 0
            if in_a_row >= 3:
                raise SystemExit(f"{name}: {in_a_row} calls failed in a row, nothing written -- {record.get('error', '')[:200]}")
        out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


# ---- scoring -------------------------------------------------------------------------

def attribution(own, tracks):
    if own is None or not tracks:
        return "wrong"
    if own in tracks:
        return "strict" if len(tracks) == 1 else "multi"
    return "wrong"


RANK_ATTR = {"strict": 3, "multi": 2, "wrong": 1}


def arm_lines(session, span_align, arm, model_rows=None):
    """span (start, end) -> [(tracks, text, aligned)] for one arm."""
    out = defaultdict(list)
    if arm == "A0":
        for span in session.spans:
            chars = [(c, seg) for t, c, seg in session.chars
                     if span["start"] - b.live.SPAN_PAD <= t <= span["end"] + b.live.SPAN_PAD]
            # one "line" per production line inside the span, carrying its labels
            by_line = defaultdict(str)
            for c, seg in chars:
                by_line[seg] += c
            for seg, text in by_line.items():
                out[(span["start"], span["end"])].append((tuple(session.labels[seg]), text, None))
    elif arm == "A1":
        for (s, e, a, k), (text, al) in span_align.items():
            if a == "qwen-anchored":
                out[(s, e)].append(((k,), text, al))
    else:
        for (s, e), rows in model_rows.items():
            for row in rows:
                out[(s, e)].append((tuple(row["tracks"]), b.normalize(row["text"]), None))
    return out


def score_arm(sessions, label, lines_for):
    ref_side, hyp_side, decoy, floor = Counter(), Counter(), Counter(), Counter()
    for session, span_align, lines in lines_for:
        by_id = {row["id"]: row for row in session.refs}
        tracks = proj.timeline(session.name)
        active = proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)

        def sounding(k, spans):
            # Not pre-registered: does the speaker's own track sound anywhere in these spans?
            return k is not None and k < active.shape[1] and any(
                active[int(s["start"] / proj.FRAME): int(np.ceil(s["end"] / proj.FRAME)), k].any() for s in spans)
        cache = {}
        for key, items in lines.items():
            near = session.near(key[0], key[1], b.NEAR)
            cache[key] = [(tracks, text, al or b.align(text, near)) for tracks, text, al in items]
        for side, row in b.pair_members(session):
            own = session.speaker_track.get(row["spk"])
            best = ["miss"] * len(row["norm"])
            for span in b.touching(session, row):
                for tracks, _, al in cache.get((span["start"], span["end"]), ()):
                    if row["id"] not in al["refs"]:
                        continue
                    kind = attribution(own, tracks)
                    for i, value in enumerate(al["refs"][row["id"]][0]):
                        if value in HEARD and (best[i] == "miss" or RANK_ATTR[kind] > RANK_ATTR[best[i]]):
                            best[i] = kind
            ref_side.update(best)
            own_sounds = sounding(own, b.touching(session, row))
            floor[("wrong", own_sounds)] += best.count("wrong")
        kinds = {(s["start"], s["end"]): s["kind"] for s in session.spans}
        for key, items in cache.items():
            for tracks, text, al in items:
                bad = 0
                for j in range(len(text)):
                    if j in al["claimed"]:
                        owner = session.speaker_track.get(by_id[al["claimed"][j]]["spk"])
                        what = attribution(owner, tracks)
                        what = {"strict": "own", "multi": "multi incl. own", "wrong": "someone else's"}[what]
                    elif j in al["near"]:
                        what = "near-miss"
                    else:
                        what, bad = "nothing near", bad + 1
                    if kinds[key] == "real":
                        hyp_side[what] += 1
                if kinds[key] == "decoy" and text:
                    decoy["lines"] += 1
                    decoy["confabulated lines"] += bad / len(text) > 0.5
        decoy["decoy spans"] += sum(k == "decoy" for k in kinds.values())
        decoy["decoy spans with a confabulated line"] += len({key for key, items in cache.items() if kinds[key] == "decoy"
                                                               and any(t and sum(j not in a["claimed"] and j not in a["near"] for j in range(len(t))) / len(t) > 0.5
                                                                       for _, t, a in items)})
    total = sum(ref_side.values())
    heard = ref_side["strict"] + ref_side["multi"] + ref_side["wrong"]
    print(f"  {label:<18} heard {heard / total:6.1%} of reference chars | of heard: strict {ref_side['strict'] / max(heard, 1):6.1%}  "
          f"multi {ref_side['multi'] / max(heard, 1):6.1%}  wrong {ref_side['wrong'] / max(heard, 1):6.1%}  (chars {total})")
    written = sum(hyp_side.values())
    print(f"  {'':<18} written {written}: " + "  ".join(f"{k} {v / max(written, 1):.1%}" for k, v in hyp_side.most_common()))
    wrong_total = floor[("wrong", True)] + floor[("wrong", False)]
    print(f"  {'':<18} [post hoc] of the wrong-speaker chars, the speaker's own track never sounds in those spans: "
          f"{floor[('wrong', False)]}/{wrong_total} ({floor[('wrong', False)] / max(wrong_total, 1):.0%}) -- the acoustic floor")
    confab = decoy["decoy spans with a confabulated line"] / max(decoy["decoy spans"], 1)
    print(f"  {'':<18} decoys: {dict(decoy)} -> confabulation {confab:.1%}")
    return {"heard": heard / total, "strict": ref_side["strict"] / max(heard, 1), "wrong": ref_side["wrong"] / max(heard, 1), "confab": confab}


def model_rows(name, model, run):
    path = proj.ALI / name / f"{PREFIX}-{model}-run{run}.json"
    if not path.exists():
        return None, 0, (0, 0)
    rows, failures, multi, count = defaultdict(list), 0, 0, 0
    for record in json.loads(path.read_text(encoding="utf-8")):
        if record["rows"] is None:
            failures += 1
            continue
        spans = {s["id"]: s for s in record["spans"]}
        for row in record["rows"]:
            span = spans.get(row["span"])
            if span and row["tracks"]:
                rows[(span["start"], span["end"])].append(row)
                count += 1
                multi += len(row["tracks"]) > 1
    return rows, failures, (multi, count)


def score():
    loaded = []
    for name in ref.SESSIONS:
        session = b.Session(name)
        span_align, _ = b.collect(session)
        loaded.append((session, span_align))
    results = {}
    print("REFERENCE SIDE (overlapped pair members touching a real span); decoys as in 5h.4")
    results["A0"] = score_arm(None, "A0 production", [(s, a, arm_lines(s, a, "A0")) for s, a in loaded])
    results["A1"] = score_arm(None, "A1 per-track text", [(s, a, arm_lines(s, a, "A1")) for s, a in loaded])
    for model in ("lite", "agy"):
        runs = []
        for run in (1, 2):
            parts, fails, multis = [], 0, [0, 0]
            for session, span_align in loaded:
                rows, failures, (multi, count) = model_rows(session.name, model, run)
                if rows is None:
                    break
                fails += failures
                multis[0] += multi
                multis[1] += count
                parts.append((session, span_align, arm_lines(session, span_align, "A2", rows)))
            else:
                print(f"  [{model} run{run}] parse failures {fails}/40, multi-track lines {multis[0]}/{multis[1]}")
                runs.append(score_arm(None, f"A2 {model} run{run}", parts))
        if runs:
            mean = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
            a0 = results["A0"]
            checks = [("内容不丢 heard >= A0 - 2 pp", mean["heard"] >= a0["heard"] - 0.02, f"{mean['heard']:.1%} vs {a0['heard']:.1%}"),
                      ("严格归对 >= A0 + 20 pp", mean["strict"] >= a0["strict"] + 0.20, f"{mean['strict']:.1%} vs {a0['strict']:.1%}"),
                      ("归错人 <= 10%", mean["wrong"] <= 0.10, f"{mean['wrong']:.1%}"),
                      ("诱饵编造 <= 10%", mean["confab"] <= 0.10, f"{mean['confab']:.1%}")]
            for text, ok, value in checks:
                print(f"    {'PASS' if ok else 'FAIL'}  {text}: {value}")
            print("    => " + ("可行" if all(ok for _, ok, _ in checks) else "不做"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--ask", nargs=2, metavar=("MODEL", "RUN"))
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="v1")
    arguments = parser.parse_args()
    global PREFIX, SYSTEM, CANDIDATES, TEXT_ONLY
    PREFIX, SYSTEM = VARIANTS[arguments.variant]
    CANDIDATES = "timecut" if arguments.variant.startswith("v2t") else "anchored"
    TEXT_ONLY = arguments.variant.endswith("-text")
    if arguments.show:
        name = ref.SESSIONS[1]
        session = b.Session(name)
        tracks = proj.timeline(name)
        active = proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)
        print(build_messages(session, llm.chunks(name)[2], active)[1]["content"])
    if arguments.ask:
        ask(arguments.ask[0], int(arguments.ask[1]))
    if arguments.score:
        score()


if __name__ == "__main__":
    main()
