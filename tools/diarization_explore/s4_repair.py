"""Plan 5h.6: masked-projection candidates, then a multimodal model keeps what it hears.

Same chunks, real spans, decoys and scoring as 5h.4 (`s4_llm`). Real spans get
the 5h.3 per-track masked decodes as candidates (the two most active tracks);
decoys get candidates from the *same* mask: one decode of their only speaker,
one with that speaker attenuated -- so "has candidates" cannot give a real span
away, and the decoy candidates carry the same artefacts.

  --decode-decoys        GPU, once
  --ask MODEL RUN        lite | agy
  --score MODEL          5h.4 criteria + copy / `from` / redundancy report
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

import s4_headroom as ref
import s4_llm as llm
import s4_projection as proj

HERE = Path(__file__).resolve().parent
BLOCK = re.compile(r"<overlap_repair>(.*?)</overlap_repair>", re.DOTALL)
PREFIX = "repair"


def decoy_path(session):
    return proj.ALI / session / "decoy_projection.json"


def decode_decoys():
    import soundfile

    from finesub.speech.recognition.fw_refine_backend import RefinedWhisperModel
    from finesub.speech.recognition.transcribe import REFINE_SEC

    model = RefinedWhisperModel("large-v3-turbo", device="cuda", compute_type="float16", refine_sec=REFINE_SEC)
    for session in ref.SESSIONS:
        if decoy_path(session).exists():
            continue
        tracks = proj.timeline(session)
        audio, rate = soundfile.read(str(proj.ALI / session / f"{session}-vocal.ogg"), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        active = proj.activity(tracks, int(len(audio) / proj.SR / proj.FRAME) + 2)
        records = []
        for chunk in llm.chunks(session):
            for span in chunk["spans"]:
                if span["kind"] != "decoy":
                    continue
                begin = max(0, int((span["start"] - proj.CONTEXT) * proj.SR))
                finish = min(len(audio), int((span["end"] + proj.CONTEXT) * proj.SR))
                first, last = int(span["start"] / proj.FRAME), int(np.ceil(span["end"] / proj.FRAME))
                speaker = int(np.argmax(active[first:last].sum(axis=0)))
                other = (speaker + 1) % active.shape[1]  # any track that is not talking here
                clip, offset = audio[begin:finish], begin / proj.SR
                records.append(
                    {
                        "start": span["start"],
                        "end": span["end"],
                        "a": proj.decode(model, clip * proj.gain_for(active, speaker, begin, finish), offset),
                        "b": proj.decode(model, clip * proj.gain_for(active, other, begin, finish), offset),
                    }
                )
        decoy_path(session).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(session, "decoys", len(records), flush=True)


def _raw(words, start, end):
    return "".join(
        w["word"].strip() for w in words if start - ref.PAD <= (w["start"] + w["end"]) / 2 <= end + ref.PAD
    )


def candidates(session):
    """(start, end) -> (a, b) raw candidate strings, real spans and decoys alike."""

    tracks = proj.timeline(session)
    frames = int(max(t["end"] for t in tracks) / proj.FRAME) + 2
    active = proj.activity(tracks, frames)
    out = {}
    for record in json.loads((proj.ALI / session / "projection.json").read_text(encoding="utf-8")):
        first, last = int(record["start"] / proj.FRAME), int(np.ceil(record["end"] / proj.FRAME))
        # Two candidates per span, as the decoys have: the two most active tracks.
        ranked = sorted(record["masked"], key=lambda k: -int(active[first:last, int(k)].sum()))[:2]
        texts = [_raw(record["masked"][k], record["start"], record["end"]) for k in ranked] + ["", ""]
        out[(record["start"], record["end"])] = (texts[0], texts[1])
    for record in json.loads(decoy_path(session).read_text(encoding="utf-8")):
        out[(record["start"], record["end"])] = (
            _raw(record["a"], record["start"], record["end"]),
            _raw(record["b"], record["start"], record["end"]),
        )
    return out


def build_messages(chunk, segments, cands):
    lines = "\n".join(
        f"{line}|{max(0.0, start - chunk['begin']):.1f}|{end - chunk['begin']:.1f}|{text.strip()}"
        for line, start, end, text in segments
        if end > chunk["begin"] and start < chunk["end"]
    )
    blocks = []
    for span in chunk["spans"]:
        a, b = cands[(span["start"], span["end"])]
        blocks.append(
            f"{span['id']}|{span['start'] - chunk['begin']:.1f}|{span['end'] - chunk['begin']:.1f}\n"
            f"  a: {a or '（空）'}\n  b: {b or '（空）'}"
        )
    system = (HERE / "overlap_repair_system.md").read_text(encoding="utf-8")
    user = string.Template((HERE / "overlap_repair_user.md").read_text(encoding="utf-8")).substitute(
        duration=f"{chunk['end'] - chunk['begin']:.1f}", lines=lines or "（本段没有现有字幕）", spans="\n".join(blocks)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse(text):
    match = BLOCK.search(text or "")
    if not match:
        return None
    rows = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "text" in row:
            rows.append({**row, "kind": "add"})  # the scorer in s4_llm reads `kind`
    return rows


def ask(model, run):
    from finesub.llm.media_upload import window_media_ref
    from finesub.llm.routing.config import LLMRole

    client = llm.client_for(model)
    for session in ref.SESSIONS:
        out = proj.ALI / session / f"{PREFIX}-{model}-run{run}.json"
        if out.exists():
            continue
        segments = llm.stable_segments(session)
        cands = candidates(session)
        records = []
        for index, chunk in enumerate(llm.chunks(session)):
            clip = proj.ALI / session / "llm-clips" / f"c{index:02d}.wav"  # the 5h.5 clips, byte-identical
            record = dict(chunk)
            try:
                media = window_media_ref(clip, execution_settings=client.execution_settings, routes=client.router.routes)
                reply = client.complete(
                    LLMRole.LIGHTWEIGHT_MULTIMODAL,
                    build_messages(chunk, segments, cands),
                    file_ref=media,
                    **llm.CALL_ROUTING[model],
                )
                record["raw"] = reply.content
                record["model"] = reply.target_id
                record["rows"] = parse(record["raw"])
            except Exception as error:  # a failed chunk counts as "nothing added"
                record["raw"], record["rows"], record["error"] = "", None, repr(error)
            records.append(record)
            added = len(record["rows"] or ())
            print(f"{session} c{index:02d} adds={added} {'PARSE-FAIL' if record['rows'] is None else ''} {record.get('error', '')[:120]}", flush=True)
        out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


def report_copying(model):
    """Not criteria: how the additions relate to the candidates and to what is already there."""

    for run in (1, 2):
        if not (proj.ALI / ref.SESSIONS[0] / f"{PREFIX}-{model}-run{run}.json").exists():
            continue
        origin, copied, total, redundant, grams = Counter(), 0, 0, 0, 0
        for session in ref.SESSIONS:
            cands = candidates(session)
            words = ref.hypothesis(session)
            for record in json.loads((proj.ALI / session / f"{PREFIX}-{model}-run{run}.json").read_text(encoding="utf-8")):
                spans = {s["id"]: s for s in record["spans"]}
                for row in record["rows"] or ():
                    text = ref.normalize(str(row.get("text", "")))
                    if not text:
                        continue
                    total += 1
                    origin[str(row.get("from", "?"))] += 1
                    span = spans.get(row.get("span"))
                    if span and text in {ref.normalize(c) for c in cands[(span["start"], span["end"])] if c}:
                        copied += 1
                    try:
                        start, end = record["begin"] + float(row["start"]), record["begin"] + float(row["end"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    have = set(ref.bigrams(ref.window_text(words, start, end)))
                    added = ref.bigrams(text)
                    grams += sum(added.values())
                    redundant += sum(n for g, n in added.items() if g in have)
        print(
            f"  {PREFIX}-{model} run{run}: adds {total}, verbatim copy of a candidate {copied}/{total or 1} "
            f"({copied / (total or 1):.0%}), from {dict(origin)}, redundancy {redundant / (grams or 1):.0%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-decoys", action="store_true")
    parser.add_argument("--ask", nargs=2, metavar=("MODEL", "RUN"))
    parser.add_argument("--score", metavar="MODEL")
    parser.add_argument("--show", action="store_true", help="print one chunk's user prompt")
    arguments = parser.parse_args()
    if arguments.decode_decoys:
        decode_decoys()
    if arguments.show:
        session = ref.SESSIONS[1]
        print(build_messages(llm.chunks(session)[2], llm.stable_segments(session), candidates(session))[1]["content"])
    if arguments.ask:
        ask(arguments.ask[0], int(arguments.ask[1]))
    if arguments.score:
        llm.score(arguments.score, prefix=PREFIX)
        report_copying(arguments.score)


if __name__ == "__main__":
    main()
