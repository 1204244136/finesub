"""MOSS-Transcribe-Diarize (local 0.9B) through a serving engine (vLLM or SGLang-Omni, in WSL2).

The engine exposes the OpenAI-compatible `POST <url>/v1/audio/transcriptions`; this client uploads one wav,
greedy, and reduces the answer to the survey schema. Both engines return the raw transcript markup
`[start][Sxx]text[end]` in `text` (vLLM refuses `verbose_json` for this model; SGLang-Omni's `verbose_json`
folds the speaker into the segment text and clamps ends to duration + 5 s), so the markup is parsed here with
the upstream parser -- the same one `moss_run.py` uses -- and engine segments are only a fallback. No word timing.

    python -m tools.ts_diar_explore.moss_serve <audio> <out.json> --url http://localhost:8000 [--offset S]
        [--engine vllm|sglang] [--max-new-tokens N] [--format json|verbose_json|text]

`wall` is the request round trip (upload + encode + prefill + decode), server startup excluded.
⚠ SGLang-Omni stops silently when the KV pool cannot hold prompt + output (a 37 min clip at
`--mem-fraction-static 0.40` came back cut at 29 min, HTTP 200): check `last_end` against the duration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import requests

from .common import TMP, probe_duration, save

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
#: moss_run.py's cap: ~15 generated tokens per audio second plus slack, so a repetition loop cannot run to 65 536.
TOKENS_PER_SECOND = 15


def _parser():
    """`moss_transcribe_diarize.transcript_parser` straight from the upstream checkout: the package `__init__`
    imports torch / transformers, the parser module itself is pure Python."""
    path = TMP / "moss" / "src" / "moss_transcribe_diarize" / "transcript_parser.py"
    spec = importlib.util.spec_from_file_location("moss_transcript_parser", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass(slots=True) looks its module up
    spec.loader.exec_module(mod)
    return mod.parse_transcript


def call(url: str, audio: Path, *, fmt: str, max_new_tokens: int, timeout: float) -> tuple[dict, float]:
    data = {"model": MODEL, "response_format": fmt, "temperature": "0",
            "max_new_tokens": str(max_new_tokens), "max_completion_tokens": str(max_new_tokens)}
    t0 = time.monotonic()
    with audio.open("rb") as f:
        r = requests.post(f"{url.rstrip('/')}/v1/audio/transcriptions", data=data,
                          files={"file": (audio.name, f, "audio/wav")}, timeout=timeout)
    wall = time.monotonic() - t0
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:2000]}")
    return (r.json() if fmt != "text" else {"text": r.text}), wall


def parse(body: dict, *, offset: float) -> tuple[list[dict], str]:
    """Segments from the transcript markup when the answer carries it, else from the engine's segments."""
    text = body.get("text") or ""
    if "[S" in text:
        segs = [{"start": round(s.start + offset, 3), "end": round(s.end + offset, 3), "speaker": s.speaker, "text": s.text}
                for s in _parser()(text)]
        if segs:
            return segs, "markup"
    segs = []
    for s in body.get("segments", []) or []:
        t, spk = s.get("text", ""), s.get("speaker")
        if spk is None and t.startswith("[S") and "]" in t:  # SGLang-Omni: "[S01]..."
            spk, t = t[1:t.index("]")], t[t.index("]") + 1:]
        segs.append({"start": round(float(s["start"]) + offset, 3), "end": round(float(s["end"]) + offset, 3),
                     "speaker": spk, "text": t})
    return segs, "segments"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--engine", default="", help="label stored in the result (vllm / sglang)")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--format", default="json", choices=["json", "verbose_json", "text"])
    ap.add_argument("--max-new-tokens", type=int, default=65536,
                    help="upper bound; the request asks for min(this, 15 tokens per audio second + 256)")
    ap.add_argument("--timeout", type=float, default=7200.0)
    a = ap.parse_args()
    dur = probe_duration(a.audio)
    cap = min(a.max_new_tokens, int(TOKENS_PER_SECOND * dur) + 256)
    body, wall = call(a.url, a.audio, fmt=a.format, max_new_tokens=cap, timeout=a.timeout)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".raw.json").write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    segs, source = parse(body, offset=a.offset)
    usage = body.get("usage") or {}
    save({"model": f"moss-{a.engine}" if a.engine else "moss-serve",
          "config": {"url": a.url, "format": a.format, "max_new_tokens": cap, "parsed_from": source},
          "audio": str(a.audio), "offset": a.offset, "duration": dur, "wall": round(wall, 2), "usage": usage,
          "segments": segs, "words": None}, a.out)
    print(f"{a.engine or 'serve'} {a.audio.name}: {len(segs)} segments ({source}), "
          f"{len({s['speaker'] for s in segs})} labels, {wall:.1f}s wall (RTF {wall / dur:.3f}) -> {a.out}")


if __name__ == "__main__":
    main()
