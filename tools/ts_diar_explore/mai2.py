"""MAI-Transcribe-2 through Azure fast transcription.

Authentication and endpoint come from `~/.agents/azure/transcribe.py` (Azure hard rule 1: only its scripts
touch Azure). That script does not expose `timestamps`, so the request is rebuilt here with the same
endpoint, token and multipart shape, adding only `definition` fields its README allows.

    python -m tools.ts_diar_explore.mai2 <audio> <out.json> [--offset S] [--diarize] [--timestamps word|segment|none]
                                          [--locale ja] [--clean] [--phrases a,b]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from .common import ledger, mai2_hours_used, probe_duration, save

sys.path.insert(0, str(Path.home() / ".agents" / "azure"))
import transcribe as azure  # noqa: E402

API = f"{azure.ENDPOINT}/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
#: Owner budget is 20 audio hours for this survey; stop and report before the last two.
BUDGET_STOP_HOURS = 18.0


def definition(*, diarize: bool, timestamps: str, locale: str | None, clean: bool, phrases: list[str]) -> dict:
    options: dict = {"timestamps": timestamps}
    if clean:
        options["transcribeStyle"] = "clean"
    d: dict = {"enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2", "modelOptions": options}}
    if diarize:
        d["diarization"] = {"enabled": True}
    if locale:
        d["locales"] = [locale]
    if phrases:
        d["phraseList"] = {"phrases": phrases}
    return d


def call(audio: Path, defn: dict, *, config: str) -> dict:
    seconds = probe_duration(audio)
    if mai2_hours_used() + seconds / 3600 > BUDGET_STOP_HOURS:
        raise SystemExit(f"mai2 budget: {mai2_hours_used():.2f} h used, stop at {BUDGET_STOP_HOURS} h -- report first")
    t0 = time.monotonic()
    status = None
    try:
        with audio.open("rb") as f:
            r = requests.post(
                API,
                headers={"Authorization": f"Bearer {azure._token()}"},
                files={"audio": f, "definition": (None, json.dumps(defn))},
                timeout=1800,
            )
        status = r.status_code
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {status}: {e.response.text[:2000]}") from None
    finally:
        ledger(model="mai2", config=config, audio=str(audio), audio_seconds=round(seconds, 2),
               status=status, wall=round(time.monotonic() - t0, 2))


def parse(resp: dict, *, offset: float) -> tuple[list[dict], list[dict] | None]:
    segments, words = [], []
    for ph in resp.get("phrases", []):
        spk = ph.get("speaker")
        spk = None if spk is None else f"S{spk}"
        start = ph["offsetMilliseconds"] / 1000 + offset
        segments.append({"start": start, "end": start + ph["durationMilliseconds"] / 1000,
                         "speaker": spk, "text": ph.get("text", ""), "locale": ph.get("locale")})
        for w in ph.get("words", []) or []:
            ws = w["offsetMilliseconds"] / 1000 + offset
            words.append({"start": ws, "end": ws + w["durationMilliseconds"] / 1000, "speaker": spk, "text": w["text"]})
    return segments, (words or None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--timestamps", default="word", choices=["word", "segment", "none"])
    ap.add_argument("--locale")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--phrases", default="")
    a = ap.parse_args()
    phrases = [p for p in a.phrases.split(",") if p]
    defn = definition(diarize=a.diarize, timestamps=a.timestamps, locale=a.locale, clean=a.clean, phrases=phrases)
    config = f"ts={a.timestamps},diar={int(a.diarize)},locale={a.locale},clean={int(a.clean)},phrases={len(phrases)}"
    t0 = time.monotonic()
    resp = call(a.audio, defn, config=config)
    wall = time.monotonic() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".raw.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
    segments, words = parse(resp, offset=a.offset)
    save({"model": "mai2", "config": defn, "audio": str(a.audio), "offset": a.offset,
          "duration": probe_duration(a.audio), "wall": round(wall, 2), "segments": segments, "words": words}, a.out)
    print(f"mai2 {config}: {len(segments)} segments, {len(words or [])} words, {wall:.1f}s wall, "
          f"budget used {mai2_hours_used():.2f} h -> {a.out}")


if __name__ == "__main__":
    main()
