"""MOSS-Transcribe-Diarize (local 0.9B). Runs under its own venv -- it needs transformers 5.x, envs/asr has 4.x:

    tmp/ts-diar/moss/.venv/Scripts/python.exe -m tools.ts_diar_explore.moss_run <out_dir> <audio>[@offset] ...

One model load, many files. Per file: `<stem>.raw.txt` (the generated transcript verbatim) and `<stem>.json`
in the survey schema. Segment granularity only: `[start][Sxx]text[end]`, no word timing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription
from transformers import AutoModelForCausalLM, AutoProcessor

from .common import probe_duration, save

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
TOKENS_PER_SECOND = 15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("audio", nargs="+", help="path, optionally path@offset_seconds")
    ap.add_argument("--max-new-tokens", type=int, default=65536)
    ap.add_argument("--prompt", default=None, help="replaces the default Chinese instruction")
    ap.add_argument("--tag", default="", help="suffix for output names (e.g. mix / vocal)")
    ap.add_argument("--sdpa", default="auto", choices=["auto", "efficient", "flash"],
                    help="force an SDPA kernel; auto falls back to the O(n^2) math kernel on long audio")
    a = ap.parse_args()
    if a.sdpa == "efficient":
        # With no mask, transformers passes `enable_gqa=True`; the memory-efficient kernel does not take GQA,
        # so SDPA silently falls back to the math kernel, which materialises the L x L scores (51 GiB at
        # 37 min of audio). Expanding K/V first keeps the efficient kernel eligible.
        import transformers.integrations.sdpa_attention as hf_sdpa
        hf_sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False

    dtype = torch.bfloat16
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype="auto",
                                                 attn_implementation="sdpa").to(dtype=dtype).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print(f"loaded in {time.monotonic() - t0:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    for item in a.audio:
        path, _, off = item.partition("@")
        path, offset = Path(path), float(off or 0)
        stem = path.stem + (f"-{a.tag}" if a.tag else "")
        messages = (build_transcription_messages(str(path), a.prompt) if a.prompt
                    else build_transcription_messages(str(path)))
        dur = probe_duration(path)
        # A 5 min window needs ~1 650 tokens; a window that falls into a repetition loop would otherwise run
        # to 65 536 (hina-68 window 0 ran two hours and filled the card). Cap by duration, flag the hit.
        cap = min(a.max_new_tokens, int(TOKENS_PER_SECOND * dur) + 256)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.monotonic()
        kernel = {"auto": contextlib.nullcontext(), "efficient": sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
                  "flash": sdpa_kernel(SDPBackend.FLASH_ATTENTION)}[a.sdpa]
        with kernel:
            result = generate_transcription(model, processor, messages, max_new_tokens=cap,
                                            do_sample=False, device=torch.device("cuda"), dtype=dtype)
        wall = time.monotonic() - t1
        peak = torch.cuda.max_memory_allocated() / 2**30
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / f"{stem}.raw.txt").write_text(result["text"], encoding="utf-8")
        segments = [{"start": s.start + offset, "end": s.end + offset, "speaker": s.speaker, "text": s.text}
                    for s in parse_transcript(result["text"])]
        save({"model": "moss", "hit_cap": result["generated_tokens"] >= cap, "config": {"prompt": a.prompt, "max_new_tokens": a.max_new_tokens, "sdpa": a.sdpa},
              "audio": str(path), "offset": offset, "duration": dur, "wall": round(wall, 2),
              "peak_gib": round(peak, 2), "prompt_len": result["prompt_len"],
              "generated_tokens": result["generated_tokens"], "segments": segments, "words": None},
             a.out_dir / f"{stem}.json")
        print(f"{path.name}: {len(segments)} segments, {result['generated_tokens']} tokens, "
              f"{wall:.1f}s (RTF {wall / dur:.3f}), peak {peak:.2f} GiB")
        print(json.dumps(segments[:3], ensure_ascii=False))


if __name__ == "__main__":
    main()
