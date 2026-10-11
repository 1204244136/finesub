"""Smoke-test a freshly built patched CTranslate2 wheel on the CPU.

The minimum acceptance of ../ct2-patches/README.md that a runner without a GPU
can check:

1. it is our build (local label) and carries the refine extension, and it
   has flash attention exactly when the variant should (CT2_EXPECT_FLASH_ATTN,
   set by the workflow: a forgotten -DWITH_FLASH_ATTN builds fine and only
   leaves the switch with nothing to turn on);
2. a model decodes on the CPU -- `get_supported_compute_types("cpu")` reports
   float32 even for builds whose CPU GEMM does not work, so only a real decode
   proves it;
3. deleting the model returns within a time limit -- the Ruy build decoded
   correctly and then hung forever in the destructor.

Usage: python smoke.py   (needs faster-whisper and numpy; downloads whisper tiny)
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
import threading
import time

DESTRUCTOR_LIMIT_SEC = 120.0


def main() -> int:
    import ctranslate2
    import numpy as np
    from faster_whisper import WhisperModel

    version = ctranslate2.__version__
    print(f"ctranslate2 {version}")
    if "+finesub" not in version:
        print("FAIL: not the patched build (no +finesub local label)")
        return 1
    doc = inspect.getdoc(ctranslate2.models.Whisper.generate) or ""
    for marker in ("return_refine_paths", "return_refine_weights"):
        if marker not in doc:
            print(f"FAIL: generate() lacks {marker}")
            return 1
    if not hasattr(ctranslate2, "flash_attention_available"):
        print("FAIL: no flash_attention_available() (patch 0012 missing)")
        return 1
    has_flash = bool(ctranslate2.flash_attention_available())
    print(f"flash attention compiled: {has_flash}")
    expected = os.environ.get("CT2_EXPECT_FLASH_ATTN", "").strip().lower()
    if expected in {"true", "false"} and has_flash != (expected == "true"):
        print(f"FAIL: flash attention compiled is {has_flash}, expected {expected}")
        return 1

    model = WhisperModel("tiny", device="cpu", compute_type="float32")
    rate = 16000
    t = np.arange(rate * 3, dtype=np.float32) / rate
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    started = time.perf_counter()
    segments, info = model.transcribe(audio, language="en", beam_size=1)
    segments = list(segments)
    print(f"decoded {len(segments)} segment(s) in {time.perf_counter() - started:.1f}s")

    holder = {"model": model}
    del model, segments, info
    done = threading.Event()

    def drop() -> None:
        holder.clear()
        gc.collect()
        done.set()

    threading.Thread(target=drop, daemon=True).start()
    if not done.wait(DESTRUCTOR_LIMIT_SEC):
        print(f"FAIL: model destructor did not return within {DESTRUCTOR_LIMIT_SEC:.0f}s")
        sys.stdout.flush()
        os._exit(1)
    print("destructor returned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
