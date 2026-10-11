"""Where MOSS decode time goes: one decoder attention call at q_len = 1 over an L-token KV cache.

MOSS's decoder is Qwen3-0.6B-shaped (28 layers, 16 query heads, 8 KV heads, head_dim 128, bf16). Single-pass
runs pay 90-117 ms per generated token at L ~ 20-35k, where weights + KV cache (~1.5 GB + 112 KB/token)
bound it at ~5-10 ms on this card. `moss_run.py --sdpa efficient` expands K/V (no GQA) so the prefill avoids
the O(L^2) math kernel; this measures what that costs every decode step, against the alternatives:

  efficient+expand   what moss_run does today: repeat_kv copy, then the memory-efficient kernel
  math+gqa           transformers' default decode call (enable_gqa=True -> math kernel; at q_len 1 the
                     score matrix is 1 x L, so no memory problem)
  flash+expand       FlashAttention-2 kernel, if this build has it for the card
  cudnn+expand       cuDNN attention, if available

Times are per call (one layer); x28 for a decode step's attention. Short-task protocol: warm-up dropped,
arms alternated, order swapped between rounds.

Then the same call with the KV length growing by one per call, as in real decoding: cuDNN builds an
execution plan per shape, so at a new length every step it costs ~360 ms per call (2026-09-25) -- it is
only usable behind a fixed-shape (static) KV cache.

    tmp/ts-diar/moss/.venv/Scripts/python.exe -m tools.ts_diar_explore.moss_attn_bench
"""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HQ, HKV, D = 16, 8, 128
LENGTHS = (5_000, 20_000, 35_000)
ROUNDS, ITERS = 6, 50


def expand(x: torch.Tensor) -> torch.Tensor:  # transformers.repeat_kv
    b, h, l, d = x.shape
    return x[:, :, None].expand(b, h, HQ // HKV, l, d).reshape(b, HQ, l, d)


def arms(q, k, v):
    def eff():
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, expand(k), expand(v))

    def math_gqa():
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    def flash():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, expand(k), expand(v))

    def cudnn():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, expand(k), expand(v))

    return {"efficient+expand": eff, "math+gqa": math_gqa, "flash+expand": flash, "cudnn+expand": cudnn}


def time_ms(fn) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(0.5)  # ~8 GiB: one layer's K/V at 35k is ~150 MB
    print(torch.__version__, torch.cuda.get_device_name())
    for L in LENGTHS:
        q = torch.randn(1, HQ, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, HKV, L, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        fns = {}
        for name, fn in arms(q, k, v).items():
            try:
                ref = fn()
                torch.cuda.synchronize()
                fns[name] = fn
                first = time_ms(fn)  # warm-up, reported, not aggregated
            except RuntimeError as e:
                print(f"L={L:6d} {name:<17} unavailable: {str(e).splitlines()[0][:80]}")
                continue
            print(f"L={L:6d} {name:<17} warm-up {first:.3f} ms")
            del ref
        samples = {n: ([], []) for n in fns}
        order = list(fns)
        for r in range(ROUNDS):
            seq = order if r % 2 == 0 else order[::-1]
            for n in seq:
                samples[n][r % 2].append(time_ms(fns[n]))
        for n, (fwd, rev) in samples.items():
            g = (statistics.median(fwd) * statistics.median(rev)) ** 0.5
            print(f"L={L:6d} {n:<17} {g:.3f} ms/call  x28 = {28 * g:.1f} ms/step")
    growing()


def growing() -> None:
    q = torch.randn(1, HQ, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, HKV, 20_100, D, device="cuda", dtype=torch.bfloat16)
    for name in ("efficient+expand", "math+gqa", "cudnn+expand"):
        arms(q, k[:, :, :20_000], k[:, :, :20_000])[name]()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(20):
            arms(q, k[:, :, :20_001 + i], k[:, :, :20_001 + i])[name]()
        torch.cuda.synchronize()
        print(f"L=20000+i {name:<17} {1000 * (time.perf_counter() - start) / 20:.2f} ms/call with a new length every call")


if __name__ == "__main__":
    main()
