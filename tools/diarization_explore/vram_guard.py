"""Hard 8 GiB VRAM ceiling for exploration runs (owner 2026-09-16: "加个 8G 显存的守卫，不要再超了").

Two diarization runs in a row spilled past the 16.3 GiB card into Windows *shared* GPU memory. That spill is
the dangerous part: it raises no error, only runs 2-3x slower, and `nvidia-smi --query-gpu=memory.used` cannot
see it -- so the only warning the owner got was the machine grinding.

Two layers, because each misses something the other catches:

1. **Allocator cap** (`torch.cuda.set_per_process_memory_fraction`). Tensors, TorchScript module weights,
   and the cuDNN/cuBLAS workspaces all go through PyTorch's caching allocator, so past the cap PyTorch raises
   `OutOfMemoryError` instead of asking the driver for more -- the spill never starts. It cannot see the CUDA
   context or the kernels, which live outside the allocator; hence the headroom subtracted below.
2. **OS watchdog.** Polls this process's `\\GPU Process Memory(pid_*)\\Dedicated Usage` + `Shared Usage`
   -- what Windows actually charges the process, shared memory included -- and `os._exit`s the whole process
   the moment it crosses the limit. It is the backstop for whatever the allocator does not account for.

⚠ The two layers are not interchangeable, and layer 2 cannot be replaced by `torch.cuda.memory_reserved()`:
that is the very quantity layer 1 already caps (the cap fires on allocator-reserved bytes -- its message reads
"N GiB allowed"), so polling it can never fire first. Measured 2026-09-19 with a bare context plus one 1 GiB
tensor: allocator reserved 1.00 GiB, Windows charged 1.19 dedicated + 0.07 shared. Those 0.26 GiB, and the
shared column in particular, are the entire reason this layer exists.

    import vram_guard
    vram_guard.install()          # before the first CUDA allocation
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time

LIMIT_GIB = 8.0
#: Memory kept outside PyTorch's allocator for the CUDA context, kernels and workspace spikes. Measured
#: 2026-09-16 with the three Sortformer artifacts loaded: 0.31 GiB (Windows charged 0.78, allocator held 0.47).
#: Deliberately left at ~5x that -- the cap then sits at 6.5 GiB and a spike has room before the watchdog,
#: which is what actually enforces the 8 GiB, has to kill anything. Verified: a 7 GiB allocation is refused.
CONTEXT_HEADROOM_GIB = 1.5
GIB = 1024 ** 3

#: One PowerShell per sample is what this used to do, and it is why the watchdog was expensive: the *first*
#: `Get-Counter` in a process spends ~0.64 s enumerating the GPU counter set, every one after that costs ~1 ms
#: (measured 2026-09-19; a whole tick was 0.83 core-s and 84 MiB of RSS, ~28% of a core at the old 3 s period).
#: So the PowerShell is kept and fed one line per sample instead, which is also what makes a sub-second period
#: affordable. It writes one integer per request; an unreadable counter measures as 0, as it did before.
_SESSION_SCRIPT = """
$paths = '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage', '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage'
while ($null -ne [Console]::In.ReadLine()) {{
  $sum = (Get-Counter $paths -ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Sum
  [Console]::Out.WriteLine([int64]$sum.Sum)
  [Console]::Out.Flush()
}}
"""


class _CounterSession:
    """A PowerShell kept alive across samples, one request line in and one integer out.

    Replies are pumped into a queue by a reader thread so that `sample` can time out: a wedged `Get-Counter`
    must not silently retire the watchdog (the old per-sample `subprocess.run` had `timeout=20` for that).
    """

    def __init__(self, pid: int) -> None:
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", _SESSION_SCRIPT.format(pid=pid)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1)
        self.replies: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True, name="vram-guard-pump").start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.replies.put(line)
        self.replies.put("")          # PowerShell is gone; unblock whoever is waiting

    def sample(self, timeout: float) -> int | None:
        """Bytes Windows charges the process on the GPU, dedicated plus shared. None if unreadable."""
        try:
            self.proc.stdin.write("\n")
            self.proc.stdin.flush()
            return int(self.replies.get(timeout=timeout))
        except Exception:
            return None

    def close(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass


class _Watchdog(threading.Thread):
    def __init__(self, limit_bytes: int, period: float) -> None:
        super().__init__(daemon=True, name="vram-guard")
        self.limit, self.period = limit_bytes, period
        self.peak = 0

    def run(self) -> None:
        pid = os.getpid()
        session: _CounterSession | None = None
        while True:
            if session is None:
                session = _CounterSession(pid)
            used = session.sample(timeout=20.0)
            if used is None:                       # wedged or dead -- start a fresh one next round
                session.close()
                session = None
            else:
                self.peak = max(self.peak, used)
                if used > self.limit:
                    sys.stderr.write(
                        f"\nvram_guard: process {pid} is using {used / GIB:.2f} GiB of GPU memory "
                        f"(dedicated + shared), over the {self.limit / GIB:.1f} GiB limit -- killing it\n")
                    sys.stderr.flush()
                    os._exit(3)
            time.sleep(self.period)


_watchdog: _Watchdog | None = None


def install(limit_gib: float = LIMIT_GIB, *, period: float = 0.5) -> None:
    """Cap this process at `limit_gib`. Call before the first CUDA allocation."""
    global _watchdog
    import torch

    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    allocator = max(0.5, limit_gib - CONTEXT_HEADROOM_GIB) * GIB
    torch.cuda.set_per_process_memory_fraction(min(1.0, allocator / total), 0)
    if _watchdog is None:
        _watchdog = _Watchdog(int(limit_gib * GIB), period)
        _watchdog.start()


def peak_gib() -> float:
    """Highest dedicated + shared usage the watchdog has seen so far."""
    return (_watchdog.peak if _watchdog else 0) / GIB
