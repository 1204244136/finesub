"""One table of every labelling measured on the windows of rounds 3-4 (owner 2026-10-06), all scored the same way.

Each window's rows (production labelled them, they have truth, no profile took them as a reference) are scored as a
partition: the labels a labelling ends on are matched one-to-one to people inside the window (`window_partition_score`).
Columns: production; the judges (text-closed, text-open_gen, audio-open_gen); 35t (300 s / 540 s windows) and MOSS Pro
(`window_external`), whose speakers label production's rows through `assign_all`. A model not run on a window is "—".

    python tools/diarization_explore/window_table.py r3   # one process per round (P2 loads hina first)
    python tools/diarization_explore/window_table.py r4
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_external as wx  # noqa: E402
import window_partition_score as ps  # noqa: E402

JUDGES = ("text-closed", "text-open_gen", "audio-open_gen")
NAMES = {"manosaba-a": "魔裁 a", "manosaba-b": "魔裁 b", "onetonton-07": "姐姐咚咚 07", "onetonton-13": "姐姐咚咚 13"}


def main(rnd):
    r3, _, _ = ps._round(rnd)
    base = r3.outcome({})
    picks = {arm: r3.answers(arm) for arm in JUDGES}
    print("| 窗口（计分行） | 生产 | 只改现有轨（纯文本） | 先挑人 + 新建轨（纯文本） | 先挑人 + 新建轨（带音频） | 35t 300 秒 | 35t 540 秒 | MOSS Pro |")
    for ident, a, b in r3.windows():
        m = r3.load(ident)
        rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
        cells = []
        for got in ({}, *(picks[arm] for arm in JUDGES)):
            right = ps.partition_right(rows, ps.final_labels(r3, m, ident, a, b, got), m.truth)
            cells.append(f"{sum(right.values()) / len(rows):.1%}")
        lo, hi = wx.span(m, a, b)
        for _, name in wx.MODELS:
            if not wx.A.path(ident, name, span=(lo, hi)).exists():
                cells.append("—")
                continue
            lead = wx.leaders(m, ident, name, lo, hi)
            right = ps.partition_right(rows, {i: lead[i] for i in rows}, m.truth)
            cells.append(f"{sum(right.values()) / len(rows):.1%}")
        print(f"| {NAMES.get(ident, ident)}（{len(rows)}） | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
