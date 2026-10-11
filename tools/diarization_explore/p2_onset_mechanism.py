"""Post-hoc (not a gate): what the rows look like where the onset truth and the covered truth disagree.

A disputed row: one person's reference line starts inside the row (the onset truth) and that person is not among the
people tied for the most coverage (the covered truth). For those rows, how much of the row the onset person's line
covers, by corpus family -- a small share means the row is mostly someone else's speech; a large one means two people
genuinely share the row. (That the covered person's line is already running at the row's start follows from the rule
itself: it is not reported as a finding.)

    python tools/diarization_explore/p2_onset_mechanism.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_features as pf  # noqa: E402
import p2_text_truth as tt  # noqa: E402


def family(ident):
    return "bilingual (hina-61, 63-74)" if ident in tt.BILINGUAL else "onetonton" if ident.startswith("onetonton") else "other hina"


def main():
    import relabel_eval as ev
    rows, _ = pf.load_rows("covered")
    share = defaultdict(list)
    total = defaultdict(int)
    for ident in sorted(rows, key=lambda i: i.startswith("onetonton")):  # ⚠ hina first
        stable, grid, people, _ = pf.reference(ident)
        for r in rows[ident]:
            if r["state"] != "ordinary":
                continue
            total[family(ident)] += 1
            onset, covered = r["truths"]["onset"], r["truths"]["covered"]
            if set(onset) & set(covered):
                continue
            cover, n = ev.span_cover(stable[r["index"]], grid)
            share[family(ident)].append(cover[people.index(onset[0])] / n)
    for fam in sorted(total):
        s = np.array(share[fam])
        print(f"{fam}: {len(s)} disputed of {total[fam]} ordinary rows ({len(s) / total[fam]:.1%}); onset person's coverage of the row "
              f"median {np.median(s) if len(s) else 0:.2f}, >= 0.4 on {(s >= 0.4).mean() if len(s) else 0:.0%}")
    allshare = sum(len(v) for v in share.values())
    print(f"all: {allshare} disputed rows; bilingual family holds {len(share['bilingual (hina-61, 63-74)']) / allshare:.1%} of them")


if __name__ == "__main__":
    main()
