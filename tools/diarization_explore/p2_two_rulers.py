"""Post-hoc (not a gate): round 2 re-scored with two rulers (peer, 2026-10-05), after the owner heard the audio judge
right on rows our ruler called "broken".

  strict   score only clean rows: one person holds >= 70% of all reference coverage of the row, no tie
  lenient  a label is right when its person's reference line covers >= 30% of the row (several people can be right)

The two nets bound the judge's real net. Also counts the clean flagged rows of the dev split (for a clean-only rerun).

    python tools/diarization_explore/p2_two_rulers.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_retrack_judge as lj  # noqa: E402
import p2_features as pf  # noqa: E402

LENIENT = 0.30
CLEAN = 0.70


def shares():
    """{(file, index): {person: share of the row's frames}} for every dev flagged row (hina first)."""
    import relabel_eval as ev
    rows, _ = pf.load_rows("covered")
    out = {}
    for ident in sorted(pf.SPLIT["dev"], key=lambda i: i.startswith("onetonton")):
        stable, grid, people, _ = pf.reference(ident)
        for r in lj.flagged_rows(rows, ident):
            cover, n = ev.span_cover(stable[r["index"]], grid)
            out[(ident, r["index"])] = {people[i]: float(cover[i]) / n for i in np.flatnonzero(cover)}
    return rows, out


def clean(share):
    if not share:
        return False
    top = max(share.values())
    return top / sum(share.values()) >= CLEAN and sum(v == top for v in share.values()) == 1


def main():
    rows, share = shares()
    by_key = {(r["file"], r["index"]): r for f in pf.SPLIT["dev"] for r in lj.flagged_rows(rows, f)}
    owner = lambda key, track: by_key[key]["owner"].get(track)  # noqa: E731
    strict_right = lambda key, track: owner(key, track) in by_key[key]["truth"]  # noqa: E731
    lenient_right = lambda key, track: share[key].get(owner(key, track), 0.0) >= LENIENT  # noqa: E731
    flagged = list(by_key)
    clean_keys = [k for k in flagged if clean(share[k])]
    print(f"dev flagged rows {len(flagged)}; clean {len(clean_keys)} ({len(clean_keys) / len(flagged):.0%}), "
          f"of which wrong {sum(not strict_right(k, by_key[k]['leader']) for k in clean_keys)}")
    bases = {"strict": np.mean([not strict_right(k, by_key[k]["leader"]) for k in clean_keys]),
             "lenient": np.mean([not lenient_right(k, by_key[k]["leader"]) for k in flagged])}
    print(f"dev flagged error rate: strict (clean rows) {bases['strict']:.1%}; lenient (all rows) {bases['lenient']:.1%}")
    picks = {arm: lj.picks("v2", arm) for arm in ("text", "audio", "mosspro")}
    sample = [(f, it["index"]) for f, items in lj.sample("v2")["files"].items() for it in items]
    for ruler, keys, right in (("strict", [k for k in sample if clean(share[k])], strict_right), ("lenient", sample, lenient_right)):
        print(f"== {ruler}: {len(keys)} sample rows ==")
        for arm, pk in picks.items():
            if not pk:
                continue
            keys_arm = [k for k in keys if k in pk]
            tally = Counter()
            n_wrong = sum(not right(k, by_key[k]["leader"]) for k in keys_arm)
            n_right = len(keys_arm) - n_wrong
            for k in keys_arm:
                p = pk[k]
                was = right(k, by_key[k]["leader"])
                if not p.startswith("s") or not p[1:].isdigit() or int(p[1:]) == by_key[k]["leader"]:
                    tally["kept/abstain"] += 1
                    continue
                now = right(k, int(p[1:]))
                tally["fixed" if now and not was else "broken" if was and not now else "neutral"] += 1
            base = bases[ruler]
            net = 100 * (base * tally["fixed"] / max(1, n_wrong) - (1 - base) * tally["broken"] / max(1, n_right))
            print(f"{arm:8s}: {len(keys_arm)} rows ({n_wrong} wrong / {n_right} right) {dict(tally)}; net {net:+.1f} per 100 flagged rows")


if __name__ == "__main__":
    main()
