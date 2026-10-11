"""Is the masked projection at least a net information gain, if kept apart?

5h.3 scored the arms as whole texts. A candidate layer shown to a person only
adds what the baseline did not already have, so this scores that increment:
bigrams in an arm's region text beyond the baseline's (multiset difference),
and how many of them match what someone nearby actually said. Post-hoc
(not pre-registered) -- a reading of 5h.3's data, not an acceptance.
"""

from __future__ import annotations

import json
from collections import Counter

import s4_headroom as ref
import s4_projection as proj


def main():
    totals = {arm: Counter() for arm in ("sham", "masked")}
    regions_with_true_new = {arm: 0 for arm in totals}
    region_count = 0
    for session in ref.SESSIONS:
        records = json.loads((proj.ALI / session / "projection.json").read_text(encoding="utf-8"))
        references = ref.reference(session)
        words = [{"start": m, "end": m, "word": t} for m, t in ref.hypothesis(session)]
        for record in records:
            region_count += 1
            nearby = Counter()
            for row in references:
                if row["start"] < record["end"] + 1 and row["end"] > record["start"] - 1:
                    nearby |= ref.bigrams(row["text"])
            base = ref.bigrams(proj.region_text(words, record["start"], record["end"]))
            arms = {
                "sham": proj.region_text(record["sham"], record["start"], record["end"]),
                "masked": "".join(proj.region_text(w, record["start"], record["end"]) for w in record["masked"].values()),
            }
            for arm, text in arms.items():
                new = ref.bigrams(text) - base
                true_new = new & (nearby - base)  # right, and not something the baseline already had
                totals[arm]["new"] += sum(new.values())
                totals[arm]["true"] += sum(true_new.values())
                if true_new:
                    regions_with_true_new[arm] += 1
    for arm, total in totals.items():
        print(
            f"{arm:<7} new bigrams {total['new']:5d}  of which correct {total['true']:5d} "
            f"({total['true'] / max(total['new'], 1):.1%})  | regions gaining >=1 correct bigram "
            f"{regions_with_true_new[arm]}/{region_count}"
        )


if __name__ == "__main__":
    main()
