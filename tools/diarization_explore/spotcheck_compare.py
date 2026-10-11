"""Compare Sortformer output against the manual spotcheck table
(out/nepolabo/nepolabo-spotcheck.md, window 1200-1350s).

The table rows are "| 20:02.3 | 2.0s | S02* | <真实说话人> | 文本 |" where the
<真实说话人> column was filled by hand with 1/2/3/4 (= ねね/ポルカ/ぼたん/ラミィ in
some order) for a subset of rows. We map Sortformer speaker labels to those
manual labels greedily and report agreement.
"""
import json
import re
import sys
from collections import defaultdict

MM_TO_SEC = lambda mmss: int(mmss.split(":")[0]) * 60 + float(mmss.split(":")[1])


def parse_spotcheck(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or not re.match(r"^\d+:\d\d\.\d$", cells[0]):
            continue
        t_sec = MM_TO_SEC(cells[0])
        dur = float(cells[1].rstrip("s"))
        text = cells[4] if len(cells) > 4 else ""
        # manual label sits in the LAST column (trailing cell) as 1/2/3/4 or 4(mix)
        truth = cells[-1] if len(cells) >= 6 else ""
        m = re.match(r"^(\d+)", truth)
        if m:
            rows.append({"t": t_sec, "dur": dur, "truth": int(m.group(1)), "text": text})
    return rows


def main():
    sf_path, spotcheck_path = sys.argv[1], sys.argv[2]
    sf = json.load(open(sf_path, encoding="utf-8"))["segments"]
    rows = parse_spotcheck(spotcheck_path)
    print(f"spotcheck rows with manual labels: {len(rows)}")

    # For each labeled row, find the Sortformer segment covering its midpoint.
    pairs = []
    for r in rows:
        mid = r["t"] + r["dur"] / 2
        hit = None
        for s in sf:
            if s["start"] <= mid < s["end"]:
                hit = s
                break
        if hit:
            pairs.append((r["truth"], hit["label"], r["t"], r["text"][:20]))

    # Greedy mapping: manual label -> sortformer label by majority vote
    votes = defaultdict(lambda: defaultdict(int))
    for truth, lbl, _, _ in pairs:
        votes[truth][lbl] += 1
    mapping = {}
    used = set()
    for truth in sorted(votes, key=lambda k: -sum(votes[k].values())):
        best = None; bv = 0
        for lbl, v in votes[truth].items():
            if lbl not in used and v > bv:
                best, bv = lbl, v
        if best:
            mapping[truth] = best
            used.add(best)
    print(f"mapping manual->sortformer: {mapping}")

    agree = 0
    for truth, lbl, t, text in pairs:
        ok = mapping.get(truth) == lbl
        agree += ok
        print(f"  t={t:7.1f}s manual={truth} sortformer={lbl} {'OK' if ok else 'XX'}  {text}")
    print(f"agreement: {agree}/{len(pairs)} = {agree/len(pairs)*100:.1f}%")


if __name__ == "__main__":
    main()
