"""Cross-window speaker label propagation + consistency check vs full-run.

Windows (50% overlap) are diarized independently; labels are chained across
overlapping regions by co-active-time voting (no embeddings needed).

Usage:
    python window_propagation.py
"""
import json
from collections import defaultdict, Counter


WINDOWS = [
    {"name": "w0", "path": "nepolabo-chunks/w0.v2.1.json", "offset": 0.0},
    {"name": "w1", "path": "nepolabo-chunks/w1.v2.1.json", "offset": 900.0},
    {"name": "w2", "path": "nepolabo-chunks/w2.v2.1.json", "offset": 1800.0},
]
OVERLAPS = [("w0", "w1", 900, 1800), ("w1", "w2", 1800, 2700)]


def load(path):
    return json.load(open(path, encoding="utf-8"))["segments"]


def segs_at(segs, t, offset):
    gt = t + offset
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s["label"], gt
    return None, gt


def main():
    data = {w["name"]: (w, load(w["path"])) for w in WINDOWS}

    # 1) pairwise mapping via overlap voting
    maps = {}  # (winA, winB) -> {labelA: labelB}
    for wa, wb, t0, t1 in OVERLAPS:
        wa_cfg, sa = data[wa]
        wb_cfg, sb = data[wb]
        votes = defaultdict(Counter)
        t = t0
        while t < t1:
            la, _ = segs_at(sa, t - wa_cfg["offset"], wa_cfg["offset"])
            lb, _ = segs_at(sb, t - wb_cfg["offset"], wb_cfg["offset"])
            if la is not None and lb is not None:
                votes[la][lb] += 1
            t += 0.5
        mapping = {}
        used = set()
        for la in sorted(votes, key=lambda k: -sum(votes[k].values())):
            best = None; bv = 0
            for lb, v in votes[la].items():
                if lb not in used and v > bv:
                    best, bv = lb, v
            if best:
                mapping[la] = best
                used.add(best)
        maps[(wa, wb)] = mapping
        print(f"overlap {wa}->{wb}: {mapping}")

    # 2) global ID assignment: w0 labels = A/B/C/D in first-appearance order
    w0_cfg, s0 = data["w0"]
    order0 = sorted({s["label"] for s in s0}, key=lambda l: min(s["start"] for s in s0 if s["label"] == l))
    global_id = {l: f"G{i}" for i, l in enumerate(order0)}
    print(f"w0 global ids: {global_id}")

    # chain w1, w2
    w1_cfg, s1 = data["w1"]
    map01 = maps[("w0", "w1")]
    w1_to_global = {}
    for l1 in sorted({s["label"] for s in s1}):
        l0 = [k for k, v in map01.items() if v == l1]  # w0 labels mapping into l1
        w1_to_global[l1] = global_id[l0[0]] if l0 else f"NEW:{l1}"
    # any w1 label not matched -> new global id (rare)
    print(f"w1->global: {w1_to_global}")

    w2_cfg, s2 = data["w2"]
    map12 = maps[("w1", "w2")]
    w2_to_global = {}
    for l2 in sorted({s["label"] for s in s2}):
        l1 = [k for k, v in map12.items() if v == l2]  # w1 labels mapping into l2
        w2_to_global[l2] = w1_to_global.get(l1[0]) or f"NEW:{l2}"
    print(f"w2->global: {w2_to_global}")

    # 3) build windowed global segments and compare with full-run labels
    full = json.load(open("nepolabo.v2.1.json", encoding="utf-8"))["segments"]
    win_global = []
    for w in WINDOWS:
        cfg, segs = data[w["name"]]
        gl = {"w0": global_id, "w1": w1_to_global, "w2": w2_to_global}[w["name"]]
        for s in segs:
            win_global.append({
                "start": s["start"] + cfg["offset"],
                "end": s["end"] + cfg["offset"],
                "label": gl.get(s["label"], s["label"]),
            })

    # compare on 0.5s sampling where both systems say speech,
    # after greedy pairing windowed G-labels with full-run labels
    full_labels = sorted({s["label"] for s in full})
    win_labels = sorted({s["label"] for s in win_global})
    pair_votes = defaultdict(Counter)
    t = 0.0
    tend = max(s["end"] for s in full)
    while t < tend:
        lw = None; lf = None
        for s in win_global:
            if s["start"] <= t < s["end"]:
                lw = s["label"]; break
        for s in full:
            if s["start"] <= t < s["end"]:
                lf = s["label"]; break
        if lw and lf:
            pair_votes[lw][lf] += 1
        t += 0.5
    pairing = {}
    used_f = set()
    for lw in sorted(pair_votes, key=lambda k: -sum(pair_votes[k].values())):
        best = None; bv = 0
        for lf, v in pair_votes[lw].items():
            if lf not in used_f and v > bv:
                best, bv = lf, v
        if best:
            pairing[lw] = best
            used_f.add(best)
    print(f"greedy pairing windowed->full: {pairing}")

    agree = tot = 0
    per_win = defaultdict(lambda: [0, 0])
    for w in WINDOWS:
        cfg, segs = data[w["name"]]
        gl = {"w0": global_id, "w1": w1_to_global, "w2": w2_to_global}[w["name"]]
        t = cfg["offset"]
        tend = cfg["offset"] + max((s["end"] for s in segs), default=0)
        while t < tend:
            lw = None; lf = None
            for s in segs:
                if s["start"] <= t - cfg["offset"] < s["end"]:
                    lw = pairing.get(gl[s["label"]]); break
            for s in full:
                if s["start"] <= t < s["end"]:
                    lf = s["label"]; break
            if lw and lf:
                tot += 1
                per_win[w["name"]][1] += 1
                if lw == lf:
                    agree += 1
                    per_win[w["name"]][0] += 1
            t += 0.5
    print(f"\n=== windowed vs full-run label agreement (paired) ===")
    print(f"overall: {agree}/{tot} = {agree/tot*100:.1f}%")
    for wn, (a, n) in sorted(per_win.items()):
        print(f"  {wn}: {a}/{n} = {a/n*100:.1f}%")
    # speaker count global
    print(f"global speakers (windowed): {len({s['label'] for s in win_global})}  vs full-run: {len({s['label'] for s in full})}")


if __name__ == "__main__":
    main()
