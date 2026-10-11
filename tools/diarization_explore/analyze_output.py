"""Analyze Sortformer diarization output vs pyannote baseline.

Usage:
    python analyze_output.py <sortformer.json> <pyannote.json> <name>
"""
import json
import sys
from collections import defaultdict


def load_sortformer(p):
    d = json.load(open(p, encoding="utf-8"))
    return d["segments"]  # [{start,end,label}]


def load_pyannote(p):
    return json.load(open(p, encoding="utf-8"))  # [{start,end,spk}]


def speaker_stats(segs, key_label="label"):
    n_spk = sorted({s[key_label] for s in segs})
    per = defaultdict(float)
    turns = defaultdict(int)
    for s in segs:
        per[s[key_label]] += s["end"] - s["start"]
        turns[s[key_label]] += 1
    total = sum(per.values())
    print(f"  speakers={len(n_spk)} total_speech={total:.1f}s")
    for lbl in n_spk:
        print(f"    {lbl}: {turns[lbl]} turns, {per[lbl]:.1f}s ({per[lbl]/total*100:.1f}%)")
    return n_spk


def align_and_score(sf, pa, iou_thr=0.5):
    """Greedy segment-level alignment between Sortformer (sf) and pyannote (pa).

    For each pair of overlapping segments (IoU >= thr), vote the label-pairing.
    Then solve a best permutation between speaker sets and report agreement.
    """
    # build overlap votes
    votes = defaultdict(lambda: defaultdict(int))  # sf_label -> pa_spk -> weight
    sf_cov = 0.0
    for s in sf:
        for q in pa:
            st = max(s["start"], q["start"]); en = min(s["end"], q["end"])
            if en - st <= 0:
                continue
            inter = en - st
            union = (s["end"] - s["start"]) + (q["end"] - q["start"]) - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou >= iou_thr:
                votes[s["label"]][q["spk"]] += inter
    # greedy one-to-one pairing (sort by vote mass desc)
    pairs = []
    used_pa = set()
    for sf_lbl in sorted(votes, key=lambda k: -sum(votes[k].values())):
        best = None; best_v = 0
        for pa_spk, v in votes[sf_lbl].items():
            if pa_spk not in used_pa and v > best_v:
                best, best_v = pa_spk, v
        if best:
            pairs.append((sf_lbl, best, best_v))
            used_pa.add(best)
    mapping = {a: b for a, b, _ in pairs}
    # agreement over time: sample every 0.5s within speech regions
    total = 0; agree = 0
    for s in sf:
        if s["label"] not in mapping:
            continue
        t = s["start"]
        while t < s["end"] - 0.01:
            total += 1
            ref = None
            for q in pa:
                if q["start"] <= t < q["end"]:
                    ref = q["spk"]; break
            if ref == mapping[s["label"]]:
                agree += 1
            t += 0.5
    agree_pct = agree / total * 100 if total else 0.0
    print(f"  aligned pairs (greedy): {[(a, b) for a, b, _ in pairs]}")
    print(f"  time-agreement with mapped labels: {agree}/{total} = {agree_pct:.1f}%")
    return mapping, agree_pct


def spotcheck_window(segs, t0, t1, name):
    """Print segments overlapping [t0,t1] window for manual inspection."""
    print(f"  --- spotcheck window {t0}-{t1}s ---")
    for s in sorted(segs, key=lambda x: x["start"]):
        if s["end"] >= t0 and s["start"] <= t1:
            st = max(s["start"], t0); en = min(s["end"], t1)
            print(f"    [{st:8.1f}-{en:7.1f}] {s['label']}  ({en-st:.1f}s)")


def main():
    sf_p, pa_p, name = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"===== {name} =====")
    sf = load_sortformer(sf_p)
    pa = load_pyannote(pa_p)
    print(f"[Sortformer] {len(sf)} segments")
    sf_spk = speaker_stats(sf)
    print(f"[pyannote] {len(pa)} segments")
    pa_spk = speaker_stats(pa, "spk")
    align_and_score(sf, pa)


if __name__ == "__main__":
    main()
