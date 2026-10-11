"""Precise recall/precision/F1 of Sortformer VAD vs project stable.json segments.

Uses interval-level intersection (no frame-grid rounding).

Usage:
    python vad_metrics.py <sortformer.json> <stable.json> <audio_sec>
"""
import json
import sys


def inter_len(a, b):
    """Total intersection length (seconds) between two interval lists."""
    out = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
        if hi > lo:
            out += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def main():
    sf_path, stable_path, audio_sec = sys.argv[1], sys.argv[2], float(sys.argv[3])

    sf = json.load(open(sf_path, encoding="utf-8"))["vad_segments"]
    sf = [(s, e) for s, e in sf]
    d = json.load(open(stable_path, encoding="utf-8"))
    segs = d if isinstance(d, list) else d.get("segments", [])
    st = sorted((float(s["start"]), float(s["end"])) for s in segs)

    sf_len = sum(e - s for s, e in sf)
    st_len = sum(e - s for s, e in st)
    inter = inter_len(sf, st)
    audio = audio_sec

    # Treat stable.json as reference (project-approved speech)
    recall = inter / st_len if st_len else 0.0        # what fraction of project speech Sortformer covers
    precision = inter / sf_len if sf_len else 0.0     # of Sortformer's speech, how much is project speech
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0

    # Global coverage stats
    print(f"audio total:            {audio:.1f}s")
    print(f"project VAD (stable):   {st_len:.1f}s ({st_len/audio*100:.1f}%)  [{len(st)} segs]")
    print(f"Sortformer VAD:         {sf_len:.1f}s ({sf_len/audio*100:.1f}%)  [{len(sf)} segs]")
    print(f"intersection:           {inter:.1f}s")
    print()
    print(f"RECALL (proj speech covered):  {recall*100:.1f}%   (Sortformer misses {st_len-inter:.1f}s)")
    print(f"PRECISION (Sortformer speech within proj): {precision*100:.1f}%")
    print(f"F1:                     {f1*100:.1f}%")
    print(f"Sortformer-only extra:  {sf_len-inter:.1f}s")
    print()
    # Recall at different views
    print(f"covered_by_sortformer_of_stable = {inter/st_len*100:.1f}%")
    print(f"stable_covered_by_sortformer = {inter/sf_len*100:.1f}%")


if __name__ == "__main__":
    main()
