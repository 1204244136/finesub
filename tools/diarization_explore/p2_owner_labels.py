"""Owner's hand labels on the first 20 clips of the review video (2026-10-05), scored against every arm and the ruler.

The owner listened to clips A01–A20 of `out/p2-review/p2-review.mp4` (labelled flagged rows of sample v2) and named the
track(s) actually speaking in the judged row -- for THIS run's ASR row, as the owner notes: an ASR change could move it,
and it need not match the refined reference. A row where two tracks speak counts both; clip 13 is "too chaotic" and is
left out. A label is right when its track is one the owner named (tracks, not people: no owner-map involved).

    python tools/diarization_explore/p2_owner_labels.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_retrack_judge as lj  # noqa: E402
import p2_features as pf  # noqa: E402

VIDEO_ASS = lj.lr.n3.MAIN / "out" / "p2-review" / "p2-review.ass"
#: clip number -> tracks the owner heard speaking in the judged row (owner 2026-10-05, verbatim notes in the comment)
OWNER = {
    1: {6, 9},     # 前半s6 后半s9
    2: {5},
    3: {5},
    4: {1},
    5: {6},
    6: {5, 4},     # 第一句 前半s5 后半s4（第二句紧挨着的 s5，ASR 和裁判都错了）
    7: {5},
    8: {9},        # 前半s9
    9: {5},
    10: {5},
    11: {9},
    12: {3},
    13: None,      # 过于混乱
    14: {6},
    15: {9, 6},    # s9 s6 同时同一句话
    16: {9, 8},    # 前半s9 后半s8
    17: {5},
    18: {7, 3},    # 同一句话 s7 先说了一遍，s3 重读了一遍，ASR 没有转录两次
    19: {4},
    20: {1},
}


def clips():
    """[(n, file, row start)] from the video's header lines."""
    out = []
    for line in VIDEO_ASS.read_text(encoding="utf-8-sig").splitlines():
        m = re.search(r",head,.*?A(\d+) · (\S+) · ([\d.]+)s", line)
        if m:
            out.append((int(m.group(1)), m.group(2), float(m.group(3))))
    return out


def main():
    rows, _ = pf.load_rows("covered")
    picks = {arm: lj.picks("v2", arm) for arm in ("text", "audio", "mosspro")}
    tally = {k: [0, 0] for k in ("ruler", "production", "text", "audio", "mosspro")}
    files = sorted({f for _, f, _ in clips()}, key=lambda i: i.startswith("onetonton"))
    stables = {f: pf.reference(f)[0] for f in files}
    for n, ident, start in clips():
        if n not in OWNER or OWNER[n] is None:
            continue
        stable = stables[ident]
        index = min(range(len(stable)), key=lambda i: abs(float(stable[i]["start"]) - start))
        r = next(x for x in rows[ident] if x["index"] == index)
        heard = OWNER[n]
        truth_tracks = {k for k, v in r["owner"].items() if v in r["truth"]}
        line = [f"A{n:02d} {ident} row {index}: owner heard {sorted(heard)}; production s{r['leader']}"]
        tally["ruler"][0] += bool(truth_tracks & heard)
        tally["ruler"][1] += 1
        tally["production"][0] += r["leader"] in heard
        tally["production"][1] += 1
        for arm, pk in picks.items():
            p = pk.get((ident, index))
            if p is None:
                line.append(f"{arm} -")
                continue
            track = int(p[1:]) if re.fullmatch(r"s\d+", p) else r["leader"]
            tally[arm][0] += track in heard
            tally[arm][1] += 1
            line.append(f"{arm} s{track}{'✓' if track in heard else '✗'}")
        print("  ".join(line))
    print("right / scored: " + "  ".join(f"{k} {a}/{b}" for k, (a, b) in tally.items()))
    print("ruler = the covered-truth person's tracks overlap what the owner heard")


if __name__ == "__main__":
    main()
