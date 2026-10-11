"""Plan §5p.2: measure our diarization against the 姐姐咚咚 refined track's speaker labels.

⚠ Read §5p.2 first. The text is a Chinese translation, so **no content measure is meaningful
here**; what this material carries that nothing else we have does is **a human speaker label on
every line of real Japanese multi-person audio**.

⚠ Their end times are snapped to the next line (87% of adjacent lines), so a line's span is not
its speech. Everything below leans on **starts** and on the 399 pairs whose start *and* end match
exactly -- the only overlap annotation whose extent is not a guess.

  python refined_compare.py OUT_DIR EPISODE   # e.g. ../../out/<stem> 1
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import refined_ass as ra

FRAME = 0.08  #: the diarization frame, so both sides land on one grid


def activity_grid(spans, frames):
    """0/1 occupancy on the frame grid."""
    signal = np.zeros(frames, dtype=np.float32)
    for start, end in spans:
        a, b = int(max(0.0, start) / FRAME), int(np.ceil(max(0.0, end) / FRAME))
        signal[a:min(b, frames)] = 1.0
    return signal


def start_offsets(theirs, ours_starts):
    """How far each of their line starts sits from our nearest segment start (M1, amended).

    ⚠ Amendment, 2026-09-12, written after the first run and before looking at M2-M5 (plan §5p.5):
    the pre-registered gate correlated the two occupancy signals and demanded >= 0.5. It came out
    at 0.392 -- but occupancy is the wrong quantity: their track covers 95.6% of the timeline and
    ours 85.7%, and two near-constant signals cannot correlate however well they line up. Starts
    are the reliable end of their axis (§5p.1), so the drift is measured on starts instead.
    """
    import bisect

    gaps = []
    for start, _, _, _ in theirs:
        index = bisect.bisect_left(ours_starts, start)
        near = [ours_starts[j] for j in (index - 1, index) if 0 <= j < len(ours_starts)]
        if near:
            gaps.append(min((n - start for n in near), key=abs))
    return np.array(sorted(gaps))


def best_offset(theirs, ours, limit=30.0):
    """The pre-registered M1: global drift by correlating occupancy. Kept for the record; it is
    reported alongside the amended measure and no longer gates anything."""
    span = int(limit / FRAME)
    theirs = theirs - theirs.mean()
    ours = ours - ours.mean()
    width = min(len(theirs), len(ours))
    theirs, ours = theirs[:width], ours[:width]
    denominator = np.sqrt((theirs ** 2).sum() * (ours ** 2).sum()) or 1.0
    scores = []
    for shift in range(-span, span + 1):
        if shift >= 0:
            value = float((theirs[shift:] * ours[:width - shift]).sum())
        else:
            value = float((theirs[:width + shift] * ours[-shift:]).sum())
        scores.append((value / denominator, shift * FRAME))
    return max(scores)


def load_ours(out_dir):
    out_dir = Path(out_dir)
    stem = out_dir.name
    payload = json.loads((out_dir / f"{stem}-stable.json").read_text(encoding="utf-8"))
    timeline = json.loads((out_dir / f"{stem}-diar.json").read_text(encoding="utf-8"))
    return payload["segments"], timeline


def report(out_dir, episode):
    theirs = [r for r in ra.events(ra.episodes()[episode - 1]) if r[2] != ra.NOTE and r[3]]
    segments, timeline = load_ours(out_dir)
    tracks = timeline["tracks"]
    frames = int(max(max(t["end"] for t in tracks), max(r[1] for r in theirs)) / FRAME) + 2

    # ---- M1 drift ---------------------------------------------------------
    mine = activity_grid([(s["start"], s["end"]) for s in segments], frames)
    yours = activity_grid([(r[0], r[1]) for r in theirs], frames)
    score, _ = best_offset(yours, mine)
    starts = sorted(s["start"] for s in segments)
    gaps = start_offsets(theirs, starts)
    offset = float(np.median(gaps))
    spread = float(np.median(np.abs(gaps)))
    print(f"M1 整体漂移（修正后的判据）：起点差中位 {offset:+.3f} s，|差| 中位 {spread:.3f} s，"
          f"|差| < 0.25 s 的占 {np.mean(np.abs(gaps) < 0.25):.1%}")
    print(f"   覆盖率 他们 {yours.mean():.1%} / 我们 {mine.mean():.1%}；"
          f"预注册那条占用信号互相关的峰值 {score:.3f}（量的是占空比，已作废，见 §5p.5）")
    worse = all(np.median(np.abs(start_offsets([(a + s, b + s, w, x) for a, b, w, x in theirs], starts))) > spread
                for s in (-0.5, 0.5))
    print(f"   整体平移 ±0.5 s 后变差：{'是' if worse else '否'}")
    if spread >= 0.5 or not worse:
        print("   ⇒ 对不上，停在这里")
        return
    shifted = [(a - offset, b - offset, who, text) for a, b, who, text in theirs]

    # ---- M2 speaker count -------------------------------------------------
    hints = timeline.get("metadata", {}).get("hints", {})
    print(f"\nM2 说话人数：真值 3，我们 {timeline['metadata']['track_count']} 条轨；"
          f"fourth_track_share {hints.get('fourth_track_share')}")

    # ---- M3 track purity --------------------------------------------------
    # A track's frames are credited to whoever the refined track says was talking then. Starts are
    # the reliable end, so a line only claims its first second -- padding cannot reach that far.
    # ⚠ Neither the 1 s window nor the overlap rule was in the 5p.2 pre-registration (audit 5q).
    # Single-label: one owner per frame, later lines overwriting earlier ones in file order -- so
    # during simultaneous speech the truth is whichever line came last. Multi-label: a frame is
    # the track's owner's if that person is AMONG those speaking. 8.8% of the trusted frames on #1
    # have two speakers; the single-label figure is the lower bound.
    CLAIM = 1.0
    owner = np.full(frames, -1, dtype=np.int8)
    people = sorted({r[2] for r in shifted})
    speaking = np.zeros((frames, len(people)), dtype=bool)
    for start, _, who, _ in shifted:
        a = int(max(0.0, start) / FRAME)
        owner[a:min(a + int(CLAIM / FRAME), frames)] = people.index(who)
        speaking[a:min(a + int(CLAIM / FRAME), frames), people.index(who)] = True
    trusted = speaking.any(axis=1)
    full = np.zeros((frames, len(people)), dtype=bool)
    for start, end, who, _ in shifted:
        full[int(max(0.0, start) / FRAME):min(int(np.ceil(max(0.0, end) / FRAME)), frames), people.index(who)] = True
    print(f"\nM3 轨的纯度（只算每行起点后 1 s 的可信区；其中 {(speaking.sum(axis=1) >= 2)[trusted].mean():.1%} 的帧有两人同时说）：")
    owner_of_track = {}
    for k in sorted({t["track"] for t in tracks}):
        mask = np.zeros(frames, dtype=bool)
        for t in tracks:
            if t["track"] == k:
                mask[int(t["start"] / FRAME):int(np.ceil(t["end"] / FRAME))] = True
        counts = collections.Counter(owner[mask & (owner >= 0)].tolist())
        total = sum(counts.values())
        if not total:
            print(f"   s{k}: 落在可信区的帧太少")
            continue
        who, top = counts.most_common(1)[0]
        multi = speaking[mask & trusted]
        best = int(multi.sum(axis=0).argmax())
        # Read the full-line-span figures. Their ends are snapped to the next line; the same snap
        # applied to meeting truth moves mixing by -0.4..+0.9 pp (track_mixing --snap), so the
        # figure is close, not a bound. The 1 s window reads 74-82% purity even on meeting
        # corpora that are 93-98% pure by full truth (track_mixing --chop 2.1 --claim 1), so it
        # cannot judge a track (5p.5 re-correction, 5q).
        spans = full[mask][full[mask].any(axis=1)]
        if not len(spans):
            print(f"   s{k}: 完整行跨度里没有有人说话的帧")
            continue
        shares = spans.sum(axis=0) / len(spans)
        full_best = int(shares.argmax())
        owner_of_track[k] = people[full_best]
        rest = "，".join(f"{people[i]} {shares[i]:.0%}" for i in np.argsort(-shares) if i != full_best and shares[i])
        print(f"   s{k}: 主人 {people[full_best]}  完整行跨度纯度 {shares[full_best]:.1%}（{len(spans) * FRAME:.0f} s）"
              f"  其余在说的人 {rest}  ｜1 s 口径 单标签 {top / total:.1%} / 多标签 {multi[:, best].mean():.1%}（量法偏低、判不了）")

    # ---- M4 overlap -------------------------------------------------------
    exact = []
    order = sorted(shifted)
    for i, (a0, a1, wa, _) in enumerate(order):
        for b0, b1, wb, _ in order[i + 1:]:
            if b0 >= a1 - 0.005:
                break
            if wa != wb and abs(a0 - b0) < 0.005 and abs(a1 - b1) < 0.005:
                exact.append((a0, a1))
    active = np.zeros((frames, 1 + max(t["track"] for t in tracks)), dtype=bool)
    for t in tracks:
        active[int(t["start"] / FRAME):int(np.ceil(t["end"] / FRAME)), t["track"]] = True
    hit = two_people = 0
    for a, b in exact:
        window = active[int(a / FRAME):int(np.ceil(b / FRAME))]
        if len(window) and (window.sum(axis=1) >= 2).any():
            hit += 1
            # Two tracks sounding is not two people when both tracks belong to one (M3's s2/s3).
            if any(len({owner_of_track.get(k) for k in np.nonzero(row)[0]}) >= 2
                   for row in window if row.sum() >= 2):
                two_people += 1
    print(f"\nM4 同时发话：人工明确标注 {len(exact)} 处（起止完全相同的行对），"
          f"我们在其中 {hit} 处报了 ≥2 条轨同时在响 = {hit / max(len(exact), 1):.1%}；"
          f"去掉只靠同一个人的两条轨撑起来的：{two_people} = {two_people / max(len(exact), 1):.1%}")

    # ---- M6 where the lines are cut (audit 5q; NOT pre-registered, descriptive) -------------
    # The gap scorer v2 leaves on purpose: it is blind to line breaks, and D3's product is line
    # breaks. A human turn is a run of consecutive lines by one speaker, spanning until the next
    # speaker starts (starts only -- 5p.1). Count how many runs of leader track our labelled
    # stable segments make inside it. The human makes one by construction. ⚠ This is production's
    # own segmentation with its labels (A0), not T4, which has not been run on this material.
    turns, order = [], sorted(shifted)
    for i, (start, _, who, _) in enumerate(order):
        if turns and turns[-1][2] == who:
            continue
        turns.append([start, None, who])
    for i in range(len(turns)):
        turns[i][1] = turns[i + 1][0] if i + 1 < len(turns) else float("inf")
    ours = sorted((s["start"], (s.get("speaker") or {}).get("track")) for s in segments)
    pieces, switches_person, switches_split, owned = [], 0, 0, 0
    for begin, finish, who in turns:
        inside = [track for start, track in ours if begin - 0.25 <= start < finish - 0.25 and track is not None]
        if not inside:
            continue
        changes = [(x, y) for x, y in zip(inside, inside[1:]) if x != y]
        # A switch between one person's two tracks is the split (M2 / M3), not a cut in the wrong
        # place -- counting both as one number mixed two defects (audit 5q, Fable round 2).
        split = sum(1 for x, y in changes if owner_of_track.get(x) is not None and owner_of_track.get(x) == owner_of_track.get(y))
        pieces.append(1 + len(changes) - split)
        switches_person += len(changes) - split
        switches_split += split
        owned += sum(owner_of_track.get(track) == who for track in inside) / len(inside)
    if pieces:
        ordered = sorted(pieces)
        print(f"\nM6 一轮里我们换了几次人（生产分句 + 标签；人工恒为 1）：{len(pieces)} 轮，"
              f"中位 {ordered[len(ordered) // 2]}，只有 1 个人的 {sum(p == 1 for p in pieces) / len(pieces):.0%}；"
              f"轮内换人 {switches_person} 次，另有同一人两条轨之间的切换 {switches_split} 次；"
              f"轮内分句的轨归给这一轮说话人的平均 {owned / len(pieces):.0%}")

    # ---- M5 not_same ------------------------------------------------------
    pairs = hints.get("co_activity", {})
    flagged = set(hints.get("not_same", ()))
    print(f"\nM5 `not_same`：{len(flagged)}/{len(pairs)} 对被判为「不同的人」"
          f"（真值：三位是真的不同人）")
    for pair, share in sorted(pairs.items()):
        print(f"   {pair} 共活 {share:.1%}{'  判为不同的人' if pair in flagged else '  未判'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("episode", type=int)
    arguments = parser.parse_args()
    report(arguments.out_dir, arguments.episode)


if __name__ == "__main__":
    main()
