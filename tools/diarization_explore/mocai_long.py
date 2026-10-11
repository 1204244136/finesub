"""Does a Sortformer track keep its identity across a 6-11 hour stream?

Plan 5r.2 「摸清 Sortformer 本身」. NVIDIA writes that the arrival-order speaker cache degrades on long
recordings; everything we owned before 魔裁 stopped at 1.41 h, so the claim was untested here.

Scores ONE whole-file timeline in hourly windows, two ways:

  global  a track's owner is its majority character over the WHOLE file  -- what a consumer sees, since
          `-diar.json` gives a track one number for the whole recording
  local   a track's owner is its majority character within THAT hour     -- separation quality alone

`local` is an upper bound on `global` for every hour (the local owner is chosen to fit that hour). So the
gap `local - global` is the cost of identity drift specifically, with the per-window separation quality
held out -- a track that is clean every hour but means a different person in hour 7 than in hour 1 shows up
as a large gap, not as bad separation.

    python tools/diarization_explore/mocai_long.py 1 [--window 3600]

⚠ Same structural limits as mocai_diar.py: the streamer is not in the truth, cue spans are upper bounds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import mocai_diar as MD  # noqa: E402

FRAME = MD.FRAME


def build(part: int):
    tracks = json.loads((WT / "out" / f"mocai-in{part}" / f"mocai-in{part}-diar-nosep.json")
                        .read_text(encoding="utf-8"))["tracks"]
    cues = MD.truth_cues(f"lower{part}")
    end = max(max(t["end"] for t in tracks), max(b for _, b, _ in cues))
    people = sorted({w for *_, w in cues})
    frames = int(end / FRAME) + 2
    truth = np.full(frames, -1, dtype=np.int32)          # one character per frame; overlaps are rare in a VN
    for a, b, w in cues:
        truth[int(a / FRAME):int(np.ceil(b / FRAME))] = people.index(w)
    track_of = np.full((frames, 4), False)
    for t in tracks:
        track_of[int(t["start"] / FRAME):int(np.ceil(t["end"] / FRAME)), t["track"]] = True
    return tracks, cues, people, truth, track_of


def owners(truth, track_of, lo, hi, n_people):
    """Majority character per track inside frames [lo, hi)."""
    out = {}
    for k in range(4):
        sel = track_of[lo:hi, k] & (truth[lo:hi] >= 0)
        if sel.sum():
            out[k] = int(np.bincount(truth[lo:hi][sel], minlength=n_people).argmax())
    return out


def cue_score(cues, track_of, lo_s, hi_s, owner, people):
    """Duration-weighted: most active track over each cue belongs to that cue's character."""
    right = total = 0.0
    for a, b, w in cues:
        if not (lo_s <= a < hi_s):
            continue
        lo, hi = int(a / FRAME), max(int(a / FRAME) + 1, int(np.ceil(b / FRAME)))
        activity = track_of[lo:hi].sum(axis=0)
        total += b - a
        if activity.any():
            right += (b - a) * (owner.get(int(activity.argmax())) == people.index(w))
    return right / total if total else float("nan"), total


def transitions(truth, track_of, people, window, min_speech=120.0, still=60.0):
    """Every change of a track's local owner, classified as REUSE or SWAP.

    With more characters than slots a track SHOULD change person when one leaves the story and another
    arrives -- that is correct. A change is labelled REUSE when the old owner has gone quiet, else SWAP.

    ⚠ **SWAP is an UPPER BOUND on identity errors, not a count of them.** Measured on input1/input2 it
    reads 20 and 22, but most are not errors:
      - a SHARED slot oscillates: `s1 Ema -> Sherry -> Ema -> Sherry` every window is two people on one track
        whose majority flips on a close vote, not an identity event;
      - the protagonist is in nearly every window, so every change away from her passes "still speaking".
    REUSE is reliable (the old owner really did leave -- input2's Alisa, 0.0 min/hour after 5.2 h). For the
    rest use `classify`, which asks where the old owner's speech GOES.
    """
    n = len(people)
    step = int(window / FRAME)
    held = {}
    out = []
    for lo in range(0, len(truth), step):
        hi = min(len(truth), lo + step)
        if (truth[lo:hi] >= 0).sum() * FRAME < min_speech:
            continue
        local = owners(truth, track_of, lo, hi, n)
        for k, who in local.items():
            prev = held.get(k)
            if prev is not None and prev != who:
                after = int(((truth[lo:min(len(truth), lo + step)]) == prev).sum()) * FRAME
                out.append((lo * FRAME, k, people[prev], people[who], after, "SWAP" if after >= still else "reuse"))
            held[k] = who
    return out


def classify(truth, track_of, people, change, window, still=60.0, moved=0.5, held=0.20):
    """reuse / never-held / shared / MOVED for one owner change, by where the old owner's speech goes.

    Compares the window before the change with the window after it, on the OLD owner's frames only:
      reuse       the old owner speaks < `still` s afterwards -- they left; the slot was correctly handed on
      never-held  before the change less than `held` of their speech was on this track at all, so it was
                  never really theirs: the owner label was a plurality won by a hair. A first version counted
                  these as MOVED ("Hanna's share on s0: 1% -> 0%"), which is noise, not an identity event
      shared      they keep speaking and most of it stays on the same track -- two people on one slot, and the
                  owner label merely flipped on a close majority
      MOVED       the track did hold them, they keep speaking, and their share on it drops below `moved` of
                  what it was -- their speech went elsewhere while they were present. The real identity error.
    """
    at, k, old, _new, _after, _ = change
    i = people.index(old)
    step = int(window / FRAME)
    at_f = int(at / FRAME)

    def share_on_k(lo, hi):
        fr = truth[lo:hi] == i
        return (track_of[lo:hi, k][fr].sum() / fr.sum()) if fr.sum() else float("nan"), fr.sum() * FRAME

    before, _ = share_on_k(max(0, at_f - step), at_f)
    after, spoken = share_on_k(at_f, min(len(truth), at_f + step))
    if spoken < still:
        return "reuse", before, after
    if not before >= held:          # also catches NaN: they did not speak in the window before
        return "never-held", before, after
    if after < moved * before:
        return "MOVED", before, after
    return "shared", before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", type=int)
    parser.add_argument("--window", type=float, default=3600.0)
    parser.add_argument("--transitions", type=float, metavar="SECONDS",
                        help="also list owner changes at this resolution, classified REUSE vs SWAP")
    args = parser.parse_args()
    tracks, cues, people, truth, track_of = build(args.part)
    n = len(people)
    whole = owners(truth, track_of, 0, len(truth), n)
    span = len(truth) * FRAME
    print(f"input{args.part}: {span / 3600:.2f} h, {len(people)} characters, {len(cues)} truth cues")
    print(f"whole-file owners: " + ", ".join(f"s{k}={people[v]}" for k, v in sorted(whole.items())))
    print(f"\n{'window':>8}{'speech':>8}{'global':>9}{'local':>8}{'drift':>8}   hour owners (local)")
    rows = []
    start = 0.0
    while start < span:
        lo, hi = int(start / FRAME), int(min(span, start + args.window) / FRAME)
        local = owners(truth, track_of, lo, hi, n)
        g, secs = cue_score(cues, track_of, start, start + args.window, whole, people)
        l, _ = cue_score(cues, track_of, start, start + args.window, local, people)
        if secs >= 60:
            rows.append((start, g, l, secs))
            changed = [f"s{k}={people[v]}" + ("*" if whole.get(k) != v else "") for k, v in sorted(local.items())]
            print(f"{start / 3600:>7.1f}h{secs / 60:>7.1f}m{g:>9.1%}{l:>8.1%}{100 * (l - g):>+7.1f}   {', '.join(changed)}")
        start += args.window
    if rows:
        w = np.array([r[3] for r in rows])
        g = np.array([r[1] for r in rows])
        l = np.array([r[2] for r in rows])
        print(f"\nweighted: global {np.average(g, weights=w):.1%}, local {np.average(l, weights=w):.1%}, "
              f"identity-drift cost {100 * np.average(l - g, weights=w):+.1f} pp   (* = hour owner differs from whole-file owner)")
        # ⚠ Test degradation on LOCAL, never on global. A first-half/second-half split of `global` is decided
        # by the majority vote: if the stream has two stable slot mappings (a mid-stream identity swap), the
        # whole-file owner is fitted to the bigger regime and the other one scores badly under it. input2 reads
        # 32.9% -> 53.5% that way, which looks like the model warming up and is only the vote.
        hours = np.array([r[0] for r in rows]) / 3600
        slope = np.polyfit(hours, l, 1, w=w)[0] if len(rows) > 2 else float("nan")
        half = len(rows) // 2
        print(f"separation over time (local): first half {np.average(l[:half], weights=w[:half]):.1%}, "
              f"second half {np.average(l[half:], weights=w[half:]):.1%}, trend {100 * slope:+.2f} pp/hour")
    if args.transitions:
        changes = transitions(truth, track_of, people, args.transitions)
        swaps = [c for c in changes if c[5] == "SWAP"]
        print(f"\nowner changes at {args.transitions:.0f} s resolution: {len(changes)} "
              f"({len(changes) - len(swaps)} reuse, {len(swaps)} SWAP)")
        print("  ⚠ SWAP is an upper bound, NOT an error count: shared-slot oscillation and the always-present "
              "protagonist both land here (see transitions.__doc__)")
        kinds = [(c, *classify(truth, track_of, people, c, args.transitions)) for c in changes]
        tally = {name: sum(1 for _, kind, *_ in kinds if kind == name)
                 for name in ("reuse", "never-held", "shared", "MOVED")}
        print(f"  by where the old owner's speech goes: {tally['reuse']} reuse, {tally['never-held']} never-held, "
              f"{tally['shared']} shared, {tally['MOVED']} MOVED (the track held them, they kept talking, it let go)")
        for (at, k, old, new, *_), kind, before, after in kinds:
            if kind == "MOVED":
                print(f"    {at / 3600:5.2f}h  s{k}: {old} -> {new}   ({old}'s share on s{k}: {before:.0%} -> {after:.0%})")


if __name__ == "__main__":
    main()
