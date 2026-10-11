"""C.1: the acceptance set for track naming, and the scorer that uses it.

Without this, every change to the role prompt is a matter of impression. onetonton has per-line
speaker truth and three named people, so "which person does track s1 actually belong to" has an
answer, and a model's reply can be scored against it.

Two numbers, and ⚠ **the second one is why this exists**:

* **naming accuracy**, weighted by the speech time each track holds -- getting the streamer right
  matters more than getting a track that says four words right;
* **abstention rate**, reported separately and never folded in. A prompt that guesses when it is
  unsure scores better on accuracy alone and is worse in the product, where a confident wrong name
  costs a person more than an admitted blank.

⚠ This scores an ANSWER FILE. It calls no model and spends no quota; producing answers is C.2/C.4.

    python tools/diarizen_export/roles_bed.py --build        # write the truth mapping
    python tools/diarizen_export/roles_bed.py --score a.json # score a reply against it
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TRUTH = REPO / "tmp" / "diarizen-bed" / "roles-truth.json"
TIMELINES = REPO / "tmp" / "diarizen-export" / "port-timelines"

#: A track holding less of the speech than this is not scored: its "true owner" is decided by a
#: handful of frames, so a model getting it wrong says more about the label than about the model.
MIN_SHARE = 0.02

#: ⚠ **The truth is a person, not a string.** The first version compared `reply == truth` byte for
#: byte, and the ONLY two misses in 30 scored episode-runs were `櫻川惠` (the roster name in
#: traditional characters) and `桜川めぐ` (the same person's Japanese name). Both identify the
#: person correctly; a scorer that calls them wrong is measuring orthography. Written as an
#: explicit table rather than a normalisation rule because 樱 / 櫻 / 桜 are three separate
#: codepoints that no unicodedata form folds together, and because a research bed with three
#: named people is better served by a list somebody can check than by a rule nobody can.
#: ⚠ This was added AFTER seeing the results, which is the shape `eval_contract` exists to
#: prevent -- recorded here because it is admissible only for a specific reason: it moves BOTH
#: arms to 100%, so it removes the metric's only discrimination rather than creating any.
ALIASES = {
    "樱川惠": ("櫻川惠", "桜川惠", "桜川めぐ", "樱川めぐ", "櫻川めぐ"),
    "青木佑磨": ("青木佑磨",),
    "小日向美香": ("小日向美香",),
}
#: Every spelling that means a given roster person, folded one way for lookup.
_PERSON_OF = {spelling: person
              for person, spellings in ALIASES.items()
              for spelling in (person, *spellings)}


def same_person(said: str, owner: str) -> bool:
    """Whether `said` names the person `owner` names, allowing a known alternate spelling."""
    said = str(said).strip()
    return _PERSON_OF.get(said, said) == _PERSON_OF.get(owner, owner)


def truth_for(episode: int) -> dict | None:
    """`{track: {"owner": name, "share": float}}` for one episode, or None."""
    import backend_cost as bc

    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import FRAME_SECONDS, segments_from_activity

    data = bc.ev.episode(episode)
    path = TIMELINES / f"onetonton-{episode:02d}.npz"
    if data is None or not path.exists():
        return None
    stable, _, grid, people, _ = data
    probs = np.load(path)["probs"].astype(np.float32)

    tracks = segments_from_activity(probs.tolist())
    seconds: Counter = Counter()
    owners: dict[str, Counter] = {}
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        cell = speaker.to_dict()
        held = cell.get("tracks") or ([cell["track"]] if cell.get("track") is not None else [])
        if not held:
            continue
        low = int(float(row["start"]) / FRAME_SECONDS)
        high = max(low + 1, int(np.ceil(float(row["end"]) / FRAME_SECONDS)))
        span = grid[low:high]
        counts = {people[p]: int(span[:, p].sum()) for p in range(span.shape[1])
                  if span[:, p].any()}
        if not counts:
            continue
        who = max(counts, key=counts.get)
        duration = max(0.0, float(row["end"]) - float(row["start"]))
        # ⚠ Only the leading track is credited. A row covered by two tracks tells us nothing about
        # which of them owns it, and splitting the credit would let overlap decide ownership.
        lead = str(held[0])
        seconds[lead] += duration
        owners.setdefault(lead, Counter())[who] += duration

    total = sum(seconds.values()) or 1.0
    return {
        track: {"owner": counts.most_common(1)[0][0],
                "share": round(seconds[track] / total, 4),
                "purity": round(counts.most_common(1)[0][1] / max(sum(counts.values()), 1e-9), 4)}
        for track, counts in owners.items()
    }


def score(reply: dict, truth: dict) -> dict:
    """One episode's naming accuracy and abstention rate, weighted by speech time."""
    scored = {t: v for t, v in truth.items() if v["share"] >= MIN_SHARE}
    if not scored:
        return {"weight": 0.0}
    weight = sum(v["share"] for v in scored.values())
    correct = abstained = 0.0
    for track, facts in scored.items():
        answer = reply.get(track) or reply.get(f"s{track}")
        name = (answer or {}).get("name") if isinstance(answer, dict) else answer
        if not name:
            abstained += facts["share"]
        elif same_person(name, facts["owner"]):
            correct += facts["share"]
    # ⚠ The ignored tracks are counted for BLANKS even though they are not scored. `abstention`
    # over scored tracks alone read a flat 0.0% across 30 episode-runs and looked like a model
    # that never declines -- while every blank it produced sat on a track holding 0.02%-0.19% of
    # the speech, i.e. the abstention was working perfectly and `MIN_SHARE` was filtering out the
    # only evidence of it. A number whose zero cannot be distinguished from "not measured here"
    # is worse than no number, so the tiny tracks report their own.
    ignored = [t for t in truth if t not in scored]
    blank_tiny = sum(1 for t in ignored
                     if not ((reply.get(t) or reply.get(f"s{t}") or {}) or {}).get("name"))
    return {
        "weight": round(weight, 4),
        "accuracy": round(correct / weight, 4),
        "abstention": round(abstained / weight, 4),
        "tracks_scored": len(scored),
        "tracks_ignored": len(ignored),
        "tiny_tracks_declined": blank_tiny,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--score")
    parser.add_argument("--episodes", nargs="+", type=int,
                        default=list(range(1, 16)))
    args = parser.parse_args()

    if args.build:
        truth = {}
        for episode in args.episodes:
            per_episode = truth_for(episode)
            if per_episode is None:
                print(f"#{episode}: no timeline or no axis, skipped")
                continue
            truth[str(episode)] = per_episode
            listed = ", ".join(
                f"s{t}->{v['owner']}({v['share']:.0%}, purity {v['purity']:.0%})"
                for t, v in sorted(per_episode.items(), key=lambda kv: -kv[1]["share"]))
            print(f"#{episode:2d}: {listed}")
        TRUTH.parent.mkdir(parents=True, exist_ok=True)
        TRUTH.write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {TRUTH} ({len(truth)} episodes)")
        # ⚠ Purity is reported so a low one is visible: a track whose "owner" holds only 60% of it
        # is a weak label, and a model marked wrong on it may be right about the audio.
        purities = [v["purity"] for e in truth.values() for v in e.values()
                    if v["share"] >= MIN_SHARE]
        if purities:
            print(f"  scored tracks: {len(purities)}, purity median {np.median(purities):.0%}, "
                  f"min {min(purities):.0%}")
        return

    if args.score:
        truth = json.loads(TRUTH.read_text(encoding="utf-8"))
        replies = json.loads(Path(args.score).read_text(encoding="utf-8"))
        rows = []
        for episode, per_episode in sorted(truth.items(), key=lambda kv: int(kv[0])):
            result = score(replies.get(episode, {}), per_episode)
            if result["weight"]:
                rows.append(result)
                print(f"#{int(episode):2d}: accuracy {result['accuracy']:.1%}  "
                      f"abstention {result['abstention']:.1%}  "
                      f"({result['tracks_scored']} scored, {result['tracks_ignored']} too small)")
        if rows:
            print(f"\nn={len(rows)}: accuracy {np.mean([r['accuracy'] for r in rows]):.1%}  "
                  f"abstention {np.mean([r['abstention'] for r in rows]):.1%}")
            print("  ⚠ read them together: accuracy alone rewards guessing.")
        return

    parser.error("pass --build or --score")


if __name__ == "__main__":
    main()
