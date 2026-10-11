"""W6: if the tail tracks stop asking for a name, how much naming is left and what does it cost?

W2b measured the problem: on hina, DiariZen leads with 4.63 more tracks per episode than Sortformer
(19 of 19 worse) while cutting the edit cost by 300 (18 of 19 better). Those two cannot be added --
the owner priced line edits, never track-level naming -- so the question to put to them is not the
abstract exchange rate but "with this rule on, how many tracks still need a name, and what does it
cost". This computes exactly that, offline, on both corpora.

The rule (plan closeout section 2, row 4): a track holding less than `floor_seconds` of speech OR
less than `floor_share` of all track activity renders as `other` instead of asking for a name. The
track number is kept; the partition is untouched (P15).

    python tools/diarization_explore/w6_tail.py --write

Thresholds frozen before this ran: naming cost must fall, `lenient` (the true person is somewhere
in the label set) must not fall by more than 0.5 pp, and -- added by the reviewer, and the reason
the rule could be wrong -- the share of reference-covered row time whose LEADING track gets binned
must stay under 5%. Above that the rule is not trimming a tail, it is hiding speech.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402
import backend_cost as bc  # noqa: E402
import cost_metrics as cm  # noqa: E402
import relabel_eval as ev  # noqa: E402
from accuracy_report import bootstrap_mean_ci  # noqa: E402

FLOOR_SECONDS = 30.0
FLOOR_SHARE = 0.02

#: `accuracy_cross.configure()` rebinds `lab.rm.sources`/`lab.rm.episode` to the Onetonton corpus,
#: so a later hina pass in the SAME process finds no files and silently scores nothing. Captured at
#: import and restored before the hina pass rather than ordering the corpora carefully.
_PRISTINE_RM = (lab.rm.sources, lab.rm.episode)


def binned_tracks(probs, *, floor_seconds=FLOOR_SECONDS, floor_share=FLOOR_SHARE) -> set[int]:
    """Tracks the rule renders as `other`: too short in absolute terms, or too small a share.

    ⚠ "share" is of the summed per-track activity, the same denominator `covering` uses -- not of
    the union of speech, which would count overlap once and move the line for a different reason.
    """
    seconds = cm.track_seconds(probs)
    total = sum(seconds.values()) or 1.0
    return {t for t, v in seconds.items() if v < floor_seconds or v / total < floor_share}


def measure(probs, stable, grid, *, floor_seconds=FLOOR_SECONDS, floor_share=FLOOR_SHARE):
    """Naming left, lenient before/after, and how much covered time the rule hides."""
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    tracks = segments_from_activity(probs.tolist())
    binned = binned_tracks(probs, floor_seconds=floor_seconds, floor_share=floor_share)
    owners = cm.owner_of_tracks(probs, grid, list(range(grid.shape[1])))

    leading, leading_named = set(), set()
    covered = hidden = 0.0
    lenient_before = lenient_after = 0.0
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        lo = int(row["start"] / lab.FRAME)
        hi = max(lo + 1, int(np.ceil(row["end"] / lab.FRAME)))
        span = grid[lo:hi].sum(axis=0)
        if not span.any():
            continue
        who = int(span.argmax())
        weight = row["end"] - row["start"]
        covered += weight
        listed = [int(t) for t in (speaker.extras.get("tracks") or ([] if speaker.track is None else [speaker.track]))]
        if speaker.track is not None:
            leading.add(int(speaker.track))
            if int(speaker.track) not in binned:
                leading_named.add(int(speaker.track))
            else:
                hidden += weight
        lenient_before += weight * any(owners.get(t) == who for t in listed)
        lenient_after += weight * any(owners.get(t) == who for t in listed if t not in binned)
    covered = covered or 1.0
    return {
        "tracks": len(cm.track_seconds(probs)),
        "leading": len(leading),
        "leading_named": len(leading_named),
        "binned": len(binned),
        "lenient_before": lenient_before / covered,
        "lenient_after": lenient_after / covered,
        "hidden_share": hidden / covered,
    }


def rows_for(corpus, episodes=None, *, floor_seconds=FLOOR_SECONDS, floor_share=FLOOR_SHARE):
    if corpus == "onetonton":
        from accuracy_cross import configure

        configure()
        keys = episodes or list(range(1, 16))
        loaders = [(k, ev.episode(k), bc.V2_OUT / f"onetonton-{k:02d}" / "diarizen.npz") for k in keys]
    else:
        lab.rm.sources, lab.rm.episode = _PRISTINE_RM
        frozen = json.loads((HERE / "results" / "diarizen-v3-20260917.json").read_text(encoding="utf-8"))
        keys = episodes or frozen["groups"]["heldout_all"]
        loaders = []
        for k in keys:
            loaded = bc.hina_episode(k)
            loaders.append((k, loaded[0] if loaded else None, loaded[1] if loaded else None))
    out = []
    for key, data, cached in loaders:
        if data is None or cached is None or not Path(cached).exists():
            continue
        stable, timeline, grid, people, _ = data
        arms = {
            "sortformer": bc.timeline_probs(timeline["tracks"], len(grid)),
            "diarizen": np.load(cached)["probs"][: len(grid)],
        }
        row = {"episode": key, "people": len(people)}
        for name, probs in arms.items():
            row[name] = measure(probs[: len(grid)], stable, grid,
                                floor_seconds=floor_seconds, floor_share=floor_share)
        out.append(row)
    return out


def report(corpus, rows):
    print(f"\n=== {corpus} ({len(rows)} files) ===")
    print(f"{'file':>10}{'ppl':>4}  {'arm':<11}{'lead':>6}{'named':>7}{'binned':>8}"
          f"{'lenient':>9}{'->':>9}{'hidden':>8}")
    for r in rows:
        for arm in ("sortformer", "diarizen"):
            m = r[arm]
            print(f"{str(r['episode']):>10}{r['people']:>4}  {arm:<11}{m['leading']:>6}{m['leading_named']:>7}"
                  f"{m['binned']:>8}{m['lenient_before']*100:>8.2f}%{m['lenient_after']*100:>8.2f}%"
                  f"{m['hidden_share']*100:>7.2f}%")
    summary = {}
    for arm in ("sortformer", "diarizen"):
        drop = [r[arm]["leading"] - r[arm]["leading_named"] for r in rows]
        lenient = [(r[arm]["lenient_after"] - r[arm]["lenient_before"]) * 100 for r in rows]
        hidden = [r[arm]["hidden_share"] * 100 for r in rows]
        named = [r[arm]["leading_named"] for r in rows]
        summary[arm] = {
            "leading_before": float(np.mean([r[arm]["leading"] for r in rows])),
            "leading_named": float(np.mean(named)),
            "naming_saved": {"mean": float(np.mean(drop)), "ci95": bootstrap_mean_ci(drop)},
            "lenient_delta_pp": {"mean": float(np.mean(lenient)), "ci95": bootstrap_mean_ci(lenient)},
            "hidden_share_pct": {"mean": float(np.mean(hidden)), "max": float(np.max(hidden))},
        }
        s = summary[arm]
        print(f"\n  {arm}: leading {s['leading_before']:.2f} -> named {s['leading_named']:.2f} "
              f"(saves {s['naming_saved']['mean']:.2f}/file)")
        print(f"    lenient {s['lenient_delta_pp']['mean']:+.3f} pp"
              f"   hidden time mean {s['hidden_share_pct']['mean']:.2f}% max {s['hidden_share_pct']['max']:.2f}%")
        if s["naming_saved"]["mean"] == 0:
            # Nothing to trim is not a failure: Sortformer emits four substantial tracks, so the
            # rule is inapplicable there rather than rejected.
            print("    [n/a ] the rule bins nothing on this arm")
            continue
        for name, ok in (("naming falls", s["naming_saved"]["mean"] > 0),
                         ("lenient drop <= 0.5 pp", s["lenient_delta_pp"]["mean"] >= -0.5),
                         ("hidden time < 5%", s["hidden_share_pct"]["mean"] < 5.0)):
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpora", nargs="*", default=["onetonton", "hina"])
    p.add_argument("--episodes", nargs="*", default=None,
                   help="score these instead of the frozen set -- for VALIDATING the candidate on "
                        "material that took no part in the split that produced it")
    p.add_argument("--floor-seconds", type=float, default=FLOOR_SECONDS)
    p.add_argument("--floor-share", type=float, default=FLOOR_SHARE,
                   help="0 turns the frozen two-clause rule into the `<30 s` single clause")
    p.add_argument("--label", default=None, help="suffix for the written file, so a validation "
                                                 "run cannot overwrite the frozen one")
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    payload = {}
    for corpus in args.corpora:
        rows = rows_for(corpus, args.episodes,
                        floor_seconds=args.floor_seconds, floor_share=args.floor_share)
        if not rows:
            raise SystemExit(f"{corpus}: scored nothing -- check the corpus wiring, do not report a blank pass")
        payload[corpus] = {"rows": rows, "summary": report(corpus, rows)}
    if args.write:
        out = bc.V2_OUT / f"w6-tail{'-' + args.label if args.label else ''}.json"
        out.write_text(json.dumps({**payload, "rule": {"floor_seconds": args.floor_seconds,
                                                       "floor_share": args.floor_share},
                                   "episodes": args.episodes,
                                   "script": lab.digest(__file__)}, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
