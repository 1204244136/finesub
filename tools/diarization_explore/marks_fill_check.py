"""Acceptance 3 of the marks' production port (`docs/plans/diar-marks-plan.md`, 「验收（预注册）」): what does filling
missed lines do to lines the reference has nothing on (music, effects mis-recognised as speech)?

On the fresh held-out: every missed line (no track reaches `min_share`), with or without a reference truth, split by
back-channel; the rule is judged on non-back-channel lines only.
* share of missed lines with no reference line at all <= 30%: fill them all, as step 1 did;
* > 30%: try filling only lines whose highest per-slot mean probability over the line is >= 0.05 (unfilled lines stay
  empty). Adopt it when it removes more than half of the no-reference lines and the fill-right rate on missed lines with
  a truth (empty counts as not right) drops by no more than 3 points; otherwise stop and ask the owner.

    python tools/diarization_explore/marks_fill_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marks_port_check as pc  # noqa: E402
import marks_step1 as ms  # noqa: E402
import p2_features as pf  # noqa: E402

from finesub.speech.diarization import marks  # noqa: E402
from finesub.speech.diarization.assign import assign_all  # noqa: E402
from finesub.speech.diarization.segments import TrackSegment, segments_from_activity  # noqa: E402

GATE = 0.05
LIMIT = 0.30


def main():
    import relabel_eval as ev

    groups_, _ = ms.groups()
    fresh = groups_["fresh"]
    counts = {bc: {"no_ref": 0, "truth": 0, "right_all": 0, "right_gated": 0, "no_ref_gated": 0} for bc in (False, True)}
    for ident, records in fresh.items():
        stable, grid_ref, people, starts_ref = pf.reference(ident)
        by_index = {r["index"]: r for r in records}
        data = np.load(pf.CACHE / f"{ident}.npz")
        native = data["native"]
        slot_probs = marks.resample(data["probs"], float(data["frame_seconds"]), native.shape[0])
        evidence = pc.Evidence(native_activity=native, soft=data["soft"], centroid_of_track=data["centroid_of_track"],
                               hard=data["hard"], slot_probs=slot_probs, slot_grid=pc.grid_for(data["probs"], data["local"]))
        timeline = [TrackSegment(track=t.track, start=t.start, end=t.end)
                    for t in segments_from_activity(native.tolist(), frame_seconds=marks.STEP)]
        speakers = assign_all(stable, timeline)
        marked = marks.row_marks(stable, timeline, speakers, evidence)
        frames = native.shape[0]
        for index, (row, speaker, mark) in enumerate(zip(stable, speakers, marked)):
            if speaker.track is not None:
                continue
            lo = int(float(row["start"]) / marks.STEP)
            hi = min(max(lo + 1, int(np.ceil(float(row["end"]) / marks.STEP))), frames)
            if hi <= lo:
                continue
            bc = marks.characters(str(row.get("text", ""))) <= marks.BACKCHANNEL_CHARS
            gated = float(slot_probs[lo:hi].mean(axis=0).max()) >= GATE
            starters = [w for a, w in starts_ref if float(row["start"]) - 0.3 <= a < float(row["end"])]
            cover, _ = ev.span_cover(row, grid_ref)
            if not starters and not cover.any():
                counts[bc]["no_ref"] += 1
                counts[bc]["no_ref_gated"] += gated
                continue
            record = by_index.get(index)
            if record is None or record["state"] != "missed":
                continue
            counts[bc]["truth"] += 1
            right = mark is not None and pf.right(record, mark["track"])
            counts[bc]["right_all"] += right
            counts[bc]["right_gated"] += right and gated
        print(ident, "done", flush=True)

    for bc, c in counts.items():
        total = c["no_ref"] + c["truth"]
        print(f"{'back-channel' if bc else 'non-back-channel'} missed lines: {total}; no reference line {c['no_ref']} "
              f"({c['no_ref'] / max(1, total):.1%}); with a truth {c['truth']}: filled right {c['right_all'] / max(1, c['truth']):.1%}; "
              f"gate >= {GATE}: keeps {c['no_ref_gated']} of the no-reference lines, right {c['right_gated'] / max(1, c['truth']):.1%}")
    c = counts[False]
    share = c["no_ref"] / max(1, c["no_ref"] + c["truth"])
    if share <= LIMIT:
        print(f"VERDICT: {share:.1%} <= {LIMIT:.0%} -> fill every missed line (step 1's rule)")
        return
    removed = 1 - c["no_ref_gated"] / max(1, c["no_ref"])
    drop = (c["right_all"] - c["right_gated"]) / max(1, c["truth"])
    ok = removed > 0.5 and drop <= 0.03
    print(f"VERDICT: {share:.1%} > {LIMIT:.0%}; the gate removes {removed:.1%} of no-reference lines, right drops {100 * drop:.1f} pts "
          f"-> {'adopt the gate' if ok else 'stop and ask the owner'}")


if __name__ == "__main__":
    main()
