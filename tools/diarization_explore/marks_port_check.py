"""Acceptance 1 of the marks' production port (`docs/plans/diar-marks-plan.md`, 「验收（预注册）」): does
`finesub.speech.diarization.marks` reproduce step 1 line by line?

Two passes over step 1's caches (P2 dev hina, the fresh held-out, 姐姐咚咚), compared with step 1's own tiers
(`marks_step1` models, p, q, back-channel lines unmarked, mixed fill), on the lines both sides have (step 1 keeps only
lines with a reference truth):

1. in memory, from the cached float32 probabilities and float64 soft scores, on step 1's unrounded timeline;
2. through the product path: the timeline and a sidecar written to disk, `apply_to_stable`, the marks read back.

Both must agree on every line; a difference whose score sits within 1e-9 of p or q is counted and passed. For every
other difference on the product path the script also says whether `assign` itself labels the line differently on the
timeline as `-diar.json` stores it (times rounded to 3 decimals): such a line differs upstream of the marks.

    python tools/diarization_explore/marks_port_check.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import marks_export as me  # noqa: E402
import marks_step1 as ms  # noqa: E402
import p2_features as pf  # noqa: E402

from finesub.speech.diarization import marks, nemotron3_backend  # noqa: E402
from finesub.speech.diarization.assign import assign_all  # noqa: E402
from finesub.speech.diarization.evidence import ClusterEvidence, evidence_path, timeline_key  # noqa: E402
from finesub.speech.diarization.segments import TrackSegment, segments_from_activity, segment_overlap_ratio  # noqa: E402
from finesub.speech.diarization.stage import DiarizationResult, apply_to_stable, write_timeline  # noqa: E402

NEAR = 1e-9


class Evidence:
    """The sidecar's fields, in memory."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def grid_for(probs, local):
    """The rasterised slot grid whose windows are the cached `local` (the cache kept the windows, not the grid)."""
    base = int(np.ceil(len(probs) / 2))
    for frames in (base, base - 1, base + 1, base + 2):
        grid = nemotron3_backend.rasterised(probs, samples=frames * 320, source_step=0.01)
        if np.array_equal(nemotron3_backend.windows(grid), local):
            return grid
    raise AssertionError("no grid reproduces the cached windows")


def expected(records, model, cmodel, p, q, slots, ident):
    """{line index: mark} as step 1 decides it, on the lines with a truth."""
    out = {}
    labelled = [r for r in records if r["leader"] is not None]
    scores = dict(zip((r["index"] for r in labelled), model.predict_proba(pf.matrix(labelled))[:, 1])) if labelled else {}
    for r in records:
        if r["leader"] is None:
            s = slots.get((ident, r["index"]))
            if s and s["slot_track"] >= 0 and s["slot_purity"] >= marks.FILL_PURITY:
                out[r["index"]] = ("filled", s["slot_track"], "slot")
            elif r["nearest"] is not None:
                out[r["index"]] = ("filled", r["nearest"], "nearest")
            else:
                out[r["index"]] = None
            continue
        score = scores[r["index"]]
        if r["chars"] <= marks.BACKCHANNEL_CHARS or score < p:
            out[r["index"]] = ("none", abs(score - p))
            continue
        k, s = ms.alternative(cmodel, r)
        out[r["index"]] = ("low", k if (k is not None and s >= q) else None, min(abs(score - p), abs(s - q)))
    return out


def as_tuple(mark):
    if mark is None:
        return ("none",)
    if mark["tier"] == "filled":
        return ("filled", mark["track"], mark["by"])
    return ("low", mark.get("alternative"))


def same(want, got):
    """(agree, near-threshold)."""
    if want is None:
        return got == ("none",), False
    if want[0] == "filled":
        return got == want, False
    if want[0] == "none":
        return got == ("none",), want[1] <= NEAR
    return got == ("low", want[1]), want[2] <= NEAR


def compare(name, per_file, upstream):
    lines = sum(len(v) for v in per_file.values())
    diffs = [(f, i, w, g, near) for f, rows in per_file.items() for i, (w, g, ok, near) in rows.items() if not ok]
    near = sum(1 for *_, n in diffs if n)
    moved = sum(1 for f, i, *_ in diffs if i in upstream.get(f, ()))
    print(f"  {name}: {lines} lines, {len(diffs)} differ ({near} within {NEAR} of p or q; "
          f"{moved} where `assign` itself differs on the rounded timeline)")
    for f, i, w, g, _ in diffs[:15]:
        print(f"    {f}#{i}: step 1 {w} vs port {g}{'  [assign differs]' if i in upstream.get(f, ()) else ''}")
    return len(diffs) - near, moved


def main():
    groups_, slots = ms.groups()
    model, cmodel, p, q = me.fitted()
    assert abs(p - marks.P_WRONG) < 1e-12 and abs(q - marks.ALTERNATIVE) < 1e-12, "constants out of date: rerun marks_export"
    files = {name: groups_[name] for name in ("dev", "fresh", "姐姐咚咚")}
    order = [f for g in files.values() for f in g]
    order.sort(key=lambda f: f.startswith("onetonton"))  # p2_features.reference: hina first
    failures = 0
    passes = {"in memory": {}, "product path": {}}
    upstream = {"in memory": {}, "product path": {}}
    for ident in order:
        records = next(g[ident] for g in files.values() if ident in g)
        stable = pf.reference(ident)[0]
        data = np.load(pf.CACHE / f"{ident}.npz")
        native = data["native"]
        slot_probs = marks.resample(data["probs"], float(data["frame_seconds"]), native.shape[0])
        assert np.array_equal(slot_probs, pf.resample(data["probs"], float(data["frame_seconds"]), native.shape[0])), ident
        grid = grid_for(data["probs"], data["local"])
        evidence = Evidence(native_activity=native, soft=data["soft"], centroid_of_track=data["centroid_of_track"],
                            hard=data["hard"], slot_probs=slot_probs, slot_grid=grid)
        want = expected(records, model, cmodel, p, q, slots, ident)

        timeline = [TrackSegment(track=t.track, start=t.start, end=t.end)
                    for t in segments_from_activity(native.tolist(), frame_seconds=marks.STEP)]
        speakers = assign_all(stable, timeline)
        got = marks.row_marks(stable, timeline, speakers, evidence)
        passes["in memory"][ident] = {i: (w, as_tuple(got[i]), *same(w, as_tuple(got[i]))) for i, w in want.items()}

        with tempfile.TemporaryDirectory() as tmp:
            stable_path = Path(tmp) / "x-stable.json"
            diar_path = Path(tmp) / "x-diar.json"
            stable_path.write_text(json.dumps({"segments": stable}, ensure_ascii=False), encoding="utf-8")
            result = DiarizationResult(tracks=tuple(timeline), frame_seconds=marks.STEP,
                                       overlap=segment_overlap_ratio(timeline), backend="nemotron3")
            write_timeline(result, diar_path)
            ClusterEvidence(soft=data["soft"], centroids=np.zeros((1, 1)), native_activity=native, native_step=marks.STEP,
                            centroid_of_track=data["centroid_of_track"], slot_grid=grid, slot_probs=slot_probs,
                            hard=data["hard"], timeline=timeline_key(timeline)).save(evidence_path(diar_path))
            apply_to_stable(stable_path, diar_path)
            payload = json.loads(stable_path.read_text(encoding="utf-8"))
            assert payload["metadata"]["speaker"]["marks"] is not None, ident
            read = [(segment.get("speaker") or {}).get("mark") for segment in payload["segments"]]
        passes["product path"][ident] = {i: (w, as_tuple(read[i]), *same(w, as_tuple(read[i]))) for i, w in want.items()}
        stored = assign_all(stable, result.tracks)
        upstream["product path"][ident] = {i for i, (a, b) in enumerate(zip(speakers, stored))
                                           if a.track != b.track or a.extras.get("tracks") != b.extras.get("tracks")}
        print(ident, "done", flush=True)

    explained = 0
    for name, per_file in passes.items():
        print(f"{name}:")
        for group, members in files.items():
            failed, moved = compare(group, {f: per_file[f] for f in members}, upstream[name])
            failures += failed
            explained += moved
        tiers = Counter(g[0] for rows in per_file.values() for (_, g, _, _) in rows.values())
        print(f"  tiers on compared lines: {dict(tiers)}")
    if failures == 0:
        print("PASS")
    else:
        print(f"FAIL by the pre-registered rule: {failures} lines differ away from the thresholds; "
              f"{explained} of them are lines `assign` labels differently on the stored (rounded) timeline")


if __name__ == "__main__":
    main()
