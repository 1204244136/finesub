"""A per-row identity confidence from the clustering itself, against the time-coverage margin (owner 2026-10-05).

Production's per-row confidences are TIME coverage: how much of the row each track is active for. About six
in ten wrong labels sit on rows a single track covers almost fully -- the timing is right, the person is not --
and no coverage threshold sees them. VBx already computes, for every (16 s chunk, local speaker), how close
that voiceprint sits to each cluster centroid (`ClusterResult.soft`, 2 - cosine distance). Per row: the
active-frame-weighted mean, over the chunk-speakers assigned to the row's track, of (soft to own centroid -
best soft to any other centroid). Low = the voiceprint is not clearly this track's person.

Material: 姐姐咚咚 15 (oversplit_sweep cache; production's default setting reproduces production exactly),
W2 truth, owners and "merged" exclusion as in `retrack_lowconf.py` -- but tracks that switched hands midway
are NOT excluded here, so its time-margin numbers do not line up with that script's table; the soft-vs-time
comparison is on the same rows and stands. At equal flag budgets it compares the soft margin, the time margin,
and half of each.

    python tools/diarization_explore/crosstalk_soft.py
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oversplit_sweep as osw  # noqa: E402

BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30)
MERGED = 0.80


def rows_for(ident):
    import nemotron3_cost as nc
    import relabel_eval as ev
    from finesub.speech.diarization import diarizen_assemble as da
    from finesub.speech.diarization import diarizen_cluster
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.nemotron3_backend import NATIVE_STEP
    from finesub.speech.diarization.segments import segments_from_activity
    from finesub.speech.diarization.stage import resolve_backend

    nc.corpus_keys("onetonton")
    data = np.load(osw.OUT / f"{ident}.npz")
    local, embeddings = data["local"], data["embeddings"]
    plda = resolve_backend(None).ensure(None, "cpu").plda_dir
    clustered = diarizen_cluster.cluster(embeddings, local, plda)
    hard, soft, centroid_of_track = clustered.hard, clustered.soft, clustered.centroid_of_track
    native = da.assemble(local, hard).astype(np.float32)
    tracks = segments_from_activity(native.tolist(), frame_seconds=NATIVE_STEP)
    stable, _, grid, people, _ = ev.episode(int(ident[-2:]))
    import backend_cost as bc
    starts = bc.refined_starts(int(ident[-2:]))
    from finesub.speech.diarization.nemotron3_backend import STEP_SECONDS
    # The frame each chunk is put back at -- the same rule `nemotron3_backend.windows` cut it with.
    chunk_start = np.array([da.closest_frame(c * STEP_SECONDS + 0.5 * da.FRAME_DURATION) for c in range(local.shape[0])])
    per_window = local.shape[1]
    records = []
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        d = speaker.to_dict()
        ids = d.get("tracks") or []
        if not ids:
            continue
        starters = {w for a, w in starts if float(row["start"]) - 0.3 <= a < float(row["end"])}
        cover, _ = ev.span_cover(row, grid)
        truth = next(iter(starters)) if len(starters) == 1 else (people[int(cover.argmax())] if cover.any() else None)
        if truth is None:
            continue
        track = int(ids[0])
        centroid = int(centroid_of_track[track]) if track < len(centroid_of_track) else -1
        lo, hi = int(float(row["start"]) / NATIVE_STEP), max(int(float(row["start"]) / NATIVE_STEP) + 1,
                                                            int(np.ceil(float(row["end"]) / NATIVE_STEP)))
        total = weight = 0.0
        if centroid >= 0:
            for c in np.flatnonzero((chunk_start < hi) & (chunk_start + per_window > lo)):
                a, b = max(lo, chunk_start[c]) - chunk_start[c], min(hi, chunk_start[c] + per_window) - chunk_start[c]
                for j in np.flatnonzero(hard[c] == track):
                    active = float(local[c, a:b, j].sum())
                    if active <= 0:
                        continue
                    scores = soft[c, j]
                    others = np.delete(scores, centroid)
                    total += active * (scores[centroid] - (others.max() if others.size else 0.0))
                    weight += active
        shares = sorted(d.get("coverage", {}).values(), reverse=True)
        time_margin = (shares[0] - (shares[1] if len(shares) > 1 else 0.0)) / shares[0] if shares and shares[0] else 0.0
        records.append(dict(track=track, truth=truth, seconds=float(row["end"]) - float(row["start"]),
                            soft=total / weight if weight else np.nan, time=time_margin))
    totals = defaultdict(Counter)
    for r in records:
        totals[r["track"]][r["truth"]] += r["seconds"]
    owner = {k: c.most_common(1)[0][0] for k, c in totals.items()}
    purity = {k: c.most_common(1)[0][1] / sum(c.values()) for k, c in totals.items()}
    for r in records:
        r["wrong"] = owner[r["track"]] != r["truth"]
        r["merged"] = purity[r["track"]] < MERGED
    return records


def flag(rows, key, budget):
    """The `budget` share of rows with the lowest `key` (NaN soft counts as lowest: no voiceprint evidence)."""
    values = np.array([(-np.inf if np.isnan(r[key]) else r[key]) for r in rows])
    order = np.argsort(values, kind="stable")
    return set(order[: int(round(budget * len(rows)))].tolist())


def main():
    rows = []
    for ident in osw.CONFIRMATION:
        if (osw.OUT / f"{ident}.npz").exists():
            rows += rows_for(ident)
    rows = [r for r in rows if not r["merged"]]
    wrong = sum(r["wrong"] for r in rows)
    missing = sum(np.isnan(r["soft"]) for r in rows)
    print(f"姐姐咚咚 15: {len(rows)} labelled rows with a truth (merged tracks excluded), wrong {wrong} ({wrong / len(rows):.1%}); "
          f"no voiceprint evidence on {missing}")
    corr = np.corrcoef([0 if np.isnan(r['soft']) else r['soft'] for r in rows], [r['time'] for r in rows])[0, 1]
    print(f"correlation soft vs time margin: {corr:+.2f}")
    print(f"{'budget':>7}  {'soft margin':>26}  {'time margin':>26}  {'half each (union)':>26}")
    for budget in BUDGETS:
        cells = []
        soft_set, time_set = flag(rows, "soft", budget), flag(rows, "time", budget)
        half = flag(rows, "soft", budget / 2) | flag(rows, "time", budget / 2)
        for chosen in (soft_set, time_set, half):
            caught = sum(rows[i]["wrong"] for i in chosen)
            cells.append(f"err {caught / max(1, len(chosen)):4.0%} catch {caught / wrong:5.1%} ({len(chosen) / len(rows):.0%})")
        print(f"{budget:>7.0%}  " + "  ".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
