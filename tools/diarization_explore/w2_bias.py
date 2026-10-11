"""A3: how much of the edit-cost gap is the ruler favouring whichever arm has more tracks?

W2/W2b priced two backends against each other and reported DiariZen about 300 points per file
cheaper on hina. The mapping that pricing rests on -- `owner_of_tracks` -- is a many-to-one oracle
fitted on the very rows it is then scored on, so an arm with more tracks gets more chances at a
flattering assignment. findings 7 controlled for exactly this when it measured recursive
splitting, and built a time-block null for it; W2/W2b did not (fable 2026-09-19).

Two controls, both pure functions of artifacts already on disk:

  crossfit    the mapping is fitted on alternating minutes and scored on the others, the same
              trick `fixed` already carries. What it removes is in-sample fitting.
  time-block  every track is cut in two by time, doubling the track count WITHOUT adding any
              information. Whatever that gains is the arithmetic free lunch a real comparison
              has to beat.

⚠ Neither is expected to flip the direction -- `fixed`'s own crossfit gap is under 1 pp -- but the
MAGNITUDE of -300 is what Q2 is being decided on, so its in-sample component has to be stated.

    python tools/diarization_explore/w2_bias.py --corpus hina
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
from accuracy_report import bootstrap_mean_ci  # noqa: E402
from eval_contract import DEFAULT_POLICY, announce  # noqa: E402

#: ⚠ The same number `owner_of_tracks_folded` holds out with. Two copies of a fold length is two
#: chances for the rows scored and the frames fitted to disagree about where a fold ends.
FOLD_SECONDS = DEFAULT_POLICY.fold_seconds


def cost_with(probs, stable, grid, people, starts, owner, *, fold: int | None = None) -> float:
    """Price the timeline under `owner`; with `fold`, price ONLY the rows in that fold.

    ⚠ The `fold` argument is the whole point and the first version did not have it. Fitting the
    mapping on the other fold and then pricing *every* row leaves half of each pass in-sample, so
    the average of the two passes is half an in-sample number wearing the word "crossfit"
    (fable 2026-09-19). The rows priced here must be exactly the rows the mapping never saw.

    ⚠ **A row belongs to the fold its START is in, and that leaves a residual** (2026-09-20 audit).
    Rows are atomic for pricing, so they cannot be split the way `owner_of_tracks_folded` now
    splits frames; a row crossing a minute boundary is therefore priced under a mapping that did
    see the truth of its tail. The overlap is at most one row long -- on this material about 2% of
    rows, each contributing a few seconds -- against the frame-level leak this replaced, which put
    whole segments of held-out truth into the fit. Assigning by start is also what keeps the two
    folds a partition of the rows, which is the invariant the test above asserts.
    """
    import refined_multi as rm

    segments = cm.labelled_segments(probs, stable)
    if fold is not None:
        segments = [row for row in segments
                    if int(row["start"] // FOLD_SECONDS) % 2 == fold]
    # ⚠ Takes the POLICY, not a literal: this whole file exists to say how much of W2's gap is
    # ruler artefact, so it has to be the same ruler W2 quotes. Writing `True` here is how it
    # drifted once already -- the policy flipped and this did not follow.
    total, _ = rm.strong_cost(segments, grid, people, owner, starts,
                              DEFAULT_POLICY.add, merge=DEFAULT_POLICY.merge,
                              price_unlabelled=DEFAULT_POLICY.price_unlabelled)
    return float(total)


def rows_for(corpus: str, keys) -> list[dict]:
    rows = []
    for key in keys:
        if corpus == "hina":
            loaded = bc.hina_episode(key)
            if loaded is None:
                continue
            (stable, timeline, grid, people, _), cached, starts = loaded
        else:
            data = bc.ev.episode(key)
            cached = bc.V2_OUT / f"onetonton-{key:02d}" / "diarizen.npz"
            if data is None or not Path(cached).exists():
                continue
            stable, timeline, grid, people, _ = data
            starts = bc.refined_starts(key)
        arms = {
            "sortformer": bc.timeline_probs(timeline["tracks"], len(grid)),
            "diarizen": np.load(cached)["probs"],
        }
        row = {"episode": key, "people": len(people)}
        for backend, probs in arms.items():
            probs = probs[: len(grid)]
            entry = {
                "in_sample": cost_with(probs, stable, grid, people, starts,
                                       cm.owner_of_tracks(probs, grid, people)),
            }
            # Each fold's rows priced by a mapping fitted on the OTHER fold, then added: every row
            # is priced exactly once, by a mapping that never saw it. The two folds partition the
            # timeline by alternating minutes, so the sum is comparable to the in-sample total.
            entry["crossfit"] = sum(
                cost_with(probs, stable, grid, people, starts,
                          cm.owner_of_tracks_folded(probs, grid, people, fold=fold), fold=fold)
                for fold in (0, 1)
            )
            # The null: more tracks, not one bit more information. ⚠ Swept rather than fixed at
            # two, because a null that leaves Sortformer at 8 tracks is not a fair control for
            # DiariZen's 10.8 -- the free lunch grows with the track count, so one point
            # understates it (fable 2026-09-19).
            for pieces in (2, 3, 4):
                split = cm.time_block_split(probs, pieces=pieces)
                entry[f"time_block_null_{pieces}"] = cost_with(
                    split, stable, grid, people, starts, cm.owner_of_tracks(split, grid, people))
                entry[f"null_tracks_{pieces}"] = int(split.shape[1])
            entry["tracks"] = int(probs.shape[1])
            row[backend] = entry
        rows.append(row)
    return rows


def report(corpus: str, rows: list[dict]) -> None:
    print(f"\n{corpus}，{len(rows)} 份 —— 编辑代价的两个偏差对照\n")
    print(f"{'后端':<12}{'样本内':>10}{'crossfit':>11}{'拟合红利':>11}"
          + "".join(f"{f'null×{p}':>11}" for p in (2, 3, 4)))
    for backend in ("sortformer", "diarizen"):
        inside = np.array([row[backend]["in_sample"] for row in rows])
        cross = np.array([row[backend]["crossfit"] for row in rows])
        cells = "".join(
            f"{np.mean([row[backend][f'time_block_null_{p}'] - row[backend]['in_sample'] for row in rows]):>+11.1f}"
            for p in (2, 3, 4))
        print(f"{backend:<12}{inside.mean():>10.1f}{cross.mean():>11.1f}"
              f"{(cross - inside).mean():>+11.1f}{cells}")
    print("  （null×N 列是「白捡」：把每条轨按文件时间切成 N 段之后省下多少，越负=白捡越多）")

    print(f"\n后端差（DiariZen − Sortformer）\n")
    for label, key in (("样本内（W2b 报的那个）", "in_sample"), ("crossfit（真留出）", "crossfit")):
        diff = np.array([row["diarizen"][key] - row["sortformer"][key] for row in rows])
        interval = bootstrap_mean_ci(list(diff))
        band = f"[{interval[0]:+.0f}, {interval[1]:+.0f}]" if interval else "(n < 5)"
        print(f"  {label:<22}{diff.mean():>+9.1f}  中位 {np.median(diff):>+8.1f}  {band}"
              f"  DiariZen 更好 {int((diff < 0).sum())}/{len(rows)}")
    inside = np.array([row["diarizen"]["in_sample"] - row["sortformer"]["in_sample"] for row in rows])
    cross = np.array([row["diarizen"]["crossfit"] - row["sortformer"]["crossfit"] for row in rows])
    if abs(inside.mean()) > 1e-9:
        print(f"  ⇒ 拟合贡献了后端差的 {100 * (1 - cross.mean() / inside.mean()):+.1f}%")

    print("\n  去掉轨数的算术红利之后（DiariZen 对「被切过的 Sortformer」）：")
    for pieces in (2, 3, 4):
        diff = np.array([row["diarizen"]["in_sample"] - row["sortformer"][f"time_block_null_{pieces}"]
                         for row in rows])
        tracks = np.mean([row["sortformer"][f"null_tracks_{pieces}"] for row in rows])
        print(f"    Sortformer 切 {pieces} 段（{tracks:.0f} 轨）  {diff.mean():>+8.1f}"
              f"  中位 {np.median(diff):>+7.1f}  DiariZen 更好 {int((diff < 0).sum())}/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=("hina", "onetonton", "both"), default="both")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    payload = {}
    for corpus in (("hina", "onetonton") if args.corpus == "both" else (args.corpus,)):
        if corpus == "onetonton":
            from accuracy_cross import configure

            configure()
            keys = list(range(1, 16))
        else:
            frozen = json.loads((HERE / "results" / "diarizen-v3-20260917.json").read_text(encoding="utf-8"))
            keys = list(frozen["groups"]["heldout_all"])
        announce(corpus)
        rows = rows_for(corpus, keys)
        if not rows:
            raise SystemExit(f"{corpus}: scored nothing -- do not report a blank result")
        report(corpus, rows)
        payload[corpus] = rows
    if args.write:
        # ⚠ `results/`, not `tmp/` -- the closeout plan quotes these (2026-09-20 audit §5.3).
        out = HERE / "results" / "w2-bias-20260920.json"
        out.write_text(json.dumps({**payload, "script": lab.digest(__file__)}, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
