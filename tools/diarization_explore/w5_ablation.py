"""Are the extra second labels worth their keep? The ablation W5 should have run.

W5 asked this with overlap minutes, which the reference axis cannot support. The revision then
asked it with the edit-cost breakdown -- and got it wrong a second time, in the other direction:
comparing DiariZen's total against Sortformer's answers "which backend is cheaper", and 80% of
that gap is the `move` term, which is the LEADING track being the right person. Second labels have
nothing to do with it (fable 2026-09-19, and the same document says so two paragraphs earlier).

Holding the backend fixed and removing only the extra labels is the comparison that answers it:

    full   production assignment, every track the rule keeps
    top-1  the same timeline, same segments, only the leading label

Everything else -- segments, owner mapping, pricing -- is identical, so the difference is the
second labels and nothing else. `refined_multi.strong_cost` is the same function W2/W2b used.

    python tools/diarization_explore/w5_ablation.py
    python tools/diarization_explore/w5_ablation.py --corpus onetonton
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


def episode_rows(corpus: str, keys) -> list[dict]:
    import refined_multi as rm

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
            owner = cm.owner_of_tracks(probs, grid, people)
            entry = {}
            for arm in ("full", "top1"):
                segments = cm.labelled_segments(probs, stable, top_one=(arm == "top1"))
                total, merges = rm.strong_cost(segments, grid, people, owner, starts, 2, merge=3)
                entry[arm] = {"total": float(total), "labels": float(total - 3 * merges)}
                entry[f"{arm}_terms"] = cm.cost_terms(segments, grid, people, owner, starts)
                # The verdict depends on the owner pricing a missed speaker at 2 against a
                # spurious label at 1. Sweeping that price rather than testing one alternative:
                # a single point at 1.0 shows the sign flips somewhere below 2 but not where, and
                # "the conclusion needs exactly 2:1" is a much stronger claim than the data makes
                # (fable 2026-09-19). ⚠ Re-pricing re-runs the minimisation, so `move` re-pairs
                # and a linear extrapolation from two points is only approximate.
                for price in PRICES:
                    priced, _ = rm.strong_cost(segments, grid, people, owner, starts, price, merge=3)
                    entry[f"{arm}_p{price}"] = float(priced)
            # How much of the reference actually marks two people talking. Where it marks none,
            # a second label can neither earn nor lose, so the file carries no information about
            # this question at all -- and eleven of the nineteen hina references are like that.
            entry["reference_overlap_pct"] = 100.0 * float((grid.sum(axis=1) > 1).sum()) / max(
                float(grid.any(axis=1).sum()), 1.0)
            row[backend] = entry
        rows.append(row)
    return rows


OVERLAP_FLOOR = 5.0
#: Prices for a missed speaker, against a spurious label fixed at 1. The owner's is 2.
PRICES = (1.0, 1.25, 1.5, 1.75, 2.0)


def _group(rows, backend, annotated):
    """Split by whether the reference actually marks two people talking.

    ⚠ **Post-hoc**: the 5% line was drawn after seeing the readings. What makes it defensible is
    that the data is a cliff and not a slope -- 20.84% median against 0.51%, forty times -- so the
    exact threshold cannot matter much; it is not a knob that was tuned until the answer improved.
    Say so beside any table built from it.

    Where the reference annotates no overlap a second label can neither earn nor lose, so those
    files carry NO information about this question.
    Eleven of the nineteen hina references are like that -- median 0.51% of speech against 20.84%
    in the rest, a gap of forty times and not a gradient -- and pooling them drags every estimate
    toward zero and makes the corpus look bimodal when it is really two different measurements.
    """
    return [row for row in rows
            if (row[backend]["reference_overlap_pct"] >= OVERLAP_FLOOR) == annotated]


def report(corpus: str, rows: list[dict]) -> None:
    print(f"\n{corpus}，{len(rows)} 份 —— 同一后端内，去掉第二标签会怎样"
          f"（正 = 去掉之后更贵 ⇒ 第二标签有价值）\n")
    print(f"{'后端':<12}{'分组':<14}{'n':>3}{'均值':>9}{'中位':>8}"
          f"{'95% 区间':>18}{'有用':>8}{'盈亏平衡价':>12}")
    for backend in ("sortformer", "diarizen"):
        for annotated, label in ((True, "参考有重叠"), (False, "参考几乎没有")):
            group = _group(rows, backend, annotated)
            if not group:
                continue
            diff = np.array([row[backend]["top1"]["total"] - row[backend]["full"]["total"]
                             for row in group])
            curve = [np.mean([row[backend][f"top1_p{price}"] - row[backend][f"full_p{price}"]
                              for row in group]) for price in PRICES]
            # Where the sign changes, read off the swept curve rather than extrapolated from two
            # points: re-pricing re-runs the minimisation, so `move` re-pairs and the curve is not
            # exactly linear.
            crossing = (f"{np.interp(0.0, curve, PRICES):.2f}"
                        if min(curve) < 0 < max(curve) else ("永不" if max(curve) <= 0 else "<1.0"))
            interval = bootstrap_mean_ci(list(diff))
            band = f"[{interval[0]:+.0f}, {interval[1]:+.0f}]" if interval else "(n < 5)"
            print(f"{backend:<12}{label:<14}{len(group):>3}{diff.mean():>+9.1f}"
                  f"{np.median(diff):>+8.1f}{band:>18}"
                  f"{f'{int((diff > 0).sum())}/{len(group)}':>8}{crossing:>12}")
    print("\n⚠ 末列 = 让结论翻号所需的「漏标」价（删除固定 1 分，owner 给的是 2）。"
          "\n   ⇒ 结论**不需要恰好 2:1**：漏标只要值到约 1.2–1.55 倍就成立。"
          "\n   ⚠ 但方向要记住：精修轴漏写重叠，会把真的第二标签记成 delete ⇒"
          "\n   低价一侧的负号是对第二标签**不利的上界**，与 W5 已记的偏差同向。")
    print("\n按项拆（top-1 减 全标签，正 = 去掉第二标签之后这一项更贵）\n")
    print(f"{'后端':<12}{'delete':>9}{'move':>9}{'add':>9}{'duplicate':>11}")
    for backend in ("sortformer", "diarizen"):
        cells = []
        for term in ("delete", "move", "add", "duplicate"):
            values = np.array([row[backend]["top1_terms"][term] - row[backend]["full_terms"][term]
                               for row in rows])
            cells.append(f"{values.mean():>+9.1f}" if term != "duplicate" else f"{values.mean():>+11.1f}")
        print(f"{backend:<12}" + "".join(cells))


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
        rows = episode_rows(corpus, keys)
        if not rows:
            raise SystemExit(f"{corpus}: scored nothing -- do not report a blank result")
        report(corpus, rows)
        payload[corpus] = rows
    if args.write:
        out = bc.V2_OUT / "w5-ablation.json"
        out.write_text(json.dumps({**payload, "script": lab.digest(__file__)}, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
