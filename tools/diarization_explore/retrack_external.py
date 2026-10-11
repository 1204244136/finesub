"""MOSS Pro, MOSS (open), Gemini 3.5 Transcribe next to production and agy on the same rows (owner 2026-10-04).

Two row sets per episode, both from production's own assignment (Nemotron-3 + vbx, `assign_all`):
  missed    rows production left unlabelled where exactly one person's refined line starts (the P1 rows)
  lowconf   labelled rows whose margin (leader - runner-up) / leader <= 0.3 (the P2 candidates)
Each external arm's segments become a track timeline, the SAME `assign_all` labels the same rows, and
every arm's tracks get their majority person row by row (time-weighted, on rows with a truth) -- so an
arm is right on a row when its track's person is the row's person. No model call; reads caches only.

    python tools/diarization_explore/retrack_external.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402

sys.path.insert(0, str(lr.n3.WT))
from tools.ts_diar_explore import arms as A  # noqa: E402

#: ⚠ hina first: loading any onetonton item rebinds refined_multi.sources for the rest of the process.
ITEMS = ("hina-25", "hina-27", "hina-64", "hina-67", "hina-73",
         "onetonton-01", "onetonton-05", "onetonton-02", "onetonton-03", "onetonton-04", "onetonton-06",
         "onetonton-07", "onetonton-08")
EXTERNAL = ("mosspro", "g35t", "moss-single")
MARGIN = 0.3


def load(item):
    import nemotron3_cost as nc
    import refined_multi as rm
    import backend_cost as bc
    import relabel_eval as ev
    if item.startswith("hina-"):
        nc.corpus_keys("hina")
        stable, _, grid, people, _, lines = rm.episode(item, rm.sources()[item])
        starts = sorted((a, w) for a, _, w in lines)
    else:
        nc.corpus_keys("onetonton")
        stable, _, grid, people, _ = ev.episode(int(item[-2:]))
        starts = bc.refined_starts(int(item[-2:]))
    tracks = json.loads((lr.ABSTAIN / item / "nemotron3-diar.json").read_text(encoding="utf-8"))["tracks"]
    return stable, grid, people, starts, tracks


def truth_of(row, grid, people, starts):
    import relabel_eval as ev
    starters = {w for a, w in starts if float(row["start"]) - 0.3 <= a < float(row["end"])}
    if len(starters) == 1:
        return next(iter(starters)), True
    cover, _ = ev.span_cover(row, grid)
    return (people[int(cover.argmax())], False) if cover.any() else (None, False)


def labels(stable, tracks):
    """Per row: (leading track or None, margin)."""
    out = []
    for speaker in lr._assign(stable, tracks):
        d = speaker.to_dict()
        ids = d.get("tracks") or []
        shares = sorted(d.get("coverage", {}).values(), reverse=True)
        margin = (shares[0] - (shares[1] if len(shares) > 1 else 0.0)) / shares[0] if shares and shares[0] else 0.0
        out.append((ids[0] if ids else None, margin))
    return out


def owners(stable, assigned, truths):
    weight = defaultdict(Counter)
    for row, (track, _), (truth, _) in zip(stable, assigned, truths):
        if track is not None and truth is not None:
            weight[track][truth] += float(row["end"]) - float(row["start"])
    return {k: c.most_common(1)[0][0] for k, c in weight.items()}


def external_tracks(item, name):
    path = A.path(item, name) if hasattr(A, "path") else None
    if path is not None and not Path(path).exists():
        return None
    try:
        arm = A.load(item, name)
    except Exception:  # noqa: BLE001 - the arm was not run on this item
        return None
    segments = [s for s in arm.get("segments", []) if s.get("speaker") and float(s["end"]) > float(s["start"])]
    if not segments:
        return None
    ids = {label: i for i, label in enumerate(sorted({s["speaker"] for s in segments}))}
    return [{"start": float(s["start"]), "end": float(s["end"]), "track": ids[s["speaker"]]} for s in segments]


def agy_answers(item):
    out = {}
    for name, folder in (("agy·文", "fill-agy-r1"), ("agy·文", "voices-text-agy"),
                         ("agy·音", "fill-av-agy-r1"), ("agy·音", "voices-audio-agy")):
        for path in (lr.OUT / folder).glob(f"{item}-*.json"):
            for key, answer in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items():
                out.setdefault(name, {})[int(key[1:])] = str(answer.get("track", "abstain"))
    return out


def main():
    table = {"missed": defaultdict(lambda: defaultdict(list)), "lowconf": defaultdict(lambda: defaultdict(list)),
             "all": defaultdict(lambda: defaultdict(list))}
    for item in ITEMS:
        try:
            stable, grid, people, starts, tracks = load(item)
        except Exception as error:  # noqa: BLE001
            print("skip", item, error)
            continue
        truths = [truth_of(row, grid, people, starts) for row in stable]
        production = labels(stable, tracks)
        prod_owner = owners(stable, production, truths)
        arms = {"生产": (production, prod_owner)}
        for name in EXTERNAL:
            ext = external_tracks(item, name)
            if ext is None:
                continue
            assigned = labels(stable, ext)
            arms[name] = (assigned, owners(stable, assigned, truths))
        agy = agy_answers(item)
        centres = np.array([(float(r["start"]) + float(r["end"])) / 2 for r in stable])
        labelled = [i for i, (t, _) in enumerate(production) if t is not None]
        for index, row in enumerate(stable):
            truth, onset = truths[index]
            if truth is None:
                continue
            track, margin = production[index]
            sets = ["all"] if track is not None else []
            if track is None and onset:
                sets.append("missed")
            if track is not None and margin <= MARGIN:
                sets.append("lowconf")
            for s in sets:
                cell = table[s][item]
                for name, (assigned, owner) in arms.items():
                    t = assigned[index][0]
                    cell[name].append(None if t is None else owner.get(t) == truth)
                if s == "missed":
                    nearest = min(labelled, key=lambda j: abs(centres[j] - centres[index]))
                    cell["基线"].append(prod_owner.get(production[nearest][0]) == truth)
                    for name, answers in agy.items():
                        if index in answers:
                            answer = answers[index]
                            cell[name].append(prod_owner.get(int(answer[1:])) == truth
                                              if answer.startswith("s") and answer[1:].isdigit() else None)

    def show(values):
        filled = [v for v in values if v is not None]
        return f"{sum(filled)}/{len(filled)}/{len(values)}" if values else "—"

    for which, title in (("missed", "漏检行（生产没标签，「开口的人」口径）"), ("lowconf", f"已挂轨、边际 ≤ {MARGIN} 的行"),
                         ("all", "全部已挂轨的行")):
        print(f"\n== {title}  格式：对/给了标签/行数")
        names = ["生产", "基线", "agy·文", "agy·音", *EXTERNAL]
        for item, cell in table[which].items():
            print(f"  {item:13} " + "  ".join(f"{n} {show(cell.get(n, []))}" for n in names if n in cell))
        # Paired totals: for each arm, only the items that arm covers, with the other arms on the SAME items.
        for name in names:
            covered = [item for item, cell in table[which].items() if name in cell]
            if not covered or name in ("生产", "基线"):
                continue
            row = []
            for other in ("生产", "基线", "agy·文", "agy·音", name):
                values = [v for item in covered for v in table[which][item].get(other, [])]
                if values and other not in [r[0] for r in row]:
                    filled = [v for v in values if v is not None]
                    row.append((other, f"{sum(filled)}/{len(filled)}/{len(values)} ({sum(filled) / max(1, len(filled)):.0%})"))
            print(f"  合计（{name} 覆盖的 {len(covered)} 集）: " + "  ".join(f"{a} {b}" for a, b in row))


if __name__ == "__main__":
    main()
