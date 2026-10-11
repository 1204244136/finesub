"""Dump every P1 row the model got wrong or left blank, with its context, for reading (owner 2026-10-04).

One JSON line per row: where it came from (run, file, row), what it says, the context the model saw
(with tracks), what the model answered and why, the truth person ("onset": exactly one person's
refined line starts inside the row; else the person covering most of it), which track that person
owns, what the neighbour baseline would have said, and whether the row is filler. No model call.

    python tools/diarization_explore/retrack_dump.py > rows.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import retrack_voices as rv  # noqa: E402
from retrack_split import is_filler  # noqa: E402


def rows_of(run, ident, stable, speakers, tracks, grid, people, starts, items, folder):
    import backend_cost as bc
    import cost_metrics as cm
    import relabel_eval as ev
    owner = lr.owners(tracks, grid, people)
    tracks_of_person = {}
    for track, person in owner.items():
        tracks_of_person.setdefault(person, []).append(f"s{track}")
    answers = {}
    for path in folder.glob(f"{ident}-*.json"):
        answers.update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
    cells = [lr.cell(s) for s in speakers]
    centres = np.array([(float(r["start"]) + float(r["end"])) / 2 for r in stable])
    labelled = [i for i, s in enumerate(speakers) if s.to_dict().get("tracks")]
    for index in items:
        row = stable[index]
        starters = {w for a, w in starts if float(row["start"]) - 0.3 <= a < float(row["end"])}
        cover, _ = ev.span_cover(row, grid)
        truth = next(iter(starters)) if len(starters) == 1 else (people[int(cover.argmax())] if cover.any() else None)
        if truth is None:
            continue
        answer = answers.get(f"q{index}", {})
        track = str(answer.get("track", "abstain"))
        filled = track.startswith("s") and track[1:].isdigit()
        right = filled and owner.get(int(track[1:])) == truth
        if right:
            continue
        nearest = min(labelled, key=lambda j: abs(centres[j] - centres[index]))
        lo, hi = max(0, index - lr.CONTEXT), min(len(stable), index + lr.CONTEXT + 1)
        yield {
            "run": run, "file": ident, "row": index, "outcome": "wrong" if filled else ("other" if track == "other" else "abstain"),
            "text": str(row.get("text", "")).strip(), "seconds": round(float(row["end"]) - float(row["start"]), 2),
            "filler": is_filler(str(row.get("text", ""))), "truth_kind": "onset" if len(starters) == 1 else "covered",
            "truth_person": truth, "truth_tracks": tracks_of_person.get(truth, []),
            "model_track": track, "model_track_owner": owner.get(int(track[1:])) if filled else None,
            "model_conf": answer.get("conf"), "model_why": answer.get("why", ""),
            "neighbour_track": f"s{speakers[nearest].to_dict()['tracks'][0]}",
            "neighbour_right": owner.get(speakers[nearest].to_dict()["tracks"][0]) == truth,
            "track_owner": {f"s{k}": v for k, v in owner.items()},
            "context": [("▶ " if j == index else "") + lr.line(stable[j], cells[j]) for j in range(lo, hi)],
        }


def main():
    out = sys.stdout
    # ⚠ Order matters: anything that loads onetonton rebinds refined_multi.sources, so the 放送局 files
    # (rv.material) load first and fill_material runs once.
    voices = rv.material()
    material = lr.fill_material()
    for run, folder in (("text-agy", lr.OUT / "fill-agy-r1"),):
        for ident, corpus, stable, speakers, tracks, grid, people, starts, items in material:
            for record in rows_of(f"{run}:{corpus}", ident, stable, speakers, tracks, grid, people, starts, items, folder):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    for task in ("text", "audio"):
        for ident, stable, speakers, tracks, grid, people, starts, rows in voices:
            for record in rows_of(f"voices-{task}-agy", ident, stable, speakers, tracks, grid, people, starts, rows,
                                  lr.OUT / f"voices-{task}-agy"):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
