"""P1 accuracy split by what the row says: filler / short vs a meaningful sentence (owner 2026-10-04).

Reads the cached answers of `llm_retrack.py` (text arms) -- no model call. Two cuts, both reported per
corpus and per truth (covered: the person covering most of the row; onset: exactly one person's
refined line starts inside it):

  length   >= 8 letters (punctuation and spaces removed) vs shorter
  filler   the row is nothing but backchannel / interjection words vs anything else

    python tools/diarization_explore/retrack_split.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402

#: Backchannel and interjection words, longest first so 「なるほどね」 is eaten before 「なる」 could be.
FILLER = sorted("""はい うん ええ えー えっ え あー あ ああ おお おー お へえ へー ほう ふーん ふん そう そうそう そうそうそう
そうだね そうですね そうなんだ そうなんです そっか そうか 本当 ほんと ほんとに 本当に まじ マジ なるほど なるほどね
確かに たしかに わかる わかります わかった いや いやいや いやー ね ねえ ねー よ よね だよね だね あら あれ ん んー
うわ うわー わー わあ やば やばい すごい すげー すご 笑 ありがとう ありがとうございます お願いします よろしく""".split(),
                key=len, reverse=True)


def is_filler(text):
    rest = re.sub(r"[\W_ー〜~!?！？、。…・]", "", text)
    for word in FILLER:
        rest = rest.replace(word, "")
    return rest == ""


def main():
    import backend_cost as bc
    import cost_metrics as cm
    import relabel_eval as ev

    # ⚠ fill_material() once: a second call finds hina gone (onetonton's corpus_keys rebinds sources).
    material = lr.fill_material()
    arms = {arm: {} for arm in lr.ARMS}
    for arm in arms:
        for path in (lr.OUT / f"fill-{arm}-r1").glob("*.json"):
            arms[arm].update({f"{path.stem.rsplit('-', 1)[0]}:{k}": v
                              for k, v in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items()})
    # cells[(corpus, truth, cut, bucket)][name] = [right?...]; None = not filled
    cells = defaultdict(lambda: defaultdict(list))
    for ident, corpus, stable, speakers, tracks, grid, people, starts, items in material:
        owner = lr.owners(tracks, grid, people)
        centres = np.array([(float(r["start"]) + float(r["end"])) / 2 for r in stable])
        labelled = [i for i, s in enumerate(speakers) if s.to_dict().get("tracks")]
        for index in items:
            row = stable[index]
            text = str(row.get("text", "")).strip()
            cover, _ = ev.span_cover(row, grid)
            starters = {w for a, w in starts if float(row["start"]) - 0.3 <= a < float(row["end"])}
            truths = {"covered": people[int(cover.argmax())] if cover.any() else None,
                      "onset": next(iter(starters)) if len(starters) == 1 else None}
            nearest = min(labelled, key=lambda j: abs(centres[j] - centres[index]))
            buckets = {"length": ">=8 字" if len(re.sub(r"[\W_]", "", text)) >= 8 else "<8 字",
                       "filler": "填充词" if is_filler(text) else "有意义的句子"}
            for truth_name, truth in truths.items():
                if truth is None:
                    continue
                for cut, bucket in buckets.items():
                    cell = cells[(corpus, truth_name, cut, bucket)]
                    cell["基线"].append(owner.get(speakers[nearest].to_dict()["tracks"][0]) == truth)
                    for arm, answers in arms.items():
                        answer = answers.get(f"{ident}:q{index}", {})
                        track = str(answer.get("track", "abstain"))
                        cell[arm].append(owner.get(int(track[1:])) == truth
                                         if track.startswith("s") and track[1:].isdigit() else None)
                        if answer.get("conf") == "high":
                            cell[f"{arm} high"].append(owner.get(int(track[1:])) == truth
                                                       if track.startswith("s") and track[1:].isdigit() else None)

    def show(values):
        filled = [v for v in values if v is not None]
        return f"{sum(filled)}/{len(filled)} ({sum(filled) / len(filled):.0%})" if filled else "—"

    for key in sorted(cells):
        corpus, truth, cut, bucket = key
        cell = cells[key]
        print(f"{corpus:9} {truth:7} {bucket:7} n={len(cell['基线']):4}  " +
              "  ".join(f"{name} {show(cell[name])}" for name in ("基线", "agy", "agy high", "lite")))


if __name__ == "__main__":
    main()
