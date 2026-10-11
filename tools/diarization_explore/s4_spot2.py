"""Plan 5o.9 spot checks C / D: is scorer v2 right? Prints a sheet to read by hand.

Fixed seed: 30 real windows (W = span +-1 s), each with one arm drawn from A0-W / T4 / T2-Q, and
20 decoy windows. Per reference sentence the sheet shows the scorer's verdict per character:

  status   = right   ~ homophone   x misheard   _ missed
  owner    S only on the speaker's own line   M multi-label incl. own   W someone else's   U unlabelled   . not heard

and each output line with its labels. Decoys: the characters the scorer found nothing near are
shown in 【】. The reader writes down only the characters they disagree with.

  --sheet C|D     (D: set DIAR_S4_CORPUS=callhome-ja)
"""

from __future__ import annotations

import argparse
import json
import random

import s4_breakdown as b
import s4_headroom as ref
import s4_projection as proj
import s4_score2 as sc
import s4_timecut as tc

SEED = 20260912
MARK = {"right": "=", "homophone": "~", "misheard": "x", "miss": "_"}
OWN = {"strict": "S", "multi": "M", "wrong": "W", "unlabelled": "U", "no track": "N", None: "."}


def arms(item):
    tracks = item.session.tracks
    active = proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)
    results = json.loads((proj.ALI / item.name / "timecut_qwen.json").read_text(encoding="utf-8"))
    layout = json.loads((proj.ALI / item.name / "timecut_layout.json").read_text(encoding="utf-8"))
    strip = lambda d: {k: [(t, x) for t, x, _ in v] for k, v in d.items()}  # noqa: E731
    return {"A0-W": strip(tc.production_lines(item.session)), "T4": strip(tc.word_lines(item.session, active)),
            "T2-Q": strip(tc.variant_lines(item.session, results, layout, "T2", fallback=True, deleak=False))}


def sheet():
    rng = random.Random(SEED)
    loaded = [sc.Loaded(name) for name in ref.SESSIONS]
    real = [(item, s) for item in loaded for s in item.session.spans if s["kind"] == "real"]
    decoys = [(item, s) for item in loaded for s in item.session.spans if s["kind"] == "decoy"]
    cache = {}
    for index, (item, span) in enumerate(rng.sample(real, 30)):
        arm = rng.choice(["A0-W", "T4", "T2-Q"])
        if item.name not in cache:
            cache[item.name] = arms(item)
        key = (span["start"], span["end"])
        lines = [(tuple(t), x) for t, x in cache[item.name][arm].get(key, []) if x]
        near = item.session.near(key[0], key[1], b.NEAR)
        content, copies, kinds, _ = sc.window(lines, near, item.owner_of) if lines else ({}, {}, [], [])
        print(f"\n## C{index:02d} {item.name} {key[0]:.2f}-{key[1]:.2f} arm {arm}  (speaker -> track {item.session.speaker_track})")
        for tracks, text in lines:
            print(f"   out [{'+'.join(f's{k}' for k in tracks) or '-'}] {text}")
        for row in near:
            if not (row["start"] < key[1] + tc.W_PAD and row["end"] > key[0] - tc.W_PAD):
                continue
            status = [max((o[0] for o in content.get((row["id"], p), ())), key=b.RANK.get, default="miss") for p in range(len(row["norm"]))]
            owner = [max((o[1] for o in copies.get((row["id"], p), ())), key=sc.ATTR_RANK.get, default=None) for p in range(len(row["norm"]))]
            print(f"   ref {row['spk']} {row['start']:.2f}-{row['end']:.2f}  {row['norm']}")
            print(f"       status {''.join(MARK[s] for s in status)}")
            print(f"       owner  {''.join(OWN[o] if s != 'miss' else '.' for s, o in zip(status, owner))}")
    for index, (item, span) in enumerate(rng.sample(decoys, 20)):
        arm = rng.choice(["A0-W", "T4", "T2-Q"])
        if item.name not in cache:
            cache[item.name] = arms(item)
        key = (span["start"], span["end"])
        lines = [(tuple(t), x) for t, x in cache[item.name][arm].get(key, []) if x]
        near = item.session.near(key[0], key[1], b.NEAR)
        _, _, kinds, _ = sc.window(lines, near, item.owner_of) if lines else ({}, {}, [], [])
        joined = "".join(x for _, x in lines)
        marked = "".join(f"【{c}】" if kind == "nothing near" else c for c, (kind, _) in zip(joined, kinds))
        print(f"\n## D{index:02d} decoy {item.name} {key[0]:.2f}-{key[1]:.2f} arm {arm}")
        print(f"   out   {marked or '(nothing written)'}")
        for row in near:
            print(f"   ref {row['spk']} {row['start']:.2f}-{row['end']:.2f}  {row['norm']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", choices=("C", "D"), required=True)
    arguments = parser.parse_args()
    print(f"spot check {arguments.sheet}, corpus {ref.CORPUS}, reading {ref.READING}")
    sheet()


if __name__ == "__main__":
    main()
