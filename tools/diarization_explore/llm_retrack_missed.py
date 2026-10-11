"""P2 missed rows on the held-out split, P1-shaped (owner 2026-10-05): the pre-registered tier-C reading.

Rows production left unlabelled (state `missed`) in the 15 held-out hina files; the P1 prompt unchanged
(`retrack_fill_system.md` / `retrack_fill_user.md`: track profiles, +-5 rows of context, choose a track / other /
abstain). Arms: agy (Gemini 3.7 Flash, main) and 3.5 Flash-Lite (reference), text only. Truth = covered (ties: any tied
person counts). Criterion written in the report before the run: per file, (judge right) - (nearest-labelled baseline
right), file-paired bootstrap 95% lower bound > 0; abstain = left empty = not right. Also reported: a hybrid (judge where
it answers, else the baseline), and the stronger no-model baseline -- the row's Nemotron slot mapped to a track.

    python tools/diarization_explore/llm_retrack_missed.py build
    python tools/diarization_explore/llm_retrack_missed.py run --arm agy --shard 0 --shards 4
    python tools/diarization_explore/llm_retrack_missed.py score
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import llm_retrack_judge as lj  # noqa: E402
import p2_features as pf  # noqa: E402

OUT = lr.OUT / "missed-heldout"
SEED = 20261005
_MATERIAL = {}


def items():
    """{ident: [row index]} -- every missed row of the held-out files."""
    rows, _ = pf.load_rows("covered")
    return {f: [r["index"] for r in rows[f] if r["state"] == "missed"] for f in pf.SPLIT["heldout"]}


def material(ident):
    if not _MATERIAL:
        for name in pf.SPLIT["heldout"]:  # all hina
            stable = pf.reference(name)[0]
            _MATERIAL[name] = (stable, lj._speakers(name, stable))
    return _MATERIAL[ident]


def calls():
    out = []
    for ident, rows in items().items():
        if not rows:
            continue
        stable, speakers = material(ident)
        system, file_calls = lr.fill_prompts(ident, stable, speakers, rows)
        out += [(key, system, user, ids) for key, ids, _, user in file_calls]
    return out


def build():
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    work = calls()
    total = sum(count(system) + count(user) for _, system, user, _ in work)
    print(f"{sum(len(v) for v in items().values())} missed rows, {len(work)} calls, {total} input tokens per arm")


def run(arm, shard, shards):
    complete = lr.client_for(lr.ARMS[arm])
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    for key, system, user, ids in calls()[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        answer, text = None, ""
        for attempt in range(4):
            try:
                text = complete(system, user)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{key}: {type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            answer = lr.parse(text, "fill", "id", ids)
            if answer is not None:
                break
            print(f"{key}: malformed reply; retry", flush=True)
        path.write_text(json.dumps({"answer": answer, "raw": text}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: {'ok' if answer else 'GAVE UP'}", flush=True)


def score():
    rows, _ = pf.load_rows("covered")
    slots = pickle.loads((pf.CACHE / "slot_identity.pkl").read_bytes())
    rng = np.random.default_rng(SEED)
    files = [f for f, idx in items().items() if idx]
    for arm in lr.ARMS:
        answers = {}
        for path in (OUT / arm).glob("*.json"):
            answers.update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
        if not answers:
            continue
        per_file, tallies = {}, {"answered": 0, "right": 0, "nearest": 0, "slot": 0, "hybrid": 0, "rows": 0}
        for f in files:
            by_index = {r["index"]: r for r in rows[f]}
            judge = nearest = slot = hybrid = 0
            for index in items()[f]:
                r = by_index[index]
                pick = str(answers.get(f"q{index}", {}).get("track", "abstain"))
                answered = pick.startswith("s") and pick[1:].isdigit()
                ok = answered and pf.right(r, int(pick[1:]))
                base = r["nearest"] is not None and pf.right(r, r["nearest"])
                judge += ok
                nearest += base
                slot += pf.right(r, slots[(f, index)]["slot_track"])
                hybrid += ok if answered else base
                tallies["answered"] += answered
                tallies["rows"] += 1
            per_file[f] = (judge, nearest, hybrid)
            tallies["right"] += judge
            tallies["nearest"] += nearest
            tallies["slot"] += slot
            tallies["hybrid"] += hybrid
        diff = np.array([per_file[f][0] - per_file[f][1] for f in files])
        hyb = np.array([per_file[f][2] - per_file[f][1] for f in files])
        sims = [diff[rng.integers(0, len(files), len(files))].sum() for _ in range(5000)]
        sims_h = [hyb[rng.integers(0, len(files), len(files))].sum() for _ in range(5000)]
        n = tallies["rows"]
        print(f"{arm}: {n} rows, answered {tallies['answered']} ({tallies['answered'] / n:.0%}), right {tallies['right']} "
              f"({tallies['right'] / n:.1%} of all, {tallies['right'] / max(1, tallies['answered']):.1%} of answered); "
              f"nearest baseline {tallies['nearest']} ({tallies['nearest'] / n:.1%}), slot baseline {tallies['slot']} ({tallies['slot'] / n:.1%}), "
              f"hybrid {tallies['hybrid']} ({tallies['hybrid'] / n:.1%})")
        print(f"   judge - nearest: {diff.sum():+d} rows [{np.percentile(sims, 2.5):+.0f}, {np.percentile(sims, 97.5):+.0f}] "
              f"-> {('PASS' if np.percentile(sims, 2.5) > 0 else 'fail') if arm == 'agy' else '(reference arm: no verdict)'}; hybrid - nearest: {hyb.sum():+d} [{np.percentile(sims_h, 2.5):+.0f}, {np.percentile(sims_h, 97.5):+.0f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "run", "score"))
    parser.add_argument("--arm", choices=sorted(lr.ARMS), default="agy")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    {"build": build, "score": score}.get(args.command, lambda: run(args.arm, args.shard, args.shards))()


if __name__ == "__main__":
    main()
