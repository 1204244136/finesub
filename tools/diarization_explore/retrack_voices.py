"""P1 on voices that differ a lot: 提瓦特放送局 (owner 2026-10-04).

On 姐姐咚咚 (two similar female voices) the audio arm matched the text arm's precision (70 % vs 69 %) and
abstained less. This asks the same question on hosts whose voices are far apart -- 钟离
(deep male), 派蒙 (female), 温迪 / 男主 (lighter male), plus female guests -- on 放送局 episodes
the earlier runs never touched. Text arm and audio arm, both agy, same rows.

    python tools/diarization_explore/retrack_voices.py timelines             # GPU: production Nemotron-3
    python tools/diarization_explore/retrack_voices.py build                 # rows + token estimate
    python tools/diarization_explore/retrack_voices.py run --task text|audio --shard i --shards 4
    python tools/diarization_explore/retrack_voices.py score
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import llm_retrack_av as av  # noqa: E402

#: 钟离 + 派蒙 in most; 温迪 (35), 男主 (19, 25-27), female guests (19 宵宫, 21/20 嘉宾, 29 多莉 / 云堇). hina-18 dropped: its
#: 「男女主」 style is two people under one name.
EPISODES = ("hina-21", "hina-23", "hina-27", "hina-29", "hina-35", "hina-19", "hina-20", "hina-25", "hina-26")
ARM = "agy"


def vocal(ident):
    return lr.n3.MAIN / "out" / "refined-2609" / ident / f"{ident}-vocal.ogg"


def timelines():
    import vram_guard
    vram_guard.install()
    from finesub.speech.diarization.stage import run_diarization, write_timeline
    for ident in EPISODES:
        path = lr.ABSTAIN / ident / "nemotron3-diar.json"
        if path.exists():
            continue
        result = run_diarization(vocal(ident), device="cuda")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_timeline(result, path)
        print(ident, len({t.track for t in result.tracks}), "tracks", flush=True)


def material():
    """[(ident, stable, speakers, tracks, grid, people, starts, rows)] -- rows = onset-truth unlabelled rows."""
    import nemotron3_cost as nc
    import refined_multi as rm
    nc.corpus_keys("hina")
    sources = rm.sources()
    out = []
    for ident in EPISODES:
        stable, _, grid, people, _, lines = rm.episode(ident, sources[ident])
        starts = sorted((a, w) for a, _, w in lines)
        tracks = json.loads((lr.ABSTAIN / ident / "nemotron3-diar.json").read_text(encoding="utf-8"))["tracks"]
        speakers = lr._assign(stable, tracks)
        rows = [i for i, s in enumerate(speakers) if not (s.to_dict().get("tracks") or [])
                and len({w for a, w in starts if float(stable[i]["start"]) - 0.3 <= a < float(stable[i]["end"])}) == 1]
        out.append((ident, stable, speakers, tracks, grid, people, starts, rows))
    return out


def jobs(task):
    out = []
    for ident, stable, speakers, _, _, _, _, rows in material():
        per = lr.PER_CALL if task == "text" else av.PER_CALL
        for start in range(0, len(rows), per):
            out.append((f"{ident}-{start // per}", ident, stable, speakers, rows[start:start + per]))
    return out


def prompt(task, key, ident, stable, speakers, chunk):
    if task == "text":
        system, calls = lr.fill_prompts(ident, stable, speakers, chunk)
        return system, calls[0][3], None, 0.0
    return av.prompt(ident, stable, speakers, chunk, key, source=vocal(ident))


def build():
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    for task in ("text", "audio"):
        text = seconds = rows = 0
        work = jobs(task)
        for key, ident, stable, speakers, chunk in work:
            system, user, _, length = prompt(task, key, ident, stable, speakers, chunk)
            text += count(system) + count(user)
            seconds += length
            rows += len(chunk)
        print(f"{task}: {len(work)} calls, {rows} rows, text {text} tokens, audio {seconds:.0f} s ≈ {seconds * 32:.0f}; "
              f"total ≈ {text + seconds * 32:.0f}")


def run(task, shard, shards):
    complete = av.client_for(lr.ARMS[ARM]) if task == "audio" else lr.client_for(lr.ARMS[ARM])
    folder = lr.OUT / f"voices-{task}-{ARM}"
    folder.mkdir(parents=True, exist_ok=True)
    for key, ident, stable, speakers, chunk in jobs(task)[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        system, user, wav, _ = prompt(task, key, ident, stable, speakers, chunk)
        answer, text = None, ""
        for attempt in range(4):
            try:
                text = complete(system, user, wav) if task == "audio" else complete(system, user)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{key}: {type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            answer = lr.parse(text, "fill", "id", [f"q{i}" for i in chunk])
            if answer is not None:
                break
            print(f"{key}: malformed reply; retry", flush=True)
        path.write_text(json.dumps({"answer": answer, "raw": text}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: {'ok' if answer else 'GAVE UP'}", flush=True)


def score():
    import backend_cost as bc
    import cost_metrics as cm
    from retrack_split import is_filler
    results = {name: [] for name in ("基线", "text", "audio")}
    for ident, stable, speakers, tracks, grid, people, starts, rows in material():
        owner = lr.owners(tracks, grid, people)
        answers = {task: {} for task in ("text", "audio")}
        for task in answers:
            for path in (lr.OUT / f"voices-{task}-{ARM}").glob(f"{ident}-*.json"):
                answers[task].update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
        centres = np.array([(float(r["start"]) + float(r["end"])) / 2 for r in stable])
        labelled = [i for i, s in enumerate(speakers) if s.to_dict().get("tracks")]
        for index in rows:
            truth = next(iter({w for a, w in starts if float(stable[index]["start"]) - 0.3 <= a < float(stable[index]["end"])}))
            kind = "填充词" if is_filler(str(stable[index].get("text", ""))) else "有意义的句子"
            nearest = min(labelled, key=lambda j: abs(centres[j] - centres[index]))
            results["基线"].append((ident, kind, truth, owner.get(speakers[nearest].to_dict()["tracks"][0]) == truth))
            for task in ("text", "audio"):
                track = str(answers[task].get(f"q{index}", {}).get("track", "abstain"))
                results[task].append((ident, kind, truth, owner.get(int(track[1:])) == truth
                                      if track.startswith("s") and track[1:].isdigit() else None))

    def show(values):
        filled = [v for v in values if v is not None]
        return f"{sum(filled)}/{len(filled)} of {len(values)} ({sum(filled) / max(1, len(filled)):.0%})"

    for kind in (None, "填充词", "有意义的句子"):
        print(f"[{kind or '全部'}] " + "  ".join(
            f"{name} {show([r[3] for r in rows if kind is None or r[1] == kind])}" for name, rows in results.items()))
    for ident in EPISODES:
        print(f"  {ident}: " + "  ".join(
            f"{name} {show([r[3] for r in rows if r[0] == ident])}" for name, rows in results.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("timelines", "build", "run", "score"))
    parser.add_argument("--task", choices=("text", "audio"), default="text")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "timelines":
        timelines()
    elif args.command == "build":
        build()
    elif args.command == "run":
        run(args.task, args.shard, args.shards)
    else:
        score()


if __name__ == "__main__":
    main()
