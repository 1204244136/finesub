"""P1 (fill the rows Nemotron-3 missed) with audio, on a small subset (owner 2026-10-04).

The 0.5M token budget covers text AND audio, and is a ceiling, not a target: the text arms already
used ~0.35M, so this runs ~100 姐姐咚咚 rows -- the ones where exactly one person's refined line
starts inside the row, which is the clean truth. One audio file per call: each track's voice samples
first, then each row's context, 1.5 s of silence between pieces; the prompt's times are seconds IN
that file (the C.5d audio arm got this wrong).

    python tools/diarization_explore/llm_retrack_av.py build           # token estimate, writes the wavs
    python tools/diarization_explore/llm_retrack_av.py run --arm agy --shard 0 --shards 4
    python tools/diarization_explore/llm_retrack_av.py score            # text vs audio, agy vs lite, same rows
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402

PER_EPISODE = 7       # ~100 rows in all
PER_CALL = 8
GAP = 1.5             # silence between pieces, seconds
MAX_CONTEXT = 24.0    # longest context piece, seconds
SAMPLE = (2.0, 5.0)   # length range of a track's voice sample
SR = 16000


def items():
    """[(ident, stable, speakers, rows)] -- onetonton rows where exactly one person starts speaking."""
    out = []
    for ident, corpus, stable, speakers, _, _, _, starts, rows in lr.fill_material():
        if corpus != "onetonton":
            continue
        clean = [i for i in rows
                 if len({w for a, w in starts if float(stable[i]["start"]) - 0.3 <= a < float(stable[i]["end"])}) == 1]
        out.append((ident, stable, speakers, clean[:: max(1, len(clean) // PER_EPISODE)][:PER_EPISODE]))
    return out


def voice_samples(stable, speakers, track, count=2):
    rows = [i for i, s in enumerate(speakers) if (s.to_dict().get("tracks") or []) == [track]
            and SAMPLE[0] <= float(stable[i]["end"]) - float(stable[i]["start"]) <= SAMPLE[1]]
    rows.sort(key=lambda i: -(float(stable[i]["end"]) - float(stable[i]["start"])))
    return sorted(rows[:count])


def prompt(ident, stable, speakers, chunk, key, source=None):
    """(system, user, wav path, audio seconds) for one call; `source` is the vocal track (default: out/<ident>/)."""
    import soundfile as sf
    source = source or lr.n3.MAIN / "out" / ident / f"{ident}-vocal.ogg"
    wave, rate = sf.read(str(source), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    assert rate == SR, rate
    gap = np.zeros(int(GAP * SR), dtype=np.float32)
    pieces, layout, cursor = [], [], [0.0]

    def add(a, b):
        piece = wave[int(a * SR):int(b * SR)]
        start = cursor[0]
        pieces.extend([piece, gap])
        cursor[0] += len(piece) / SR + GAP
        return start

    profile_text, known = lr.profiles(stable, speakers)
    for track in known:
        for i in voice_samples(stable, speakers, track):
            length = float(stable[i]["end"]) - float(stable[i]["start"])
            at = add(float(stable[i]["start"]), float(stable[i]["end"]))
            layout.append(f"| s{track} 的声音样本 | {at:.1f}–{at + length:.1f} 秒 |")
    cells = [lr.cell(s) for s in speakers]
    blocks = []
    for index in chunk:
        lo, hi = max(0, index - lr.CONTEXT), min(len(stable), index + lr.CONTEXT + 1)
        centre = (float(stable[index]["start"]) + float(stable[index]["end"])) / 2
        a = max(float(stable[lo]["start"]) - 0.3, centre - MAX_CONTEXT / 2)
        b = min(float(stable[hi - 1]["end"]) + 0.3, centre + MAX_CONTEXT / 2)
        at = add(a, b)
        layout.append(f"| q{index} 的上下文 | {at:.1f}–{at + b - a:.1f} 秒 |")
        body = []
        for j in range(lo, hi):
            row = stable[j]
            moved = f"{at + float(row['start']) - a:.1f}" if a <= float(row["start"]) < b else "（不在录音里）"
            body.append(("▶ " if j == index else "") + f"{moved}|{cells[j]}|{str(row.get('text', '')).strip()}")
        blocks.append(f"### q{index}\n" + "\n".join(body))
    folder = lr.OUT / "av"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.wav"
    sf.write(str(path), np.concatenate(pieces), SR, subtype="PCM_16")
    layout_text = "| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
    user = lr.template("retrack_fill_user.md").substitute(
        title=lr.TITLE, profiles=profile_text + "\n\n音频布局：\n" + layout_text,
        items="\n\n".join(blocks), count=len(chunk))
    system = lr.template("retrack_fill_system.md").template + (HERE / "retrack_fill_av_note.md").read_text(encoding="utf-8")
    return system, user, path, cursor[0]


def jobs():
    out = []
    for ident, stable, speakers, rows in items():
        for start in range(0, len(rows), PER_CALL):
            out.append((f"{ident}-{start // PER_CALL}", ident, stable, speakers, rows[start:start + PER_CALL]))
    return out


def client_for(target):
    from dataclasses import replace

    from finesub.llm.client import RoleClient
    from finesub.llm.media_upload import window_media_ref
    from finesub.llm.routing.config import LLMRole, role_config_for

    if target.startswith("local-agy"):
        base = role_config_for("research", preset_id="agy", role=LLMRole.GENERAL_CAPABLE)
    else:
        base = role_config_for("correction-mm", "intermediate", preset_id="default", role=LLMRole.GENERAL_CAPABLE)
    chain = tuple(e for e in base.endpoint_chain if e.target_id == target)
    if not chain:
        raise SystemExit(f"{target} is not in the base chain")
    client = RoleClient(role_configs={LLMRole.GENERAL_CAPABLE: replace(
        base, endpoint_chain=chain, test_endpoint=chain[0], model_group_id="")})

    def complete(system, user, wav):
        reference = window_media_ref(wav, routes=client.router.routes)
        reply = client.complete(LLMRole.GENERAL_CAPABLE,
                                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                file_ref=reference)
        landed = str(getattr(reply, "target_id", ""))
        if landed != target:
            raise SystemExit(f"pinned to {target} but landed on {landed!r}; stopped after one call")
        return reply.content
    return complete


def build():
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    text = seconds = rows = 0
    work = jobs()
    for key, ident, stable, speakers, chunk in work:
        system, user, _, length = prompt(ident, stable, speakers, chunk, key)
        text += count(system) + count(user)
        seconds += length
        rows += len(chunk)
    print(f"av: {len(work)} calls, {rows} rows, text {text} tokens, audio {seconds:.0f} s ≈ {seconds * 32:.0f} tokens "
          f"(32 tokens/s), total ≈ {text + seconds * 32:.0f} per arm")


def run(arm, shard, shards):
    complete = client_for(lr.ARMS[arm])
    folder = lr.OUT / f"fill-av-{arm}-r1"
    folder.mkdir(parents=True, exist_ok=True)
    for key, ident, stable, speakers, chunk in jobs()[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        system, user, wav, _ = prompt(ident, stable, speakers, chunk, key)
        answer, text = None, ""
        for attempt in range(4):
            try:
                text = complete(system, user, wav)
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
    """Same rows, four answers each (text / audio x agy / lite), plus the neighbour baseline."""
    import backend_cost as bc
    import cost_metrics as cm
    material = {m[0]: m for m in lr.fill_material()}
    arms = {f"{task}-{arm}": lr.OUT / f"{folder}-{arm}-r1"
            for task, folder in (("text", "fill"), ("audio", "fill-av")) for arm in lr.ARMS}
    rights = {name: [] for name in [*arms, "neighbour"]}
    for ident, stable, speakers, rows in items():
        _, _, _, _, tracks, grid, people, starts, _ = material[ident]
        owner = lr.owners(tracks, grid, people)
        answers = {name: {} for name in arms}
        for name, folder in arms.items():
            for path in folder.glob(f"{ident}-*.json"):
                answers[name].update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
        centres = np.array([(float(r["start"]) + float(r["end"])) / 2 for r in stable])
        with_tracks = [i for i, s in enumerate(speakers) if s.to_dict().get("tracks")]
        for index in rows:
            truth = next(iter({w for a, w in starts if float(stable[index]["start"]) - 0.3 <= a < float(stable[index]["end"])}))
            nearest = min(with_tracks, key=lambda j: abs(centres[j] - centres[index]))
            rights["neighbour"].append(owner.get(speakers[nearest].to_dict()["tracks"][0]) == truth)
            for name in arms:
                track = str(answers[name].get(f"q{index}", {}).get("track", "abstain"))
                rights[name].append(owner.get(int(track[1:])) == truth if track.startswith("s") and track[1:].isdigit()
                                    else None)
    for name, values in rights.items():
        filled = [v for v in values if v is not None]
        print(f"  {name:12} filled {len(filled)}/{len(values)}  right {sum(filled)} ({sum(filled) / max(1, len(filled)):.0%})")


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
