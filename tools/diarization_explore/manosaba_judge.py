"""Judges on the 魔裁 story clips, where the speaker truth is clean (on-screen nameplates; `manosaba_truth.py`).

Owner 2026-10-05: the hina rulers could not referee the judges on mixed rows; this material has one voice per line,
every character line voiced, and a nameplate per line. Same prompt, options and 8-row calls as P2 round 2
(`llm_retrack_judge.py`), four answers per row:
  text       agy (Gemini 3.7 Flash) text only
  audio      + one audio file per call (voice samples per track, then each row's context)
  audio_sem  the audio arm with the owner's idea: judge by content first, audio only as support
             (`retrack_judge_semantic_note.md`)
  production the label as it stands (the baseline every arm is scored against)
Rows: production-labelled ASR rows over a character's line, 48 wrong + 48 right (fixed seed, all three clips), wrong
only where the right person owns some track of the clip (otherwise no track answer can be right). Owners: each
track's majority person by seconds (truth-derived, used only for scoring). Profiles never use a sampled row or its
+-5 rows. Criteria are in the report before the run.

    python tools/diarization_explore/manosaba_judge.py build
    python tools/diarization_explore/manosaba_judge.py run --arm text|audio|audio_sem --shard 0 --shards 2
    python tools/diarization_explore/manosaba_judge.py score
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import llm_retrack_judge as lj  # noqa: E402
import manosaba_truth as mt  # noqa: E402

OUT = lr.OUT / "manosaba"
SEED = 20261005
PER_SIDE = 48
PER_CALL = 8
ARMS = ("text", "audio", "audio_sem")
_MAT = {}


def material(clip):
    """(stable, speakers, truth rows, owners) per clip."""
    if clip not in _MAT:
        from finesub.speech.diarization.assign import track_coverage
        from finesub.speech.diarization.segments import TrackSegment
        table = mt.names()
        stable, speakers = mt.production(clip)
        rows = mt.rows_with_truth(clip, table)
        weight = defaultdict(Counter)
        for r in rows:
            if r["leader"] is not None:
                weight[r["leader"]][r["truth"]] += r["end"] - r["start"]
        owner = {k: c.most_common(1)[0][0] for k, c in weight.items()}
        tracks = json.loads((mt.RUNS / f"{clip}-diar.json").read_text(encoding="utf-8"))["tracks"]
        segs = [TrackSegment(track=int(t["track"]), start=float(t["start"]), end=float(t["end"])) for t in tracks]
        for r in rows:
            cov = track_coverage(r["start"], r["end"], segs)
            r["others"] = [int(k) for k, _ in sorted(cov.items(), key=lambda kv: -kv[1]) if int(k) != r["leader"]]
        _MAT[clip] = (stable, speakers, rows, owner)
    return _MAT[clip]


def sample():
    path = OUT / "sample.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    wrong, right = [], []
    for clip in mt.CLIPS:
        _, _, rows, owner = material(clip)
        owned = set(owner.values())
        for r in rows:
            if r["leader"] is None:
                continue
            item = {"clip": clip, "index": r["index"], "leader": r["leader"], "others": r["others"], "truth": r["truth"]}
            if owner[r["leader"]] == r["truth"]:
                right.append(item)
            elif r["truth"] in owned:
                wrong.append(item)
    rng = np.random.default_rng(SEED)
    picks = [wrong[i] for i in rng.choice(len(wrong), PER_SIDE, replace=False)] + \
            [right[i] for i in rng.choice(len(right), PER_SIDE, replace=False)]
    files = defaultdict(list)
    for p in sorted(picks, key=lambda p: (p["clip"], p["index"])):
        files[p["clip"]].append(p)
    out = {"files": dict(files), "pool": {"wrong_fixable": len(wrong), "right": len(right)}}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def system(arm):
    text = lj.system_prompt("v2", audio=arm != "text")
    if arm == "audio_sem":
        text += (HERE / "retrack_judge_semantic_note.md").read_text(encoding="utf-8")
    return text


def calls(arm):
    """[(key, system, user, ids, wav or None, audio seconds)]."""
    import soundfile as sf
    out = []
    for clip, items in sample()["files"].items():
        stable, speakers, _, _ = material(clip)
        cells = [lr.cell(s) for s in speakers]
        exclude = set()
        for it in items:
            exclude.update(range(max(0, it["index"] - lr.CONTEXT), min(len(stable), it["index"] + lr.CONTEXT + 1)))
        profile_text, known = lj.profiles(stable, speakers, exclude)
        bare = [line.split("：", 1)[0].lstrip("- ") for line in profile_text.splitlines() if line.endswith("例句：")]
        if bare:
            print(f"warning: {clip}: tracks {bare} have no example line ({len(exclude)}/{len(stable)} rows excluded)", file=sys.stderr)
        wave = None
        if arm != "text":
            wave, rate = sf.read(str(mt.RUNS / f"{clip}-vocal.ogg"), dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if rate != lj.SR:
                import librosa
                wave = librosa.resample(wave, orig_sr=rate, target_sr=lj.SR)
        for start in range(0, len(items), PER_CALL):
            chunk = items[start:start + PER_CALL]
            key = f"{clip}-{start // PER_CALL}"
            pieces, layout, cursor = [], [], [0.0]
            gap = np.zeros(int(lj.GAP * lj.SR), dtype=np.float32)

            def add(a, b):
                piece = wave[int(a * lj.SR):int(b * lj.SR)]
                pieces.extend([piece, gap])
                at = cursor[0]
                cursor[0] += len(piece) / lj.SR + lj.GAP
                return at
            if wave is not None:
                for track in known:
                    picks = [i for i, s in enumerate(speakers) if i not in exclude and (s.to_dict().get("tracks") or []) == [track]
                             and lj.SAMPLE[0] <= float(stable[i]["end"]) - float(stable[i]["start"]) <= lj.SAMPLE[1]]
                    picks.sort(key=lambda i: -(float(stable[i]["end"]) - float(stable[i]["start"])))
                    for i in sorted(picks[:2]):
                        length = float(stable[i]["end"]) - float(stable[i]["start"])
                        at = add(float(stable[i]["start"]), float(stable[i]["end"]))
                        layout.append(f"| s{track} 的声音样本 | {at:.1f}–{at + length:.1f} 秒 |")
            blocks = []
            for it in chunk:
                index = it["index"]
                lo, hi = max(0, index - lr.CONTEXT), min(len(stable), index + lr.CONTEXT + 1)
                if wave is None:
                    body = [("▶ " if j == index else "") + lr.line(stable[j], cells[j]) for j in range(lo, hi)]
                else:
                    centre = (float(stable[index]["start"]) + float(stable[index]["end"])) / 2
                    a = max(float(stable[lo]["start"]) - 0.3, centre - lj.MAX_CONTEXT / 2)
                    b = min(float(stable[hi - 1]["end"]) + 0.3, centre + lj.MAX_CONTEXT / 2)
                    at = add(a, b)
                    layout.append(f"| q{index} 的上下文 | {at:.1f}–{at + b - a:.1f} 秒 |")
                    body = []
                    for j in range(lo, hi):
                        row = stable[j]
                        moved = f"{at + float(row['start']) - a:.1f}" if a <= float(row["start"]) < b else "（不在录音里）"
                        body.append(("▶ " if j == index else "") + f"{moved}|{cells[j]}|{str(row.get('text', '')).strip()}")
                blocks.append(f"### q{index}\n" + "\n".join(body) + f"\n{lj.allowed('text', it)}")
            profiles = profile_text
            wav = None
            if wave is not None:
                folder = OUT / "av"
                folder.mkdir(parents=True, exist_ok=True)
                wav = folder / f"{key}.wav"
                sf.write(str(wav), np.concatenate(pieces), lj.SR, subtype="PCM_16")
                profiles += "\n\n音频布局：\n| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
            user = lr.template("retrack_judge_user.md").substitute(title=lr.TITLE, profiles=profiles, items="\n\n".join(blocks), count=len(chunk))
            out.append((key, system(arm), user, [f"q{x['index']}" for x in chunk], wav, cursor[0]))
    return out


def build():
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    data = sample()
    print(f"sample: {sum(len(v) for v in data['files'].values())} rows {({k: len(v) for k, v in data['files'].items()})}; pool {data['pool']}")
    total = 0
    for arm in ARMS:
        cs = calls(arm)
        text = sum(count(c[1]) + count(c[2]) for c in cs)
        seconds = sum(c[5] for c in cs)
        total += text + seconds * 32
        (OUT / f"prompt-{arm}-example.md").write_text(cs[0][2], encoding="utf-8")
        print(f"{arm}: {len(cs)} calls, text {text} tokens, audio {seconds:.0f} s ≈ {seconds * 32:.0f}; arm total ≈ {text + seconds * 32:.0f}")
    print(f"all arms ≈ {total:.0f} tokens")


def run(arm, shard, shards):
    import llm_retrack_av as av
    complete = av.client_for(lj.TARGET) if arm != "text" else lr.client_for(lj.TARGET)
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    for key, sys_text, user, ids, wav, _ in calls(arm)[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        answer, text = None, ""
        for attempt in range(4):
            try:
                text = complete(sys_text, user, wav) if wav is not None else complete(sys_text, user)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{key}: {type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            answer = lr.parse(text, "judge", "id", ids)
            if answer is not None:
                break
            print(f"{key}: malformed reply; retry", flush=True)
        path.write_text(json.dumps({"answer": answer, "raw": text}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: {'ok' if answer else 'GAVE UP'}", flush=True)


def score():
    """Net per 100 labelled rows: fixed share of wrong rows x fixable base - broken share of right rows x (1 - error base).

    Pre-registered: the fixed term weighted by the whole production error rate. The sample's wrong rows are drawn only
    from fixable ones (the true person owns a track), so the corrected reading (post-hoc, peer 2026-10-05) weights it by
    the fixable share of labelled rows instead; the broken term is the same in both.
    """
    data = sample()
    owners = {c: material(c)[3] for c in mt.CLIPS}
    labelled = [(c, r) for c in mt.CLIPS for r in material(c)[2] if r["leader"] is not None]
    wrong = [(c, r) for c, r in labelled if owners[c][r["leader"]] != r["truth"]]
    base = len(wrong) / len(labelled)
    fixable = sum(r["truth"] in set(owners[c].values()) for c, r in wrong) / len(labelled)
    print(f"labelled rows with truth {len(labelled)}; production error rate {base:.1%}, fixable (the true person owns a track) {fixable:.1%}")
    keys = [(c, it["index"]) for c, items in data["files"].items() for it in items]
    item = {(c, it["index"]): it for c, items in data["files"].items() for it in items}
    ceiling = np.mean([any(owners[c][t] == it["truth"] for t in it["others"][:2]) for (c, _), it in item.items() if owners[c][it["leader"]] != it["truth"]])
    print(f"sample {len(keys)} rows; wrong rows whose true person owns one of the two offered tracks: {ceiling:.0%}")
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(keys), len(keys)) for _ in range(5000)]

    def net(res, sel, weight):
        f = sum(res[k][0] == "fixed" for k in sel)
        b = sum(res[k][0] == "broken" for k in sel)
        nw = sum(res[k][1] for k in sel)
        return 100 * (weight * f / max(1, nw) - (1 - base) * b / max(1, len(sel) - nw))

    sims = {}
    for arm in ARMS:
        answers = {}
        for path in (OUT / arm).glob("*.json"):
            clip = path.stem.rsplit("-", 1)[0]
            for k, v in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items():
                answers[(clip, int(k[1:]))] = v
        if not answers:
            continue
        res = {}
        for key in keys:
            c, _ = key
            it = item[key]
            was_wrong = owners[c][it["leader"]] != it["truth"]
            pick = str((answers.get(key) or {}).get("track", "")).strip()
            if key not in answers:
                o = "missing"
            elif pick in ("keep", f"s{it['leader']}"):
                o = "kept"
            elif not re.fullmatch(r"s\d+", pick):
                o = "abstain"
            else:
                now = owners[c].get(int(pick[1:])) == it["truth"]
                o = ("fixed" if now else "wrong_to_wrong") if was_wrong else ("same_person" if now else "broken")
            res[key] = (o, was_wrong)
        n_wrong = sum(w for _, w in res.values())
        tally = Counter(o for o, _ in res.values())
        # bootstrap over rows (only 3 clips: a file bootstrap would be meaningless); the same draws for every arm
        sims[arm] = {w: np.array([net(res, [keys[i] for i in d], w) for d in draws]) for w in (base, fixable)}
        print(f"{arm:9s}: {dict(tally)}; fixes {tally['fixed'] / max(1, n_wrong):.0%} of wrong, breaks {tally['broken'] / max(1, len(res) - n_wrong):.0%} of right")
        for w, label in ((base, "pre-registered"), (fixable, "fixable-weighted (post-hoc)")):
            s = sims[arm][w]
            print(f"           net {net(res, keys, w):+.1f} [{np.percentile(s, 2.5):+.1f}, {np.percentile(s, 97.5):+.1f}] per 100 labelled rows, {label}")
    for a, b in (("audio_sem", "audio"), ("audio", "text")):
        if a in sims and b in sims:
            d = sims[a][base] - sims[b][base]
            print(f"paired {a} - {b} (pre-registered weight): mean {d.mean():+.1f} [{np.percentile(d, 2.5):+.1f}, {np.percentile(d, 97.5):+.1f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "run", "score"))
    parser.add_argument("--arm", choices=ARMS, default="text")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    {"build": build, "score": score}.get(args.command, lambda: run(args.arm, args.shard, args.shards))()


if __name__ == "__main__":
    main()
