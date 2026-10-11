"""P2 judges on flagged labelled rows: small paired comparisons on the dev split (owner 2026-10-05).

Pilot (sample v1, `sample.json`): 120 rows, agy text only, two option sets -- `three` (keep / the two other tracks
with the most coverage / abstain) and `open` (keep / any track / abstain). Same picks; both lost rows.

Round 2 (sample v2, `sample-v2.json`), three arms on the SAME rows, criteria written in the report before the run:
  text     agy (Gemini 3.7 Flash), three options, prompt v2 (judge who says MOST of the row; a tie may go either way)
  audio    the same prompt and options plus one audio file per call: each track's voice samples, then each row's context
  mosspro  MOSS Pro on the whole file, its speakers mapped to production tracks WITHOUT the truth: by co-occurrence on
           confidently labelled rows (time margin > 0.5); on a flagged row it changes the label to the track its
           speaker maps to, and abstains where it has no label or no mapping.
Rows: dev, ordinary, time margin <= 0.5, truth = covered (ties: any tied person counts, `p2_features`). Files are drawn
among those MOSS Pro can cover today (already run, or fitting the free quota); 60 wrong + 60 right rows.
Track profiles (example lines and voice samples) never use a flagged row of the file, nor a sampled row's +-5 rows;
both v2 LLM arms judge 8 rows per call.

    python tools/diarization_explore/llm_retrack_judge.py build  [--sample v2]
    python tools/diarization_explore/llm_retrack_judge.py run --arm text|audio --shard 0 --shards 4
    python tools/diarization_explore/llm_retrack_judge.py moss          # map MOSS Pro outputs to decisions (no call)
    python tools/diarization_explore/llm_retrack_judge.py score [--sample v1|v2]
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
import p2_features as pf  # noqa: E402

OUT = lr.OUT / "judge"
TARGET = lr.ARMS["agy"]
SEED = 20261005
PER_SIDE_V2 = 60
#: Same batch for both v2 arms: rows judged together and prompt length must not differ between text and audio.
PER_CALL_TEXT, PER_CALL_AUDIO = 8, 8
SR, GAP, MAX_CONTEXT, SAMPLE = 16000, 1.5, 24.0, (2.0, 5.0)
SAMPLES = {"v1": "sample.json", "v2": "sample-v2.json", "v3": "sample-v3.json"}
ARMS = {"v1": ("three", "open"), "v2": ("text", "audio", "mosspro"), "v3": ("text", "audio")}
#: Files with a MOSS Pro whole-file output from earlier rounds (`retrack_external.py`, the 2026-09 survey).
MOSS_DONE = ("hina-25", "hina-27", "hina-67", "hina-68", "hina-70", "hina-71", "hina-72", "hina-73", "onetonton-01", "onetonton-05")
#: Free MOSS Pro audio left when the v2 sample was drawn (two keys x 1.25 h after the owner's refresh, minus a margin).
MOSS_BUDGET_S = 2.4 * 3600


def vocal(ident):
    if ident.startswith("onetonton-"):
        return lr.n3.MAIN / "out" / ident / f"{ident}-vocal.ogg"
    return lr.n3.MAIN / "out" / "refined-2609" / ident / f"{ident}-vocal.ogg"


def seconds_of(ident):
    import soundfile as sf
    return sf.info(str(vocal(ident))).duration


def _speakers(ident, stable):
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import TrackSegment, segments_from_activity
    native = np.load(pf.CACHE / f"{ident}.npz")["native"]
    tracks = segments_from_activity(native.tolist(), frame_seconds=pf.STEP)
    return assign_all(stable, [TrackSegment(track=t.track, start=t.start, end=t.end) for t in tracks])


def flagged_rows(rows, ident):
    return [r for r in rows[ident] if r["state"] == "ordinary" and r["margin"] <= 0.5]


def others(r):
    return [c["track"] for c in sorted(r["candidates"], key=lambda c: (-c["cover"], -c["prob"])) if c["track"] != r["leader"]]


def sample(version):
    """{"files": {ident: [items]}, ...} -- frozen at first call."""
    path = OUT / SAMPLES[version]
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if version == "v1":
        raise SystemExit("sample v1 is the frozen pilot; it is never redrawn")
    if version == "v3":
        return _sample_clean(path)
    rows, _ = pf.load_rows("covered")
    rng = np.random.default_rng(SEED + 2)
    counts = {f: (sum(r["wrong"] for r in flagged_rows(rows, f)), sum(not r["wrong"] for r in flagged_rows(rows, f))) for f in pf.SPLIT["dev"]}
    # files MOSS Pro can cover: already run, plus new ones (most wrong flagged rows first) while the free quota lasts --
    # the new ones spread the sample over more files (the file-level bootstrap needs them), not just more rows
    chosen, budget = [f for f in MOSS_DONE if f in counts], MOSS_BUDGET_S
    for f in sorted((f for f in counts if f not in MOSS_DONE), key=lambda f: -counts[f][0]):
        length = seconds_of(f)
        if length <= budget:
            chosen.append(f)
            budget -= length
    pool_wrong = [r for f in chosen for r in flagged_rows(rows, f) if r["wrong"]]
    pool_right = [r for f in chosen for r in flagged_rows(rows, f) if not r["wrong"]]
    picks = [pool_wrong[i] for i in rng.choice(len(pool_wrong), min(PER_SIDE_V2, len(pool_wrong)), replace=False)] + \
            [pool_right[i] for i in rng.choice(len(pool_right), min(PER_SIDE_V2, len(pool_right)), replace=False)]
    files = defaultdict(list)
    for r in sorted(picks, key=lambda r: (r["file"], r["index"])):
        files[r["file"]].append({"index": r["index"], "leader": r["leader"], "others": others(r)})
    out = {"files": dict(files), "moss_new": [f for f in chosen if f not in MOSS_DONE],
           "pool": {f: counts[f] for f in chosen}}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _sample_clean(path):
    """v3: dev flagged rows that are CLEAN (one person holds >= 70% of the reference coverage, no tie;
    `p2_two_rulers`), 60 wrong + 60 right, none from sample v2; any dev file (no MOSS Pro arm)."""
    import p2_two_rulers as tr
    rows, share = tr.shares()
    v2 = {(f, it["index"]) for f, items in sample("v2")["files"].items() for it in items}
    pool = [r for f in pf.SPLIT["dev"] for r in flagged_rows(rows, f)
            if tr.clean(share[(f, r["index"])]) and (f, r["index"]) not in v2]
    wrong = [r for r in pool if r["wrong"]]
    right = [r for r in pool if not r["wrong"]]
    rng = np.random.default_rng(SEED + 3)
    picks = [wrong[i] for i in rng.choice(len(wrong), PER_SIDE_V2, replace=False)] + \
            [right[i] for i in rng.choice(len(right), PER_SIDE_V2, replace=False)]
    files = defaultdict(list)
    for r in sorted(picks, key=lambda r: (r["file"], r["index"])):
        files[r["file"]].append({"index": r["index"], "leader": r["leader"], "others": others(r)})
    out = {"files": dict(files), "pool": {"wrong": len(wrong), "right": len(right)}}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


_MATERIAL = {}


def material(version, ident):
    """(stable, speakers) per file, loaded once per process: `reference` may not reload hina after onetonton."""
    if not _MATERIAL:
        for name in sorted(sample(version)["files"], key=lambda i: i.startswith("onetonton")):  # ⚠ hina first
            stable = pf.reference(name)[0]
            _MATERIAL[name] = (stable, _speakers(name, stable))
    return _MATERIAL[ident]


_FLAGGED = {}


def blocked(ident, n, items):
    """Row indices a profile may not use: every flagged row of the file itself (sampled or not; about a fifth of their
    labels are wrong), plus the +-5-row context of the SAMPLED rows (so the judged line never shows up as an example).
    ⚠ Not the context of every flagged row: that covers 80-94% of a file and leaves ~3x more option tracks with no
    example line or voice sample (peer round 3)."""
    if not _FLAGGED:
        rows, _ = pf.load_rows("covered")
        _FLAGGED.update({f: [r["index"] for r in flagged_rows(rows, f)] for f in rows})
    out = set(_FLAGGED[ident])
    for item in items:
        out.update(range(max(0, item["index"] - lr.CONTEXT), min(n, item["index"] + lr.CONTEXT + 1)))
    return out


def profiles(stable, speakers, exclude):
    """`lr.profiles` without the excluded rows as example lines (shares still over the whole file)."""
    seconds, rows = {}, {}
    for index, (row, speaker) in enumerate(zip(stable, speakers)):
        tracks = speaker.to_dict().get("tracks") or []
        if not tracks:
            continue
        seconds[tracks[0]] = seconds.get(tracks[0], 0.0) + float(row["end"]) - float(row["start"])
        if index not in exclude and len(tracks) == 1 and len(re.sub(r"[\W_]", "", str(row.get("text", "")))) >= 8:
            rows.setdefault(tracks[0], []).append(index)
    total = sum(seconds.values()) or 1.0
    out = []
    for track in sorted(seconds, key=lambda k: -seconds[k]):
        picks = rows.get(track, [])
        picks = [picks[round(i * (len(picks) - 1) / max(1, lr.PROFILE_LINES - 1))] for i in range(min(lr.PROFILE_LINES, len(picks)))]
        out.append(f"- s{track}：占语音 {seconds[track] / total * 100:.0f}%。例句：" +
                   " / ".join(f"「{str(stable[i].get('text', '')).strip()}」" for i in dict.fromkeys(picks)))
    return "\n".join(out), sorted(seconds)


def allowed(arm, item):
    if arm == "open":
        return f"可选：keep（现在是 s{item['leader']}），或任意一条已有的轨，或 abstain"
    options = "、".join(f"s{t}" for t in item["others"][:2]) or "（无）"
    return f"可选：keep（现在是 s{item['leader']}），或 {options}，或 abstain"


def system_prompt(version, audio=False):
    text = lr.template("retrack_judge_system.md").template
    if version != "v1":
        text += (HERE / "retrack_judge_v2_note.md").read_text(encoding="utf-8")
    if audio:
        text += (HERE / "retrack_judge_av_note.md").read_text(encoding="utf-8")
    return text


def text_calls(version, arm):
    """[(key, system, user, ids, wav or None)]."""
    files = sample(version)["files"]
    calls = []
    for ident in sorted(files):
        stable, speakers = material(version, ident)
        cells = [lr.cell(s) for s in speakers]
        items = files[ident]
        per_call = len(items) if version == "v1" else PER_CALL_TEXT
        profile_text, _ = profiles(stable, speakers, blocked(ident, len(stable), items) if version != "v1" else set())
        for start in range(0, len(items), per_call):
            chunk = items[start:start + per_call]
            blocks = []
            for item in chunk:
                index = item["index"]
                lo, hi = max(0, index - lr.CONTEXT), min(len(stable), index + lr.CONTEXT + 1)
                body = [("▶ " if j == index else "") + lr.line(stable[j], cells[j]) for j in range(lo, hi)]
                blocks.append(f"### q{index}\n" + "\n".join(body) + f"\n{allowed(arm, item)}")
            user = lr.template("retrack_judge_user.md").substitute(title=lr.TITLE, profiles=profile_text, items="\n\n".join(blocks), count=len(chunk))
            key = ident if version == "v1" else f"{ident}-{start // per_call}"
            calls.append((key, system_prompt(version), user, [f"q{x['index']}" for x in chunk], None))
    return calls


def audio_calls(version):
    """[(key, system, user, ids, wav, audio seconds)]."""
    import soundfile as sf
    files = sample(version)["files"]
    calls = []
    for ident in sorted(files):
        stable, speakers = material(version, ident)
        cells = [lr.cell(s) for s in speakers]
        items = files[ident]
        exclude = blocked(ident, len(stable), items)
        wave, rate = sf.read(str(vocal(ident)), dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        assert rate == SR, rate
        profile_text, known = profiles(stable, speakers, exclude)
        for start in range(0, len(items), PER_CALL_AUDIO):
            chunk = items[start:start + PER_CALL_AUDIO]
            key = f"{ident}-{start // PER_CALL_AUDIO}"
            gap = np.zeros(int(GAP * SR), dtype=np.float32)
            pieces, layout, cursor = [], [], [0.0]

            def add(a, b):
                piece = wave[int(a * SR):int(b * SR)]
                pieces.extend([piece, gap])
                at = cursor[0]
                cursor[0] += len(piece) / SR + GAP
                return at
            for track in known:
                picks = [i for i, s in enumerate(speakers) if i not in exclude and (s.to_dict().get("tracks") or []) == [track]
                         and SAMPLE[0] <= float(stable[i]["end"]) - float(stable[i]["start"]) <= SAMPLE[1]]
                picks.sort(key=lambda i: -(float(stable[i]["end"]) - float(stable[i]["start"])))
                for i in sorted(picks[:2]):
                    length = float(stable[i]["end"]) - float(stable[i]["start"])
                    at = add(float(stable[i]["start"]), float(stable[i]["end"]))
                    layout.append(f"| s{track} 的声音样本 | {at:.1f}–{at + length:.1f} 秒 |")
            blocks = []
            for item in chunk:
                index = item["index"]
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
                blocks.append(f"### q{index}\n" + "\n".join(body) + f"\n{allowed('text', item)}")
            folder = OUT / "av"
            folder.mkdir(parents=True, exist_ok=True)
            wav = folder / f"{key}.wav"
            sf.write(str(wav), np.concatenate(pieces), SR, subtype="PCM_16")
            layout_text = "| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
            user = lr.template("retrack_judge_user.md").substitute(
                title=lr.TITLE, profiles=profile_text + "\n\n音频布局：\n" + layout_text, items="\n\n".join(blocks), count=len(chunk))
            calls.append((key, system_prompt(version, audio=True), user, [f"q{x['index']}" for x in chunk], wav, cursor[0]))
    return calls


def build(version):
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    data = sample(version)
    print(f"sample {version}: {sum(len(v) for v in data['files'].values())} rows in {len(data['files'])} files; "
          f"MOSS Pro to run on {data.get('moss_new', [])}; pool (wrong, right flagged rows) {data.get('pool', {})}")
    for arm in [a for a in ARMS[version] if a in ("three", "open", "text")]:
        calls = text_calls(version, arm)
        total = sum(count(system) + count(user) for _, system, user, _, _ in calls)
        (OUT / f"prompt-{version}-{arm}-example.md").write_text(calls[0][2], encoding="utf-8")
        print(f"{arm}: {len(calls)} calls, {total} input tokens")
    if "audio" in ARMS[version]:
        calls = audio_calls(version)
        text = sum(count(c[1]) + count(c[2]) for c in calls)
        seconds = sum(c[5] for c in calls)
        (OUT / f"prompt-{version}-audio-example.md").write_text(calls[0][2], encoding="utf-8")
        print(f"audio: {len(calls)} calls, text {text} tokens, audio {seconds:.0f} s ≈ {seconds * 32:.0f} tokens; total ≈ {text + seconds * 32:.0f}")


def run(version, arm, shard, shards):
    if arm == "audio":
        import llm_retrack_av as av
        complete = av.client_for(TARGET)
        calls = [c[:5] for c in audio_calls(version)]
    else:
        complete = lr.client_for(TARGET)
        calls = text_calls(version, arm)
    folder = OUT / (arm if version == "v1" else f"{version}-{arm}")
    folder.mkdir(parents=True, exist_ok=True)
    for key, system, user, ids, wav in calls[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        answer, text = None, ""
        for attempt in range(4):
            try:
                text = complete(system, user, wav) if wav is not None else complete(system, user)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{key}: {type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            # ⚠ agy's answer channel is malformed ~23% of the time: retried, never scored.
            answer = lr.parse(text, "judge", "id", ids)
            if answer is not None:
                break
            print(f"{key}: malformed reply; retry", flush=True)
        path.write_text(json.dumps({"answer": answer, "raw": text}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: {'ok' if answer else 'GAVE UP'}", flush=True)


def moss(version="v2"):
    """MOSS Pro decisions on the sample rows, mapped to production tracks without the truth."""
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import TrackSegment
    import retrack_external as rx
    rows, _ = pf.load_rows("covered")
    data = sample(version)
    folder = OUT / "v2-mosspro"
    folder.mkdir(parents=True, exist_ok=True)
    for ident, items in data["files"].items():
        stable, _ = material(version, ident)
        tracks = rx.external_tracks(ident, "mosspro")
        if tracks is None:
            print(f"{ident}: no MOSS Pro output")
            continue
        moss_speakers = assign_all(stable, [TrackSegment(track=t["track"], start=t["start"], end=t["end"]) for t in tracks])
        moss_leader = [(s.to_dict().get("tracks") or [None])[0] for s in moss_speakers]
        # mapping: MOSS speaker -> production track, by seconds on confidently labelled rows (no truth used)
        weight = defaultdict(Counter)
        for r in rows[ident]:
            if r["leader"] is not None and r["margin"] > 0.5 and moss_leader[r["index"]] is not None:
                weight[moss_leader[r["index"]]][r["leader"]] += r["seconds"]
        mapping = {m: c.most_common(1)[0][0] for m, c in weight.items()}
        answer = {}
        for item in items:
            m = moss_leader[item["index"]]
            track = mapping.get(m) if m is not None else None
            pick = "abstain" if track is None else "keep" if track == item["leader"] else f"s{track}"
            answer[f"q{item['index']}"] = {"id": f"q{item['index']}", "track": pick, "conf": "high", "why": f"moss speaker {m}"}
        (folder / f"{ident}.json").write_text(json.dumps({"answer": answer, "mapping": {str(k): v for k, v in mapping.items()}},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{ident}: {len(mapping)} MOSS speakers mapped, {sum(a['track'] not in ('keep', 'abstain') for a in answer.values())} changes")


def outcomes(version, arm, rows):
    """{(file, index): (outcome, was_wrong)} -- fixed / broken / wrong_to_wrong / same_person / kept / abstain / missing."""
    data = sample(version)
    folder = OUT / (arm if version == "v1" else f"{version}-{arm}")
    out = {}
    for ident, items in data["files"].items():
        answer = {}
        for path in folder.glob(f"{ident}*.json"):
            if path.stem == ident or re.fullmatch(rf"{re.escape(ident)}-\d+", path.stem):
                answer.update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
        by_index = {r["index"]: r for r in rows[ident]}
        for item in items:
            r = by_index[item["index"]]
            reply = answer.get(f"q{item['index']}")
            pick = str((reply or {}).get("track", "")).strip()
            if reply is None:
                result = "missing"
            elif pick in ("keep", f"s{r['leader']}"):
                result = "kept"
            elif not re.fullmatch(r"s\d+", pick):
                result = "abstain"
            else:
                now = pf.right(r, int(pick[1:]))
                result = ("fixed" if now else "wrong_to_wrong") if r["wrong"] else ("same_person" if now else "broken")
            out[(ident, item["index"])] = (result, r["wrong"])
    return out


def picks(version, arm):
    """{(file, index): picked track string} for the changed rows of one arm."""
    folder = OUT / (arm if version == "v1" else f"{version}-{arm}")
    out = {}
    for ident in sample(version)["files"]:
        for path in folder.glob(f"{ident}*.json"):
            if path.stem == ident or re.fullmatch(rf"{re.escape(ident)}-\d+", path.stem):
                for key, reply in (json.loads(path.read_text(encoding="utf-8"))["answer"] or {}).items():
                    out[(ident, int(key[1:]))] = str(reply.get("track", "")).strip()
    return out


def mixed_rows(version):
    """{(file, index): True when two people share the row}: the top person holds < 70% of all reference coverage of
    the row, or several people tie for the top. ⚠ Not `max / frames`: extended reference end times let two lines both
    cover the whole row, which that ratio would call clean."""
    import relabel_eval as ev
    out = {}
    for ident in sorted(sample(version)["files"], key=lambda i: i.startswith("onetonton")):  # ⚠ hina first
        stable, grid, _, _ = pf.reference(ident)
        for item in sample(version)["files"][ident]:
            cover, n = ev.span_cover(stable[item["index"]], grid)
            top = cover.max()
            out[(ident, item["index"])] = bool(top == 0 or top / cover.sum() < 0.7 or (cover == top).sum() > 1)
    return out


_CLEAN_SHARE = {}


def net_of(counts, base):
    """Net per 100 flagged rows from summed (fixed, broken, n_wrong, n_right), re-weighted to the dev error rate."""
    fixed, broken, n_wrong, n_right = counts
    return 100 * (base * fixed / max(1, n_wrong) - (1 - base) * broken / max(1, n_right))


def score(version):
    data = sample(version)
    files = sorted(data["files"])
    print(f"sample {version}: {', '.join(files)} ({sum(len(v) for v in data['files'].values())} rows)")

    def subset_draws(keep):
        """5000 file resamples over `keep` (indices into `files`), the same draws for every arm on the same subset."""
        rng = np.random.default_rng(SEED)
        return [keep[rng.integers(0, len(keep), len(keep))] for _ in range(5000)]
    for truth in ("covered", "onset"):
        rows, _ = pf.load_rows(truth)
        dev_flagged = [r for f in pf.SPLIT["dev"] for r in flagged_rows(rows, f)]
        if version == "v3":  # the sample is clean rows only: re-weight to the error rate of clean flagged rows
            import p2_two_rulers as tr
            if not _CLEAN_SHARE:
                _CLEAN_SHARE.update(tr.shares()[1])
            dev_flagged = [r for r in dev_flagged if tr.clean(_CLEAN_SHARE[(r["file"], r["index"])])]
        base = float(np.mean([r["wrong"] for r in dev_flagged]))
        print(f"== truth: {truth}; dev flagged error rate {base:.1%} ==")
        per_file, present = {}, {}
        for arm in ARMS[version]:
            res = outcomes(version, arm, rows)
            if not any(v[0] != "missing" for v in res.values()):
                continue
            present[arm] = {k[0] for k, (o, _) in res.items() if o != "missing"}
            if len(present[arm]) < len(files):
                print(f"{arm}: no answer for {sorted(set(files) - present[arm])}; its counts and net cover the other files only")
                res = {k: v for k, v in res.items() if k[0] in present[arm]}
            counts = np.array([[sum(o == "fixed" for k, (o, _) in res.items() if k[0] == f),
                                sum(o == "broken" for k, (o, _) in res.items() if k[0] == f),
                                sum(w for k, (_, w) in res.items() if k[0] == f),
                                sum(not w for k, (_, w) in res.items() if k[0] == f)] for f in files], dtype=float)
            per_file[arm] = counts
            tally = Counter(o for o, _ in res.values())
            changed = tally["fixed"] + tally["broken"] + tally["wrong_to_wrong"] + tally["same_person"]
            own = np.array([i for i, f in enumerate(files) if f in present[arm]])
            sims = [net_of(counts[d].sum(axis=0), base) for d in subset_draws(own)]
            n_wrong, n_right = int(counts[:, 2].sum()), int(counts[:, 3].sum())
            print(f"{arm:8s}: {dict(tally)}; sample {n_wrong} wrong / {n_right} right; fixes {tally['fixed'] / max(1, n_wrong):.0%} of wrong, "
                  f"breaks {tally['broken'] / max(1, n_right):.0%} of right; changes on wrong rows (sample) "
                  f"{(tally['fixed'] + tally['wrong_to_wrong']) / max(1, changed):.0%}; net {net_of(counts.sum(axis=0), base):+.1f} "
                  f"[{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}] per 100 flagged rows (file bootstrap)")
        if version == "v2" and {"text", "mosspro"} <= set(per_file):
            # variant: change only where MOSS Pro and the text judge pick the same new track
            res_t = outcomes(version, "text", rows)
            picks_t, picks_m = picks(version, "text"), picks(version, "mosspro")
            agreed = Counter(res_t[k][0] for k in res_t if res_t[k][0] in ("fixed", "broken", "wrong_to_wrong", "same_person")
                             and picks_t.get(k) == picks_m.get(k))
            counts = per_file["text"].sum(axis=0).copy()
            counts[0], counts[1] = agreed["fixed"], agreed["broken"]
            print(f"text AND mosspro agree on the change: {dict(agreed)}; net {net_of(counts, base):+.1f} per 100 flagged rows")
        if version == "v2" and truth == "covered":
            mixed = mixed_rows(version)
            print(f"   mixed rows in the sample: {sum(mixed.values())} of {len(mixed)}")
            for arm in per_file:
                res = outcomes(version, arm, rows)
                for label, keep in (("clean (one person holds >= 70% of the reference coverage)", False), ("mixed", True)):
                    part = Counter(res[k][0] for k in res if mixed[k] == keep)
                    print(f"   {arm:8s} {label}: {dict(part)}")
        if version != "v1" and "text" in per_file:
            for arm in ("audio", "mosspro"):
                if arm not in per_file:
                    continue
                # only the files this arm answered (MOSS Pro failed server-side on hina-37), text restricted to the same
                keep = np.array([i for i, f in enumerate(files) if f in present[arm]])
                sub = subset_draws(keep)
                point = net_of(per_file[arm][keep].sum(axis=0), base) - net_of(per_file["text"][keep].sum(axis=0), base)
                sims = [net_of(per_file[arm][d].sum(axis=0), base) - net_of(per_file["text"][d].sum(axis=0), base) for d in sub]
                print(f"   {arm} - text on the {len(keep)} files {arm} answered (file-paired bootstrap): {point:+.1f} "
                      f"[{np.percentile(sims, 2.5):+.1f}, {np.percentile(sims, 97.5):+.1f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "run", "moss", "score"))
    parser.add_argument("--sample", choices=sorted(SAMPLES), default="v2")
    parser.add_argument("--arm", choices=("three", "open", "text", "audio"), default="text")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        build(args.sample)
    elif args.command == "run":
        run(args.sample, args.arm, args.shard, args.shards)
    elif args.command == "moss":
        moss(args.sample)
    else:
        score(args.sample)


if __name__ == "__main__":
    main()
