"""Speaker judge round 9 (owner 2026-10-07): does the judge do better on production's corrected text than on raw ASR?

The open question for putting the judge into production is where it sits: on the raw ASR rows (what every round so far
measured) or after the correction windows (cleaner text: back-channels and fragments dropped, mishearings fixed). Same
windows as round 8 (marks step 2), both modalities on all eleven; the cast block is each window's true speakers (owner:
truth controls that variable); no marks on either input; one prompt for both inputs. Criteria: the marks plan,
「对照实验：裁判读原始 ASR 还是纠错后的文本」, written first.

Corrected text comes from production's own correction (`finesub.llm.correction_translation`, the default preset) run on
a slice of each file: the window plus SLICE_MARGIN rows either side, segment ids kept as the file's 1-based positions,
`--fast off` so a slice is planned like the whole file it comes from. 魔裁 b and c are short clips that production runs
whole (and fast), so they run whole with the defaults.

Both inputs share everything but the window's rows: cast block, track profiles and voice samples (from the raw stable),
the recording's span. A corrected row is one output row of the correction (a merge of up to three sources); it starts
at its first source, its track cell is production's `assign` over the merged span (so `s5+s4` and `??` keep round 7's
meaning), and it is judged when any of its sources is inside the window. Its answer applies to each of its sources in
the window; `keep` / `abstain` leave every source on its acoustic label. Sources the correction dropped keep theirs.

    python tools/diarization_explore/window_judge_r9.py slices                 # write the slices
    python tools/diarization_explore/window_judge_r9.py correct [--execute]    # plan (dry run) or run the correction
    python tools/diarization_explore/window_judge_r9.py build                  # render, count tokens, write examples
    python tools/diarization_explore/window_judge_r9.py run --arm audio-corrected --pass 1 [--shard 0 --shards 2]
    python tools/diarization_explore/window_judge_r9.py score
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_judge_r5 as r5  # noqa: E402  (only for its name table; imported first so round 8 patches last)
import window_judge_r8 as r8  # noqa: E402  (patches round 3's load / windows to round 8's)
import window_partition_score as ps  # noqa: E402

r3 = r8.r3
lr, lj = r3.lr, r3.lj
MAIN = lr.OUT.parents[1]
ROOT = lr.OUT / "r9"
SEED = 20261015
SLICE_MARGIN = 100
WHOLE = ("manosaba-b", "manosaba-c")  # short clips production corrects whole
MODALITIES = ("text", "audio")
SOURCES = ("raw", "corrected")
ARMS = tuple(f"{m}-{s}" for m in MODALITIES for s in SOURCES)
FRESH = ("hina 普通", "魔裁 c")
DELTA = 3.0          # non-inferiority margin, points per 100 rows
SINGLE_PASS = 2.0    # extra margin a single pass gives up: step 2's largest same-input two-run gap
BACKCHANNEL = 4      # characters after punctuation is dropped (production's marks rule)


def plan():
    """Round 8's eleven windows, every one judged by both modalities here."""
    return r8.plan()


def windows():
    return [(w["ident"], w["a"], w["b"]) for w in plan()]


def load(ident):
    m = r3.load(ident)
    names = set(m.people) | {p for s in m.truth.values() for p in s}
    m.aliases = {n: a for n, a in r5.ALIASES.items() if n in names}  # owner 2026-10-06: names with their Japanese form
    return m


# --- correction --------------------------------------------------------------------------------------

def run_dir(ident):
    return (MAIN / "out" / "manosaba") if ident.startswith("manosaba-") else (MAIN / "out" / "refined-2609" / ident)


def source_audio(ident):
    source = Path(json.loads((run_dir(ident) / f"{ident}-metadata.json").read_text(encoding="utf-8"))["source"])
    return source if source.is_absolute() else MAIN / source


def slice_dir(w):
    return ROOT / "correction" / f"{w['ident']}-{w['a']}"


def slice_bounds(w):
    if w["ident"] in WHOLE:
        return 0, None
    return max(0, w["a"] - SLICE_MARGIN), w["b"] + SLICE_MARGIN


def slices():
    for w in plan():
        payload = json.loads((run_dir(w["ident"]) / f"{w['ident']}-stable.json").read_text(encoding="utf-8"))
        lo, hi = slice_bounds(w)
        hi = len(payload["segments"]) if hi is None else min(hi, len(payload["segments"]))
        # the loader takes an explicit id over the position; 1-based so id 0 (falsy) never falls back to the position
        segments = [{**{k: v for k, v in s.items() if k != "speaker"}, "id": i + 1}
                    for i, s in enumerate(payload["segments"]) if lo <= i < hi]
        metadata = {k: v for k, v in (payload.get("metadata") or {}).items() if k != "speaker"}
        folder = slice_dir(w)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{folder.name}-stable.json").write_text(
            json.dumps({"segments": segments, "metadata": metadata}, ensure_ascii=False), encoding="utf-8")
        print(f"{folder.name}: rows {lo}-{hi} ({len(segments)}), audio {source_audio(w['ident']).name}")


def correct(execute, preset=None, parallel=1):
    """The correction of every slice not corrected yet. `preset` agy (owner 2026-10-07): the free Gemini API kept
    answering 503 on the later slices, so they ran on agy with retrieval kept `local`; a slice whose research already
    finished reuses it (the reuse gates do not include the model)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(w):
        folder = slice_dir(w)
        command = [sys.executable, "-m", "finesub.llm.correction_translation", str(folder / f"{folder.name}-stable.json"),
                   "--audio", str(source_audio(w["ident"])),
                   "-o", str(folder / f"{folder.name}.srt"), "--task-artifact-dir", str(folder / "artifacts"),
                   # production names a task after its source, the episode title here: context the correction reads
                   "--prompt-dir", str(folder / "prompts"), "--task-id", source_audio(w["ident"]).stem]
        if w["ident"] not in WHOLE:
            command += ["--fast", "off"]
        if preset:
            command += ["--llm-preset", preset, "--retrieval", "local"]
        if execute:
            command.append("--execute")
        with open(folder / "correction.log", "a", encoding="utf-8") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"== {folder.name}: exit {code}", flush=True)
        return code

    todo = [w for w in plan() if not (execute and (slice_dir(w) / f"{slice_dir(w).name}-annotated.csv").exists())]
    with ThreadPoolExecutor(parallel) as pool:
        codes = list(pool.map(one, todo))
    if any(codes):
        raise SystemExit(f"{sum(1 for c in codes if c)} slice(s) failed")


def corrected_rows(w):
    """[(source rows, corrected text)] in output order, from the correction's `-annotated.csv` (positions are 1-based)."""
    folder = slice_dir(w)
    lines = (folder / f"{folder.name}-annotated.csv").read_text(encoding="utf-8").splitlines()
    header = lines[0].lstrip("# ").split("|")
    position, text = header.index("position"), header.index("corrected")
    out = []
    for line in lines[1:]:
        if line.strip():
            cols = line.split("|", len(header) - 1)
            out.append(([int(x) - 1 for x in cols[position].split(",")], cols[text].strip()))
    return out


_TIMELINES = {}


def timeline(ident):
    """The production timeline the round's labels come from (round 3's `load`)."""
    if ident not in _TIMELINES:
        from finesub.speech.diarization.segments import TrackSegment, segments_from_activity
        if ident.startswith("manosaba-"):
            tracks = json.loads((run_dir(ident) / f"{ident}-diar.json").read_text(encoding="utf-8"))["tracks"]
            _TIMELINES[ident] = [TrackSegment(track=int(t["track"]), start=float(t["start"]), end=float(t["end"])) for t in tracks]
        else:
            native = np.load(r3.pf.CACHE / f"{ident}.npz")["native"]
            _TIMELINES[ident] = [TrackSegment(track=t.track, start=t.start, end=t.end)
                                 for t in segments_from_activity(native.tolist(), frame_seconds=r3.pf.STEP)]
    return _TIMELINES[ident]


def merged_cell(m, ident, sources):
    from finesub.speech.diarization.assign import assign_all
    span = {"start": float(m.stable[sources[0]]["start"]), "end": float(m.stable[sources[-1]]["end"])}
    return lr.cell(assign_all([span], timeline(ident))[0])


def rows_for(w, source):
    """[(id or None, sources, start seconds, cell, text)] shown to the judge; id set on the rows it judges."""
    ident, a, b = w["ident"], w["a"], w["b"]
    m = load(ident)
    shown = set(r3.shown_rows(m, a, b))
    cells = [lr.cell(s) for s in m.speakers]
    if source == "raw":
        return [(f"q{i}" if a <= i < b else None, [i], float(m.stable[i]["start"]), cells[i], str(m.stable[i].get("text", "")).strip())
                for i in sorted(shown)]
    out = []
    for sources, text in corrected_rows(w):
        if not any(s in shown for s in sources):
            continue
        cell = cells[sources[0]] if len(sources) == 1 else merged_cell(m, ident, sources)
        if len(sources) == 1:
            assert cell == merged_cell(m, ident, sources), (ident, sources)  # the merged cell is production's rule
        judged = any(a <= s < b for s in sources)
        out.append((f"q{sources[0]}" if judged else None, sources, float(m.stable[sources[0]]["start"]), cell, text))
    return out


def truth_cast(m, a, b):
    return sorted({p for i in r3.shown_rows(m, a, b) if i in m.truth for p in m.truth[i]})


# --- prompts and calls -------------------------------------------------------------------------------

def system_prompt(modality, m):
    spec = r3.MATERIALS[m.kind]
    return lr.template("window_judge_r9_system.md").substitute(
        scene=spec["scene"], options=(HERE / "window_judge_r3_options_open.md").read_text(encoding="utf-8"),
        audio=(HERE / "window_judge_r9_audio_note.md").read_text(encoding="utf-8") if modality == "audio" else "",
        example="\n".join(r3.EXAMPLE["open"]), tracks=r3.TRACKS["open"])


def calls(arm, folder=None):
    """[(key, system, user, ids, wav or None, audio seconds, {id: sources})]."""
    modality, source = arm.split("-")
    out = []
    for w in plan():
        ident, a, b = w["ident"], w["a"], w["b"]
        m = load(ident)
        spec = r3.MATERIALS[m.kind]
        profile, voice, _ = r3.profiles(m, a, b)
        shown = r3.shown_rows(m, a, b)
        wav, seconds = None, 0.0
        if modality == "audio":
            import soundfile as sf
            wave = r3._wave(m)
            pieces, layout, cursor = [], [], 0.0
            gap = np.zeros(int(lj.GAP * lj.SR), dtype=np.float32)
            for track in sorted(voice):
                for i in voice[track]:
                    piece = wave[int(float(m.stable[i]["start"]) * lj.SR):int(float(m.stable[i]["end"]) * lj.SR)]
                    layout.append(f"| s{track} 的声音样本 | {cursor:.1f}–{cursor + len(piece) / lj.SR:.1f} 秒 |")
                    pieces.extend([piece, gap])
                    cursor += len(piece) / lj.SR + lj.GAP
            lo = max(0.0, float(m.stable[shown[0]]["start"]) - 0.5)  # the raw rows' span for both inputs
            hi = float(m.stable[shown[-1]]["end"]) + 0.5
            at = cursor
            pieces.append(wave[int(lo * lj.SR):int(hi * lj.SR)])
            layout.append(f"| 这一段字幕的录音 | {at:.1f}–{at + hi - lo:.1f} 秒 |")
            seconds = at + hi - lo
            target = folder or ROOT / "av"
            target.mkdir(parents=True, exist_ok=True)
            wav = target / f"{ident}-{a}.wav"
            sf.write(str(wav), np.concatenate(pieces), lj.SR, subtype="PCM_16")
            silent = [f"s{track}" for track in sorted(voice) if not voice[track]]
            profile += ("\n\n音频布局：\n| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
                        + (f"\n\n没有声音样本的轨：{'、'.join(silent)}" if silent else ""))
            when = lambda t: f"{at + t - lo:.1f}"  # noqa: E731
        else:
            when = lambda t: f"{t:.1f}"  # noqa: E731
        rows = rows_for(w, source)
        text = "\n".join((f"▶ {q} " if q else "") + f"{when(start)}|{cell}|{line}" for q, _, start, cell, line in rows)
        ids = {q: sources for q, sources, *_ in rows if q}
        cast = truth_cast(m, a, b)
        user = lr.template("window_judge_r3_user.md").substitute(
            title=spec["title"], cast="、".join(m.display(n) for n in cast) or "（无）", profiles=profile, rows=text, count=len(ids))
        out.append((f"{ident}-{a}", system_prompt(modality, m), user, list(ids), wav, seconds, ids))
    return out


def build():
    from finesub.llm.token_budget import default_token_counter
    counter = default_token_counter()
    count = counter.count_text
    total = 0
    for w in plan():
        m = load(w["ident"])
        raw, fixed = rows_for(w, "raw"), rows_for(w, "corrected")
        inside = range(w["a"], w["b"])
        kept = {s for _, sources, *_ in fixed for s in sources if s in inside}
        merged = sum(1 for q, sources, *_ in fixed if q and len(sources) > 1)
        print(f"  {w['group']} {w['ident']} {w['a']}-{w['b']}: judged raw {sum(1 for r in raw if r[0])}, corrected {sum(1 for r in fixed if r[0])} "
              f"(merged rows {merged}); sources dropped by the correction {len(inside) - len(kept)}; cast {truth_cast(m, w['a'], w['b'])}")
    for arm in ARMS:
        cs = calls(arm, ROOT / "build")
        text = sum(count(c[1]) + count(c[2]) for c in cs)
        audio = sum(c[5] for c in cs) * 32
        rows = sum(len(c[3]) for c in cs)
        total += text + audio
        print(f"{arm}: {len(cs)} calls, text input {text}, audio {audio:.0f} ({sum(c[5] for c in cs):.0f} s), judged rows {rows}")
        (ROOT / f"prompt-{arm}-example.md").write_text(cs[-1][1] + "\n\n=====\n\n" + cs[-1][2], encoding="utf-8")
    print(f"one pass, input ~{total:.0f} tokens (a floor: thinking, retries not counted); counted by {counter.source}")


def client_for(modality):
    """complete(system, user, wav) -> the call's result (content and usage), pinned to round 3's agy target."""
    from dataclasses import replace

    from finesub.llm.client import RoleClient
    from finesub.llm.media_upload import window_media_ref
    from finesub.llm.routing.config import LLMRole, role_config_for

    base = role_config_for("research", preset_id="agy", role=LLMRole.GENERAL_CAPABLE)
    chain = tuple(e for e in base.endpoint_chain if e.target_id == lj.TARGET)
    if not chain:
        raise SystemExit(f"{lj.TARGET} is not in the base chain")
    # ⚠ model_group_id="" is the pin: a group id makes the router ignore endpoint_chain
    # owner 2026-10-08: the judge thinks as hard as the correction (medium); rounds 9 and 10's pass 1 ran on the
    # research role's high (budget 39,321)
    client = RoleClient(role_configs={LLMRole.GENERAL_CAPABLE: replace(
        base, endpoint_chain=chain, test_endpoint=chain[0], model_group_id="", thinking_level="medium", thinking_budget=0)})

    def complete(system, user, wav=None):
        extra = {"file_ref": window_media_ref(wav, routes=client.router.routes)} if wav is not None else {}
        reply = client.complete(LLMRole.GENERAL_CAPABLE,
                                [{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
        landed = str(getattr(reply, "target_id", ""))
        if landed != lj.TARGET:
            raise SystemExit(f"pinned to {lj.TARGET} but landed on {landed!r}; stopped after one call")
        return reply
    return complete


def pass_dir(number, arm):
    return ROOT / f"pass{number}" / arm


def run(arm, number, shard, shards):
    complete = client_for(arm.split("-")[0])
    folder = pass_dir(number, arm)
    folder.mkdir(parents=True, exist_ok=True)
    for key, system, user, ids, wav, *_ in calls(arm, folder)[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        answer, text, attempts = None, "", []
        for attempt in range(4):
            try:
                reply = complete(system, user, wav)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            text = reply.content
            attempts.append(dict(reply.raw_response.get("usage") or {}))
            answer = lr.parse(text, "judge", "id", ids)
            if answer is not None:
                break
            print("malformed reply; retry", flush=True)
        # a call that gave up is kept apart, so a rerun tries again and scoring never reads it as "keep everything"
        target = path if answer else path.with_suffix(".failed.json")
        target.write_text(json.dumps({"answer": answer, "raw": text, "usage": attempts}, ensure_ascii=False, indent=1), encoding="utf-8")
        if answer:
            path.with_suffix(".failed.json").unlink(missing_ok=True)
        print(f"{arm} pass {number} {key}: {'ok' if answer else 'GAVE UP'} after {len(attempts)} replies", flush=True)


# --- scoring -----------------------------------------------------------------------------------------

def picks_of(arm, number):
    """{(ident, a, source row): pick}: a corrected row's answer spread over its sources inside the window."""
    out = {}
    sources_of = {f"{w['ident']}-{w['a']}": {q: s for q, s, *_ in rows_for(w, "corrected") if q} for w in plan()} \
        if arm.endswith("corrected") else {}
    for (ident, a, i), pick in r3.answers(arm, pass_dir(number, arm)).items():
        if arm.endswith("raw"):
            out[(ident, a, i)] = pick
            continue
        b = next(w["b"] for w in plan() if (w["ident"], w["a"]) == (ident, a))
        for s in sources_of[f"{ident}-{a}"][f"q{i}"]:
            if a <= s < b:
                out[(ident, a, s)] = pick
    return out


def backchannel(m, i):
    return len(re.sub(r"[\W_]", "", str(m.stable[i].get("text", "")))) <= BACKCHANNEL


def score():
    rng = np.random.default_rng(SEED)
    r3.OUT = ROOT / "build"
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    clusters = [base[k]["cluster"] for k in keys]
    group = {(w["ident"], w["a"]): w["group"] for w in plan()}
    fresh = np.array([group[k[:2]] in FRESH for k in keys])
    short = np.array([backchannel(load(k[0]), k[2]) for k in keys])
    kept = set()
    for w in plan():
        kept |= {(w["ident"], w["a"], s) for _, sources, *_ in rows_for(w, "corrected") for s in sources}
    common = np.array([k in kept for k in keys])

    def labelled_right(picks):
        right, missed = {}, {}
        for ident, a, b in windows():
            m = load(ident)
            final = ps.final_labels(r3, m, ident, a, b, picks)
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            found = ps.match(rows, final, m.truth)
            right.update({(ident, a, i): found.get(final[i]) in m.truth[i] for i in rows})
            missed.update({(ident, a, i): final[i] is not None and found.get(final[i]) in m.truth[i]
                           for i in range(a, b) if i in m.truth and m.leader(i) is None})
        return np.array([right[k] for k in keys], float), missed

    def interval(d, mask):
        groups = defaultdict(list)
        for j in np.flatnonzero(mask):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return d[mask].mean(), np.percentile(sims, 2.5), np.percentile(sims, 97.5)

    production, _ = labelled_right({})
    print(f"labelled rows with truth {len(keys)} (fresh {fresh.sum()}, back-channel {short.sum()}, dropped by the correction "
          f"{(~common).sum()}); production right {production.mean():.1%}")
    for w in plan():
        m = load(w["ident"])
        mixed = sum(1 for q, sources, *_ in rows_for(w, "corrected")
                    if q and len(sources) > 1 and len({frozenset(m.truth[s]) for s in sources if s in m.truth}) > 1)
        if mixed:
            print(f"  {w['ident']}-{w['a']}: {mixed} judged corrected rows merge sources whose true speakers differ")
    passes = sorted(int(p.name[4:]) for p in ROOT.glob("pass*") if p.name[4:].isdigit())
    for modality in MODALITIES:
        cols, missed, usage = {}, {}, {}
        for number in passes:
            for source in SOURCES:
                arm = f"{modality}-{source}"
                folder = pass_dir(number, arm)
                got = picks_of(arm, number) if folder.exists() else {}
                if not got:
                    continue
                cols[(number, source)], missed[(number, source)] = labelled_right(got)
                replies = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
                judged = sum(len(r["answer"] or {}) for r in replies)
                last = [r["usage"][-1] for r in replies if r.get("usage")]
                every = [u for r in replies for u in r.get("usage") or []]
                completion = sum(u.get("completion_tokens", 0) for u in last)
                thinking = sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in last)
                usage[(number, source)] = (f"{len(replies)} calls ({len(every) - len(last)} retries, {sum(u.get('completion_tokens', 0) for u in every)} "
                                           f"output tokens with them); accepted replies: output {completion}, thinking {thinking}, per judged row "
                                           f"{completion / max(judged, 1):.1f} / {thinking / max(judged, 1):.1f}, per source row "
                                           f"{completion / max(len(keys), 1):.2f}")
        for (number, source), col in cols.items():
            ok = [v for v in missed[(number, source)].values()]
            print(f"{modality}-{source} pass {number}: right {col.mean():.1%} (fresh {col[fresh].mean():.1%}); missed rows with truth "
                  f"{len(ok)}, right {np.mean(ok) if ok else 0:.1%}; {usage[(number, source)]}")
        pairs = [n for n in passes if (n, "raw") in cols and (n, "corrected") in cols]
        if not pairs:
            continue
        diffs = {n: 100 * (cols[(n, "corrected")] - cols[(n, "raw")]) for n in pairs}
        d = np.mean([diffs[n] for n in pairs], axis=0)
        main = fresh & ~short
        for name, mask in (("MAIN fresh, not back-channel", main), ("fresh, all rows", fresh), ("fresh, back-channel", fresh & short),
                           ("fresh, sources both inputs show (mechanism)", main & common), ("merged (in sample), not back-channel", ~fresh & ~short)):
            if mask.any():
                v, lo, hi = interval(d, mask)
                per = ", ".join(f"pass {n} {diffs[n][mask].mean():+.1f}" for n in pairs)
                print(f"  {modality} {name} ({mask.sum()} rows): corrected - raw {v:+.1f} [{lo:+.1f}, {hi:+.1f}] ({per})")
        v, lo, hi = interval(d, main)
        margin = DELTA if len(pairs) > 1 else DELTA - SINGLE_PASS
        worse = DELTA if len(pairs) > 1 else DELTA + SINGLE_PASS
        if lo > -margin and all(diffs[n][main].mean() > -DELTA for n in pairs):
            verdict = "corrected is not worse"
        elif hi < -worse:
            verdict = "corrected is worse"
        else:
            verdict = "uncertain" + (" (one pass: a second pass may settle it)" if len(pairs) == 1 else "")
        print(f"  VERDICT {modality} ({len(pairs)} pass{'es' if len(pairs) > 1 else ''}): {verdict}")


def dump(number=1):
    """Per window and modality, one table row per source row in the window: both inputs' text and cell, both replies'
    answer and reason, truth, and right / wrong under the score's partition rule; plus each call's usage. For reading
    the two inputs' replies side by side (`compare/<modality>/<window>.md`)."""
    r3.OUT = ROOT / "build"
    base = r3.outcome({})
    for modality in MODALITIES:
        replies, right = {}, {}
        for source in SOURCES:
            arm = f"{modality}-{source}"
            replies[source] = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in pass_dir(number, arm).glob("*.json")
                               if not p.name.endswith(".failed.json")}
            picks = picks_of(arm, number)
            for ident, a, b in windows():
                m = load(ident)
                final = ps.final_labels(r3, m, ident, a, b, picks)
                rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
                found = ps.match(rows, final, m.truth)
                right.update({(source, ident, a, i): found.get(final[i]) in m.truth[i] for i in rows})
        folder = ROOT / "compare" / modality
        folder.mkdir(parents=True, exist_ok=True)
        for w in plan():
            ident, a, b = w["ident"], w["a"], w["b"]
            key = f"{ident}-{a}"
            m = load(ident)
            fixed = {s: (q, sources, cell, text) for q, sources, _, cell, text in rows_for(w, "corrected") for s in sources}
            answer = {src: replies[src][key]["answer"] for src in SOURCES}
            usage = {src: replies[src][key]["usage"] for src in SOURCES}
            lines = [f"# {key}（{w['group']}，{modality}）", "",
                     "每行一个源段。raw = 纠错前那次调用，fix = 纠错后那次调用。`对?` 按窗内最有利的认人方式计（Y 对 / N 错 / - 不计分：没挂轨、没真值或是画像参考行）。",
                     "fix 一格里的 `[并 q…]` 表示这个源段被纠错并进了另一行，那一行的回答作用到它；`(删)` 表示纠错删掉了它，保留声学标签。", "",
                     "用量（每次尝试一项，最后一项是被采用的回答）：" + "；".join(
                         f"{src}: " + ", ".join(f"输入 {u.get('prompt_tokens')} 输出 {u.get('completion_tokens')} 思考 "
                                                f"{(u.get('completion_tokens_details') or {}).get('reasoning_tokens')}" for u in usage[src])
                         for src in SOURCES), "",
                     "| 行 | 真值 | 声学轨 | raw 文字 | raw 回答 | raw 理由 | raw 对? | fix 文字 | fix 回答 | fix 理由 | fix 对? |",
                     "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            clean = lambda s: str(s).replace("|", "｜").replace("\n", " ")  # noqa: E731
            for i in range(a, b):
                raw = answer["raw"].get(f"q{i}") or {}
                if i in fixed:
                    q, sources, cell, text = fixed[i]
                    got = answer["corrected"].get(q) or {}
                    head = "" if q == f"q{i}" else f"[并 {q}] "
                    fix_text, fix_pick, fix_why = head + clean(text) + (f"（轨 {cell}）" if cell != lr.cell(m.speakers[i]) else ""), got.get("track", ""), got.get("why", "")
                else:
                    fix_text, fix_pick, fix_why = "(删)", "", ""
                mark = lambda src: {True: "Y", False: "N"}.get(right.get((src, ident, a, i)), "-")  # noqa: E731
                truth = "、".join(sorted(m.truth.get(i, ()))) or "-"
                lines.append(f"| q{i} | {truth} | {lr.cell(m.speakers[i])} | {clean(m.stable[i].get('text', '')).strip()} | {raw.get('track', '')} | "
                             f"{clean(raw.get('why', ''))} | {mark('raw')} | {fix_text} | {fix_pick} | {clean(fix_why)} | {mark('corrected')} |")
            (folder / f"{key}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{modality}: {len(plan())} windows -> {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("slices", "correct", "build", "run", "score", "dump"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preset", help="correct: the LLM preset (agy for the slices the API kept refusing)")
    parser.add_argument("--parallel", type=int, default=1, help="correct: slices at once")
    parser.add_argument("--arm", choices=ARMS, default="text-raw")
    parser.add_argument("--pass", dest="number", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "correct":
        correct(args.execute, args.preset, args.parallel)
    elif args.command == "run":
        run(args.arm, args.number, args.shard, args.shards)
    else:
        {"slices": slices, "build": build, "score": score, "dump": dump}[args.command]()


if __name__ == "__main__":
    main()
