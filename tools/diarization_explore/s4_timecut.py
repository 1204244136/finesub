"""Plan 5h.17: cut by time -- each track's own stretch decoded alone -- and its variants.

Window W = span +-1 s over the 5h.4 spans (real + decoys). Per track k active in W:
  full/k   every frame of k in W (runs padded 0.1 s, joined by 0.2 s silence)     -> T1
  fullA/k  anchor (k's solo speech >=30 s outside W, <=4 s) + 0.6 s + full/k       -> T2
  excl/k   only the frames where k is the *only* track sounding                    -> T6 (single label)
  ovl      the frames where two or more tracks sound, decoded once                 -> T6 (multi label)
  whole    all of W, uncut (plan 5o.4)                                             -> T0 (content only)
T4 needs no decode: production's characters in W regrouped by which tracks sound at their time.
Plan 5o.4 re-scores all of these with `s4_score2.py`; `--score` here is the 5h.17 scorer, kept as it was.

  --whisper        decode with production's RefinedWhisperModel (asr env, GPU)
  --qwen           decode with production's QwenReferee (qwen-asr env, GPU)
  --score          5h.17's criteria against A0-W (production lines in the same windows)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import s4_headroom as ref
import s4_live as live
import s4_llm as llm
import s4_projection as proj

W_PAD, EDGE, JOIN, MIN_TRACK, ANCHOR_FAR, ANCHOR_TOTAL, GAP = 1.0, 0.1, 0.2, 0.3, 30.0, 4.0, 0.6
SR, FRAME = proj.SR, proj.FRAME


def runs(mask):
    out, start = [], None
    for index, flag in enumerate(np.append(mask, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            out.append((start, index))
            start = None
    return out


def seconds(frame_runs, offset):
    spans = [(max(0.0, (offset + a) * FRAME - EDGE), (offset + b) * FRAME + EDGE) for a, b in frame_runs]
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def clip(audio, spans):
    gap = np.zeros(int(JOIN * SR), dtype=np.float32)
    pieces = []
    for s, e in spans:
        pieces += [audio[int(s * SR): int(e * SR)], gap]
    return np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)


def window(span):
    return max(0.0, span["start"] - W_PAD), span["end"] + W_PAD


def far_anchor(audio, active, k, w0, w1):
    candidates = [r for r in live.single_runs(active, k) if r[1] < w0 - ANCHOR_FAR or r[0] > w1 + ANCHOR_FAR]
    candidates.sort(key=lambda r: min(abs(r[0] - w1), abs(r[1] - w0)))
    pieces, total = [], 0.0
    for s, e in candidates:
        take = min(e - s, ANCHOR_TOTAL - total)
        if take <= 0:
            break
        pieces.append(audio[int(s * SR): int((s + take) * SR)])
        total += take
    return np.concatenate(pieces) if pieces else None


def plan(session):
    """(job_id, audio, anchor_seconds) for every decode this session needs; plus the layout."""
    import soundfile

    audio, rate = soundfile.read(str(proj.ALI / session / f"{session}-vocal.ogg"), dtype="float32", always_2d=True)
    assert rate == SR
    audio = audio.mean(axis=1)
    tracks = proj.timeline(session)
    active = proj.activity(tracks, int(len(audio) / SR / FRAME) + 2)
    jobs, layout = [], {}
    for chunk in llm.chunks(session):
        for span in chunk["spans"]:
            sid = f"{span['start']:.2f}-{span['end']:.2f}"
            w0, w1 = window(span)
            f0, f1 = int(w0 / FRAME), int(np.ceil(w1 / FRAME))
            sub = active[f0:f1]
            on = [k for k in range(sub.shape[1]) if sub[:, k].sum() * FRAME >= MIN_TRACK]
            multi = sub.sum(axis=1) >= 2
            layout[sid] = {"tracks": on, "ovl_tracks": sorted(on, key=lambda k: -int((sub[:, k] & multi).sum()))}
            for k in on:
                full = seconds(runs(sub[:, k]), f0)
                jobs.append((f"{sid}/full/{k}", clip(audio, full), 0.0))
                excl = seconds(runs(sub[:, k] & ~multi), f0)
                if excl:
                    jobs.append((f"{sid}/excl/{k}", clip(audio, excl), 0.0))
                anchor = far_anchor(audio, active, k, w0, w1)
                if anchor is not None:
                    jobs.append((f"{sid}/A/{k}", anchor, 0.0))
                    body = clip(audio, full)
                    jobs.append((f"{sid}/fullA/{k}", np.concatenate([anchor, np.zeros(int(GAP * SR), np.float32), body]), len(anchor) / SR))
            overlap = seconds(runs(multi), f0)
            if overlap:
                jobs.append((f"{sid}/ovl/x", clip(audio, overlap), 0.0))
            # T0 (plan 5o.4): the whole window, uncut -- what re-decoding alone buys
            jobs.append((f"{sid}/whole/x", audio[int(w0 * SR): int(w1 * SR)], 0.0))
            # T7 (5h.19): exclusive stretches >= 1 s alone; shorter ones join the uncertain part
            uncertain = multi.copy()
            long_excl = {}
            for k in on:
                alone = sub[:, k] & ~multi
                kept = []
                for a, b in _bridged(runs(alone), int(0.2 / FRAME)):
                    if (b - a) * FRAME >= MIN_EXCL:
                        kept.append((a, b))
                    else:
                        uncertain[a:b] = True
                long_excl[k] = kept
                if kept:
                    jobs.append((f"{sid}/excl1/{k}", clip(audio, seconds(kept, f0)), 0.0))
            layout[sid]["t7_tracks"] = sorted(on, key=lambda k: -int((sub[:, k] & uncertain).sum()))
            unsure = seconds(runs(uncertain), f0)
            if unsure:
                jobs.append((f"{sid}/ovl1/x", clip(audio, unsure), 0.0))
    return jobs, layout


MIN_EXCL = 1.0


def _write_atomic(path, text):
    temporary = path.with_name(f".{path.name}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _bridged(frame_runs, gap):
    out = []
    for a, b in frame_runs:
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def run_whisper():
    from finesub.speech.recognition.fw_refine_backend import RefinedWhisperModel
    from finesub.speech.recognition.transcribe import REFINE_SEC

    model = RefinedWhisperModel("large-v3-turbo", device="cuda", compute_type="float16", refine_sec=REFINE_SEC)
    for session in ref.SESSIONS:
        path = proj.ALI / session / "timecut_whisper.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        jobs, layout = plan(session)
        for job_id, audio, anchor_seconds in jobs:
            if job_id in done or not len(audio):
                continue
            words = proj.decode(model, audio, 0.0)
            if "/fullA/" in job_id:
                cut = anchor_seconds + GAP - 0.15
                done[job_id] = "".join(w["word"] for w in words if w["start"] >= cut)
            elif "/A/" in job_id:
                done[job_id] = ""  # cut by timestamp already
            else:
                done[job_id] = "".join(w["word"] for w in words)
        _write_atomic(path, json.dumps(done, ensure_ascii=False))
        _write_atomic(proj.ALI / session / "timecut_layout.json", json.dumps(layout))
        print(session, "whisper jobs", len(done), flush=True)


def run_qwen(batch=48):
    from finesub.speech.verification.qwen_referee import QwenReferee

    referee = QwenReferee(device="cuda")
    try:
        for session in ref.SESSIONS:
            path = proj.ALI / session / "timecut_qwen.json"
            done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            jobs, layout = plan(session)
            todo = [(j, a) for j, a, _ in jobs if j not in done and len(a)]
            for start in range(0, len(todo), batch):
                part = todo[start: start + batch]
                for (job_id, _), (text, _) in zip(part, referee.transcribe_batch([a for _, a in part])):
                    done[job_id] = text
                _write_atomic(path, json.dumps(done, ensure_ascii=False))
            _write_atomic(proj.ALI / session / "timecut_layout.json", json.dumps(layout))
            print(session, "qwen jobs", len(done), flush=True)
    finally:
        referee.close()


# ---- scoring ---------------------------------------------------------------------------

def variant_lines(session, results, layout, variant, *, fallback=False, deleak=True, placebo=False):
    """(span start, end) -> [(tracks, normalized text, None)] for one T variant.

    The defaults reproduce 5h.17 exactly (they are also what 5h.18 showed the model).
    Plan 5o.4 scores with `fallback=True` (a track with no far anchor keeps its T1
    line instead of vanishing) and `deleak=False` (an anchor >=30 s away has nothing
    to leak; de-leaking only deletes words the two share). `placebo=True` puts T1
    through the same de-leak against the anchor's text, to see what it alone does.
    """
    import s4_breakdown as b

    out = {}
    for span in session.spans:
        sid = f"{span['start']:.2f}-{span['end']:.2f}"
        info = layout.get(sid, {"tracks": [], "ovl_tracks": []})
        lines = []
        if variant == "T1":
            for k in info["tracks"]:
                text = b.normalize(results.get(f"{sid}/full/{k}") or "")
                if placebo:
                    text = b.deleak(text, b.normalize(results.get(f"{sid}/A/{k}") or ""))
                lines.append(((k,), text))
        elif variant == "T2":
            for k in info["tracks"]:
                text, anchor = results.get(f"{sid}/fullA/{k}"), results.get(f"{sid}/A/{k}") or ""
                if text is not None:
                    text = b.normalize(live.strip_anchor(text, anchor))
                    lines.append(((k,), b.deleak(text, b.normalize(anchor)) if deleak else text))
                elif fallback:
                    lines.append(((k,), b.normalize(results.get(f"{sid}/full/{k}") or "")))
        elif variant == "T0":
            lines = [((), b.normalize(results.get(f"{sid}/whole/x") or ""))]
        elif variant == "T6":
            lines = [((k,), b.normalize(results.get(f"{sid}/excl/{k}") or "")) for k in info["tracks"]]
            if info["ovl_tracks"]:
                lines.append((tuple(info["ovl_tracks"]), b.normalize(results.get(f"{sid}/ovl/x") or "")))
        elif variant == "T7":
            lines = [((k,), b.normalize(results.get(f"{sid}/excl1/{k}") or "")) for k in info["tracks"]]
            if info.get("t7_tracks"):
                lines.append((tuple(info["t7_tracks"]), b.normalize(results.get(f"{sid}/ovl1/x") or "")))
        out[(span["start"], span["end"])] = [(tracks, text, None) for tracks, text in lines if text]
    return out


def word_lines_merged(session, active):
    """Post hoc: T4 with one line per track set per window, not one per consecutive run --
    to see how much of T4's content loss is the scorer failing on fragments."""
    out = {}
    for key, lines in word_lines(session, active).items():
        merged = {}
        for tracks, text, _ in lines:
            merged[tracks] = merged.get(tracks, "") + text
        out[key] = [(tracks, text, None) for tracks, text in merged.items()]
    return out


def word_lines(session, active):
    """T4: production's characters in W, regrouped by the tracks sounding at their time."""
    out = {}
    for span in session.spans:
        w0, w1 = window(span)
        groups = []
        for t, c, seg in session.chars:
            if not w0 <= t <= w1:
                continue
            frame = int(t / FRAME)
            on = tuple(k for k in range(active.shape[1]) if frame < len(active) and active[frame, k]) or tuple(session.labels[seg])
            if groups and groups[-1][0] == on:
                groups[-1][1].append(c)
            else:
                groups.append((on, [c]))
        # A character no track sounds under, on a line production left unlabelled,
        # stays -- unattributed, not deleted (plan 5o.1 M3: 47 AMI words went missing).
        out[(span["start"], span["end"])] = [(on, "".join(chars), None) for on, chars in groups]
    return out


def production_lines(session):
    """A0-W: production's lines inside W, each with its production labels."""
    out = {}
    for span in session.spans:
        w0, w1 = window(span)
        by_line = {}
        for t, c, seg in session.chars:
            if w0 <= t <= w1:
                by_line[seg] = by_line.get(seg, "") + c
        out[(span["start"], span["end"])] = [(tuple(session.labels[seg]), text, None) for seg, text in by_line.items()]
    return out


def score():
    import s4_breakdown as b
    import s4_resegment as rsg

    loaded = []
    for name in ref.SESSIONS:
        session = b.Session(name)
        span_align, _ = b.collect(session)
        tracks = proj.timeline(name)
        active = proj.activity(tracks, int(max(t["end"] for t in tracks) / FRAME) + 2)
        layout = json.loads((proj.ALI / name / "timecut_layout.json").read_text(encoding="utf-8"))
        results = {model: json.loads((proj.ALI / name / f"timecut_{model}.json").read_text(encoding="utf-8"))
                   for model in ("whisper", "qwen") if (proj.ALI / name / f"timecut_{model}.json").exists()}
        loaded.append((session, span_align, active, layout, results))
    print("REFERENCE SIDE, windows W = span +-1 s")
    base = rsg.score_arm(None, "A0-W production", [(s, a, production_lines(s)) for s, a, *_ in loaded])
    rsg.score_arm(None, "A1 (+-0.3 s text)", [(s, a, rsg.arm_lines(s, a, "A1")) for s, a, *_ in loaded])
    arms = {"T4 words": [(s, a, word_lines(s, act)) for s, a, act, _, _ in loaded],
            "T4 merged (post hoc)": [(s, a, word_lines_merged(s, act)) for s, a, act, _, _ in loaded]}
    for model, tag in (("whisper", "W"), ("qwen", "Q")):
        if all(model in r for *_, r in loaded):
            for variant in ("T1", "T2", "T6", "T7"):
                arms[f"{variant}-{tag}"] = [(s, a, variant_lines(s, r[model], lay, variant)) for s, a, _, lay, r in loaded]
    verdicts = {}
    for label, parts in arms.items():
        got = rsg.score_arm(None, label, parts)
        checks = [got["heard"] >= base["heard"] - 0.02, got["strict"] >= base["strict"] + 0.20, got["wrong"] <= 0.10, got["confab"] <= 0.10]
        verdicts[label] = got
        print(f"    content {'PASS' if checks[0] else 'FAIL'} | strict {'PASS' if checks[1] else 'FAIL'} | wrong {'PASS' if checks[2] else 'FAIL'} "
              f"| decoy {'PASS' if checks[3] else 'FAIL'} => {'可行' if all(checks) else '不做'}")
    return base, verdicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--score", action="store_true")
    arguments = parser.parse_args()
    if arguments.whisper:
        run_whisper()
    if arguments.qwen:
        run_qwen()
    if arguments.score:
        score()


if __name__ == "__main__":
    main()
