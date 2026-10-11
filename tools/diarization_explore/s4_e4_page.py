"""Plan 5o.9 E4: build the owner's blind annotation page for nepolabo (Japanese live).

40 units, four blind versions each (A0-W / T4 / T2-Q / T2-W), shuffled per unit and labelled
甲 / 乙 / 丙 / 丁. The owner picks the version that takes the least work to turn into the ideal
form (one person per line, only that person's characters, characters right) and the one that
takes the most; "all much the same" and "none usable" are also answers.

nepolabo has no ground truth, so `s4_breakdown.Session` cannot be built for it (it reads a
TextGrid). `Stub` carries the three things the arm builders actually use -- spans, characters
with times, and production's per-line labels -- and nothing that needs truth.

  DIAR_S4_CORPUS=ja python s4_e4_page.py --build     # clips + units.json into tmp/rescore-5o4/e4-page/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import s4_breakdown as b
import s4_headroom as ref
import s4_llm as llm
import s4_projection as proj
import s4_timecut as tc

OUT = Path(__file__).resolve().parents[2] / "tmp" / "rescore-5o4" / "e4-page"
SEED = 20260912
UNITS = 40
REF_MAX = 3.0  #: seconds of single-speaker reference audio per track
GAP = 0.5  #: silence between the window clip and each reference clip
ARMS = ("A0-W", "T4", "T2-Q", "T2-W")
BLIND = ("甲", "乙", "丙", "丁")


class Stub:
    """What `production_lines` / `word_lines` / `variant_lines` read off a session, minus truth."""

    def __init__(self, name):
        from finesub.speech.diarization.stage import labels_for, read_timeline

        self.name = name
        payload = json.loads((proj.ALI / name / f"{name}-stable.json").read_text(encoding="utf-8"))
        timeline = read_timeline(proj.ALI / name / f"{name}-diar.json")
        segments = payload["segments"]
        self.labels = [tuple(s.extras.get("tracks", ())) for s in labels_for(segments, timeline)]
        ordered = []
        for index, segment in enumerate(segments):
            for word in segment.get("words") or ():
                for c in b.normalize(word["word"]):
                    ordered.append(((word["start"] + word["end"]) / 2, len(ordered), c, index))
        self.chars = [(t, c, seg) for t, _, c, seg in sorted(ordered)]
        self.tracks = proj.timeline(name)
        self.spans = [s for c in llm.chunks(name) for s in c["spans"]]


def arms_for(session, active):
    """The four versions, each as {(span start, end): [(tracks, text)]}."""
    read = lambda f: json.loads((proj.ALI / session.name / f).read_text(encoding="utf-8"))  # noqa: E731
    layout, qwen, whisper = read("timecut_layout.json"), read("timecut_qwen.json"), read("timecut_whisper.json")
    strip = lambda d: {k: [(t, x) for t, x, _ in v] for k, v in d.items()}  # noqa: E731
    return {
        "A0-W": strip(tc.production_lines(session)),
        "T4": strip(tc.word_lines(session, active)),
        # The 5o.4 settings: a track with no far anchor keeps its T1 line, and no de-leak.
        "T2-Q": strip(tc.variant_lines(session, qwen, layout, "T2", fallback=True, deleak=False)),
        "T2-W": strip(tc.variant_lines(session, whisper, layout, "T2", fallback=True, deleak=False)),
    }, layout


LINE_PAD = 0.2   #: character times are word midpoints, so a group's span is narrower than its audio
LINE_MIN = 0.45  #: a one-character line still has to be audible


def line_times(session, active, layout, span, arm, lines):
    """(t0, t1) per line of one arm, in session seconds -- what "play this line" plays.

    ⚠ Deliberately uniform across arms: every line plays its **contiguous** span. T2's line is
    really the stitched sounding frames of one track, and playing that chopped audio would be an
    audible tell for two of the four versions -- the blind would not survive it (5o.9 E4).
    """
    w0, w1 = tc.window(span)
    inside = [(t, c, seg) for t, c, seg in session.chars if w0 <= t <= w1]
    spans = []
    if arm == "A0-W":  # one line per production segment, in first-character order
        order, reach = [], {}
        for t, _, seg in inside:
            if seg not in reach:
                order.append(seg)
                reach[seg] = [t, t]
            reach[seg][1] = t
        spans = [tuple(reach[seg]) for seg in order]
    elif arm == "T4":  # one line per run of consecutive characters sharing a track set
        groups = []
        for t, _, seg in inside:
            frame = int(t / tc.FRAME)
            on = tuple(k for k in range(active.shape[1]) if frame < len(active) and active[frame, k]) or tuple(session.labels[seg])
            if groups and groups[-1][0] == on:
                groups[-1][2] = t
            else:
                groups.append([on, t, t])
        spans = [(a, b) for _, a, b in groups]
    else:  # T2: one line per track, so the span is where that track sounds inside the window.
        # Keyed by track, not by position: `variant_lines` drops a track whose text came back
        # empty, so the line list and the track list are not the same length.
        sid = f"{span['start']:.2f}-{span['end']:.2f}"
        f0, f1 = int(w0 / tc.FRAME), int(np.ceil(w1 / tc.FRAME))
        sub = active[f0:f1]
        by_track = {}
        for k in layout[sid]["tracks"]:
            runs = tc.runs(sub[:, k])
            by_track[k] = ((f0 + runs[0][0]) * tc.FRAME, (f0 + runs[-1][1]) * tc.FRAME) if runs else (w0, w1)
        spans = [by_track.get(tracks[0] if tracks else -1, (w0, w1)) for tracks, _ in lines]
    out = []
    for index in range(len(lines)):
        a, b = spans[index] if index < len(spans) else (w0, w1)
        a, b = max(w0, a - LINE_PAD), min(w1, b + LINE_PAD)
        if b - a < LINE_MIN:
            middle = (a + b) / 2
            a, b = max(w0, middle - LINE_MIN / 2), min(w1, middle + LINE_MIN / 2)
        out.append((round(a - w0, 2), round(b - w0, 2)))
    return out


def timed(session, active, layout, span, arm, lines):
    """One arm's non-empty lines for one unit, each with the clip offsets that play it."""
    kept = [(tuple(t), x) for t, x in lines if x]
    at = line_times(session, active, layout, span, arm, kept)
    return [{"tracks": list(t), "text": x, "t0": a, "t1": b} for (t, x), (a, b) in zip(kept, at)]


def reference_clip(audio, active, k, w0, w1):
    """<=3 s where track k is the only one sounding, nearest to the window but outside it."""
    alone = active[:, k] & (active.sum(axis=1) == 1)
    best = None
    for a, c in tc.runs(alone):
        s, e = a * tc.FRAME, c * tc.FRAME
        if e <= w0 or s >= w1:  # outside the window, so it cannot be the unit's own audio
            if e - s < 0.8:
                continue
            distance = w0 - e if e <= w0 else s - w1
            if best is None or distance < best[0]:
                best = (distance, s, min(e, s + REF_MAX))
    if best is None:
        return None
    _, s, e = best
    return audio[int(s * tc.SR): int(e * tc.SR)]


def build():
    import soundfile

    name = ref.SESSIONS[0]
    session = Stub(name)
    audio, rate = soundfile.read(str(proj.ALI / name / f"{name}-vocal.ogg"), dtype="float32", always_2d=True)
    assert rate == tc.SR
    audio = audio.mean(axis=1)
    active = proj.activity(session.tracks, int(len(audio) / tc.SR / tc.FRAME) + 2)
    versions, layout = arms_for(session, active)

    eligible = [s for s in session.spans if s["kind"] == "real"
                and len(layout.get(f"{s['start']:.2f}-{s['end']:.2f}", {}).get("tracks", [])) >= 2]
    if len(eligible) < UNITS:
        raise SystemExit(f"only {len(eligible)} eligible units, need {UNITS}")
    # Deterministic, evenly spaced down the session -- not random, so the units cover the whole
    # stream rather than clumping (the pre-registration says 按序号均匀取).
    picked = [eligible[round(i * (len(eligible) - 1) / (UNITS - 1))] for i in range(UNITS)]

    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    units = []
    for index, span in enumerate(picked):
        key, sid = (span["start"], span["end"]), f"{span['start']:.2f}-{span['end']:.2f}"
        w0, w1 = tc.window(span)
        pieces, marks = [audio[int(w0 * tc.SR): int(w1 * tc.SR)]], []
        offset = len(pieces[0]) / tc.SR
        for k in layout[sid]["tracks"]:
            clip = reference_clip(audio, active, k, w0, w1)
            if clip is None or not len(clip):
                continue
            pieces.append(np.zeros(int(GAP * tc.SR), np.float32))
            offset += GAP
            marks.append({"track": k, "start": round(offset, 2), "end": round(offset + len(clip) / tc.SR, 2)})
            pieces.append(clip)
            offset += len(clip) / tc.SR
        soundfile.write(str(OUT / f"u{index:02d}.ogg"), np.concatenate(pieces), tc.SR, format="OGG", subtype="VORBIS")

        order = list(ARMS)
        rng.shuffle(order)
        units.append({
            "id": index,
            "span": [round(span["start"], 2), round(span["end"], 2)],
            "window": [round(w0, 2), round(w1, 2)],
            "audio": f"u{index:02d}.ogg",
            "window_end": round(len(pieces[0]) / tc.SR, 2),
            "refs": marks,
            # blind label -> lines; the arm each label stands for is kept out of the page
            "versions": {BLIND[i]: timed(session, active, layout, span, arm, versions[arm][key])
                         for i, arm in enumerate(order)},
            "key": {BLIND[i]: arm for i, arm in enumerate(order)},
        })
    (OUT / "units.json").write_text(json.dumps(units, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum((OUT / u["audio"]).stat().st_size for u in units)
    print(f"{len(units)} units, {len(eligible)} eligible, audio {total / 1e6:.1f} MB -> {OUT}")
    print("lines per version:", {arm: sum(len(u["versions"][lbl]) for u in units for lbl, a in u["key"].items() if a == arm) for arm in ARMS})


def page():
    """Inline the units into the page template, minus the blind key -- the page must not carry it."""
    units = json.loads((OUT / "units.json").read_text(encoding="utf-8"))
    blind = [{k: v for k, v in unit.items() if k != "key"} for unit in units]
    template = (Path(__file__).resolve().parent / "e4_page.html").read_text(encoding="utf-8")
    payload = json.dumps(blind, ensure_ascii=False)
    assert '"key"' not in payload, "the blind key leaked into the page"
    body = template.replace("/*UNITS*/", payload)
    (OUT / "index.html").write_text(body, encoding="utf-8")
    print(f"{OUT / 'index.html'}  {len(body) / 1000:.0f} kB, {len(blind)} units, blind key withheld")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--page", action="store_true", help="render index.html from units.json")
    arguments = parser.parse_args()
    if arguments.build:
        build()
    if arguments.page:
        page()
    if not (arguments.build or arguments.page):
        parser.print_help()


if __name__ == "__main__":
    main()
