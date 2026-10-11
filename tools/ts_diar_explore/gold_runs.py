"""The segmentation gold set (`tools/segmentation_gold/labels/`) as a second segmentation ruler.

Every labelled window becomes one span (overlapping windows of a clip merged), padded by `PAD` seconds of
context on each side so no model sees a cold start at a labelled position. Audio and VAD come from the
production baseline `out/ts-baseline-20260831-hq/<clip>/` (vocal track, `-vad.json`), which is also the
production arm (`-stable.json`).

    python -m tools.ts_diar_explore.gold_runs spans                  # list spans
    python -m tools.ts_diar_explore.gold_runs cut                    # cut the span clips
    python -m tools.ts_diar_explore.gold_runs call mai2|g35t         # one API call per span, cached
    (MOSS, not run yet: moss_run over the span clips as `<clip>.wav@<span start>` so its times come out absolute)
    python -m tools.ts_diar_explore.gold_runs lines                  # seg arms + per-clip merged line files
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .common import MAIN, TMP, WT, clip, save

LABELS = WT / "tools" / "segmentation_gold" / "labels"
BASE = MAIN / "out" / "ts-baseline-20260831-hq"
GOLD = TMP / "gold"
PAD = 20.0


def spans() -> list[tuple[str, float, float]]:
    windows = defaultdict(list)
    for path in sorted(LABELS.glob("*.json")):
        name, a, b = path.stem.rsplit("-", 2)
        windows[name].append((float(a), float(b)))
    out = []
    for name, ws in windows.items():
        ws.sort()
        merged = [list(ws[0])]
        for a, b in ws[1:]:
            if a <= merged[-1][1] + 2 * PAD:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out += [(name, max(0.0, a - PAD), b + PAD) for a, b in merged]
    return out


def span_id(name: str, a: float, b: float) -> str:
    return f"{name}-{a:.0f}-{b:.0f}"


def span_audio(name: str, a: float, b: float) -> Path:
    return GOLD / "clips" / f"{span_id(name, a, b)}.wav"


def cmd_cut() -> None:
    for name, a, b in spans():
        print(clip(BASE / name / f"{name}-vocal.ogg", span_audio(name, a, b), a, b - a))


def cmd_call(model: str) -> None:
    extra = ["--diarize"] if model == "mai2" else []
    for name, a, b in spans():
        out = GOLD / model / f"{span_id(name, a, b)}.json"
        if out.exists():
            continue
        subprocess.run([sys.executable, "-m", f"tools.ts_diar_explore.{model}", str(span_audio(name, a, b)),
                        str(out), "--offset", str(a), *extra], check=True, encoding="utf-8", errors="replace")


#: seg arm -> seg_arm.py arguments. The `+vadend` / `+prodseam` variants borrow production's VAD for word ends
#: and production's Whisper segment starts for DP seams: the model as a text source inside production's split.
SEG_ARMS = {"punct": ["--arm", "punct"], "sbd": ["--arm", "sbd"], "strip-sbd": ["--arm", "strip-sbd"],
            "sbd+vadend": ["--arm", "sbd", "--ends", "vad"],
            "sbd+prodseam": ["--arm", "sbd", "--seams", "prod"],
            "sbd+vadend+prodseam": ["--arm", "sbd", "--ends", "vad", "--seams", "prod"]}


def cmd_lines() -> None:
    """Per model and arm: one line file per clip (spans concatenated), ready for `gold.py score --clip`."""
    by_clip = defaultdict(list)
    for name, a, b in spans():
        by_clip[name].append((a, b))
    for model in ("mai2", "g35t", "moss"):
        for arm in ("native", *SEG_ARMS):
            for name, sp in by_clip.items():
                lines = []
                for a, b in sp:
                    src = GOLD / model / f"{span_id(name, a, b)}.json"
                    if not src.exists():
                        break
                    if arm == "native":
                        lines += json.loads(src.read_text(encoding="utf-8"))["segments"]
                        continue
                    dst = GOLD / "lines" / model / arm / f"{span_id(name, a, b)}.json"
                    if not dst.exists():
                        subprocess.run([sys.executable, "-m", "tools.ts_diar_explore.seg_arm", str(src),
                                        str(BASE / name / f"{name}-vad.json"), str(dst), *SEG_ARMS[arm],
                                        "--aligned", str(BASE / name / f"{name}-aligned.json")],
                                       check=True, encoding="utf-8", errors="replace")
                    lines += json.loads(dst.read_text(encoding="utf-8"))["segments"]
                else:
                    save({"segments": sorted(lines, key=lambda s: s["start"])},
                         GOLD / "merged" / model / arm / f"{name}.json")


def _arm_files() -> dict[str, dict[str, Path]]:
    arms = {"prod": {name: BASE / name / f"{name}-stable.json" for name in {s[0] for s in spans()}}}
    for model_dir in sorted((GOLD / "merged").glob("*")):
        for arm_dir in sorted(model_dir.glob("*")):
            arms[f"{model_dir.name}/{arm_dir.name}"] = {p.stem: p for p in arm_dir.glob("*.json")}
    return arms


def cmd_score() -> None:
    """Pooled over every clip: time-anchored (tau 0.3 / 0.5) and by substrate word index (decision only)."""
    sys.path.insert(0, str(WT))
    from tools.segmentation_gold import gold
    clips = {s[0] for s in spans()}
    print(f"{'arm':<22} {'must@0.3':>9} {'viol/100':>8} {'must@0.5':>9} {'viol/100':>8} "
          f"{'|dt| med':>8} {'idx must':>9} {'idx viol':>8} {'cuts':>5}")
    for arm, files in _arm_files().items():
        if set(files) != clips:
            print(f"{arm:<22} incomplete ({len(files)}/{len(clips)} clips)")
            continue
        tot = {0.3: defaultdict(float), 0.5: defaultdict(float)}
        idx, offs = defaultdict(float), []
        for name in sorted(clips):
            golds = gold.load_golds(LABELS, name)
            cues = gold.load_cues(files[name])
            cuts = gold._cuts(cues)
            for tau in tot:
                for k, v in gold.evaluate(golds, cuts, tau).items():
                    if k not in ("recall", "viol_per100"):
                        tot[tau][k] += v
            words = gold.load_words(MAIN.parent / golds[0]["substrate_path"].removeprefix("../"))
            for k, v in gold.evaluate_by_index(golds, gold.cuts_to_indices(cues, words)).items():
                if k not in ("recall", "viol_per100"):
                    idx[k] += v
            for g in golds:
                for it in g["items"]:
                    near = [m - it["t"] for _, _, m in cuts if it["label"] == "must" and abs(m - it["t"]) <= 1.0]
                    if near:
                        offs.append(abs(min(near, key=abs)))

        def rate(r):
            judged = r["in_window"] - r["skipped"]
            return r["must_hit"] / r["must"], 100 * (r["declared"] + r["presumed"]) / max(1, judged)

        (m3, v3), (m5, v5), (mi, vi) = rate(tot[0.3]), rate(tot[0.5]), rate(idx)
        med = sorted(offs)[len(offs) // 2] if offs else float("nan")
        print(f"{arm:<22} {m3:9.1%} {v3:8.1f} {m5:9.1%} {v5:8.1f} {med:8.3f} {mi:9.1%} {vi:8.1f} "
              f"{int(tot[0.3]['in_window']):5d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spans", "cut", "call", "lines", "score"])
    ap.add_argument("model", nargs="?")
    a = ap.parse_args()
    if a.cmd == "spans":
        for name, s, e in spans():
            print(f"{span_id(name, s, e)}  {e - s:.0f}s")
    elif a.cmd == "cut":
        cmd_cut()
    elif a.cmd == "call":
        cmd_call(a.model)
    elif a.cmd == "lines":
        cmd_lines()
    else:
        cmd_score()


if __name__ == "__main__":
    main()
