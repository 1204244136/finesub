"""Score agent session-mode runs (``run_session_ab.py``) against a refined subtitle.

Reuses ``score_flow_ab``'s quality and artifact metrics, and adds what the
session-mode comparison needs (docs/llm_local_agent_experiments.md §3.5):

- token cost split into uncached input / cached input / output, summed over
  the per-call records *plus* the session totals a pseudo-conversational run
  books in ``agent-session-usage.json`` (its per-call records carry none);
- similarity of window 1 vs the later windows, since a carried-over
  conversation can only help or hurt from window 2 on;
- a refined ``.ass`` read directly: ``{...}`` blocks dropped (in these
  feedback files they hold the machine line the refiner replaced), ``\\N``
  kept as a line break, annotation and watermark styles skipped (any style
  whose name contains ``注释`` or ``水印``: ``Radio注释``, ``注释（1080p）``, ``水印-左``).

    python -m tools.bench.score_session_ab out/session-ab/x --refined <refined.ass|srt> [--md report.md]

Same reading rule as the flow scorer: the refined subtitle is not a gold
standard, so only differences between arms scored against it mean anything.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from finesub.subtitles.alignment import align_srt
from finesub.subtitles.model import parse_srt

from tools.bench.score_flow_ab import _similarity, aggregate, artifact_metrics, quality_metrics

#: A style is not dialogue when its name contains one of these.
NON_DIALOGUE_STYLE_MARKS = ("注释", "水印")
_BRACES = re.compile(r"\{[^}]*\}")


def _ass_time_to_srt(stamp: str) -> str:
    hours, minutes, rest = stamp.strip().split(":")
    seconds, _, centis = rest.partition(".")
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(centis.ljust(2, '0')[:2]) * 10:03d}"


def ass_to_srt_text(ass_text: str) -> str:
    fields: List[str] = []
    blocks: List[str] = []
    for line in ass_text.splitlines():
        if line.startswith("Format:") and not fields and "Text" in line:
            fields = [f.strip() for f in line[len("Format:"):].split(",")]
            continue
        if not line.startswith("Dialogue:") or not fields:
            continue
        values = line[len("Dialogue:"):].split(",", len(fields) - 1)
        row = dict(zip(fields, (v.strip() if k != "Text" else v for k, v in zip(fields, values))))
        if any(mark in row.get("Style", "") for mark in NON_DIALOGUE_STYLE_MARKS):
            continue
        text = _BRACES.sub("", row.get("Text", "")).replace("\\N", "\n").strip()
        if not text:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n{_ass_time_to_srt(row['Start'])} --> {_ass_time_to_srt(row['End'])}\n{text}\n"
        )
    return "\n".join(blocks)


def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    return f"{millis // 3_600_000:02d}:{millis // 60_000 % 60:02d}:{millis // 1000 % 60:02d},{millis % 1000:03d}"


def refined_srt(path: Path, workdir: Path, clip_end: float | None = None) -> Path:
    """The refined subtitle as SRT, cut to ``clip_end`` when the run used a clip.

    A refined line past the clip has nothing to match, so leaving it in would
    lower every arm's match rate by the same amount and hide nothing -- but it
    would make the numbers incomparable with an unclipped run.
    """

    text = (
        ass_to_srt_text(path.read_text(encoding="utf-8-sig"))
        if path.suffix.lower() == ".ass"
        else path.read_text(encoding="utf-8-sig")
    )
    if clip_end is not None:
        kept = [seg for seg in parse_srt(text) if seg.start < clip_end]
        text = "\n".join(
            f"{i}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n{seg.text}\n" for i, seg in enumerate(kept, 1)
        )
    out = workdir / (path.stem + ".srt")
    out.write_text(text, encoding="utf-8")
    return out


def _windows(run_dir: Path, stable: Path) -> List[tuple[float, float]]:
    """Window spans in seconds: the plan names 1-based segment ids of the stable."""

    plan = next(iter(run_dir.glob("**/correction-window-plan.json")), None)
    if plan is None or not stable.is_file():
        return []
    windows = json.loads(plan.read_text(encoding="utf-8"))["plan"]["windows"]
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    spans = []
    for window in windows:
        own = [int(i) for i in window["segment_ids"] if i not in set(window.get("overlap_ids") or ())]
        if own:
            spans.append((float(segments[min(own) - 1]["start"]), float(segments[max(own) - 1]["end"])))
    return spans


def window_split(machine_srt: Path, refined: Path, windows: List[tuple[float, float]]) -> Dict[str, Any]:
    """Mean similarity of the pairs whose refined line starts in window 1 vs later."""

    if len(windows) < 2:
        return {}
    machine = parse_srt(machine_srt.read_text(encoding="utf-8-sig"))
    ref = parse_srt(refined.read_text(encoding="utf-8-sig"))
    first_end = windows[1][0]
    early: List[float] = []
    late: List[float] = []
    for pair in align_srt(machine, ref).matched:
        sim = _similarity(pair.machine_text, pair.refined_segment.text)
        (early if pair.refined_segment.start < first_end else late).append(sim)
    return {
        "windows": len(windows),
        "similarity_window1": statistics.fmean(early) if early else None,
        "similarity_later": statistics.fmean(late) if late else None,
    }


_INPUT_LINE = re.compile(r"^- input_tokens: (\d+) / (\d+) / (\d+)", re.M)
_OUTPUT_LINE = re.compile(r"^- output_tokens_breakdown: \d+ / \d+ / (\d+)", re.M)


def usage_metrics(run_dir: Path) -> Dict[str, Any]:
    """Token use per run, counted once per provider session.

    A resumed session's per-call usage is the session's running total, not the
    call's own (measured 2026-09-30: Codex and WorkBuddy grow 33k, 89k, 169k,
    ... over one ``resume`` session), so calls are grouped by session id and
    each session contributes its largest figure; a ``per-window`` call has a
    session of its own, so for it this is the plain sum. A pseudo-conversational
    run books its sessions in ``agent-session-usage.json`` instead.
    ⚠ agy's figures do not add up even within one call (cached above total), so
    its token columns are not comparable -- read them, do not rank on them.
    """

    sessions: Dict[str, Dict[str, int]] = {}
    for exchange in sorted(run_dir.glob("**/exchanges/*.md")):
        text = exchange.read_text(encoding="utf-8", errors="replace")
        row = next((l for l in text.splitlines() if l.startswith("| local")), None)
        found = _INPUT_LINE.search(text)
        if row is None or found is None:
            continue
        session = row.split("|")[5].strip() or exchange.name
        out = _OUTPUT_LINE.search(text)
        figures = {
            "uncached_input_tokens": int(found.group(1)),
            "cached_input_tokens": int(found.group(2)),
            "total_input_tokens": int(found.group(3)),
            "total_output_tokens": int(out.group(1)) if out else 0,
        }
        held = sessions.setdefault(session, dict.fromkeys(figures, 0))
        for key, value in figures.items():
            held[key] = max(held[key], value)
    totals = dict.fromkeys(("uncached_input_tokens", "cached_input_tokens", "total_input_tokens", "total_output_tokens"), 0)
    for held in sessions.values():
        for key in totals:
            totals[key] += held[key]
    booked = next(iter(run_dir.glob("**/agent-session-usage.json")), None)
    pseudo = 0
    if booked is not None:
        for row in json.loads(booked.read_text(encoding="utf-8")).get("sessions") or []:
            pseudo += 1
            for key in totals:
                value = (row.get("usage") or {}).get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    return {**totals, "provider_sessions": len(sessions), "booked_sessions": pseudo}


def score_run(run_dir: Path, refined: Path) -> Dict[str, Any] | None:
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        return None  # still running: the runner writes run.json last
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("exit_code") != 0 or not meta.get("execute", True):
        return None
    srt = next(iter(run_dir.glob("*-translated.srt")), None) or next(iter(run_dir.glob("*.srt")), None)
    artifacts = next(iter(run_dir.glob("**/task-artifacts.jsonl")), None)
    if srt is None or artifacts is None:
        return None
    return {
        "run": run_dir.name,
        "arm": meta.get("arm", run_dir.parent.name),
        "wall_seconds": meta.get("wall_seconds"),
        **quality_metrics(srt, refined),
        **window_split(srt, refined, _windows(run_dir, Path(meta.get("stable", "")))),
        **artifact_metrics(artifacts),
        **usage_metrics(run_dir),
    }


AGG_EXTRA = (
    "similarity_window1",
    "similarity_later",
    "uncached_input_tokens",
    "cached_input_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "retries",
    "call_errors",
)


def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = aggregate(scores)
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for score in scores:
        by_arm.setdefault(score["arm"], []).append(score)
    for arm, rows in by_arm.items():
        for key in AGG_EXTRA:
            values = [float(r[key]) for r in rows if r.get(key) is not None]
            if values:
                out[arm][key] = {
                    "mean": statistics.fmean(values),
                    "min": min(values),
                    "max": max(values),
                    "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
                }
    return out


def render_markdown(scores: List[Dict[str, Any]], agg: Dict[str, Dict[str, Any]]) -> str:
    def cell(s: Dict[str, Any], key: str, fmt: str = "{:.3f}", spread: bool = False) -> str:
        v = s.get(key)
        if not v:
            return "—"
        text = fmt.format(v["mean"])
        if spread:
            text += f" ±{fmt.format(v['stdev'])}"
        return text

    lines = [
        "| arm | n | 相似度 mean ±sd | 窗1 | 窗2+ | 对上率 | 低相似行 | 首次通过 | 重试 | 输入合计 | uncached in | cached in | 输出 | 墙钟 s |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, s in agg.items():
        lines.append(
            f"| {arm} | {s['n']} | {cell(s, 'mean_similarity', spread=True)} | {cell(s, 'similarity_window1')} | "
            f"{cell(s, 'similarity_later')} | {cell(s, 'matched_rate')} | {cell(s, 'low_similarity', '{:.1f}')} | "
            f"{cell(s, 'first_attempt_ok_rate')} | {cell(s, 'retries', '{:.1f}')} | {cell(s, 'total_input_tokens', '{:.0f}')} | "
            f"{cell(s, 'uncached_input_tokens', '{:.0f}')} | "
            f"{cell(s, 'cached_input_tokens', '{:.0f}')} | {cell(s, 'total_output_tokens', '{:.0f}')} | {cell(s, 'wall_seconds', '{:.0f}')} |"
        )
    lines += [
        "",
        "| run | arm | 相似度 | 窗1 | 窗2+ | 对上率 | 首次通过 | 重试 | 调用错误 | uncached in | cached in | 输出 | 会话账 | 墙钟 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in scores:
        def num(key: str, fmt: str = "{:.3f}") -> str:
            value = r.get(key)
            return fmt.format(value) if value is not None else "—"

        lines.append(
            f"| {r['run']} | {r['arm']} | {num('mean_similarity')} | {num('similarity_window1')} | {num('similarity_later')} | "
            f"{num('matched_rate')} | {num('first_attempt_ok_rate', '{:.2f}')} | {r['retries']} | {r['call_errors']} | "
            f"{r['uncached_input_tokens']} | {r['cached_input_tokens']} | {r['total_output_tokens']} | {r['provider_sessions']}+{r['booked_sessions']} | "
            f"{r.get('wall_seconds') or '—'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", help="<root>/<arm>/<run>/ as run_session_ab.py writes it")
    parser.add_argument("--refined", required=True, help="the refiner's .ass or .srt")
    parser.add_argument("--clip-end", type=float, help="seconds: drop refined lines starting after the clipped input")
    parser.add_argument("--json", help="write per-run scores + aggregates here")
    parser.add_argument("--md", help="write the markdown tables here")
    args = parser.parse_args(argv)
    root = Path(args.root)
    with tempfile.TemporaryDirectory() as tmp:
        refined = refined_srt(Path(args.refined), Path(tmp), args.clip_end)
        scores = [
            score
            for arm_dir in sorted(p for p in root.iterdir() if p.is_dir())
            for run_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir())
            if (score := score_run(run_dir, refined)) is not None
        ]
    agg = _aggregate(scores)
    text = render_markdown(scores, agg)
    print(text)  # product output
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": scores, "arms": agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
