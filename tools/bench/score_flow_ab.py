"""Score task-flow acceptance runs against a manually refined subtitle.

The task-flow acceptance (`git show 3a3dce18:docs/plans/llm-task-execution-plan.md` §11): every run directory produced by
``run_flow_ab.py`` holds the translated SRT and the ``task-artifacts.jsonl``
of one full correction run; this script turns them into the pre-registered
metrics and aggregates them per arm.

    python -m tools.bench.score_flow_ab out/flow-ab --refined <refined.srt> [--md report.md]

Quality is measured against a refiner's subtitle, which is NOT a gold
standard (data/manually-refined-subs/0.4.2实操/README.md 判读规则): the
numbers only mean something as a *difference between arms scored against
the same reference*, never as an absolute.
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

from finesub.subtitles.alignment import align_srt
from finesub.subtitles.model import parse_srt


def _normalize(text: str) -> str:
    return "".join((text or "").split())


def _similarity(a: str, b: str) -> float:
    a, b = _normalize(a), _normalize(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def quality_metrics(machine_srt: Path, refined_srt: Path) -> Dict[str, Any]:
    machine = parse_srt(machine_srt.read_text(encoding="utf-8-sig"))
    refined = parse_srt(refined_srt.read_text(encoding="utf-8-sig"))
    aligned = align_srt(machine, refined)
    sims = [_similarity(pair.machine_text, pair.refined_segment.text) for pair in aligned.matched]
    return {
        "machine_lines": len(machine),
        "refined_lines": len(refined),
        "matched": len(aligned.matched),
        "matched_rate": (len(aligned.matched) / len(refined)) if refined else 0.0,
        "refined_only": len(aligned.refined_only),
        "machine_only": len(aligned.machine_only),
        "mean_similarity": statistics.fmean(sims) if sims else 0.0,
        "low_similarity": sum(1 for s in sims if s < 0.3),
    }


def artifact_metrics(artifacts: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in artifacts.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = [r.get("kind") for r in rows]
    windows = [r for r in rows if r.get("kind") == "correction_window_response"]
    first = [w for w in windows if int(w.get("payload", {}).get("attempt", 0)) == 0]
    ok_first = [w for w in first if w.get("payload", {}).get("validation_ok")]
    model_calls = sum(
        1
        for k in kinds
        if k in (
            "correction_window_response",
            "correction_query_response",
            "research_round1_response",
            "research_round2_response",
            "prepare_task_request",
            "prepare_task_response",
            "fast_round1_response",
            "search_loop_round",
        )
    )
    searches = 0
    for r in rows:
        if r.get("kind") in ("research_search_results", "correction_search_results"):
            searches += len(r.get("payload", {}).get("executed") or [])
    totals: Dict[str, int] = {}
    for r in rows:
        if r.get("kind") == "token_distribution_report":
            for key, value in (r.get("payload", {}).get("totals") or {}).items():
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0) + int(value)
    graph = next((r["payload"] for r in rows if r.get("kind") == "task_graph"), {})
    return {
        "model_calls": model_calls,
        "window_calls": len(windows),
        "first_attempt_ok_rate": (len(ok_first) / len(first)) if first else 0.0,
        "retries": kinds.count("correction_window_retry"),
        "call_errors": sum(1 for k in kinds if k and k.endswith("_call_error")),
        "search_queries": searches,
        "prompt_tokens": totals.get("prompt_tokens", 0) or totals.get("total_input_tokens", 0),
        "output_tokens": totals.get("total_output_tokens", 0) or totals.get("output_tokens", 0),
        "thinking_tokens": totals.get("thinking_tokens", 0),
        "strategy": graph.get("strategy", "legacy"),
        "executed_by": graph.get("executed_by", "legacy"),
    }


def score_run(run_dir: Path, refined: Path) -> Dict[str, Any] | None:
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("exit_code") not in (None, 0):
        # A run that died (provider 503, chain exhausted) leaves partial
        # outputs; scoring them would let an outage read as a quality delta.
        return None
    srt = next(iter(run_dir.glob("*-translated.srt")), None) or next(iter(run_dir.glob("*.srt")), None)
    artifacts = next(iter(run_dir.glob("**/task-artifacts.jsonl")), None)
    if srt is None or artifacts is None:
        return None
    return {
        "run": run_dir.name,
        "arm": meta.get("arm", run_dir.parent.name),
        "wall_seconds": meta.get("wall_seconds"),
        "exit_code": meta.get("exit_code"),
        **quality_metrics(srt, refined),
        **artifact_metrics(artifacts),
    }


AGG_KEYS = (
    "mean_similarity",
    "matched_rate",
    "low_similarity",
    "first_attempt_ok_rate",
    "model_calls",
    "search_queries",
    "prompt_tokens",
    "output_tokens",
    "thinking_tokens",
    "wall_seconds",
)


def aggregate(scores: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for score in scores:
        by_arm.setdefault(score["arm"], []).append(score)
    out: Dict[str, Dict[str, Any]] = {}
    for arm, rows in sorted(by_arm.items()):
        summary: Dict[str, Any] = {"n": len(rows)}
        for key in AGG_KEYS:
            values = [float(r[key]) for r in rows if r.get(key) is not None]
            if not values:
                continue
            summary[key] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        out[arm] = summary
    return out


def render_markdown(scores: List[Dict[str, Any]], agg: Dict[str, Dict[str, Any]]) -> str:
    lines = ["| arm | n | 相似度 mean (min–max) | 对上率 | 低相似行 | 首次校验通过率 | 模型调用 | 搜索 | prompt tok | 输出 tok | 思考 tok | 墙钟 s |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, s in agg.items():
        def cell(key, fmt="{:.3f}", rng=False):
            v = s.get(key)
            if not v:
                return "—"
            text = fmt.format(v["mean"])
            if rng:
                text += f" ({fmt.format(v['min'])}–{fmt.format(v['max'])})"
            return text
        lines.append(
            f"| {arm} | {s['n']} | {cell('mean_similarity', rng=True)} | {cell('matched_rate')} | {cell('low_similarity', '{:.1f}')} | "
            f"{cell('first_attempt_ok_rate')} | {cell('model_calls', '{:.1f}')} | {cell('search_queries', '{:.1f}')} | "
            f"{cell('prompt_tokens', '{:.0f}')} | {cell('output_tokens', '{:.0f}')} | {cell('thinking_tokens', '{:.0f}')} | {cell('wall_seconds', '{:.0f}')} |"
        )
    lines.append("")
    lines.append("| run | arm | 相似度 | 对上率 | 低相似 | 首次通过 | 调用 | 搜索 | prompt | 输出 | 思考 | 墙钟 | 策略 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in scores:
        lines.append(
            f"| {r['run']} | {r['arm']} | {r['mean_similarity']:.3f} | {r['matched_rate']:.3f} | {r['low_similarity']} | "
            f"{r['first_attempt_ok_rate']:.2f} | {r['model_calls']} | {r['search_queries']} | {r['prompt_tokens']} | "
            f"{r['output_tokens']} | {r['thinking_tokens']} | {r.get('wall_seconds') or '—'} | {r['strategy']}/{r['executed_by']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", help="out/flow-ab-like root: <root>/<arm>/<run>/")
    parser.add_argument("--refined", required=True, help="the refiner's SRT")
    parser.add_argument("--json", help="write per-run scores + aggregates here")
    parser.add_argument("--md", help="write the markdown tables here")
    args = parser.parse_args(argv)
    root = Path(args.root)
    scores = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            score = score_run(run_dir, Path(args.refined))
            if score is not None:
                scores.append(score)
    agg = aggregate(scores)
    text = render_markdown(scores, agg)
    print(text)  # product output
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": scores, "arms": agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
