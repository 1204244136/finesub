"""Blind pairwise quality judging for the session-mode arms (docs/llm_local_agent_experiments.md §3.5).

Similarity to the refined subtitle is a coarse signal; the quality call is a
blind pairwise judgement. ``pack`` lays two runs' translations side by side,
window by window, under the labels X and Y (which run is X is random and kept
in a separate key the judge never sees), with the ASR source and the refined
subtitle of the same span for reference. A judge agent fills in the JSON the
pack asks for; ``collect`` maps X/Y back to arms through the key.

    python -m tools.bench.judge_session_ab pack <runA> <runB> --stable <stable.json> --refined <refined.ass|srt> \\
        [--clip-end S] --out <dir>
    python -m tools.bench.judge_session_ab collect <dir> <verdict.json> [<verdict.json> ...]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from finesub.subtitles.model import parse_srt

from tools.bench.score_session_ab import _windows, refined_srt

INSTRUCTIONS = """\
# 字幕译文盲评

你在比较同一段日语直播（单人）的两份机器中文译文 **X** 与 **Y**。它们来自同一个系统的两种运行方式，
你不知道哪份是哪种，也不该去猜。每个窗口给出三样东西，按时间排列：

- **原文**：日语语音识别结果（可能有识别错误，不是金标）；
- **X**、**Y**：两份译文；
- **精修参照**：人工字幕组精修过的中文。它**不是金标**：断句是精修者的自由，填充词取舍没有定规，
  译文带译者偏好（意译、口语化、改称呼都正常）。差异不等于译错；把它当作「懂行的人会怎么理解这段话」的参考。

## 你要判断的

逐窗比较 X 与 Y，标准按重要性排：

1. **意思对不对**：误译、漏译（原文有实质内容而译文没有）、凭空添加；
2. **专名与称呼**：人名、作品名、角色名译法是否正确，**跨窗口是否前后一致**；
3. **通顺**：中文是否自然、可读。

断句、行数、标点、语气词取舍上的差别**不算**优劣，除非它让意思错了。两份差不多时判 `tie`，不要硬分。

## 输出

只输出一个 JSON 对象（不要 markdown 代码块以外的任何文字），格式：

```json
{
  "windows": [
    {"window": 1, "verdict": "X|Y|tie",
     "x_errors": [{"time": "mm:ss", "kind": "误译|漏译|添加|专名|不通顺", "note": "一句话"}],
     "y_errors": [ ... ],
     "reason": "一两句话"}
  ],
  "names": [{"source": "原文专名", "x": ["X 里出现过的译法"], "y": ["Y 里出现过的译法"]}],
  "overall": "X|Y|tie",
  "overall_reason": "两三句话"
}
```

每个 `*_errors` 列出你确认的问题（带时间，好让人复核）；没有就给空数组。`names` 只列在两份里出现过、
且至少一份译法前后不一或与参照明显不同的专名。
"""


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _lines(segments: List[Any], start: float, end: float) -> List[str]:
    return [f"[{_mmss(s.start)}] {s.text.strip()}" for s in segments if start <= s.start < end]


def _translated(run_dir: Path) -> Path:
    found = next(iter(run_dir.glob("*-translated.srt")), None)
    if found is None:
        raise SystemExit(f"{run_dir}: no *-translated.srt")
    return found


def pack(run_a: Path, run_b: Path, stable: Path, refined: Path, clip_end: float | None, out: Path, seed: int) -> None:
    rng = random.Random(seed)
    x_run, y_run = (run_a, run_b) if rng.random() < 0.5 else (run_b, run_a)
    windows = _windows(run_a, stable)
    if not windows or windows != _windows(run_b, stable):
        raise SystemExit("the two runs must share one window plan")
    source = [
        type("Seg", (), {"start": float(s["start"]), "text": s.get("text", "")})()
        for s in json.loads(stable.read_text(encoding="utf-8"))["segments"]
    ]
    x = parse_srt(_translated(x_run).read_text(encoding="utf-8-sig"))
    y = parse_srt(_translated(y_run).read_text(encoding="utf-8-sig"))
    with tempfile.TemporaryDirectory() as tmp:
        ref = parse_srt(refined_srt(refined, Path(tmp), clip_end).read_text(encoding="utf-8"))
    parts = [INSTRUCTIONS]
    for index, (start, _end) in enumerate(windows, 1):
        end = windows[index][0] if index < len(windows) else float("inf")
        parts.append(f"\n## 窗口 {index}（{_mmss(start)} 起）\n")
        for label, segs in (("原文", source), ("X", x), ("Y", y), ("精修参照", ref)):
            parts.append(f"### {label}\n\n" + ("\n".join(_lines(segs, start, end)) or "（空）") + "\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack.md").write_text("\n".join(parts), encoding="utf-8")
    key = {"X": str(x_run), "Y": str(y_run), "seed": seed}
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out / "pack.md")


def _arm(run: str) -> str:
    path = Path(run)
    return f"{path.parent.name}/{path.name}"


def collect(out: Path, verdicts: List[Path]) -> Dict[str, Any]:
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    names = {"X": _arm(key["X"]), "Y": _arm(key["Y"]), "tie": "tie"}
    result: Dict[str, Any] = {"X": names["X"], "Y": names["Y"], "judges": []}
    for path in verdicts:
        text = path.read_text(encoding="utf-8")
        body = text[text.find("{"): text.rfind("}") + 1]
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # Judges drop the comma between two window objects now and then.
            data = json.loads(re.sub(r"\}(\s*)\{", r"},\1{", body))
        windows = data.get("windows") or []
        tally = {names[k]: sum(1 for w in windows if w.get("verdict") == k) for k in ("X", "Y", "tie")}
        result["judges"].append(
            {
                "file": str(path),
                "overall": names.get(data.get("overall", ""), data.get("overall")),
                "windows": tally,
                "errors": {names["X"]: sum(len(w.get("x_errors") or []) for w in windows),
                           names["Y"]: sum(len(w.get("y_errors") or []) for w in windows)},
                "per_window": [names.get(w.get("verdict", ""), w.get("verdict")) for w in windows],
            }
        )
    if len(result["judges"]) > 1:
        a, b = result["judges"][0]["per_window"], result["judges"][1]["per_window"]
        result["disagreeing_windows"] = [i + 1 for i, (p, q) in enumerate(zip(a, b)) if p != q]
    print(json.dumps(result, ensure_ascii=False, indent=2))  # product output
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--stable", required=True)
    p.add_argument("--refined", required=True)
    p.add_argument("--clip-end", type=float)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260930)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("verdicts", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(Path(args.run_a), Path(args.run_b), Path(args.stable), Path(args.refined), args.clip_end, Path(args.out), args.seed)
    else:
        collect(Path(args.out), [Path(v) for v in args.verdicts])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
