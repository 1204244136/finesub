"""Blind cross-window checks for the session-mode arms (docs/llm_local_agent_experiments.md §3.5).

A per-window judge cannot see whether a later window used what an earlier one
established. Two packs answer that, both over a probe list fixed before any
run output was read (the frozen probe list, with its FROZEN.md alongside):

- ``probes``: each probe is a place in a later window whose right rendering
  depends on an earlier window; the judge marks every run right / partly /
  wrong there (``consistency`` probes: against that run's own rendering of the
  anchor lines);
- ``entities``: every mention of each recurring name or form of address, as
  each run rendered it; the judge lists the forms each run used and counts the
  mentions off its own majority form.

Runs are labelled A-F once per pack (shuffled); the key goes to ``<out>-keys``.

    python -m tools.bench.probe_session_ab pack out/session-ab/x --arms <armA> <armB> \\
        --stable <clip-stable.json> --probes <probes.json> --entities <entities.json> --out <dir>
    python -m tools.bench.probe_session_ab collect <dir>   # reads <dir>/verdict-{probes,entities}.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import string
from pathlib import Path
from typing import Any, Dict, List

from finesub.subtitles.model import parse_srt

from tools.bench.judge_session_ab import _mmss, _translated
from tools.bench.rank_session_ab import _keys_dir, _runs, _verdict

MARGIN = 0.5

PROBE_INSTRUCTIONS = """\
# 跨窗口信息传递：逐点评审

一段日语节目被分成若干时间窗口逐窗译成中文。下面每个「点」都在后面的窗口里，它的正确译法要用到**更早窗口**
里出现过的信息（依赖说明里写着是哪一窗的什么）。同一段素材有 {n} 份译文，标为 {labels}，来自同一系统的不同
运行，你不知道哪份是怎么产生的。

对每个点、每份译文判一个结果：

- `type=absolute`：看该份在这个点的译文是否用对了依赖的信息（参照「期望译法」与「可接受变体」）：
  `对` / `半对`（意思对但称呼或写法不完全对，或只对一部分）/ `错`（没用上、译错或漏译）；
- `type=consistency`：不比期望译法，比**同一份译文**在「锚点」处的译法：这个点的译法与它自己在锚点处一致
  记 `对`，部分一致 `半对`，不一致或漏掉 `错`。

只看给出的片段，不要猜哪份是哪种运行。只输出一个 JSON 对象，不要任何其它文字：

```json
{{"probes": [{{"id": "H01", "verdicts": {{"A": "对|半对|错", "B": "..."}}, "notes": {{"A": "一句话，可省"}}}}]}}
```
"""

ENTITY_INSTRUCTIONS = """\
# 跨窗口一致性：专名与称呼

一段日语节目被分成若干时间窗口逐窗译成中文。下面列出反复出现的专名与称呼，以及它们每一次出现处（按时间，
标出所在窗口）各份译文的对应行。共 {n} 份译文，标为 {labels}，来自同一系统的不同运行。

对每个条目、每份译文：

1. 列出这份译文对该条目用过的所有中文写法及次数（漏译或无法辨认记作「漏/错」）；
2. 以这份译文自己用得最多的写法为准，数出与它不同的次数（`off_majority`，「漏/错」也算）；
3. 判断它的主写法是否与「参照写法」之一相符（`matches_reference`: true/false）。

只输出一个 JSON 对象，不要任何其它文字：

```json
{{"entities": [{{"id": "E01", "runs": {{"A": {{"forms": {{"写法": 次数}}, "off_majority": 0, "matches_reference": true}}}}}}]}}
```
"""


def _lines_between(segs: List[Any], start: float, end: float) -> str:
    hit = [s.text.strip().replace("\n", " / ") for s in segs if s.start < end + MARGIN and s.end > start - MARGIN]
    return " / ".join(hit) or "（无对应行）"


def _source(stable: Path) -> List[Dict[str, Any]]:
    return json.loads(stable.read_text(encoding="utf-8"))["segments"]


def _span(source: List[Dict[str, Any]], ids: List[int]) -> tuple[float, float]:
    rows = [source[i - 1] for i in ids if 0 < i <= len(source)]
    return min(float(r["start"]) for r in rows), max(float(r["end"]) for r in rows)


def pack(material: Path, arms: List[str], stable: Path, probes: Path, entities: Path, out: Path, seed: int) -> None:
    runs = _runs(material, arms)
    order = runs[:]
    random.Random(seed).shuffle(order)
    labels = list(string.ascii_uppercase[: len(order)])
    translated = {label: parse_srt(_translated(run).read_text(encoding="utf-8-sig")) for label, run in zip(labels, order)}
    source = _source(stable)
    head = {"n": len(order), "labels": "、".join(labels)}

    parts = [PROBE_INSTRUCTIONS.format(**head)]
    for probe in json.loads(probes.read_text(encoding="utf-8")):
        start, end = _span(source, probe["source_ids"])
        dep = probe["depends_on"]
        parts.append(f"\n## {probe['id']}（窗口 {probe['window']}，{_mmss(start)}，type={probe.get('type', 'absolute')}）\n")
        parts.append(f"- 原文：{probe['source_text']}")
        parts.append(f"- 依赖：窗口 {dep.get('window')}：{dep.get('info') or dep.get('anchor_text')}")
        parts.append(f"- 期望译法：{probe['expected']}；可接受变体：{'、'.join(probe.get('acceptable_variants') or []) or '无'}\n")
        if probe.get("type") == "consistency":
            # One block per anchor line: anchors can sit minutes apart, and a
            # span over all of them would paste whole stretches in between.
            for anchor in dep["source_ids"]:
                a_start, a_end = _span(source, [int(anchor)])
                parts.append(f"### 锚点 {_mmss(a_start)} 原文：{source[int(anchor) - 1].get('text', '').strip()}\n")
                parts += [f"- {label}：{_lines_between(translated[label], a_start, a_end)}" for label in labels]
                parts.append("")
        parts.append("### 这个点各份译文\n")
        parts += [f"- {label}：{_lines_between(translated[label], start, end)}" for label in labels]

    entity_parts = [ENTITY_INSTRUCTIONS.format(**head)]
    for entity in json.loads(entities.read_text(encoding="utf-8")):
        refs = "、".join(entity.get("zh_reference_forms") or [entity.get("zh_canonical", "")])
        entity_parts.append(f"\n## {entity['id']}：{entity.get('zh_canonical')}（参照写法：{refs}；原文写法：{'、'.join(entity.get('source_spellings') or [])}）\n")
        for mention in entity.get("mentions") or []:
            start, end = _span(source, [int(mention["id"])])
            window = "/".join(mention.get("windows") or [])
            entity_parts.append(f"### {_mmss(start)}（窗口 {window}）原文：{source[int(mention['id']) - 1].get('text', '').strip()}")
            entity_parts += [f"- {label}：{_lines_between(translated[label], start, end)}" for label in labels]

    out.mkdir(parents=True, exist_ok=True)
    (out / "pack-probes.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    (out / "pack-entities.md").write_text("\n".join(entity_parts) + "\n", encoding="utf-8")
    keys = _keys_dir(out)
    keys.mkdir(parents=True, exist_ok=True)
    key = {label: f"{run.parent.name}/{run.name}" for label, run in zip(labels, order)}
    (keys / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(order)} runs -> {out}")


SCORE = {"对": 1.0, "半对": 0.5, "错": 0.0}
#: The pre-registered primary analysis (the FROZEN.md beside the probe list):
#: strong and medium probes; weak ones are reported on their own.
PRIMARY = ("strong", "medium")


def _probe_scores(
    verdicts: List[Dict[str, Any]], key: Dict[str, str], strength: Dict[str, str], allow_unscorable: bool
) -> Dict[str, Any]:
    """Per run, the mean over the primary and the weak probes.

    Every probe must judge every label, and every verdict must be one of the
    three words: a missing label would shrink that run's denominator and an
    unknown word would count as wrong, both silently. ``allow_unscorable``
    drops an unknown word instead of failing, and lists what it dropped.
    """

    primary: Dict[str, List[float]] = {run: [] for run in key.values()}
    weak: Dict[str, List[float]] = {run: [] for run in key.values()}
    dropped = []
    for probe in verdicts:
        pid = probe.get("id")
        given = probe.get("verdicts") or {}
        if sorted(given) != sorted(key):
            raise SystemExit(f"probe {pid}: verdicts must name every label once, got {sorted(given)}")
        if pid not in strength:
            raise SystemExit(f"probe {pid} is not in the frozen probe list")
        bucket = primary if strength[pid] in PRIMARY else weak
        for label, verdict in given.items():
            if verdict not in SCORE:
                if not allow_unscorable:
                    raise SystemExit(f"probe {pid} label {label}: unknown verdict {verdict!r}")
                dropped.append({"probe": pid, "run": key[label], "verdict": verdict})
                continue
            bucket[key[label]].append(SCORE[verdict])
    judged = {probe.get("id") for probe in verdicts}
    unjudged = sorted(set(strength) - judged)
    if unjudged:
        # A probe the judge skipped (or the lenient parser lost) would change
        # every run's denominator without a word.
        raise SystemExit(f"probes in the frozen list with no verdict: {unjudged}")
    mean = lambda v: round(statistics.mean(v), 3) if v else None
    return {
        "probes": {run: mean(v) for run, v in primary.items()},
        "probes_weak": {run: mean(v) for run, v in weak.items()},
        "unscorable": dropped,
    }


def _entity_offs(entities: List[Dict[str, Any]], key: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    """Per run, mentions off its own majority form, and entities whose form matches the reference.

    Every entity must report every label with an integer count: a label the
    judge left out would read as "perfectly consistent"."""

    off = {run: 0 for run in key.values()}
    ref = {run: 0 for run in key.values()}
    for entity in entities:
        rows = entity.get("runs") or {}
        if sorted(rows) != sorted(key):
            raise SystemExit(f"entity {entity.get('id')}: runs must name every label once, got {sorted(rows)}")
        for label, row in rows.items():
            count = row.get("off_majority")
            if not isinstance(count, int):
                raise SystemExit(f"entity {entity.get('id')} label {label}: off_majority {count!r} is not an integer")
            off[key[label]] += count
            ref[key[label]] += 1 if row.get("matches_reference") else 0
    return {run: {"off_majority": off[run], "matches_reference": ref[run]} for run in key.values()}


def collect(out: Path, probes: Path, allow_unscorable: bool = False) -> Dict[str, Any]:
    key = json.loads((_keys_dir(out) / "key.json").read_text(encoding="utf-8"))
    strength = {p["id"]: p["strength"] for p in json.loads(probes.read_text(encoding="utf-8"))}
    result: Dict[str, Any] = {}
    probes_path = out / "verdict-probes.json"
    if probes_path.is_file():
        result.update(_probe_scores(_verdict(probes_path)["probes"], key, strength, allow_unscorable))
    entities_path = out / "verdict-entities.json"
    if entities_path.is_file():
        result["entities"] = _entity_offs(_verdict(entities_path)["entities"], key)
    arms: Dict[str, Dict[str, List[float]]] = {}
    for run in key.values():
        arm = run.split("/")[0]
        bucket = arms.setdefault(arm, {"probes": [], "off_majority": []})
        if "probes" in result and result["probes"][run] is not None:
            bucket["probes"].append(result["probes"][run])
        if "entities" in result:
            bucket["off_majority"].append(result["entities"][run]["off_majority"])
    result["by_arm"] = {arm: {k: (round(statistics.mean(v), 3), min(v), max(v)) if v else None for k, v in data.items()}
                        for arm, data in arms.items()}
    print(json.dumps(result, ensure_ascii=False, indent=2))  # product output
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("material")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--stable", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--entities", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261003)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("--probes", required=True, help="the frozen probe list: splits primary (strong+medium) from weak probes")
    c.add_argument("--allow-unscorable", action="store_true", help="drop (and list) a verdict that is none of 对/半对/错")
    args = parser.parse_args(argv)
    if args.cmd == "pack":
        pack(Path(args.material), args.arms, Path(args.stable), Path(args.probes), Path(args.entities), Path(args.out), args.seed)
    else:
        collect(Path(args.out), Path(args.probes), args.allow_unscorable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
