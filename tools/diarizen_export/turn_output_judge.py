"""C.5d graded the right way round: a judge scores each model SPLIT against the refined truth.

Owner 2026-09-20: 「感觉是裁判或裁判 prompt 的问题。只要裁判根据精修或 truth 判断
a) recall 对不对（宽松一些，语气词补进去了也问题不大） b) 对多少（给个百分比）。」

`turn_truth_judge.py` asked the judge to LABEL THE TRUTH -- "does this row's Japanese text already
hold two people's words?" -- and the judge answered that question well; the question was wrong.
It said NO on every row where the second person spoke but was never transcribed, which are exactly
the rows v2 exists for. So this grades the DELIVERABLE: for every row a model chose to split, the
row, the model's two halves, and the refined truth, scored as `recall` (a verdict) + `pct`.

**Judge v2 (`--judge 2`, the default) -- owner 2026-09-21, and why v1's headline was withdrawn.**
v1 decided "should this row be split" by asking whether at least two people said something
MEANINGFUL, with the rule "if the other half is only a backchannel, the cut is redundant". That
treated attribution as a content question. Owner: 「把 ASR 文本里已有的附和切出去是正确的，因为
归属两个人。补充 ASR 没收的填充词语气词则是多余的，但也不算大问题。」 Read against that, v1's 28
`不该拆` split roughly: ~10 were CORRECT cuts of another person's in-text backchannel, 2 were
supplied fillers, ~11 were one person's speech cut in two, ~5 undecidable. So "about half of every
arm's cuts are redundant" was an artifact of the rule and is withdrawn (plan C.5d-8).

v2's rules:

* **A quoted half is an ATTRIBUTION question.** Two people's words in the ASR text -- a
  backchannel counts -- means the cut is right. `不该拆` only when the text is one person's.
* **The fansub drops backchannels**, so "the truth shows one person" does not prove one person
  spoke. A short response that reads as someone else answering is counted as theirs.
* **A supplied half that disagrees with the truth** follows the owner's rule (2026-09-21):
  logical + only a backchannel/filler -> `对`; logical + informational -> `UNSURE` (a fansub rarely
  drops informational lines, so the mismatch is suspicious, but the model may still be right);
  illogical -> `不该补`.
* **Context rows**, one or two either side, marked reference-only -- "is this a reply to the line
  before" is how a backchannel is recognised, and "does this supply make sense" needs the flow.
* **Batch size**: the owner asked for 20-50 rows a call; 30 and 20 did not fit one reasoning turn
  and timed out (see `BATCH`), so the owner settled on 10 at effort high.

Readings (plan C.5d-8): on rows that should be split all three arms cut right 80-93% of the time;
the remaining failure is one person's speech cut in two (audio v1 ~45%, v2 and text ~25%, not
significant). Self-consistency 83% identical / 90% on the should-split step.

The fansub-dropped-backchannel rule for QUOTED halves began as this bed's extension of the owner's
supplement rule; the owner confirmed keeping it (2026-09-21).

⚠ **Blind**, as before: arms pooled and shuffled with a fixed seed before batching.
⚠ "Should this be split" never depends on the model's own halves: a `部分` truth line counts toward
the row only if its meaning is visible in the ASR text (v1's cold read caught the circular draft).
⚠ What this does NOT measure: rows a model left unsplit that it should have cut.

**Parallelism** (owner 2026-09-21: driver max 16, task-level 16). Production fans out with three
things together, and v1 of this script had none of them -- ten threads all landed on the default
lane and ran as one `node.exe`, serial:

    llm_task_slots(budgets)    the task's account, so claims can fan out past the mandatory lane
    agent_session_scope()      the run's session registry, which owns the lane ordinals
    initializer=bind_llm_worker  each worker leases its OWN lane ordinal

⚠ And the order matters: an agent slot pool's limit is fixed by whichever config BUILDS it first
(`_shared_in_flight_pool`; a later mismatch is only logged). `default_task_slots()` reads the
user's config (4) and would build it first, so the budgets are built from the raised settings
explicitly, before anything else touches the pool.

    python tools/diarizen_export/turn_output_judge.py --dry-run
    python tools/diarizen_export/turn_output_judge.py            # spends CodeBuddy credits
    python tools/diarizen_export/turn_output_judge.py --recheck
    python tools/diarizen_export/turn_output_judge.py --score
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = REPO / "tmp" / "diarizen-bed" / "turn-llm"
ARMS = ("audio-agy", "audio-agy-v2", "text-agy")

#: Splits per call (owner 2026-09-21: 20-50). ⚠ More neighbours per batch than v1's 10, so the
#: recheck matters more, not less: it is what says whether the extra neighbours move verdicts.
#: ⚠ 30 was tried first and BOTH batches timed out at 1680 s -- against ~200 s for v1's batches of
#: 10. Three times the rows took more than eight times as long (the rules grew and each row gained
#: four context lines), so time is not linear in batch size here.
#: ⚠ 20 then timed out at 3600 s on two of three batches (18 rows finished in 1261 s). Not
#: contention -- the CodeBuddy logs show each call reasoning into finish_reason="length" (~80k
#: characters, ~132 s a turn), the CLI re-requesting, and the model restarting its grading from #0:
#: 24 and 26 truncated turns on the two that timed out, 7 on the one that finished. The whole
#: batch's reasoning has to fit ONE turn's output cap at effort high, so the owner chose 10
#: (2026-09-21, keeping effort high over 20 at medium). Rows already graded are kept.
BATCH = 10

#: This run's per-call timeout. ⚠ Raised only here, on the resolved settings, and paired with NOT
#: retrying a timed-out batch: a call that is slow but working (its CPU time was rising the whole
#: way) should not be discarded at the default, and a batch that DID time out would only time out
#: again if resent unchanged -- the 30-row attempt was about to spend four more 28-minute tries.
CALL_TIMEOUT = 3600

#: Subtitle rows shown before and after each graded row, as reference only.
CONTEXT_ROWS = 2

#: Owner 2026-09-21: 16 at the driver and 16 at the task level.
PARALLEL = 16

SPLITS = re.compile(r"<splits>(.*?)</splits>", re.S)
GRADES = re.compile(r"<grades>(.*?)</grades>", re.S)
VERDICTS = ("对", "部分", "不对", "不该拆", "不该补", "UNSURE")
#: Verdicts that carry no coverage figure.
NO_PCT = ("不该拆", "不该补", "UNSURE")

SYSTEM = """你在给一个「字幕拆分」模型的输出打分。

【背景】日语对谈节目里，有些字幕行把两个人的话并进了同一行。一个模型会把这样的行拆成两段，
**每段应当只属于一个人**。如果其中一个人的话在原文里**根本没被转写出来**，
模型会按它从音频里听到的，把那半句**补**出来。
（两段的先后按文字顺序，**不一定**等于谁先开口。）

【每一条给你】
- `上文` / `下文`　这一行前后各一两行字幕的日语原文。**只用来理解对话走向，不打分**，
  也**不要**把它们的内容算成这一行的。
- `字幕原文`　这一行的日语原文（自动转写，可能漏字）
- `模型的拆法`　第一段 / 第二段。第二段标明是「取自原文」还是「模型按声音补出」
- `真值`　人工精修时间轴：这一行的时间跨度内有谁说了话、说了什么。**中文译文**，按开口先后排列，
  标出整句落在本行内（`完整`）还是只有一部分落进来（`部分`）

⚠ 中文是译文，和日语**不会逐字对应**，一律**看意思**。
⚠ 真值来自字幕组，**字幕组经常不记简短的附和、应声、语气词**。真值里没有某句话，不等于没人说过。

━━━━━━━━━━━━━━━━━━━━━━━━

【一、第二段「取自原文」：判的是**归属**】

原文里的每个字都是某个人说的，拆分就是把不同人的字分开。

1. 原文里有**两个人**的字 ⇒ 这一刀**该切**。
   ⚠ **附和、应声也算另一个人的字。** 把原文里别人的一句「うん」「そう」「はーい」切出去是**对的**，
   因为它确实属于另一个人。
2. 真值里只看到一个人，但被切出去的那段是**简短的附和/应声**，而且结合上下文**像是别人在应答**
   ⇒ 字幕组很可能没记它，**按另一个人的字算**，同第 1 条。
3. 原文从头到尾都是**同一个人**的话——被切出去的那段是他自己话的一部分，而不是别人的应答
   ⇒ `不该拆`。

该切时，再看分得对不对：
- `对`　　两段各属于一个人，分得干净
- `部分`　大体分开了，但切点偏了，某一段里混进了另一个人的一点字
- `不对`　切错了地方，两段的归属基本是乱的

【二、第二段「模型按声音补出」：先对真值】

- 和真值**对得上** ⇒ 照常判 `对` / `部分`。
- 和真值**对不上** ⇒ 看补出来的这句话**放进上下文说不说得通**：
  · 说得通，而且只是**附和、应声、语气词** ⇒ `对`（字幕组常不记这些，补上无妨）
  · 说得通，但**有一定信息量** ⇒ `UNSURE`
    （有信息量的话字幕组一般不会漏，对不上就可疑；但也可能是模型听对了，所以不判错）
  · **说不通**——和上下文或这一行的内容接不上 ⇒ `不该补`

【三、`pct`】

被拆出来的那个人（即不属于第一段的那个人）在这一行说的**有实际意义的内容**，
第二段覆盖了百分之多少，填 0–100 的整数，看意思估即可。
**附和、语气词不计入分母**；第二段里多出来的语气词**不扣分**。
答 `不该拆` / `不该补` / `UNSURE` 时填 null。

【关于真值里的 `部分`】
`部分` 表示那句话只有一部分落在这一行的时间里，其余在相邻行。只有当它的意思在**字幕原文**里
看得到时，才算这一行的内容；否则当它属于相邻行。
⚠ 这里**只看原文，不看模型的两段**——该不该拆是事实，不能由被打分的输出来决定。

【注意】你不知道、也不需要知道这些拆法来自哪个模型。每一条独立打分，互不参照。

━━━━━━━━━━━━━━━━━━━━━━━━

按下面的格式回答，每条一行，`id` 就是 `#` 后面的数字，不要有别的内容：

<grades>
{"id": 0, "recall": "对", "pct": 90, "who": "第二段对应的人", "why": "一句话理由"}
{"id": 1, "recall": "不该拆", "pct": null, "why": "一句话理由"}
{"id": 2, "recall": "不该补", "pct": null, "why": "一句话理由"}
{"id": 3, "recall": "UNSURE", "pct": null, "why": "一句话理由"}
</grades>

`recall` 只能是这六个之一：对、部分、不对、不该拆、不该补、UNSURE。
`who` 写第二段对应的人；答 不该拆 / 不该补 / UNSURE 时可以省略。"""


def halves(row, payload):
    """`(first, second, source)` for one split, or None if the quote is not in the row's text."""
    text = str(row.get("text", ""))
    quoted = str(payload.get("cut_after") or payload.get("first") or "")
    if not quoted or quoted not in text:
        return None
    cut = text.find(quoted) + len(quoted)
    if payload.get("second") is not None:
        source = "heard" if str(payload.get("source", "")).lower() == "heard" else "quoted"
        return text[:cut], str(payload["second"]), source
    return text[:cut], text[cut:], "quoted"


def collect(episodes=(1, 2, 3)):
    """Every split any arm delivered, with its context rows."""
    from turn_shape_bed import episode_rows

    episodes_data = {}
    for episode in episodes:
        loaded = episode_rows(episode)
        if loaded is not None:
            stable = loaded[0]
            index_of = {round(float(row["start"]), 1): position
                        for position, row in enumerate(stable)}
            episodes_data[episode] = (stable, index_of, loaded[3])

    items = []
    for arm in ARMS:
        path = OUT / f"{arm}.jsonl"
        if not path.exists():
            continue
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()):
            if record["episode"] not in episodes_data:
                continue
            stable, index_of, _ = episodes_data[record["episode"]]
            match = SPLITS.search(record["raw"] or "")
            if not match:
                continue
            for line in match.group(1).splitlines():
                try:
                    payload = json.loads(line.strip())
                except (json.JSONDecodeError, ValueError):
                    continue
                if str(payload.get("answer", "")).upper() != "A":
                    continue
                key = round(float(payload["time"]), 1)
                position = index_of.get(key)
                if position is None:
                    continue
                row = stable[position]
                parts = halves(row, payload)
                if parts is None:
                    continue
                before = [str(r.get("text", "")).strip()
                          for r in stable[max(0, position - CONTEXT_ROWS):position]]
                after = [str(r.get("text", "")).strip()
                         for r in stable[position + 1:position + 1 + CONTEXT_ROWS]]
                key_text = f"{record['episode']}-{key}"
                items.append({"id": f"{arm}|{key_text}", "arm": arm, "key": key_text,
                              "row": row, "first": parts[0], "second": parts[1],
                              "source": parts[2], "before": before, "after": after})
    return items, {episode: data[2] for episode, data in episodes_data.items()}


def prompt_for(batch, events_of) -> str:
    from turn_truth_judge import format_people, people_in

    parts = []
    for index, item in enumerate(batch):
        episode = int(item["key"].split("-")[0])
        truth = people_in(item["row"], events_of[episode])
        where = "模型按声音补出" if item["source"] == "heard" else "取自原文"
        block = [f"#{index}"]
        block += [f"  上文：{text}" for text in item["before"]] or ["  上文：（无）"]
        block += [
            f"  字幕原文：{str(item['row'].get('text', '')).strip()}",
            "  模型的拆法：",
            f"     第一段：{item['first'].strip() or '（空）'}",
            f"     第二段：{item['second'].strip() or '（空）'}　（{where}）",
        ]
        block += [f"  下文：{text}" for text in item["after"]] or ["  下文：（无）"]
        block += ["  真值：", *format_people(truth)]
        parts.append("\n".join(block))
    return "\n\n".join(parts) + f"\n\n共 {len(batch)} 条，逐条打分。"


def parse(text: str) -> dict:
    match = GRADES.search(text or "")
    if not match:
        return {}
    found = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            verdict = str(payload.get("recall", "")).strip()
            if verdict.upper() == "UNSURE":
                verdict = "UNSURE"
            if verdict not in VERDICTS:
                continue
            pct = payload.get("pct")
            found[int(payload["id"])] = {
                "recall": verdict,
                "pct": None if pct is None or verdict in NO_PCT else max(0, min(100, int(pct))),
                "who": payload.get("who"),
                "why": str(payload.get("why", ""))[:160],
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return found


def record_of(item, answer) -> dict:
    """The line written for one graded split. Rehearsed by `--dry-run` before anything is spent."""
    return {"id": item["id"], "arm": item["arm"], "key": item["key"], "source": item["source"],
            "first": item["first"], "second": item["second"], **answer}


def paths(judge: int):
    suffix = "" if judge == 1 else f"-j{judge}"
    return (OUT / f"output-grades{suffix}.jsonl", OUT / f"output-grades{suffix}-recheck.jsonl")


def score(path: Path) -> None:
    if not path.exists():
        print(f"no grades at {path.name}")
        return
    grades = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    shown = [v for v in VERDICTS if any(g["recall"] == v for g in grades)]
    print(f"{len(grades)} graded splits  ({path.name})\n")
    print(f"{'arm':<14}{'n':>3}" + "".join(f"{v:>7}" for v in shown) + f"{'mean pct':>12}")
    for arm in ARMS:
        mine = [grade for grade in grades if grade["arm"] == arm]
        if not mine:
            continue
        counts = Counter(grade["recall"] for grade in mine)
        pcts = [grade["pct"] for grade in mine if grade["pct"] is not None]
        mean = f"{np.mean(pcts):.0f}% (n={len(pcts)})" if pcts else "-"
        print(f"{arm:<14}{len(mine):>3}" + "".join(f"{counts[v]:>7}" for v in shown) + f"{mean:>12}")
    heard = [grade for grade in grades if grade.get("source") == "heard"]
    if heard:
        print(f"\nsupplied from the audio: {len(heard)} -> {dict(Counter(g['recall'] for g in heard))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--recheck", action="store_true",
                        help="grade everything again in a different shuffle, to a separate file")
    parser.add_argument("--judge", type=int, choices=(1, 2), default=2,
                        help="which rules' files to score (v1's are kept for comparison)")
    arguments = parser.parse_args()

    first_path, recheck_path = paths(arguments.judge)
    if arguments.score:
        score(first_path)
        if recheck_path.exists():
            print()
            score(recheck_path)
        return
    if arguments.judge != 2:
        raise SystemExit("only judge v2 can be RUN; v1's prompt is gone, its files are kept to score")

    out = recheck_path if arguments.recheck else first_path
    items, events_of = collect()
    # Pooled and shuffled so arms interleave inside every batch; a different seed for the recheck.
    random.Random(97 if arguments.recheck else 13).shuffle(items)
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} \
        if out.exists() else set()
    todo = [item for item in items if item["id"] not in done]
    print(f"{len(items)} splits across {len(ARMS)} arms; {len(todo)} to grade "
          f"({(len(todo) + BATCH - 1) // BATCH} calls, batch {BATCH})  -> {out.name}")

    if arguments.dry_run:
        print(prompt_for(todo[:2], events_of))
        json.dumps(record_of(todo[0], {"recall": "对", "pct": 50, "who": None, "why": ""}),
                   ensure_ascii=False)
        print(f"\n(write path rehearsed on {todo[0]['id']}: ok)")
        return

    run(todo, events_of, out)


def run(todo, events_of, out: Path) -> None:
    """Grade every batch, fanned out the way production fans out."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from turn_truth_judge import JUDGE_TARGET, pinned_roles
    from finesub.llm.agent.agent_session_host import agent_session_scope
    from finesub.llm.client import RoleClient
    from finesub.llm.routing.config import LLMRole
    from finesub.llm.routing.execution_policy import (default_agent_slot_budgets,
                                                     load_execution_settings)
    from finesub.llm.run_context import bind_llm_worker, llm_task_slots, llm_worker_context

    # ⚠ Raised on the user's resolved settings (a fresh ExecutionSettings would reset every other
    # field), and the budgets are built FROM these settings, first -- see the module docstring.
    settings = replace(load_execution_settings(), local_agent_max_parallel=PARALLEL,
                       local_agent_timeout_seconds=CALL_TIMEOUT)
    budgets = default_agent_slot_budgets(settings)
    for budget in budgets:
        if getattr(budget, "limit", PARALLEL) != PARALLEL:
            print(f"⚠ an agent pool was already built at limit {budget.limit}, not {PARALLEL}; "
                  f"fan-out will be capped there", flush=True)

    local = threading.local()
    write_lock = threading.Lock()

    def client():
        if not hasattr(local, "client"):
            local.client = RoleClient(role_configs=pinned_roles(JUDGE_TARGET),
                                      execution_settings=settings)
        return local.client

    def grade(offset: int) -> None:
        batch = todo[offset:offset + BATCH]
        answers, target = {}, ""
        began = time.perf_counter()
        for attempt in range(4):
            try:
                reply = client().complete(
                    LLMRole.GENERAL_CAPABLE,
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt_for(batch, events_of)}])
            except Exception as error:  # noqa: BLE001 - a transient provider error
                # ⚠ A TIMEOUT is not transient: resending the same rows only times out again.
                # Matched by name so this script does not import driver internals.
                if "Timeout" in type(error).__name__:
                    print(f"batch {offset}: TIMED OUT after "
                          f"{time.perf_counter() - began:.0f}s -- not retried; "
                          f"rerun with a smaller BATCH to grade these {len(batch)} rows",
                          flush=True)
                    return
                print(f"batch {offset}: {type(error).__name__} {str(error)[:90]}; "
                      f"retry {attempt + 1}", flush=True)
                time.sleep(15 * (attempt + 1))
                continue
            target = str(getattr(reply, "target_id", ""))
            if target != JUDGE_TARGET:
                raise SystemExit(f"⚠ the grading call landed on {target!r}, not {JUDGE_TARGET!r}.")
            answers = parse(reply.content)
            if answers:
                break
            print(f"batch {offset}: no <grades> rows parsed; retry {attempt + 1}", flush=True)
        if not answers:
            print(f"batch {offset}: gave up", flush=True)
            return
        with write_lock, out.open("a", encoding="utf-8") as handle:
            for index, item in enumerate(batch):
                if index in answers:
                    handle.write(json.dumps(record_of(item, answers[index]),
                                            ensure_ascii=False) + "\n")
        print(f"  batch {offset:3d} [{target}] {len(answers)}/{len(batch)} graded "
              f"in {time.perf_counter() - began:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    offsets = list(range(0, len(todo), BATCH))
    started = time.perf_counter()
    with llm_task_slots(budgets), agent_session_scope():
        context = llm_worker_context()
        try:
            with ThreadPoolExecutor(max_workers=min(PARALLEL, len(offsets)),
                                    thread_name_prefix="judge",
                                    initializer=bind_llm_worker,
                                    initargs=(context,)) as pool:
                futures = [pool.submit(grade, offset) for offset in offsets]
                for future in futures:
                    try:
                        future.result()
                    except SystemExit:
                        for pending in futures:
                            pending.cancel()
                        raise
        finally:
            context.release_lanes()
    print(f"{len(offsets)} batches in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
