"""Parse the 姐姐咚咚 refined ASS tracks and describe the axis before anyone leans on it.

These are fansub tracks for a Japanese three-person show, and the style name carries the
**speaker** (`樱川惠（1080p）`, `小日向美香（1080p）`, `青木佑磨（1080p）`, plus `注释` for the
translator's notes). That makes them the first material we have with **human speaker labels on
real Japanese multi-person audio** -- the gap §6 has been carrying.

⚠ Read `docs/knowledge.md`「精修字幕怎么读」first. The text is a Chinese translation, so it can
never answer "which characters did this person say"; this module exists to find out what the axis
*can* answer, by measuring the axis itself rather than assuming.

  python refined_ass.py describe            # the axis, before any audio
"""

from __future__ import annotations

import argparse
import collections
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "refined-multi" / "姐姐咚咚"
NOTE = "注释"
#: Aegisub appends `[1]`, `[2]` when styles collide while merging files, and #2 carries copies
#: named `…（1080p） - 复制` / `… - 复制-furigana`; all the same speaker. Without the tail the
#: copies read as three extra speakers (9 lines) -- the 14 903 vs 14 912 of plan 5p.1.
DEDUPE = re.compile(r"^(.*?)（\d+p）(?:\[\d+\])?(?:\s*-\s*复制(?:-furigana)?)?$")


def speaker(style):
    match = DEDUPE.match(style.strip())
    return match.group(1) if match else style.strip()


def seconds(stamp):
    hours, minutes, rest = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def events(path):
    """(start, end, speaker, text) per dialogue line, in file order."""
    out = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if "furigana" in parts[3]:
            continue  # a ruby overlay on someone's line, not a line of its own (audit 5q)
        text = re.sub(r"\{[^}]*\}", "", parts[9]).replace(r"\N", " ").strip()
        out.append((seconds(parts[1]), seconds(parts[2]), speaker(parts[3]), text))
    return out


def episodes():
    return sorted(glob.glob(str(ROOT / "*" / "*.ass")), key=lambda p: int(re.search(r"#(\d+)", p).group(1)))


def describe():
    print(f"{len(episodes())} 集\n")
    totals = collections.Counter()
    snapped = gaps = pairs = identical = partial = chorus = contained = 0
    notes_on_speech = lines_total = notes_total = 0
    durations, runs, switches, chars = [], [], [], []
    for path in episodes():
        rows = [r for r in events(path) if r[2] != NOTE and r[3]]
        notes = [r for r in events(path) if r[2] == NOTE]
        notes_total += len(notes)
        for _, _, who, _ in rows:
            totals[who] += 1
        order = sorted(rows)
        for (a0, a1, _, _), (b0, _, _, _) in zip(order, order[1:]):
            if abs(a1 - b0) < 0.005:
                snapped += 1
            elif a1 < b0:
                gaps += 1
            durations.append(a1 - a0)
        # Every overlapping pair, not only neighbours in start order: a long line overlaps
        # several later ones. The first version compared neighbours and reported 4171 pairs
        # where there are 4590 (audit 5q) -- plan 5p.1's numbers came from a scratch script.
        for i, (a0, a1, who_a, text_a) in enumerate(order):
            for b0, b1, who_b, text_b in order[i + 1:]:
                if b0 >= a1 - 0.005:
                    break
                if who_a == who_b:
                    continue
                pairs += 1
                if abs(a0 - b0) < 0.005 and abs(a1 - b1) < 0.005:
                    identical += 1
                    chorus += text_a == text_b
                else:
                    partial += 1
                    contained += b1 <= a1 + 0.005
        # Turn shape: a turn is a run of consecutive lines by one speaker (plan 5p.3).
        run = 1
        for (a0, _, who_a, text_a), (b0, _, who_b, _) in zip(order, order[1:]):
            chars.append(len(text_a))
            if who_a == who_b:
                run += 1
            else:
                runs.append(run)
                run = 1
                switches.append(b0 - a0)
        runs.append(run)
        notes_on_speech += sum(1 for n0, n1, _, _ in notes if any(r0 < n1 and r1 > n0 for r0, r1, _, _ in rows))
        lines_total += len(rows)
        print(f"  {Path(path).name:<34} 行 {len(rows):>5}  注释 {len(notes):>3}  "
              f"说话人 {sorted({r[2] for r in rows})}")
    print(f"\n说话人行数合计：{dict(totals)}")
    print(f"相邻行：end == 下一行 start 的 {snapped}，中间有空隙的 {gaps}  "
          f"（{snapped / max(snapped + gaps, 1):.1%} 是贴着的）")
    print(f"不同说话人时间重叠的行对：{pairs}（其中起止**完全相同** {identical}，部分重叠 {partial}）")
    print(f"  起止完全相同且两人写同一句（齐声）：{chorus}")
    print(f"  部分重叠里后一行整个落在前一行之内（插话、主句不切）：{contained}/{partial} = {contained / max(partial, 1):.0%}"
          f"  ⚠ 结束时间被顶到下一行，是上界")
    durations.sort()
    print(f"行时长：中位 {durations[len(durations) // 2]:.2f} s，"
          f"四分位 {durations[len(durations) // 4]:.2f}–{durations[3 * len(durations) // 4]:.2f} s")

    def quartiles(values):
        values = sorted(values)
        return values[len(values) // 2], values[len(values) // 4], values[3 * len(values) // 4]

    print(f"\n总行数 {lines_total}，换人 {len(switches)} 次（{len(switches) / max(lines_total, 1):.0%} 的行是换人）")
    median, low, high = quartiles(runs)
    print(f"一个人连说几行：中位 {median}，四分位 {low}–{high}；只说一行就换人的占 {sum(r == 1 for r in runs) / len(runs):.0%}")
    median, low, high = quartiles(switches)
    print(f"换人时两行的起点差：中位 {median:.2f} s，四分位 {low:.2f}–{high:.2f} s；"
          f"< 0.5 s（抢话）占 {sum(g < 0.5 for g in switches) / len(switches):.0%}")
    median, low, high = quartiles(chars)
    print(f"每行字数（中译）：中位 {median}，四分位 {low}–{high}")
    print(f"译注压在有人说话的时间上：{notes_on_speech}/{notes_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("describe",))
    if not ROOT.exists():
        raise SystemExit(f"素材不在 {ROOT}")
    {"describe": describe}[parser.parse_args().action]()


if __name__ == "__main__":
    sys.exit(main())
