"""Plan 5h: how much of an overlapped speaker's words does the pipeline already lose?

Reference: AliMeeting far-field TextGrid, per speaker, per utterance, with text.
Hypothesis: our -stable.json words. Score: character-bigram recall inside the
utterance's window (+-0.5 s), against a +120 s shifted window as the chance floor.
Everything about the metric is fixed in the plan before this ran.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from finesub.text import t2s_converter  # PYTHONPATH=<worktree>/src

# Paths: DATA = <repo>/data/diarization (hand-kept, main checkout), WORK =
# <repo>/tmp/diarization_explore (scratch; the three .ts go in WORK/export_v2).
import os as _os
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(_os.environ.get("DIAR_EXPLORE_DATA", ROOT / "data" / "diarization"))
WORK = Path(_os.environ.get("DIAR_EXPLORE_WORK", ROOT / "tmp" / "diarization_explore"))
#: Plan 5h.20: DIAR_S4_CORPUS=ami reruns the S4 scripts on three AMI meetings (English).
#: Default AliMeeting, so every earlier number reproduces unchanged.
CORPUS = _os.environ.get("DIAR_S4_CORPUS", "alimeeting")
#: Plan 5o.9 E3 adds the two holdouts: the untouched AliMeeting Eval sessions and six more AMI
#: meetings. One table answers every per-corpus question -- the work dir, the language, the audio.
#: (work dir, public audio corpus or None, sessions, Whisper language). `ja` is our own Japanese
#: stream (5o.9 E4): no text truth and no public audio -- decodes and the owner's page only.
CORPORA = {
    "alimeeting": ("ali", "alimeeting", ("R8001_M8004", "R8009_M8018", "R8007_M8010"), "zh"),
    "ali-holdout": ("ali-holdout", "alimeeting", ("R8003_M8001", "R8007_M8011", "R8008_M8013", "R8009_M8019", "R8009_M8020"), "zh"),
    "ami": ("ami", "ami", ("ES2004a", "IS1009a", "TS3003a", "EN2002a", "EN2002b", "EN2002c"), "en"),
    "ami-holdout": ("ami-holdout", "ami", ("IS1006d", "IS1003d", "ES2015d", "EN2002d", "TS3006d", "TS3005d"), "en"),
    "ja": ("ja", None, ("nepolabo",), "ja"),
    # 5o.9 E5: the calls the overlap rule picked, each cut to its transcribed region.
    # Pre-registered at 20; owner cut it to 10 on 2026-09-11 ("暂时没必要那么多").
    # Which 10: every other one down the overlap ranking (ranks 1, 3, 5 ... 19), not the top 10 --
    # truncating the ranking would leave only the most-overlapped half, while this keeps the same
    # overlap span with half the calls. The ranks left out are the extension set, in order:
    # 2188, 2235, 2225, 2218, 1967, 2223, 2096, 1642, 2210, 1928.
    "callhome-ja": ("callhome-ja", "callhome-jpn", ("0930", "1425", "1731", "1509", "1048",
                                                    "2231", "1725", "2212", "1832", "0924"), "ja"),
}
WORK_NAME, AUDIO_CORPUS, SESSIONS, LANGUAGE = CORPORA[CORPUS]
ENGLISH = LANGUAGE == "en"
#: Japanese keeps kana; DIAR_S4_READING=1 compares reading forms (both sides to hiragana, pykakasi)
#: -- 5o.9 E5's side report, since あたし / 私 are one word written two ways.
READING = LANGUAGE == "ja" and _os.environ.get("DIAR_S4_READING") == "1"
GRIDS = DATA / "public" / (AUDIO_CORPUS or "none") / "textgrid"
PAD, SHIFT, MIN_CHARS = 0.5, 120.0, 4
OVERLAPPED, CLEAN = 0.50, 0.05

INTERVAL = re.compile(r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"(.*?)\"", re.DOTALL)
TIER = re.compile(r'name = "(.*?)"')
KEEP = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
T2S = t2s_converter()


_WORDS: dict[str, str] = {}


def normalize(text: str) -> str:
    if ENGLISH:
        # English is scored per word: each word becomes one token (a CJK code point, so
        # every character-level tool downstream works unchanged). Consistent within a process.
        # Idempotent, because callers normalize already-normalized text again: a stand-in
        # code point passes through as itself.
        out = []
        for token in re.findall(r"[一-鿿]|[a-z0-9']+", text.lower()):
            if "一" <= token <= "鿿":
                out.append(token)
                continue
            word = token.strip("'")
            if word:
                out.append(_WORDS.setdefault(word, chr(0x4E00 + len(_WORDS))))
        return "".join(out)
    if LANGUAGE == "ja":  # no traditional->simplified conversion: it would rewrite Japanese kanji (過 -> 过)
        text = "".join(KEEP_JA.findall(text)).lower()
        if READING:
            text = "".join(item["hira"] for item in _kakasi().convert(text))
        return text
    text = T2S.convert(text) if T2S else text
    return "".join(KEEP.findall(text)).lower()


KEEP_JA = re.compile(r"[ぁ-ゖァ-ヺー一-鿿々A-Za-z0-9]")
_KAKASI = []


def _kakasi():
    if not _KAKASI:
        import pykakasi

        _KAKASI.append(pykakasi.kakasi())
    return _KAKASI[0]


def bigrams(text: str) -> Counter:
    return Counter(text[i : i + 2] for i in range(len(text) - 1))


def reference(session):
    raw = (GRIDS / f"{session}.TextGrid").read_text(encoding="utf-8", errors="replace")
    rows = []
    for block in re.split(r"item \[\d+\]:", raw)[1:]:
        name = TIER.search(block)
        for start, end, text in INTERVAL.findall(block):
            clean = normalize(text)
            if clean:
                rows.append({"spk": name.group(1) if name else "?", "start": float(start), "end": float(end), "text": clean})
    return rows


def hypothesis(session):
    payload = json.loads((WORK / WORK_NAME / session / f"{session}-stable.json").read_text(encoding="utf-8"))
    words = []
    for segment in payload["segments"]:
        for word in segment.get("words") or ():
            words.append(((word["start"] + word["end"]) / 2, normalize(word["word"])))
    words.sort()
    return words


def window_text(words, start, end):
    return "".join(text for middle, text in words if start - PAD <= middle <= end + PAD)


def covered_by_others(row, rows):
    spans = sorted(
        (max(row["start"], other["start"]), min(row["end"], other["end"]))
        for other in rows
        if other["spk"] != row["spk"] and other["start"] < row["end"] and other["end"] > row["start"]
    )
    total, cursor = 0.0, row["start"]
    for begin, finish in spans:
        begin = max(begin, cursor)
        if finish > begin:
            total += finish - begin
            cursor = finish
    return total / max(row["end"] - row["start"], 1e-6)


def recall(ref: str, hyp: str) -> float:
    wanted = bigrams(ref)
    return sum((wanted & bigrams(hyp)).values()) / max(sum(wanted.values()), 1)


def score(session):
    rows = [row for row in reference(session) if len(row["text"]) >= MIN_CHARS]
    words = hypothesis(session)
    duration = max(row["end"] for row in rows)
    for index, row in enumerate(rows):
        row["id"] = index
        row["cover"] = covered_by_others(row, rows)
        row["recall"] = recall(row["text"], window_text(words, row["start"], row["end"]))
        shifted = (row["start"] + SHIFT) % duration
        row["chance"] = recall(row["text"], window_text(words, shifted, shifted + row["end"] - row["start"]))
    return rows


def pairs(rows):
    """For each overlapped utterance, its most-overlapping partner; unique pairs."""
    seen = set()
    for row in rows:
        if row["cover"] < OVERLAPPED:
            continue
        partners = [
            (min(row["end"], other["end"]) - max(row["start"], other["start"]), other)
            for other in rows
            if other["spk"] != row["spk"] and other["start"] < row["end"] and other["end"] > row["start"]
        ]
        if not partners:
            continue
        _, partner = max(partners, key=lambda item: item[0])
        key = tuple(sorted((row["id"], partner["id"])))
        if key not in seen:
            seen.add(key)
            yield row, partner


def report(label, rows):
    clean = [row for row in rows if row["cover"] <= CLEAN]
    overlapped = [row for row in rows if row["cover"] >= OVERLAPPED]
    rc, ro = mean(r["recall"] for r in clean), mean(r["recall"] for r in overlapped)
    cc, co = mean(r["chance"] for r in clean), mean(r["chance"] for r in overlapped)
    duo = list(pairs(rows))
    better = [max(a["recall"], b["recall"]) for a, b in duo]
    worse = [min(a["recall"], b["recall"]) for a, b in duo]
    one_side = sum(1 for a, b in duo if max(a["recall"], b["recall"]) >= 0.6 and min(a["recall"], b["recall"]) < 0.3)
    print(
        f"{label:<12} clean n={len(clean):4d} recall={rc:.3f} (chance {cc:.3f}) | "
        f"overlapped n={len(overlapped):4d} recall={ro:.3f} (chance {co:.3f}) | H={100 * (rc - ro):+.1f} pp"
    )
    if duo:
        print(
            f"{'':<12} pairs n={len(duo)}: better {mean(better):.3f}, worse {mean(worse):.3f}; "
            f"one side kept (>=0.6 vs <0.3): {one_side}/{len(duo)} = {one_side / len(duo):.1%}"
        )


def main():
    pooled = []
    for session in SESSIONS:
        rows = score(session)
        report(session, rows)
        pooled.extend(rows)
    report("POOLED", pooled)


if __name__ == "__main__":
    main()
