"""魔裁 (yuka) galgame stream: timestamped SPEAKER truth, recovered from the game's own script.

Plan 5r.2 「摸清 Sortformer 本身」 needs two things this corpus has and nothing else we own does:
recordings that are **6-11 hours** long (our previous maximum was 1.41 h) and **11-15 distinct speakers per
hour** (previous maximum 8). Those are exactly the two axes the streaming speaker cache is said to degrade on.

The chain -- all of it already on disk, no OCR pass and no inference needed:

  script-raw/Scripts/**.bytes   Naninovel scenario. Every line is tagged `# <chapter>_<Speaker><NNN>`, and
                                `script-raw/CharacterNames.bytes` maps the key to a display name.
                                ⚠ Each `#` block holds TWO `;` lines -- the speaker directive
                                (`; > Ema: |#id|`) and then the text. Only the second is text; attaching the
                                tag to the first matches nothing but Narrative, which has no directive.
  game_text/lower{1..5}_jp.ass  flowocr's old matcher: OCR cue spans carrying script text.
  -> this module                normalise both sides, look the .ass text up in the script, recover the tag.

Match rate is 98.9% on `lower1` (6029/6098 cues; 188 of those are texts more than one character says and are
dropped, leaving 5841 usable spans over 8.71 h) and 99.2% on `lower5`.

⚠ **What this is not.**
1. **Not exact boundaries.** Owner 2026-09-15: the OCR span *contains* the speech but may carry blank
   padding, so a span is an UPPER BOUND on the speech interval -- the same shape of caveat as the fansub
   axis's deliberately extended end times. Do not use it for boundary-precision metrics.
2. **Not same-recording-chain.** Game voice and the streamer's microphone are different chains, so this
   cannot speak to the same-pipeline naming dead end (gap 0.036). It is a *timeline* and *headcount*
   instrument, not an identity one.
3. **Not gold.** `lower*_jp.ass` is the old matcher's output and is known to carry errors; what this module
   trusts is only that its text is verbatim from the script (99.3-99.8% per flowocr's own index), which the
   98.9% exact-match rate corroborates.
4. **`Narrative` is a tag, not a person** (28.6% of cue time on input1), and is probably unvoiced -- check
   before counting it as anybody.

The material is read-only in another project and is never copied here (that project's own rule):
videos at `D:/ClipTranslArchive/yuka/input{1..5}.mp4` (VP9, 6.2-11.0 h, 45.1 h / 53 GB), corpus at
`<flowocr>/data/corpus/mocai/`. Provenance: that project's `data/index.md` section E and
`docs/script-corpus.md`.

Usage: `python tools/diarization_explore/mocai_truth.py [lower1..lower5]`
"""
import collections
import re
import sys
from pathlib import Path

BASE = Path("C:/Users/Carl/Documents/Carl/projects/flowocr/data/corpus/mocai")
TAG = sys.argv[1] if len(sys.argv) > 1 else "lower1"


def norm(s):
    s = re.sub(r"<ruby=[^>]*>", "", s)
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)          # <br>, </ruby>, <size=..>, <cspace=..>
    s = s.replace("\\N", "").replace("\\n", "")
    s = re.sub(r"\{[^}]*\}", "", s)                   # ASS override blocks
    return re.sub(r"\s+", "", s)


def script_index():
    """normalised jp text -> set of speaker tags."""
    idx = collections.defaultdict(set)
    order = []
    cur = None
    # Each `# <id>` block carries TWO `;` lines: the Naninovel speaker directive (`; > Ema: |#id|`) and then
    # the line itself. Only the second is text -- attaching the tag to the directive matches nothing but
    # Narrative, which has no directive.
    directive = re.compile(r"^>\s*\w+\s*:\s*\|#")
    for f in sorted((BASE / "script-raw" / "Scripts").rglob("*.bytes")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#"):
                t = line[1:].strip()
                suf = t.split("_", 1)[1] if "_" in t else t
                cur = re.sub(r"\d+$", "", suf)
            elif line.startswith("; ") and cur:
                body = line[2:]
                if directive.match(body.strip()):
                    continue
                key = norm(body)
                if key:
                    idx[key].add(cur)
                    order.append((key, cur))
                cur = None
    return idx, order


def ass_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        p = line.split(",", 9)
        def sec(x):
            h, m, s = x.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        rows.append((sec(p[1]), sec(p[2]), p[9].strip()))
    return rows


def main():
    idx, _ = script_index()
    print(f"script: {len(idx)} distinct normalised lines")
    rows = ass_rows(BASE / "game_text" / f"{TAG}_jp.ass")
    print(f"{TAG}_jp.ass: {len(rows)} cues, {rows[0][0] / 3600:.2f}-{rows[-1][1] / 3600:.2f} h")

    hit = collections.Counter()
    ambiguous = exact = miss = 0
    spans = collections.defaultdict(float)
    timeline = []
    for a, b, text in rows:
        cand = idx.get(norm(text))
        if not cand:
            miss += 1
            continue
        exact += 1
        if len(cand) > 1:
            ambiguous += 1
            continue
        who = next(iter(cand))
        hit[who] += 1
        spans[who] += b - a
        timeline.append((a, b, who))

    print(f"\nmatched {exact}/{len(rows)} ({exact / len(rows):.1%}); "
          f"of those {ambiguous} ambiguous (same text, >1 speaker) -> {len(timeline)} usable")
    print(f"unmatched {miss}")
    total = sum(spans.values())
    print(f"\nspeaker distribution over {total / 3600:.2f} h of cue time:")
    for who, n in hit.most_common(14):
        print(f"  {who:<16}{n:>6} cues  {spans[who] / 60:8.1f} min  {spans[who] / total:6.1%}")

    # how many distinct speakers are live in each hour -- this is the headcount Sortformer would face
    print("\ndistinct speakers per hour of wall clock (Narrative excluded):")
    per = collections.defaultdict(set)
    for a, _, who in timeline:
        if who != "Narrative":
            per[int(a // 3600)].add(who)
    for h in sorted(per):
        print(f"  h{h:<3} {len(per[h]):>2} speakers  {sorted(per[h])}")


if __name__ == "__main__":
    main()
