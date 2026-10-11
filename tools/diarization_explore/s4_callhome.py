"""Plan 5o.9 E5: CallHome Japanese (TalkBank CABank) laid out for the S4 scripts (DIAR_S4_CORPUS=callhome-ja).

  --prepare   per call: the transcribed region (first bullet - 1 s .. last bullet + 1 s) cut out of
              <data>/public/callhome-jpn/mp3/<call>.mp3, mixed to mono, 16 kHz -> audio/<call>.wav;
              the .cha utterances, shifted to the cut, -> textgrid/<call>.TextGrid (a tier per speaker)
  --collect   = s4_holdout.py --collect with DIAR_S4_CORPUS=callhome-ja
Truth text is the spoken form: `あたし [: 私]` keeps あたし; events (`&=laugh`), unintelligible
(`xxx`), codes in brackets, intonation arrows and the bullets themselves are dropped.
Run the pipeline in between (serially):
  python -m finesub.pipeline <data>/public/callhome-jpn/audio/<call>.wav --stage stable --diarize acoustics --language ja
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import s4_headroom as ref

ROOT = ref.DATA / "public" / "callhome-jpn"
BULLET = re.compile(r"\x15?(\d+)_(\d+)\x15?")
CODES = re.compile(r"\[[^\]]*\]|&=\S+|\b(?:xxx|yyy|www)\b")
EDGE = 1.0


def utterances(cha: Path):
    """(speaker, start, end, spoken text) per bullet-closed utterance."""
    out, speaker, text = [], None, ""
    for line in cha.read_text(encoding="utf-8").splitlines():
        if line.startswith("*"):
            speaker, text = line[1:line.index(":")], line[line.index(":") + 1:]
        elif line.startswith("\t") and speaker:
            text += " " + line.strip()
        else:
            continue
        found = BULLET.search(text)
        if found:
            spoken = CODES.sub(" ", BULLET.sub(" ", text)).replace("&-", "").replace("&", "")
            out.append((speaker, int(found.group(1)) / 1000, int(found.group(2)) / 1000, " ".join(spoken.split())))
            speaker, text = None, ""
    return out


def prepare():
    (ROOT / "audio").mkdir(exist_ok=True)
    (ROOT / "textgrid").mkdir(exist_ok=True)
    for call in ref.SESSIONS:
        rows = utterances(ROOT / "transcripts" / f"{call}.cha")
        begin = max(0.0, min(s for _, s, _, _ in rows) - EDGE)
        end = max(e for _, _, e, _ in rows) + EDGE
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{begin:.3f}", "-to", f"{end:.3f}",
                        "-i", str(ROOT / "mp3" / f"{call}.mp3"), "-ac", "1", "-ar", "16000", str(ROOT / "audio" / f"{call}.wav")], check=True)
        tiers = {}
        for speaker, s, e, text in rows:
            if text:
                tiers.setdefault(speaker, []).append((s - begin, e - begin, text.replace('"', "'")))
        lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", f"size = {len(tiers)}", "item []:"]
        for index, (speaker, intervals) in enumerate(sorted(tiers.items()), start=1):
            lines += [f"    item [{index}]:", '        class = "IntervalTier"', f'        name = "{call}.{speaker}"',
                      f"        intervals: size = {len(intervals)}"]
            for j, (s, e, text) in enumerate(intervals, start=1):
                lines += [f"        intervals [{j}]:", f"            xmin = {s:.3f}", f"            xmax = {e:.3f}", f'            text = "{text}"']
        (ROOT / "textgrid" / f"{call}.TextGrid").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(call, f"{(end - begin) / 60:.1f} min", {k: len(v) for k, v in tiers.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    arguments = parser.parse_args()
    assert ref.CORPUS == "callhome-ja", "run with DIAR_S4_CORPUS=callhome-ja"
    if arguments.prepare:
        prepare()


if __name__ == "__main__":
    main()
