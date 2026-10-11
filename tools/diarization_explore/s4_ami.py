"""Plan 5h.20: put three AMI meetings where the S4 scripts expect AliMeeting (DIAR_S4_CORPUS=ami).

  --download   annotations (ami_public_manual_1.6.2.zip) + three Mix-Headset wavs
               into <data>/public/ami/ (CC BY 4.0, University of Edinburgh)
  --textgrid   segments + words XML -> one TextGrid per meeting (a tier per speaker),
               the format s4_headroom.reference reads
  --collect    copy the production run's -stable.json / -diar.json / -vocal.ogg from
               out/<meeting>/ into <work>/ami/<meeting>/ and write projection.json with the
               5h.2 overlap regions only (no decodes -- 5h.20 needs just the regions)
Run the pipeline in between:
  python -m finesub.pipeline <data>/public/ami/audio/<m>.wav --stage stable --diarize acoustics --language en
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import s4_headroom as ref
import s4_projection as proj

#: DIAR_S4_CORPUS=ami: the 5h.20 six (EN2002a-c added before scoring); =ami-holdout: the six
#: plan 5o.9 E3's rule picked (highest annotated overlap share, unused, Mix-Headset available).
assert ref.ENGLISH, "run with DIAR_S4_CORPUS=ami or ami-holdout"
MEETINGS = ref.SESSIONS
BASE = "https://groups.inf.ed.ac.uk/ami"
ROOT = ref.DATA / "public" / "ami"
ZIP = ROOT / "ami_public_manual_1.6.2.zip"
REPO = Path(__file__).resolve().parents[2]


def download():
    (ROOT / "audio").mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        urllib.request.urlretrieve(f"{BASE}/AMICorpusAnnotations/ami_public_manual_1.6.2.zip", ZIP)
    for m in MEETINGS:
        target = ROOT / "audio" / f"{m}.wav"
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE}/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav", target)
    print("ami", round(sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) / 1e6, 1), "MB")


WORD = re.compile(r'<w\b[^>]*?starttime="([\d.]+)"[^>]*?endtime="([\d.]+)"[^>]*>([^<]*)</w>')
SEGMENT = re.compile(r'<segment\b[^>]*?transcriber_start="([\d.]+)"[^>]*?transcriber_end="([\d.]+)"')


def textgrid():
    (ROOT / "textgrid").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        for m in MEETINGS:
            tiers = []
            for spk in "ABCDE":
                words_name = next((n for n in names if n.endswith(f"words/{m}.{spk}.words.xml")), None)
                segs_name = next((n for n in names if n.endswith(f"segments/{m}.{spk}.segments.xml")), None)
                if not words_name or not segs_name:
                    continue
                words = [(float(s), float(e), t.strip()) for s, e, t in WORD.findall(archive.read(words_name).decode("utf-8"))]
                intervals = []
                for s, e in SEGMENT.findall(archive.read(segs_name).decode("utf-8")):
                    s, e = float(s), float(e)
                    text = " ".join(t for ws, we, t in words if t and s - 0.05 <= (ws + we) / 2 <= e + 0.05)
                    if text:
                        intervals.append((s, e, text.replace('"', "'")))
                tiers.append((spk, intervals))
            lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", f"size = {len(tiers)}", "item []:"]
            for index, (spk, intervals) in enumerate(tiers, start=1):
                lines += [f"    item [{index}]:", '        class = "IntervalTier"', f'        name = "{m}.{spk}"',
                          f"        intervals: size = {len(intervals)}"]
                for j, (s, e, text) in enumerate(intervals, start=1):
                    lines += [f"        intervals [{j}]:", f"            xmin = {s:.3f}", f"            xmax = {e:.3f}", f'            text = "{text}"']
            (ROOT / "textgrid" / f"{m}.TextGrid").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(m, [(spk, len(iv)) for spk, iv in tiers])


def collect():
    for m in MEETINGS:
        source, target = REPO / "out" / m, proj.ALI / m
        target.mkdir(parents=True, exist_ok=True)
        for suffix in ("-stable.json", "-diar.json", "-vocal.ogg"):
            shutil.copy2(source / f"{m}{suffix}", target / f"{m}{suffix}")
        tracks = json.loads((target / f"{m}-diar.json").read_text(encoding="utf-8"))["tracks"]
        active = proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)
        regions = [{"start": s, "end": e, "sham": [], "masked": {}} for s, e in proj.regions(active)]
        (target / "projection.json").write_text(json.dumps(regions), encoding="utf-8")
        print(m, "regions", len(regions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--textgrid", action="store_true")
    parser.add_argument("--collect", action="store_true")
    arguments = parser.parse_args()
    if arguments.download:
        download()
    if arguments.textgrid:
        textgrid()
    if arguments.collect:
        collect()


if __name__ == "__main__":
    main()
