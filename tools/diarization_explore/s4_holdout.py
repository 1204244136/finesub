"""Plan 5o.9 E3: the five AliMeeting Eval sessions nothing has touched yet, laid out for the S4 scripts.

  --extract   far-field wav + TextGrid of the five sessions out of <data>/public/Eval_Ali.tar.gz;
              the 8-channel array is mixed to mono exactly as the first three were
              (`ffmpeg -ac 1 -ar 16000`, checked bit-for-bit against R8001_M8004.wav)
  --collect   copy each production run's -stable.json / -diar.json / -vocal.ogg from
              out/<session>/ into <work>/ali-holdout/<session>/ and write projection.json
              with the 5h.2 overlap regions (no decodes), as s4_ami does
Run the pipeline in between (serially, one session at a time):
  python -m finesub.pipeline <data>/public/alimeeting/audio/<s>.wav --stage stable --diarize acoustics --language zh
Then every S4 script with DIAR_S4_CORPUS=ali-holdout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import s4_headroom as ref
import s4_projection as proj

SESSIONS = ("R8003_M8001", "R8007_M8011", "R8008_M8013", "R8009_M8019", "R8009_M8020")
ROOT = ref.DATA / "public" / "alimeeting"
ARCHIVE = ref.DATA / "public" / "Eval_Ali.tar.gz"
REPO = Path(__file__).resolve().parents[2]


def extract():
    (ROOT / "audio").mkdir(parents=True, exist_ok=True)
    (ROOT / "textgrid").mkdir(parents=True, exist_ok=True)
    wanted = {s for s in SESSIONS if not (ROOT / "audio" / f"{s}.wav").exists() or not (ROOT / "textgrid" / f"{s}.TextGrid").exists()}
    if not wanted:
        print("all five already there")
        return
    with tempfile.TemporaryDirectory(dir=ref.WORK) as scratch, tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:  # one streaming pass; the far wavs are 8-channel, ~0.5 GB each
            name = Path(member.name).name
            session = next((s for s in wanted if name.startswith(s)), None)
            if session is None:
                continue
            if "/Eval_Ali_far/textgrid_dir/" in member.name and name.endswith(".TextGrid"):
                (ROOT / "textgrid" / f"{session}.TextGrid").write_bytes(archive.extractfile(member).read())
                print(session, "TextGrid")
            elif "/Eval_Ali_far/audio_dir/" in member.name and name.endswith(".wav"):
                raw = Path(scratch) / name
                with archive.extractfile(member) as source, raw.open("wb") as target:
                    shutil.copyfileobj(source, target)
                target = ROOT / "audio" / f"{session}.wav"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-ac", "1", "-ar", "16000", str(target)], check=True)
                raw.unlink()
                print(session, "audio", round(target.stat().st_size / 1e6, 1), "MB")


def collect():
    """Also lays out our own Japanese stream (DIAR_S4_CORPUS=ja, plan 5o.9 E4) the same way."""
    assert ref.CORPUS in ("ali-holdout", "ja", "callhome-ja"), "run with DIAR_S4_CORPUS=ali-holdout (or ja / callhome-ja)"
    for s in ref.SESSIONS:
        source, target = REPO / "out" / s, proj.ALI / s
        target.mkdir(parents=True, exist_ok=True)
        for suffix in ("-stable.json", "-diar.json", "-vocal.ogg"):
            shutil.copy2(source / f"{s}{suffix}", target / f"{s}{suffix}")
        tracks = json.loads((target / f"{s}-diar.json").read_text(encoding="utf-8"))["tracks"]
        active = proj.activity(tracks, int(max(t["end"] for t in tracks) / proj.FRAME) + 2)
        regions = [{"start": a, "end": b, "sham": [], "masked": {}} for a, b in proj.regions(active)]
        (target / "projection.json").write_text(json.dumps(regions), encoding="utf-8")
        print(s, "regions", len(regions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--collect", action="store_true")
    arguments = parser.parse_args()
    if arguments.extract:
        extract()
    if arguments.collect:
        collect()


if __name__ == "__main__":
    main()
