"""Pull the three sessions' far-field TextGrids (with text) out of Eval_Ali.tar.gz."""

import sys
import tarfile
from pathlib import Path

# Paths: DATA = <repo>/data/diarization (hand-kept, main checkout), WORK =
# <repo>/tmp/diarization_explore (scratch; the three .ts go in WORK/export_v2).
import os as _os
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(_os.environ.get("DIAR_EXPLORE_DATA", ROOT / "data" / "diarization"))
WORK = Path(_os.environ.get("DIAR_EXPLORE_WORK", ROOT / "tmp" / "diarization_explore"))
archive = DATA / "public" / "Eval_Ali.tar.gz"
destination = DATA / "public" / "alimeeting" / "textgrid"
destination.mkdir(parents=True, exist_ok=True)
wanted = set(sys.argv[1:])

with tarfile.open(archive) as handle:
    for member in handle:  # streaming: no full member list, one pass
        if "/Eval_Ali_far/textgrid_dir/" in member.name and member.name.endswith(".TextGrid"):
            stem = Path(member.name).stem
            if stem in wanted:
                (destination / f"{stem}.TextGrid").write_bytes(handle.extractfile(member).read())
                print("extracted", stem)
                wanted.discard(stem)
                if not wanted:
                    break
print("missing:", sorted(wanted))
