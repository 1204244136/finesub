"""Signal 1 for 串轨: the share of `review` rows per large production track, against row-level track purity.

`review` marks a row whose leading and runner-up tracks cover it almost equally. Aggregated per track, it
was the plan's first candidate for "this track holds several people". Measured 2026-10-04 on the 22 files of
`oversplit_sweep.py`: it does not separate them (a merged track has no competing track, so no review rows).

    python tools/diarization_explore/crosstalk_review.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oversplit_sweep as osw, crosstalk_signal as cs
from finesub.speech.diarization.assign import assign_all
from finesub.speech.diarization.segments import TrackSegment
for ident in osw.DEVELOPMENT + osw.CONFIRMATION:
    base = osw.timeline(ident, 0.6, 0.8)
    if ident.startswith("hina-"):
        stable = json.loads((osw.n3.MAIN / "out" / "refined-2609" / ident / f"{ident}-stable.json").read_text(encoding="utf-8"))["segments"]
    else:
        import nemotron3_cost as nc, relabel_eval as ev
        nc.corpus_keys("onetonton"); stable = ev.episode(int(ident[-2:]))[0]
    tracks = [TrackSegment(start=x["start"], end=x["end"], track=x["track"]) for x in base]
    time, rev = defaultdict(float), defaultdict(float)
    for row, sp in zip(stable, assign_all(stable, tracks)):
        d = sp.to_dict(); ids = d.get("tracks") or []
        if not ids: continue
        w = row["end"] - row["start"]; time[ids[0]] += w; rev[ids[0]] += w * bool(d.get("review"))
    total = sum(time.values())
    big = {k: rev[k] / time[k] for k in time if time[k] >= 0.10 * total}
    purity = cs.row_purity(ident, base)
    print(f"{ident:14} max review share in a large track {max(big.values()):.3f}  file review share {sum(rev.values())/total:.3f}  worst purity {min(purity.values()):.2f}")
