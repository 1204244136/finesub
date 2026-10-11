"""A per-file signal for 串轨 (several people in one track), from re-clustering alone.

Both fixes measured on 2026-10-04 -- a looser vbx clustering and MOSS -- help only on files whose
tracks hold several people, and cost on the rest, so neither can be applied blind. Candidate signal:
re-cluster the cached voiceprints one notch looser (AHC 0.5, `oversplit_sweep.py`) and see whether a
LARGE production track comes apart into large pieces. A clean track either stays whole or sheds
crumbs; a track holding two people should split along them.

Truth here is the track's purity against the reference (share of its speech time that belongs to
its majority person), on the same time grid for both corpora: the survey's 星穹铁道 reference lines
and the W2 姐姐咚咚 grid. Reads only caches; no model.

    python tools/diarization_explore/crosstalk_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oversplit_sweep as osw  # noqa: E402

STEP = 0.08
LARGE = 0.10   # a track worth asking about: >= 10 % of the file's track time
PIECE = 0.15   # a piece of it that counts: >= 15 % of that track


def raster(tracks, frames):
    ids = sorted({t["track"] for t in tracks})
    grid = np.zeros((frames, len(ids)), dtype=bool)
    index = {k: i for i, k in enumerate(ids)}
    for t in tracks:
        grid[int(t["start"] / STEP):int(np.ceil(t["end"] / STEP)), index[t["track"]]] = True
    return grid


def reference_grid(ident, frames):
    if ident.startswith("hina-"):
        sys.path.insert(0, str(osw.n3.WT))
        from tools.ts_diar_explore import score as S
        lines = [x for x in S.reference(ident) if x.get("speaker")]
        people = sorted({x["speaker"] for x in lines})
        grid = np.zeros((frames, len(people)), dtype=bool)
        for x in lines:
            grid[int(x["start"] / STEP):int(np.ceil(x["end"] / STEP)), people.index(x["speaker"])] = True
        return grid
    import nemotron3_cost as nc
    import relabel_eval as ev
    nc.corpus_keys("onetonton")
    _, _, grid, _, _ = ev.episode(int(ident.rsplit("-", 1)[1]))
    out = np.zeros((frames, grid.shape[1]), dtype=bool)
    n = min(frames, len(grid))
    out[:n] = grid[:n]
    return out


def row_purity(ident, base):
    """Purity per production track at ROW level -- the unit `fixed` scores -- not per time frame.

    ⚠ The first draft read purity off a time grid and put every 姐姐咚咚 track at 0.63-0.73: that
    show's reference lines overlap heavily in time (57 % of rows have two people under them), so a
    pure track still "owns" the other person's frames. Rows are what a proofreader moves.
    `{track: purity}` for tracks holding >= LARGE of the rows' time.
    """
    from collections import Counter, defaultdict
    weights = defaultdict(Counter)
    if ident.startswith("hina-"):
        sys.path.insert(0, str(osw.n3.WT))
        from tools.ts_diar_explore import score as S
        arm = {"segments": [{"start": x["start"], "end": x["end"], "speaker": f"T{x['track']}", "text": ""} for x in base]}
        lines, overlaps = S.line_overlaps(S.reference(ident), arm, 0.0, 1e9)
        pairs, _ = S.pairs_from(lines, overlaps)
        for person, label, seconds, _ in pairs:
            if not label.startswith("∅"):
                weights[label][person] += seconds
    else:
        import nemotron3_cost as nc
        import relabel_eval as ev
        from finesub.speech.diarization.assign import assign_all
        from finesub.speech.diarization.segments import TrackSegment
        nc.corpus_keys("onetonton")
        stable, _, grid, people, _ = ev.episode(int(ident.rsplit("-", 1)[1]))
        tracks = [TrackSegment(start=x["start"], end=x["end"], track=x["track"]) for x in base]
        for row, speaker in zip(stable, assign_all(stable, tracks)):
            ids = speaker.to_dict().get("tracks") or []
            cover, _ = ev.span_cover(row, grid)
            if ids and cover.any():
                weights[f"T{ids[0]}"][people[int(cover.argmax())]] += row["end"] - row["start"]
    total = sum(sum(c.values()) for c in weights.values()) or 1.0
    return {k: max(c.values()) / sum(c.values()) for k, c in weights.items() if sum(c.values()) >= LARGE * total}


def main():
    rows = []
    for ident in osw.DEVELOPMENT + osw.CONFIRMATION:
        if not (osw.OUT / f"{ident}.npz").exists():
            continue
        base = osw.timeline(ident, 0.6, 0.8)
        loose = osw.timeline(ident, 0.5, 0.8)
        frames = int(max(t["end"] for t in base + loose) / STEP) + 2
        b, l = raster(base, frames), raster(loose, frames)
        purity = row_purity(ident, base)
        worst = min(purity.values(), default=1.0)
        total = b.sum()
        signal = 0.0
        for k in range(b.shape[1]):
            size = b[:, k].sum()
            if size < LARGE * total:
                continue
            pieces = np.sort((b[:, k][:, None] & l).sum(axis=0))[::-1] / size
            if len(pieces) > 1 and pieces[1] >= PIECE:
                signal = max(signal, float(pieces[1]))

        rows.append((ident, signal, worst))
        print(f"{ident:14} split piece {signal:.2f}   worst large-track purity {worst:.2f}", flush=True)
    flagged = [r for r in rows if r[1] > 0]
    print(f"\nflagged {len(flagged)}/{len(rows)}; purity of flagged {[round(r[2], 2) for r in flagged]}; "
          f"lowest purity among unflagged {min((r[2] for r in rows if r[1] == 0), default=float('nan')):.2f}")


if __name__ == "__main__":
    main()
