"""Evaluate native final annotations before Sortformer-specific raster/smoothing."""
from __future__ import annotations

import argparse
import json

import numpy as np
import accuracy_lab as lab
from accuracy_identity import identity_score
from accuracy_v2 import OUT, DEV


def score_timeline(tracks, columns, stable, grid, people):
    from finesub.speech.diarization import segments
    original = segments.segments_from_activity
    try:
        # Both frozen scorers import this function on entry; change only the
        # supplied timeline in this isolated, single-threaded scoring process.
        segments.segments_from_activity = lambda *args, **kwargs: tracks
        placeholder = np.zeros((1, columns))
        return lab.score(placeholder, stable, grid, people), identity_score(placeholder, stable, grid)
    finally:
        segments.segments_from_activity = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", default=DEV)
    args = p.parse_args()
    if any(i.startswith("onetonton-") for i in args.ids):
        if not all(i.startswith("onetonton-") for i in args.ids):
            p.error("Run development and Onetonton corpora separately")
        from accuracy_cross import configure
        configure()
    from finesub.speech.diarization.segments import TrackSegment
    for ident in args.ids:
        stable, _, grid, people, *_ = lab.rm.episode(ident, lab.rm.sources()[ident])
        for parent in ("community", "diarizen", "dz_community"):
            if parent == "community":
                source = lab.OUT/ident/"community-raw.json"
                if not source.exists():
                    continue
                rows = [(r["start"], r["end"], r["speaker"]) for r in json.loads(source.read_text())["diarization"]]
            else:
                source = OUT/ident/f"{parent}.rttm"
                if not source.exists():
                    continue
                rows = []
                for line in source.read_text().splitlines():
                    f = line.split()
                    rows.append((float(f[3]), float(f[3])+float(f[4]), f[7]))
            speakers = sorted({s for _, _, s in rows}, key=(int if parent == "diarizen" else str))
            tracks = [TrackSegment(speakers.index(s), a, b) for a, b, s in rows]
            tracks.sort(key=lambda t: (t.start, t.track))
            metrics, bcubed = score_timeline(tracks, max(1, len(speakers)), stable, grid, people)
            arm = f"{parent}_native"
            result = dict(arm=arm, metrics=metrics, bcubed=bcubed,
                identity=dict(audio=lab.digest(lab.source(ident)), source=lab.digest(source), script=lab.digest(__file__)))
            (OUT/ident/f"{arm}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(ident, arm, metrics["fixed"], bcubed["f1"], flush=True)


if __name__ == "__main__":
    main()
