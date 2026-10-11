"""Production's current speaker default (DiariZen) on the evaluation episodes' vocal tracks.

The `out/refined-2609` runs predate the DiariZen default -- their `-diar.json` is Sortformer v2.1 (4 slots),
which is not what production has today. This reruns only the diarization stage, the same call
`finesub.stages` makes (`run_diarization` + `write_timeline`, default smoothing, no speaker cap), on the same
vocal track, into `data/ts-diar/episodes/<id>/diarizen-diar.json` (main checkout). Needs the [asr] env:

    PYTHONPATH=<worktree>/src C:/Users/Carl/miniconda3/envs/asr/python.exe -m tools.ts_diar_explore.dz_baseline hina-72 ...
"""

from __future__ import annotations

import sys
import time

from finesub.speech.diarization import stage

from .common import EPISODES, vocal_track


def main() -> None:
    for item in sys.argv[1:]:
        out = EPISODES / item / "diarizen-diar.json"
        if out.exists():
            continue
        t0 = time.monotonic()
        result = stage.run_diarization(vocal_track(item), device="cuda", backend="diarizen")
        stage.write_timeline(result, out)
        tracks = len({seg.track for seg in result.tracks})
        print(f"{item}: {tracks} tracks, {time.monotonic() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
