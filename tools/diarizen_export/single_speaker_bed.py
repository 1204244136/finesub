"""Does the single-speaker evidence separate one voice from two? Cut the test bed out of truth.

⚠ No collection. onetonton has per-line speaker truth, so concatenating one person's lines gives
genuinely single-speaker audio with the SAME channel, the same background music and the same
separation residue -- labelled by construction, with nothing for a human to confirm. Any two
people concatenated gives the hard case the original test bed had none of: two similar voices
from the same show, rather than the 3-to-23-speaker material on which any such threshold looks
good.

⚠ One limit, stated up front: concatenation changes the turn statistics. This bed is valid for
judging an EMBEDDING-based measure and invalid for anything that reads turn-taking. The measure
under test reads embeddings only.

⚠⚠ **What this bed does NOT test** (2026-09-20, measured): every single-speaker arm came back with
ONE cluster, and the factor is identically zero when K=1 -- `ELBO_K` and `ELBO_1` are then the same
number. So the clean separation it reports (one: max 0.0; two: min 801.4) confirms the measure is
sound and adds nothing over "how many clusters did AHC find". The case a veto exists for is
**over-splitting on single-speaker material**, and concatenated exclusive speech has none of what
causes it: no music, no game audio, no separation residue, no other voices. That case needs real
single-speaker material.

⚠ **2026-09-20: that material IS on disk**, and the note here used to say it was not.
`assets/kaguya60.flac` is the 60-minute file findings 9 measured the 2-track over-split on, and
`assets/kaguya.flac` is the same source at 3h17m. What was deleted in 2026-08-31 was the
SEPARATION output under `out/kaguya60/`, whose own README says the source is in `assets/` and
can be re-separated at any time. `assets/yui.mp3` (streamer plus movie audio) and
`assets/saki_audio.m4a` (one performer, singing and MC) are two more of the same shape.

    python tools/diarizen_export/single_speaker_bed.py --episodes 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BED = REPO / "tmp" / "diarizen-bed"
EXPORT_DIR = REPO / "tmp" / "diarizen-export"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"

#: Enough audio for the clustering to have something to say; below this the answer is noise.
MIN_SECONDS = 180.0


def spans_by_person(episode: int):
    """`{person: [(start, end)]}` from the refined axis, merged where adjacent."""
    import backend_cost as bc

    data = bc.ev.episode(episode)
    if data is None:
        return {}
    stable, _, grid, people, _ = data
    from finesub.speech.diarization.segments import FRAME_SECONDS

    spans: dict[str, list] = {}
    for index in range(grid.shape[1]):
        frames = np.flatnonzero(grid[:, index])
        if not len(frames):
            continue
        # ⚠ Exclusive frames only: a span where two people overlap belongs to neither arm.
        frames = frames[grid[frames].sum(axis=1) == 1]
        if not len(frames):
            continue
        runs, start = [], frames[0]
        for a, b in zip(frames, frames[1:]):
            if b != a + 1:
                runs.append((start * FRAME_SECONDS, (a + 1) * FRAME_SECONDS))
                start = b
        runs.append((start * FRAME_SECONDS, (frames[-1] + 1) * FRAME_SECONDS))
        spans[people[index]] = [r for r in runs if r[1] - r[0] >= 0.5]
    return spans


def build(episode: int, who: list[str], spans, destination: Path) -> float:
    """Concatenate these people's exclusive spans into one wav. Returns its duration."""
    import soundfile

    from check_port import source_for

    signal, rate = soundfile.read(source_for(f"onetonton-{episode:02d}"), dtype="float32",
                                  always_2d=True)
    mono = signal.mean(axis=1)
    pieces = []
    for person in who:
        for start, end in spans.get(person, []):
            pieces.append(mono[int(start * rate):int(end * rate)])
    if not pieces:
        return 0.0
    joined = np.concatenate(pieces)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # ⚠ float32 wav, not FLAC: FLAC is integer-only, and a requantisation of the input reshuffles
    # this chain (findings 5). This bed exists to measure the model, not the container.
    soundfile.write(destination, joined, rate, subtype="FLOAT")
    return len(joined) / rate


def measure(path: Path) -> "tuple[float, int]":
    """`(log Bayes factor, clusters)` for one built file."""
    import soundfile
    import torch

    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_embed as embedder
    from finesub.speech.diarization import diarizen_runner as runner

    signal, rate = soundfile.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(signal.mean(axis=1))[None]
    local = runner.segment(EXPORT_DIR / "diarizen_segmentation.ts", waveform).data
    filtered = runner.median_filtered(local)
    embeddings = embedder.embed(EXPORT_DIR / "diarizen_embedding.ts", waveform, filtered)
    result = clusterer.cluster(embeddings, filtered, PLDA)
    return result.single_speaker_log_bayes, int(result.centroids.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out", default=str(BED / "results.json"))
    args = parser.parse_args()

    rows = []
    for episode in args.episodes:
        spans = spans_by_person(episode)
        people = sorted(spans)
        if len(people) < 2:
            print(f"#{episode}: fewer than two people with exclusive speech, skipped")
            continue

        for arm, who in ([("one", [person]) for person in people]
                         + [("two", list(pair)) for pair in
                            zip(people, people[1:] + people[:1])][:len(people)]):
            name = f"ep{episode:02d}-{arm}-{'+'.join(who)}"
            path = BED / f"{name}.wav"
            seconds = build(episode, who, spans, path) if not path.exists() else \
                __import__("soundfile").info(str(path)).duration
            if seconds < MIN_SECONDS:
                print(f"  {name}: {seconds:.0f}s < {MIN_SECONDS:.0f}s, skipped")
                continue
            factor, clusters = measure(path)
            rows.append({"name": name, "arm": arm, "seconds": seconds,
                         "log_bayes": factor, "clusters": clusters})
            print(f"  {name}: {seconds:6.0f}s  log Bayes {factor:12.1f}  clusters {clusters}",
                  flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")

    if rows:
        one = [r["log_bayes"] for r in rows if r["arm"] == "one"]
        two = [r["log_bayes"] for r in rows if r["arm"] == "two"]
        print(f"\none speaker  n={len(one):3d}: median {np.median(one):12.1f}  "
              f"max {max(one):12.1f}" if one else "no single-speaker arms")
        print(f"two speakers n={len(two):3d}: median {np.median(two):12.1f}  "
              f"min {min(two):12.1f}" if two else "no two-speaker arms")
        if one and two:
            print(f"\nseparable? one's max {max(one):.1f} vs two's min {min(two):.1f} -> "
                  f"{'YES, no overlap' if max(one) < min(two) else 'NO, the ranges overlap'}")


if __name__ == "__main__":
    main()
