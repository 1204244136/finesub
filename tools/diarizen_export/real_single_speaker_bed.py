"""Q13, on real single-speaker material: does the evidence hold up where it has to?

The concatenated bed (`single_speaker_bed.py`) could not answer this. Its single-speaker arms all
clustered to K=1, where `ELBO_K - ELBO_1` is identically zero by construction, so its perfect
separation demonstrated nothing about the case a veto exists for: **genuinely single-speaker
material that the clustering OVER-SPLITS**. That case needs real material with the things that
cause over-splitting -- music, game audio, separation residue -- and the plan recorded it as
blocked on the owner supplying some.

⚠ It was never blocked. `assets/kaguya60.flac` is the 60-minute file `docs/diarization-findings.md`
9 measured the over-split on (2 tracks, 66/34), and a separated vocal track for it is already on
disk from the 2026-08-31 baseline run. See that section and the plan's Q13 entry.

Two arms, and only the first one needs a GPU:

* **single**: the real single-speaker file, end to end through the production DiariZen path.
* **reference band**: the multi-speaker files whose embeddings are already cached, clustered
  again on the CPU. What "several speakers" looks like on this measure, so the single-speaker
  reading has something to be near or far from.

⚠ The comparable quantity is the factor **per embedding**. The raw factor is a total over the
embeddings, so it grows with duration, and a 60-minute file against a 20-minute one would be
compared on length.

    python tools/diarizen_export/real_single_speaker_bed.py
    python tools/diarizen_export/real_single_speaker_bed.py --skip-gpu   # band only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

EXPORT_DIR = REPO / "tmp" / "diarizen-export"
V2 = REPO / "tmp" / "diarization-accuracy-v2"
DZ3 = REPO / "tmp" / "diarization-dz3"
BED = REPO / "tmp" / "single-speaker-bed"

#: ⚠ The vocal track, not the source. The over-split this is about happens on the separated
#: audio, residue included, which is what production diarizes.
#: ⚠ `people` is what the material actually holds, not what the directory is called. Getting
#: this wrong is the branch's most repeated mistake, so each entry says what it is:
#:
#:   kaguya60    one streamer, game audio. The clean single-speaker point.
#:   saki_audio  one performer, singing with spoken MC between songs. ⚠ A 1984 concert, so an
#:               announcer or audience voice is possible -- treated as PROBABLY single and
#:               reported separately from the clean point.
#:   yui         ⚠ NOT single. One live streamer reacting to a FILM, and the film's dialogue is
#:               real speech that survives separation. Included as the contamination case --
#:               "one live person plus replayed voices" -- never as a single-speaker point.
OUT = REPO.parent.parent / "out"
SINGLE = [
    ("kaguya60", OUT / "ts-baseline-20260831-hq" / "kaguya60" / "kaguya60-vocal.ogg", 1),
    ("saki_audio", OUT / "saki_audio" / "saki_audio-vocal.ogg", 1),
    ("yui", OUT / "ts-baseline-20260831-hq" / "yui" / "yui-vocal.ogg", 0),
    # ⚠ Public-domain LibriVox chapters, one reader each BY CONSTRUCTION -- LibriVox assigns a
    # section to exactly one volunteer, and the API reports who. Three different readers, so
    # three different voices AND three different home recording setups, which is the diversity
    # kaguya cannot give however many hours of it there are (one streamer, one microphone).
    #
    # ⚠ **A control, not a replacement.** These have register variation -- Dumas is mostly
    # dialogue and a reader voices the characters -- and NO music, game audio or separation
    # residue. So they isolate register from background: if these do not over-split and kaguya
    # does, the cause is the background, not the voice.
    #
    # ⚠ No separation was run on them. There is nothing to separate, so they go through the
    # production `--no-separate` path (`encode_asr_delivery`, same 16 kHz mono ogg at the same
    # spec) rather than a hand-rolled ffmpeg line.
    ("monte_cristo_003", OUT / "librivox" / "monte_cristo_003" / "monte_cristo_003-vocal.ogg", 1),
    ("monte_cristo_005", OUT / "librivox" / "monte_cristo_005" / "monte_cristo_005-vocal.ogg", 1),
    ("monte_cristo_017", OUT / "librivox" / "monte_cristo_017" / "monte_cristo_017-vocal.ogg", 1),
]


def factor_per_embedding(result) -> float | None:
    if not result.log_bayes_embeddings:
        return None
    return result.single_speaker_log_bayes / result.log_bayes_embeddings


def track_shares(tracks) -> list[float]:
    """How much of the total track activity each track holds, largest first."""
    seconds: dict[int, float] = {}
    for segment in tracks:
        seconds[segment.track] = seconds.get(segment.track, 0.0) + (segment.end - segment.start)
    total = sum(seconds.values()) or 1.0
    return sorted((value / total for value in seconds.values()), reverse=True)


def run_single(label: str, audio: Path, people: int) -> dict:
    """The full production path on one real file. The only part that needs a GPU."""
    from finesub.speech.diarization import stage as diarization
    from finesub.speech.diarization.evidence import evidence_path

    BED.mkdir(parents=True, exist_ok=True)
    timeline_path = BED / f"{label}-diar.json"
    started = time.time()
    result = diarization.run_diarization(
        audio, device="cuda", backend="diarizen", artifacts=EXPORT_DIR,
        evidence_path=evidence_path(timeline_path),
    )
    diarization.write_timeline(result, timeline_path)
    shares = track_shares(result.tracks)
    row = {
        "file": label, "people": people, "seconds": round(time.time() - started, 1),
        "tracks": len(shares), "shares": [round(s, 3) for s in shares],
        "log_bayes": result.single_speaker_log_bayes,
        "embeddings": result.log_bayes_embeddings,
        "per_embedding": factor_per_embedding(result),
    }
    print(f"  {label}: {row['tracks']} tracks {row['shares']}, "
          f"factor {row['log_bayes']:.1f} over {row['embeddings']} embeddings "
          f"= {row['per_embedding']:.3f}/emb  ({row['seconds']:.0f}s)")
    return row


def cached_rows(cases) -> list[dict]:
    """Re-cluster files whose embeddings are already on disk. CPU only, no model loaded."""
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner

    rows = []
    for label, local_path, embed_path, people in cases:
        if not (local_path.exists() and embed_path.exists()):
            continue
        filtered = runner.median_filtered(np.load(local_path)["data"])
        result = clusterer.cluster(np.load(embed_path), filtered, EXPORT_DIR / "plda")
        if not result.log_bayes_embeddings:
            continue
        per = result.single_speaker_log_bayes / result.log_bayes_embeddings
        rows.append({"file": label, "people": people,
                     "clusters": int(len(np.unique(result.hard[result.hard >= 0]))),
                     "log_bayes": result.single_speaker_log_bayes,
                     "embeddings": result.log_bayes_embeddings, "per_embedding": per})
        print(f"  {label} ({people} people): {rows[-1]['clusters']} clusters, "
              f"{per:.3f}/emb")
    return rows


def band(rows: list[dict], name: str) -> None:
    if not rows:
        print(f"{name}: nothing scored")
        return
    values = sorted(row["per_embedding"] for row in rows)
    print(f"{name} (n={len(values)}): min {values[0]:.3f}  median "
          f"{statistics.median(values):.3f}  max {values[-1]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-gpu", action="store_true",
                        help="reference band only, from cached embeddings")
    parser.add_argument("--only", nargs="+", help="run only these labels from SINGLE")
    parser.add_argument("--out", default=str(BED / "real-single-speaker.json"))
    args = parser.parse_args()

    print("A. REAL single-speaker material, full production path")
    wanted = [case for case in SINGLE if not args.only or case[0] in args.only]
    single = [] if args.skip_gpu else [run_single(*case) for case in wanted]

    print("\nB. reference band -- multi-speaker files, from cached embeddings (CPU)")
    onetonton = cached_rows([
        (f"onetonton-{n:02d}", V2 / f"onetonton-{n:02d}" / "diarizen-local.npz",
         V2 / f"onetonton-{n:02d}" / "embeddings.npy", 3) for n in range(1, 16)])
    hina = cached_rows([
        (f"hina-{n:02d}", DZ3 / f"hina-{n:02d}" / "w16" / "diarizen-local.npz",
         DZ3 / f"hina-{n:02d}" / "w16" / "embeddings.npy", 0) for n in range(10, 35)])

    print()
    band(onetonton, "onetonton (3 people)")
    band(hina, "hina (5-8 people)")
    for row in single:
        print(f"single-speaker {row['file']}: {row['per_embedding']:.3f}/emb "
              f"on {row['tracks']} tracks")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"single": single, "onetonton": onetonton, "hina": hina}, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
