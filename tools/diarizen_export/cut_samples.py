"""Cut the per-track listening samples a `-diar.json` already points at.

The run stored the expensive half -- WHICH span of each track is that track at its most
recognisable, chosen from `soft`, which only exists while the run is happening. This is the cheap
half: ffmpeg, a few seconds, repeatable at any later time from the vocal track.

    python tools/diarizen_export/cut_samples.py out/clip/clip-diar.json

Writes `out/clip/clip-speaker-samples/s0.aac`, `s1.aac`, ... -- ten seconds each, and listening to
them is faster and more accurate than any prompt that tries to name the tracks from text.

⚠ The directory is `-speaker-samples`, NOT `-speakers`: `--speaker-tracks` already owns
`<stem>-speakers/` for the per-speaker SRT and ASS export, and dropping audio into it would mix
two unrelated deliverables in one folder.

⚠ Only DiariZen timelines carry samples. Sortformer computes no embeddings, so there is no
`soft` to choose a span with, and this prints that rather than inventing one.

The clips are 16 kHz mono AAC at 32 kbps, which is what `extract_window_clip` produces because
that is what Gemini consumes. That is deliberate reuse: it is also plenty for a person to
recognise a voice, so one artifact serves both the human naming the tracks and the audio arm of
the modality experiment (`docs/archive/speaker-identity-plan.md` C.2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def samples_of(diar_json: Path) -> list[dict]:
    payload = json.loads(diar_json.read_text(encoding="utf-8"))
    return list(payload.get("metadata", {}).get("hints", {}).get("samples", ()))


def vocal_beside(diar_json: Path) -> Path:
    """The vocal track this timeline was made from, by the pipeline's own naming."""
    stem = diar_json.name.removesuffix("-diar.json")
    return diar_json.with_name(f"{stem}-vocal.ogg")


def cut(diar_json: Path, audio: Path, directory: Path) -> list[Path]:
    """One clip per sample. ⚠ Fail-soft per clip: one bad span costs that clip and no more."""
    from finesub.media.clips import CLIP_AUDIO_SUFFIX, extract_window_clip

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for sample in samples_of(diar_json):
        destination = directory / f"s{int(sample['track'])}{CLIP_AUDIO_SUFFIX}"
        try:
            extract_window_clip(audio, float(sample["start"]), float(sample["end"]), destination)
        except Exception as error:  # noqa: BLE001 - see the docstring
            print(f"  ! s{sample['track']}: {error}")
            continue
        closeness = sample.get("closeness")
        print(f"  s{sample['track']}: {sample['start']:.1f}-{sample['end']:.1f} s"
              + (f", closeness {closeness:.3f}" if closeness is not None else "")
              + f" -> {destination.name}")
        written.append(destination)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diar_json", type=Path, help="a *-diar.json from a --diarize run")
    parser.add_argument("--audio", type=Path, help="the vocal track (default: beside the timeline)")
    parser.add_argument("--out", type=Path, help="where the clips go (default: <stem>-speaker-samples/)")
    args = parser.parse_args()

    timeline = args.diar_json
    if not timeline.exists():
        print(f"no such timeline: {timeline}")
        return 1
    if not samples_of(timeline):
        backend = json.loads(timeline.read_text(encoding="utf-8")).get(
            "metadata", {}).get("backend", "?")
        print(f"{timeline.name} carries no samples (backend: {backend}). "
              "Only the diarizen backend computes them -- sortformer has no embeddings.")
        return 1

    audio = args.audio or vocal_beside(timeline)
    if not audio.exists():
        print(f"no vocal track at {audio}; pass --audio")
        return 1
    stem = timeline.name.removesuffix("-diar.json")
    directory = args.out or timeline.with_name(f"{stem}-speaker-samples")
    written = cut(timeline, audio, directory)
    print(f"{len(written)} clip(s) in {directory}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
