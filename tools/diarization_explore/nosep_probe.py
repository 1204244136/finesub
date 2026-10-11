"""Does Sortformer degrade when it reads the UN-separated mix instead of production's `-vocal.ogg`?

Owner 2026-09-15. §5m already asked a version of this, but on VoxConverse -- where the "raw" arm is
already clean speech, so separation there removes almost nothing and the answer ("no measurable harm,
single files flip between discrete attractors") cannot be carried to material that actually has music.
The 希娜 corpus does: these are game-broadcast programmes with BGM, stingers and game audio throughout,
and we hold both arms already -- the separated timelines from the refined-2609 batch, and the original
`.m4a` sources next to the refined axes.

So this only runs the cheap half: decode each source to 16 kHz mono and run the diarizer on it, writing
`out/refined-2609-nosep/<id>/<id>-diar.json`. Scoring is the existing one:

    python tools/diarization_explore/nosep_probe.py --only hina-01 hina-02
    python tools/diarization_explore/refined_multi.py agree --run-root out/refined-2609-nosep

⚠ The separated arm must not be re-run: `-diar.json` is not a deterministic function of the audio
(about 7% of subtitles change speaker on different hardware), so the comparison is only fair against the
timelines the batch actually produced. This writes a new directory and never touches those.

⚠ The segments must stay the SAME segments -- only the timeline may differ, which is what isolates the
diarizer from the rest of the pipeline. So `--relabel` copies the original run's `-stable.json` next to the
new timeline and re-applies the labels with `apply_to_stable` (cheap, deterministic, unlike the diarizer).
Skipping that step scores the OLD labels and both arms come out identical -- the comparison looks clean and
means nothing.

It also reports the diarizer's own real-time factor, which no artifact records: `run_metadata` times
vocal_separation / asr / stabilize but not diarize.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WT / "src"))
import refined_multi as rm  # noqa: E402

OUT = WT / "out" / "refined-2609-nosep"
#: Same input as the batch, fresh process -- how much does the diarizer move on its own?
CONTROL = WT / "out" / "refined-2609-resep"


def source_media(ident: str, ass: Path) -> Path | None:
    """The un-separated original next to the refined axis."""
    for suffix in (".m4a", ".mkv", ".mp4", ".webm", ".wav", ".flac"):
        candidate = ass.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--limit", type=int, default=0, help="stop after this many files")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--relabel", action="store_true",
                        help="copy each original -stable.json beside the new timeline and re-apply labels")
    parser.add_argument("--control", action="store_true",
                        help="NOISE BASELINE: re-run on the SEPARATED -vocal.ogg the batch already used, "
                             "so the same difference is measured with the input held fixed")
    args = parser.parse_args()

    from finesub.speech.diarization import stage as diarization

    out = CONTROL if args.control else OUT
    if args.relabel:
        relabel(diarization, args.only, out)
        return

    done = 0
    for ident, ass in rm.sources().items():
        if args.only and ident not in args.only:
            continue
        target = out / ident / f"{ident}-diar.json"
        if target.exists():
            print(f"{ident:<9} already there, skipping")
            continue
        media = (WT / "out" / "refined-2609" / ident / f"{ident}-vocal.ogg") if args.control             else source_media(ident, ass)
        if media is None or not media.exists():
            print(f"{ident:<9} no source media")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "mix.wav"
            t0 = time.perf_counter()
            # ⚠ `-c:a pcm_f32le` is load-bearing. ffmpeg's default for .wav is pcm_s16le, and that
            # requantisation is a ~3e-6 relative change -- enough to move the discrete speaker cache to a
            # different attractor and reorganise the whole track structure (findings 5.1). Measured:
            # on hina-07 the 16-bit decode gives [744.4, 574.8, 21.2, 745.4] where float32 gives
            # [1010.5, 660.8, 32.5, 396.1], which is what reading the .ogg directly produces.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(media),
                            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(wav)], check=True)
            decode = time.perf_counter() - t0
            t0 = time.perf_counter()
            result = diarization.run_diarization(wav, device=args.device)
            elapsed = time.perf_counter() - t0
            target.parent.mkdir(parents=True, exist_ok=True)
            diarization.write_timeline(result, target)
        tracks = json.loads(target.read_text(encoding="utf-8"))["tracks"]
        audio = max((t["end"] for t in tracks), default=0.0)
        print(f"{ident:<9} audio {audio / 60:6.1f} min  decode {decode:5.1f}s  "
              f"diarize {elapsed:6.1f}s  RTF {elapsed / max(audio, 1):.4f} ({max(audio, 1) / elapsed:.0f}x)")
        done += 1
        if args.limit and done >= args.limit:
            break


def relabel(diarization, only, out=None) -> None:
    """Give each new timeline the ORIGINAL run's segments, then re-derive the labels from it."""
    import shutil

    out = out or OUT
    for ident in sorted(d.name for d in out.iterdir() if d.is_dir()):
        if only and ident not in only:
            continue
        timeline = out / ident / f"{ident}-diar.json"
        source = WT / "out" / "refined-2609" / ident / f"{ident}-stable.json"
        target = out / ident / f"{ident}-stable.json"
        if not timeline.exists() or not source.exists():
            print(f"{ident:<9} missing timeline or original stable")
            continue
        shutil.copyfile(source, target)
        diarization.apply_to_stable(target, timeline)
        print(f"{ident:<9} relabelled from its own timeline")


if __name__ == "__main__":
    main()
