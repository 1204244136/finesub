"""Calibrate the diarization thresholds against material that has ground truth.

Unlike the probes beside it, this one is meant to be **re-run**: it is what turns
`docs/plans/speaker-diarization-plan.md`'s provisional defaults into numbers with
evidence behind them, and it has to be run again whenever those defaults move.

What it answers, in the order the plan needs it:

1. **Does our smoothing invent overlap?** The reference RTTM says how much of the
   speech really has two people in it; `segments.segment_overlap_ratio` says what
   we report after hysteresis and gap-bridging. The first real run put 58% of
   subtitle lines on two speakers, which is the symptom this measures.
2. **Where should `AssignmentParameters.second_share` sit?** Sweeping it against
   the truth gives precision and recall for "this line has two speakers" instead
   of a guess. Reference segments stand in for the subtitle grid: they are the
   units a human drew around speech, which is the same thing a subtitle is.
3. **Does the single-speaker veto's co-activity statistic separate the cases?**
   Reported per file so a threshold can be picked across many, not two.

Ground truth comes in as RTTM (`SPEAKER file 1 start dur <NA> <NA> spk <NA> <NA>`),
which is what VoxConverse, AliMeeting, AMI and DIHARD all publish.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from finesub.speech.diarization.assign import AssignmentParameters, assign_subtitle  # noqa: E402
from finesub.speech.diarization.segments import (  # noqa: E402
    SmoothingParameters,
    TrackSegment,
    segment_overlap_ratio,
    segments_from_activity,
)


@dataclass(frozen=True)
class ReferenceSegment:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_rttm(path: Path) -> list[ReferenceSegment]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start, duration = float(fields[3]), float(fields[4])
        segments.append(ReferenceSegment(speaker=fields[7], start=start, end=start + duration))
    return sorted(segments, key=lambda segment: segment.start)


def overlap_ratio_of(spans: list[tuple[float, float]]) -> float:
    """Share of covered time where at least two spans coincide."""
    edges = sorted({value for span in spans for value in span})
    speech = overlapped = 0.0
    for left, right in zip(edges, edges[1:]):
        width = right - left
        if width <= 0:
            continue
        active = sum(1 for start, end in spans if start <= left and end >= right)
        if active >= 1:
            speech += width
        if active >= 2:
            overlapped += width
    return overlapped / speech if speech else 0.0


#: How much of a line another speaker must cover to count as "in it". Audit 2.9:
#: when this equals the swept `second_share`, the F1 optimum sitting there is
#: close to an identity, so it is a parameter now and gets varied.
TRUTH_SHARE = 0.2


def true_speakers_over(
    segment: ReferenceSegment, reference: list[ReferenceSegment], truth_share: float = TRUTH_SHARE
) -> set[str]:
    """Who the reference says is audible during this segment."""
    speakers = set()
    for other in reference:
        if other.end <= segment.start:
            continue
        if other.start >= segment.end:
            break
        covered = min(segment.end, other.end) - max(segment.start, other.start)
        if covered > truth_share * segment.duration:
            speakers.add(other.speaker)
    return speakers


def dual_label_counts(
    reference: list[ReferenceSegment],
    timeline: list[TrackSegment],
    second_share: float,
    truth_share: float = TRUTH_SHARE,
) -> tuple[int, int, int, int]:
    """`(hits, predicted, wanted, total)` for "this line has two speakers".

    Counts, not rates, so the caller can pool across files. Averaging per-file
    rates would let a quiet recording -- where the truth is almost never "two
    speakers" -- contribute a precision of 0 from a single false positive and
    drag the mean around.
    """
    parameters = AssignmentParameters(second_share=second_share)
    hits = predicted = wanted = 0
    for segment in reference:
        want = len(true_speakers_over(segment, reference, truth_share)) >= 2
        speaker = assign_subtitle(segment.start, segment.end, timeline, parameters=parameters)
        got = len(speaker.extras.get("tracks", [])) >= 2
        hits += want and got
        predicted += got
        wanted += want
    return hits, predicted, wanted, len(reference)


#: Smoothing settings to compare. `raw` keeps every burst the hysteresis found:
#: it answers "is the second speaker missing because we smoothed them away, or
#: because the model never put them there?" -- which decides whether the
#: dual-label ceiling is P11's problem or the timeline's.
SMOOTHINGS = {
    "default": SmoothingParameters(),
    "raw": SmoothingParameters(min_gap=0.0, min_duration=0.0),
    "short-ok": SmoothingParameters(min_duration=0.08),
}


def diarize_file(
    audio: Path, artifacts: str, device: str, geometry: dict | None = None
) -> dict[str, list[TrackSegment]]:
    import soundfile
    import torch

    from finesub.speech.diarization.chunking import SortformerGeometry
    from finesub.speech.diarization.runner import SortformerRunner

    signal, sample_rate = soundfile.read(str(audio), dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise SystemExit(f"{audio} is {sample_rate} Hz; resample to 16 kHz first")
    waveform = torch.from_numpy(signal.mean(axis=1)).unsqueeze(0)
    activity = SortformerRunner(artifacts, device=device, geometry=SortformerGeometry(**(geometry or {}))).diarize(waveform)
    rows = activity.float().cpu().tolist()
    return {
        name: segments_from_activity(rows, parameters=parameters)
        for name, parameters in SMOOTHINGS.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="+", help="audio.wav=reference.rttm, repeated")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry", choices=("default", "high"), default="default")
    parser.add_argument("--truth-share", type=float, nargs="*", default=[TRUTH_SHARE])
    arguments = parser.parse_args()
    geometry = {} if arguments.geometry == "default" else {
        "chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40, "spkcache_update_period": 300,
    }

    # 0.10 / 0.15 added (audit 2.9): the optimum sat on the grid edge, unbracketed.
    shares = (0.10, 0.15, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
    print(f"{'file':<22} {'ref ovl':>8} {'our ovl':>8} {'ref spk':>8} {'our trk':>8}")
    totals = []
    for pair in arguments.pairs:
        audio_name, _, rttm_name = pair.partition("=")
        audio, rttm = Path(audio_name), Path(rttm_name)
        reference = parse_rttm(rttm)
        timelines = diarize_file(audio, arguments.artifacts, arguments.device, geometry)

        reference_overlap = overlap_ratio_of([(s.start, s.end) for s in reference])
        default = timelines["default"]
        print(f"{audio.stem:<22} {reference_overlap * 100:>7.1f}% "
              f"{segment_overlap_ratio(default) * 100:>7.1f}% "
              f"{len({s.speaker for s in reference}):>8} {len({s.track for s in default}):>8}")
        totals.append((audio.stem, reference, timelines))

    for truth_share, smoothing in ((t, s) for t in arguments.truth_share for s in SMOOTHINGS):
        print(f"\n--- truth share {truth_share:.2f} | smoothing: {smoothing} (pooled over every reference segment) ---")
        print(f"{'second_share':<14} {'dual-labelled':>14} {'precision':>10} {'recall':>8} {'F1':>8}")
        for value in shares:
            hits = predicted = wanted = total = 0
            for _, reference, timelines in totals:
                counts = dual_label_counts(reference, timelines[smoothing], value, truth_share)
                hits, predicted, wanted, total = (
                    hits + counts[0], predicted + counts[1], wanted + counts[2], total + counts[3]
                )
            precision = hits / predicted if predicted else 0.0
            recall = hits / wanted if wanted else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            print(f"{value:<14.2f} {predicted / total * 100:>13.1f}% {precision:>10.3f} "
                  f"{recall:>8.3f} {f1:>8.3f}")
        print(f"  (truth: {wanted}/{total} reference segments really have two speakers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
