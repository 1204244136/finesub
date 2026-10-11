"""Export a cached experimental timeline onto a COPY of stable subtitles.

Reference labels and scores never participate in assignment. The destination
must be new: original artifacts and their role/name judgments are never reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import accuracy_lab as lab


def export(ident, arm, destination):
    from finesub.speech.diarization.annotate import speaker_metadata
    from finesub.speech.diarization.segments import segments_from_activity
    metadata = json.loads((lab.OUT / ident / f"{arm}.json").read_text(encoding="utf-8"))
    if metadata.get("identity", {}).get("audio") != lab.digest(lab.source(ident)):
        raise ValueError("Cached timeline has no matching source audio identity")
    probs = np.load(lab.OUT / ident / f"{arm}.npz")["probs"]
    original = lab.WT / "out" / "refined-2609" / ident / f"{ident}-stable.json"
    payload = json.loads(original.read_text(encoding="utf-8"))
    timeline = segments_from_activity(probs.tolist())
    model = ("pyannote/speaker-diarization-community-1" if arm.startswith("community") else
             "microsoft/wavlm-base-plus-sv+Sortformer-VAD" if arm.startswith("wavlm") else
             "nvidia/diar_streaming_sortformer_4spk-v2.1")
    if arm.startswith("hybrid"):
        model = "pyannote/community-1+Sortformer"
    provenance = speaker_metadata(model=model, artifacts_revision="", track_count=probs.shape[1],
                                   parameters={"experiment_arm": arm, "frame_seconds": lab.FRAME,
                                               "source_audio_sha256": metadata["identity"]["audio"]})
    return write_preview(payload, timeline, provenance, destination, ident)


def write_preview(payload, timeline, provenance, destination, stem):
    """Write only derived copies; no reference or role/name mapping is read."""
    from finesub.speech.diarization.annotate import annotate_payload
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.export import check_completeness, group_by_speaker, render_ass, render_speaker_srts
    labels = assign_all(payload["segments"], timeline)
    annotated = annotate_payload(payload, labels, provenance)
    groups = group_by_speaker(annotated["segments"], labels)
    if check_completeness(annotated["segments"], groups):
        raise AssertionError("Export dropped a subtitle")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / f"{stem}-stable.json").write_text(json.dumps(annotated, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = {"metadata": provenance, "tracks": [{"track": t.track, "start": t.start, "end": t.end} for t in timeline]}
    (destination / f"{stem}-diar.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / f"{stem}.ass").write_text(render_ass(annotated["segments"], labels), encoding="utf-8-sig")
    for name, srt in render_speaker_srts(annotated["segments"], labels).items():
        (destination / f"{stem}-{name}.srt").write_text(srt, encoding="utf-8-sig")
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ident")
    p.add_argument("arm")
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    print(export(args.ident, args.arm, args.destination))


if __name__ == "__main__":
    main()
