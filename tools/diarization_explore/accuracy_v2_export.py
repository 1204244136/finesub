"""Export second-round cached candidates to new subtitle previews, without gold."""
import argparse
import json
from pathlib import Path

import numpy as np
import accuracy_lab as lab
from accuracy_v2 import OUT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ident")
    p.add_argument("arm", choices=["diarizen", "dz_community", "high_right80"])
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    if args.destination.exists():
        p.error("destination must be new")
    if args.ident.startswith("onetonton-"):
        from accuracy_cross import configure
        configure()
    record = json.loads((OUT/args.ident/f"{args.arm}.json").read_text(encoding="utf-8"))
    if record["identity"]["audio"] != lab.digest(lab.source(args.ident)):
        raise ValueError("cached audio fingerprint mismatch")
    source = lab.source(args.ident).with_name(f"{args.ident}-stable.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    probs = np.load(OUT/args.ident/f"{args.arm}.npz")["probs"]
    from finesub.speech.diarization.segments import segments_from_activity
    from finesub.speech.diarization.annotate import speaker_metadata
    from accuracy_export import write_preview
    model = {"diarizen": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
             "dz_community": "DiariZen-v2+community-1-global",
             "high_right80": "nvidia/diar_streaming_sortformer_4spk-v2.1"}[args.arm]
    tracks = segments_from_activity(probs.tolist())
    metadata = speaker_metadata(model=model, artifacts_revision="", track_count=probs.shape[1],
        parameters=dict(experiment_arm=args.arm, source_audio_sha256=record["identity"]["audio"],
                        source_identity=record["identity"], frame_seconds=lab.FRAME))
    write_preview(payload, tracks, metadata, args.destination, args.ident)
    print(args.destination)


if __name__ == "__main__":
    main()
