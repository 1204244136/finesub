"""S2: our embeddings against the frozen `embeddings.npy`.

⚠ Fed with the CACHED local segmentation, not ours. Our segmentation differs on about 0.02% of
frames, and feeding it in would mix that difference into this reading -- the point of comparing
stage by stage is that each stage is judged on its own input.

    python tools/diarizen_export/check_s2.py --ids hina-32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

CACHE = REPO / "tmp" / "diarization-dz3"
EXPORT = REPO / "tmp" / "diarizen-export" / "diarizen_embedding.ts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", default=["hina-32"])
    parser.add_argument("--chunks", type=int, default=40,
                        help="score only the first N chunks; 0 for all")
    args = parser.parse_args()

    import soundfile
    import torch

    import accuracy_lab as lab
    from finesub.speech.diarization import diarizen_embed as embedder
    from finesub.speech.diarization import diarizen_runner as runner

    for ident in args.ids:
        want = np.load(CACHE / ident / "w16" / "embeddings.npy")
        local = np.load(CACHE / ident / "w16" / "diarizen-local.npz")["data"]
        signal, rate = soundfile.read(lab.source(ident), dtype="float32", always_2d=True)
        assert rate == runner.SAMPLE_RATE
        waveform = torch.from_numpy(signal.mean(axis=1))[None]

        limit = args.chunks or local.shape[0]
        # ⚠ The cache is pre-filter; the research path embeds with the filtered masks.
        got = embedder.embed(EXPORT, waveform, runner.median_filtered(local)[:limit])
        reference = want[:limit]
        print(f"{ident}: ours {got.shape} vs cache {reference.shape}")

        both_nan = np.isnan(got) & np.isnan(reference)
        finite = ~np.isnan(got) & ~np.isnan(reference)
        print(f"   NaN pattern identical: {np.array_equal(np.isnan(got), np.isnan(reference))}"
              f"  ({int(both_nan.sum())} NaN cells on both sides)")
        if finite.any():
            gap = float(np.abs(got[finite] - reference[finite]).max())
            rel = float(np.abs(got[finite] - reference[finite]).max()
                        / max(float(np.abs(reference[finite]).max()), 1e-12))
            print(f"   finite cells: bit-exact {np.array_equal(got[finite], reference[finite])}"
                  f"  max|delta| {gap:.3e}  relative {rel:.3e}")
            # What matters downstream is the cosine geometry the clustering sees.
            a = got[finite].reshape(-1)
            b = reference[finite].reshape(-1)
            cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            print(f"   cosine over all finite values: {cosine:.12f}")


if __name__ == "__main__":
    main()
