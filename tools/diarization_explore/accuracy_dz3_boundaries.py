"""CPU ablation of local median filtering with frozen embeddings/global identities.

This cannot recover a local identity completely erased by the original filtering.
It isolates reconstruction/count smoothing from speaker embedding extraction.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.ndimage import median_filter

import accuracy_dz3 as dz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--widths", nargs="+", type=int, choices=[1, 5, 11], default=[1, 5])
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    from accuracy_diarizen import load_pipeline, raster
    pipeline = load_pipeline(cpu=True)
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    from pyannote.audio.utils.signal import Binarize
    for ident in args.ids:
        directory = dz.OUT / ident
        cache = directory / "w16"
        saved = np.load(cache / "diarizen-local.npz")
        sw = SlidingWindow(start=saved["window"][0], duration=saved["window"][1], step=saved["window"][2])
        local = SlidingWindowFeature(median_filter(saved["data"], size=(1, 11, 1), mode="reflect"), sw)
        baseline = json.loads((directory / "w16_default.json").read_text())
        embeddings = np.load(cache / "embeddings.npy")
        if dz.lab.digest(cache / "embeddings.npy") != baseline["identity"]["embeddings"]:
            raise ValueError("Embedding cache changed")
        clusters, _, _ = pipeline.clustering(embeddings, segmentations=local, min_clusters=1, max_clusters=20)
        clusters[local.data.sum(axis=1) == 0] = -2
        import soundfile as sf
        if ident.startswith(("R", "vox-")):
            from accuracy_dz3_meetings import dataset, save
            audio = dataset(ident)[0]
        else:
            audio = dz.lab.source(ident)
            save = dz.save_annotation
        samples = sf.info(audio).frames
        for width in [11, *[w for w in args.widths if w != 11]]:
            filtered = median_filter(saved["data"], size=(1, width, 1), mode="reflect")
            segmentation = SlidingWindowFeature(filtered, sw)
            count = pipeline.speaker_count(segmentation, pipeline._segmentation.model._receptive_field, warm_up=(0., 0.))
            count.data = np.minimum(count.data, pipeline.max_speakers).astype(np.int8)
            discrete, _ = pipeline.reconstruct(segmentation, clusters, count)
            annotation = Binarize(onset=.5, offset=.5, min_duration_on=0., min_duration_off=0.)(discrete)
            annotation.uri = "experiment"
            if width == 11:
                if ident.startswith(("R", "vox-")):
                    if annotation.to_rttm() != (directory / "w16_default.rttm").read_text():
                        raise ValueError("Default CPU reconstruction differs from original")
                elif not np.array_equal(raster(annotation, samples), np.load(directory / "w16_default.npz")["probs"]):
                    raise ValueError("Default CPU reconstruction differs from original")
                print(ident, "default CPU replay exact", flush=True)
                continue
            save(ident, f"w16_median{width}", annotation, samples, baseline["identity"] |
                 dict(boundary_script=dz.lab.digest(__file__), median_width=width, frozen_identity=True))


if __name__ == "__main__":
    main()
