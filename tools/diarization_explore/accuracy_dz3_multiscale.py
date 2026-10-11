"""Joint identity clustering of 8 s and 16 s DiariZen views, without gold.

The 8 s local view has less slot pressure; the 16 s view supplies longer voice
examples. Both keep their own native frame geometry during reconstruction.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.ndimage import median_filter

import accuracy_dz3 as dz


def frame_normalize(activity, frames):
    """Nearest-frame relative-time alignment for clustering's clean-frame filter.

    This array is only used by clustering to apply its 10% clean-activity rule,
    never as a physical timeline or reconstruction input.
    """
    indices = np.minimum((np.arange(frames) * activity.shape[1] / frames).astype(int), activity.shape[1]-1)
    return activity[:, indices, :]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", nargs="+", required=True)
    args = p.parse_args()
    import torch
    import soundfile as sf
    torch.set_num_threads(4)
    from accuracy_diarizen import load_pipeline
    pipeline = load_pipeline(cpu=True)
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    from pyannote.audio.utils.signal import Binarize
    for ident in args.ids:
        directory = dz.OUT / ident
        locals_, embeddings, provenance = [], [], []
        for duration in (16, 8):
            cache = directory / f"w{duration}"
            saved = np.load(cache / "diarizen-local.npz")
            record = json.loads((directory / f"w{duration}_default.json").read_text())
            if dz.lab.digest(cache / "embeddings.npy") != record["identity"]["embeddings"]:
                raise ValueError("Embedding cache changed")
            sw = SlidingWindow(start=saved["window"][0], duration=saved["window"][1], step=saved["window"][2])
            locals_.append(SlidingWindowFeature(median_filter(saved["data"], size=(1, 11, 1), mode="reflect"), sw))
            embeddings.append(np.load(cache / "embeddings.npy"))
            provenance.append(record["identity"])
        if provenance[0]["audio"] != provenance[1]["audio"] or provenance[0]["model"] != provenance[1]["model"]:
            raise ValueError("Multiscale source mismatch")
        frames = locals_[0].data.shape[1]
        combined = SlidingWindowFeature(np.concatenate([frame_normalize(s.data, frames) for s in locals_]),
                                        locals_[0].sliding_window)
        samples = sf.info(dz.lab.source(ident)).frames
        for suffix, scale in (("joint", 1.), ("joint_scaled", 1/3)):
            # The joint pool has ~3x as many windows as default. This heuristic
            # tests compensation for repeated evidence; it is not exact weighting.
            params = dict(pipeline.PIPELINE_PARAMS["clustering"])
            params["Fa"] *= scale
            pipeline.clustering.instantiate(params)
            clusters, _, _ = pipeline.clustering(np.concatenate(embeddings), segmentations=combined,
                                                 min_clusters=1, max_clusters=20)
            cursor = 0
            for duration, local in zip((16, 8), locals_):
                n = len(local.data)
                identities = clusters[cursor:cursor+n].copy()
                cursor += n
                identities[local.data.sum(axis=1) == 0] = -2
                count = pipeline.speaker_count(local, pipeline._segmentation.model._receptive_field, warm_up=(0., 0.))
                count.data = np.minimum(count.data, pipeline.max_speakers).astype(np.int8)
                discrete, _ = pipeline.reconstruct(local, identities, count)
                annotation = Binarize(onset=.5, offset=.5, min_duration_on=0., min_duration_off=0.)(discrete)
                annotation.uri = "experiment"
                dz.save_annotation(ident, f"w{duration}_{suffix}", annotation, samples,
                                   dict(script=dz.lab.digest(__file__), views=provenance,
                                        audio=provenance[0]["audio"], output_duration=duration,
                                        clustering=params))


if __name__ == "__main__":
    main()
