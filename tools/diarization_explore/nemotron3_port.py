"""The production Nemotron-3 port against NeMo eager: is the difference inside NeMo's own noise band?

Every exploration reading (findings 2.6) came from NeMo eager; production runs
`finesub.speech.diarization.nemotron3_model`, a plain-torch port of transformers' implementation
that loads NVIDIA's safetensors. Same weights, same algorithm, different execution form: NeMo's CUDA
attention is compiled FlexAttention, the port's is matmul + softmax; NeMo featurizes the whole file
in one STFT and pads to 16 frames, the port in ten-minute pieces. So this is an A-class change
(README_DEV.md 开发原则): bitwise agreement is not expected, noise-level jitter is not a loss, and
the job is to characterise it.

Pre-registered 2026-09-25, before the first comparison ran:

  Per file, on one float32 16 kHz mono waveform:
    nemo      NeMo eager (`nemotron3_probe.infer`, the exploration's offline profile)
    noise     NeMo eager on the waveform plus seeded N(0, 1e-7) -- the noise band; findings 2.6
              measured 0.3-0.4% self-DER for exactly this perturbation
    port      the production port on CUDA
    port-cpu  the production port on CPU (the shortest file only; a rough CPU check)

  Two levels, each reported for noise-vs-nemo and port-vs-nemo:
    frame   10 ms, same track ids, binarised at 0.5: the share of frames where the active set
            differs, over frames where either run has anyone active. Also max |dp|.
    chain   the production vbx (`nemotron3_backend.vbx`) on each run's probabilities, 20 ms 0/1
            maps, tracks matched one-to-one by overlap: the share of frames where the active set
            differs, over frames where either map has anyone active.

  Verdict, per file and level: port-vs-nemo <= max(2 x noise-vs-nemo, 0.5%) is "inside the noise
  band". Any file outside it at the chain level stops the switch until characterised and put to
  the owner; outside at the frame level only is reported, since the chain is what users see.

    python tools/diarization_explore/nemotron3_port.py hina-04 hina-10 onetonton-01 onetonton-05

Needs the exploration environment for the NeMo arms (tools/diarization_explore/README.md,
「Nemotron-3-Diarization（2026-09-24）」), and the voiceprint model and PLDA: the DiariZen hosted
build is used for those, whose two files are byte-identical to the ones the new hosted repository
carries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nemotron3_cost as nc  # noqa: E402
import nemotron3_probe as n3  # noqa: E402

OUT = n3.WT / "tmp" / "nemotron3-port"
MARGIN = 0.005


def frame_disagreement(a, b):
    frames = min(len(a), len(b))
    x, y = a[:frames] > 0.5, b[:frames] > 0.5
    busy = (x | y).any(axis=1)
    differ = (x != y).any(axis=1) & busy
    return float(differ.sum() / max(1, busy.sum())), float(np.abs(a[:frames] - b[:frames]).max())


def matched_disagreement(a, b):
    """Share of busy frames whose active set differs, after matching b's tracks onto a's."""
    from scipy.optimize import linear_sum_assignment

    frames = min(len(a), len(b))
    x, y = a[:frames] > 0.5, b[:frames] > 0.5
    width = max(x.shape[1], y.shape[1])
    x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
    y = np.pad(y, ((0, 0), (0, width - y.shape[1])))
    overlap = x.T.astype(np.int64) @ y.astype(np.int64)
    rows, cols = linear_sum_assignment(overlap, maximize=True)
    mapped = np.zeros_like(y)
    mapped[:, rows] = y[:, cols]
    busy = (x | mapped).any(axis=1)
    differ = (x != mapped).any(axis=1) & busy
    return float(differ.sum() / max(1, busy.sum()))


def verdict(port, noise):
    return "inside" if port <= max(2 * noise, MARGIN) else "OUTSIDE"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("idents", nargs="+")
    args = parser.parse_args()

    import vram_guard
    vram_guard.install()
    import soundfile as sf
    import torch

    from finesub.speech.diarization import diarizen_backend, nemotron3_backend, nemotron3_model

    dz = Path(diarizen_backend.ensure(None, "cuda"))
    handles = {device: nemotron3_backend.Handle(n3.MODEL.parent, dz_dir / "diarizen_embedding.ts", dz_dir / "plda")
               for device, dz_dir in (("cuda", dz), ("cpu", Path(diarizen_backend.ensure(None, "cpu"))))}
    OUT.mkdir(parents=True, exist_ok=True)
    nemo = n3.load_model()
    port = {"cuda": nemotron3_model.load(n3.MODEL.parent, "cuda")}

    waves = {}
    for ident in args.idents:
        wave, sr = sf.read(nc.source(ident), dtype="float32", always_2d=True)
        assert sr == 16000, sr
        waves[ident] = wave.mean(axis=1)
    shortest = min(waves, key=lambda ident: len(waves[ident]))

    for ident, wave in waves.items():
        record = {"ident": ident, "minutes": round(len(wave) / 16000 / 60, 1)}
        runs = {}
        t = time.perf_counter()
        runs["nemo"] = n3.infer(nemo, wave)[1]
        record["nemo_s"] = round(time.perf_counter() - t, 1)
        noise = np.random.default_rng(20260925).normal(0.0, 1e-7, size=wave.shape).astype(np.float32)
        runs["noise"] = n3.infer(nemo, wave + noise)[1]
        torch.cuda.empty_cache()
        tensor = torch.from_numpy(wave)[None]
        t = time.perf_counter()
        runs["port"] = nemotron3_model.speaker_probabilities(port["cuda"], tensor, device="cuda")
        record["port_s"] = round(time.perf_counter() - t, 1)
        if ident == shortest:
            cpu = nemotron3_model.load(n3.MODEL.parent, "cpu")
            t = time.perf_counter()
            runs["port-cpu"] = nemotron3_model.speaker_probabilities(cpu, tensor, device="cpu")
            record["port_cpu_s"] = round(time.perf_counter() - t, 1)
            del cpu

        chains = {}
        for arm, probs in runs.items():
            device = "cpu" if arm == "port-cpu" else "cuda"
            chains[arm] = nemotron3_backend.vbx(probs, tensor, handle=handles[device], device=device)[0]
            record[f"{arm}_tracks"] = int(chains[arm].shape[1])
        for arm in runs:
            if arm == "nemo":
                continue
            record[f"frame_{arm}"], record[f"maxdp_{arm}"] = frame_disagreement(runs["nemo"], runs[arm])
            record[f"chain_{arm}"] = matched_disagreement(chains["nemo"], chains[arm])
        record["frames"] = {"nemo": len(runs["nemo"]), "port": len(runs["port"])}
        for level in ("frame", "chain"):
            record[f"{level}_verdict"] = verdict(record[f"{level}_port"], record[f"{level}_noise"])
        np.savez_compressed(OUT / f"{ident}.npz", **{arm.replace("-", "_"): p for arm, p in runs.items()})
        with (OUT / "compare.jsonl").open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
