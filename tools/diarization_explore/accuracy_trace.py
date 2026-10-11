"""Causal probe: does replaying baseline cache selections suppress micro-perturbation divergence?"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import accuracy_lab as lab


class SelectionTrace:
    def __init__(self, original, reference=None, freeze=False):
        self.original, self.reference, self.freeze = original, reference, freeze
        self.rows = []

    def __call__(self, cache, history, has, silence, n_silence, embs, predictions, current):
        import torch
        output = self.original(cache, history, has, silence, n_silence, embs, predictions, current)
        new_cache, new_history, new_has, mean, count = output
        pool = torch.cat([cache, embs], dim=1)
        # The selected vectors are copied exactly. Four coordinates narrow matches;
        # confirm all coordinates before considering a vector a real selected frame.
        match = (new_cache[0, :, None, :4] == pool[0, None, :, :4]).all(dim=-1)
        indices = match.to(torch.int64).argmax(dim=1)
        valid = match.any(dim=1) & (pool[0, indices] == new_cache[0]).all(dim=1)
        indices = torch.where(valid, indices, -1)
        row = {"indices": indices.cpu().tolist(), "preds": predictions.detach().cpu().numpy()}
        i = len(self.rows)
        if self.reference is not None:
            ref = self.reference[i]
            row["selection_changed"] = row["indices"] != ref["indices"]
            row["prediction_max_delta"] = float(np.max(np.abs(row["preds"] - ref["preds"])))
            if self.freeze:
                selected = torch.tensor(ref["indices"], device=pool.device)
                disabled = selected < 0
                selected = selected.clamp(min=0)
                new_cache = pool[:, selected].clone()
                new_cache[:, disabled] = mean.unsqueeze(1)
                if new_has:
                    old = history if has else current
                    probability_pool = torch.cat([old, predictions], dim=1)
                    new_history = probability_pool[:, selected].clone()
                    new_history[:, disabled] = 0
                output = new_cache, new_history, new_has, mean, count
        self.rows.append(row)
        return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ident", default="hina-07", nargs="?")
    args = p.parse_args()
    import vram_guard
    vram_guard.install()
    import torch
    import soundfile as sf
    from finesub.speech.runtime.device import cuda_usable
    from finesub.speech.diarization.runner import SortformerRunner
    if not cuda_usable():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    signal, sr = sf.read(lab.source(args.ident), dtype="float32", always_2d=True)
    assert sr == 16000
    waveform = signal.mean(axis=1)
    reference = baseline = None
    report = {}
    for arm in ("baseline", "baseline_frozen", "micro", "micro_frozen", "pcm16", "pcm16_frozen"):
        runner = SortformerRunner(lab.ARTIFACTS)
        trace = SelectionTrace(runner._cache_update, reference, arm.endswith("_frozen"))
        runner._cache_update = trace
        wave = waveform if arm.startswith("baseline") else waveform * np.float32(1.0000001)
        if arm.startswith("pcm16"):
            wave = np.clip(np.round(waveform * 32768), -32768, 32767).astype(np.float32) / 32768
        start = time.perf_counter()
        probs = runner.diarize(torch.from_numpy(wave)).cpu().numpy()
        if arm == "baseline":
            reference, baseline = trace.rows, probs
            report[arm] = {"updates": len(trace.rows)}
        else:
            report[arm] = {
                "waveform_max_delta": float(np.max(np.abs(wave-waveform))),
                "max_delta": float(np.max(np.abs(probs - baseline))),
                "different_argmax_share": float(np.mean(probs.argmax(axis=1) != baseline.argmax(axis=1))),
                "first_changed_selection": next((i for i, r in enumerate(trace.rows) if r["selection_changed"]), None),
                "first_prediction_delta_over_01": next((i for i, r in enumerate(trace.rows) if r["prediction_max_delta"] > 0.1), None),
                "steps": [{k: v for k, v in row.items() if k not in ("preds", "indices")} for row in trace.rows],
            }
        print(arm, {k: v for k, v in report[arm].items() if k != "steps"}, "seconds", time.perf_counter()-start, flush=True)
        del runner
        torch.cuda.empty_cache()
    (lab.OUT / args.ident / "cache-trace.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
