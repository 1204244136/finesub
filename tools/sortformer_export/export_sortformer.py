"""Export NeMo's streaming Sortformer into TorchScript we can run without nemo.

Production must not depend on `nemo_toolkit`: it is 36 direct dependencies of
training machinery (lightning, hydra, wandb, datasets, lhotse...) that would land
in `pylock.win-py312.toml` and be installed by every managed runtime, to run one
117M model's forward pass. This script is the one place that needs nemo -- it runs
once, by us, and produces three TorchScript files the pipeline loads with plain
torch (already pinned in `[asr]`, GPU included).

    conda run -n asr python tools/sortformer_export/export_sortformer.py \
        <path to diar_streaming_sortformer_4spk-v2.1.nemo> --out-dir <dir>

The split follows NVIDIA's own export boundary (`forward_for_export`):

    sortformer_preprocessor.ts  128-mel front end                      (~145 KB)
    sortformer_core.ts          pre_encode + Conformer + Transformer   (~469 MB)
    sortformer_cache_update.ts  synchronous `streaming_update`         (~567 KB)

Two methods are copied verbatim below rather than called, because TorchScript
refuses them as written. Both copies stay *here*, in the exporter -- the pipeline
never sees them. Everything a copy bakes in is asserted before the export, so a
different checkpoint fails loudly instead of drifting:

  * `FilterbankFeatures.forward` -- an instance-level `torch.no_grad` rebind, a
    module-level `CONSTANT` closed over for the sqrt guard, and
    `log_zero_guard_value_fn`'s isinstance branch. The copy also drops the two
    `autocast(enabled=False)` guards (TorchScript wants a constant device there),
    which is why the driver must not run under autocast.
  * `_compress_spkcache` -- passes `scale_factor=2`, an int literal into a float
    parameter.

Licence: NeMo is Apache-2.0; the copies carry NVIDIA's copyright.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn


# --------------------------------------------------------------------------
# 1. mel front end
# --------------------------------------------------------------------------

def make_scriptable_featurizer(base_type):
    class ScriptableFilterbank(base_type):
        """`FilterbankFeatures.forward`, inference path only (NVIDIA, Apache-2.0)."""

        def forward(self, x: torch.Tensor, seq_len: torch.Tensor):
            seq_len_time = seq_len
            seq_len = self.get_seq_len(seq_len)

            if self.preemph is not None:
                timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
                x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
                x = x.masked_fill(~timemask, 0.0)

            x = self.stft(x)
            x = torch.view_as_real(x)
            x = torch.sqrt(x.pow(2).sum(-1))  # the guard is 0 unless use_grads

            if self.mag_power != 1.0:
                x = x.pow(self.mag_power)

            x = torch.matmul(self.fb.to(x.dtype), x)
            x = torch.log(x + self.log_zero_guard_value)

            max_len = x.size(-1)
            mask = torch.arange(max_len, device=x.device)
            mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
            pad_value = float(self.pad_value)
            x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), pad_value)
            pad_to = self.pad_to
            if pad_to > 0:
                pad_amt = x.size(-1) % pad_to
                if pad_amt != 0:
                    x = nn.functional.pad(x, (0, pad_to - pad_amt), value=pad_value)
            return x, seq_len

    return ScriptableFilterbank


def check_featurizer_assumptions(featurizer) -> list[str]:
    problems = []
    if featurizer.stft_pad_amount is not None:
        problems.append(f"stft_pad_amount={featurizer.stft_pad_amount!r}, the copy assumes None")
    if featurizer.use_grads:
        problems.append("use_grads is True, so the sqrt guard is not 0")
    if featurizer.log_zero_guard_type != "add":
        problems.append(f"log_zero_guard_type={featurizer.log_zero_guard_type!r}, the copy assumes 'add'")
    if isinstance(featurizer.log_zero_guard_value, str):
        problems.append(f"log_zero_guard_value={featurizer.log_zero_guard_value!r} is a string")
    if not featurizer.log:
        problems.append("log is False, the copy always takes the log")
    if featurizer.frame_splicing != 1:
        problems.append(f"frame_splicing={featurizer.frame_splicing}, the copy assumes 1")
    if featurizer.normalize in ("per_feature", "all_features") or isinstance(featurizer.normalize, dict):
        problems.append(f"normalize={featurizer.normalize!r} actually normalizes; the copy skips it")
    if featurizer.nb_augmentation_prob > 0:
        problems.append("nb_augmentation_prob > 0 (a training-only branch the copy dropped)")
    return problems


# --------------------------------------------------------------------------
# 2. speaker-cache bookkeeping
# --------------------------------------------------------------------------

def make_scriptable_modules(base_type):
    class ScriptableSortformerModules(base_type):
        @property
        def num_weights(self) -> int:
            # NeuralModule's version calls a plain Python helper, which TorchScript
            # compiles and then cannot serialise. Nothing at inference reads it.
            return 0

        def _compress_spkcache(self, emb_seq, preds, mean_sil_emb, permute_spk: bool = False):
            """NVIDIA's method (Apache-2.0), float literals, training branches dropped."""
            batch_size, n_frames, n_spk = preds.shape
            spkcache_len_per_spk = self.spkcache_len // n_spk - self.spkcache_sil_frames_per_spk
            strong_boost_per_spk = math.floor(spkcache_len_per_spk * self.strong_boost_rate)
            weak_boost_per_spk = math.floor(spkcache_len_per_spk * self.weak_boost_rate)
            min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * self.min_pos_scores_rate)

            scores = self._get_log_pred_scores(preds)
            scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)

            if self.scores_boost_latest > 0:
                scores[:, self.spkcache_len :, :] += self.scores_boost_latest

            scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2.0)
            scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1.0)

            if self.spkcache_sil_frames_per_spk > 0:
                pad = torch.full(
                    (batch_size, self.spkcache_sil_frames_per_spk, n_spk),
                    float("inf"),
                    device=scores.device,
                )
                scores = torch.cat([scores, pad], dim=1)

            topk_indices, is_disabled = self._get_topk_indices(scores)
            spkcache, spkcache_preds = self._gather_spkcache_and_preds(
                emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
            )
            return spkcache, spkcache_preds

    return ScriptableSortformerModules


class CacheUpdate(nn.Module):
    """`SortformerModules.streaming_update`, synchronous path, as one unit.

    Keeping the whole update here -- silence threshold, the cap check, how
    `spkcache_preds` first comes into existence -- is what keeps the driver from
    restating any of it. The driver owns the chunk loop and nothing else.
    """

    def __init__(self, modules):
        super().__init__()
        self.m = modules

    def forward(
        self,
        spkcache: torch.Tensor,
        spkcache_preds: torch.Tensor,
        has_preds: bool,
        mean_sil_emb: torch.Tensor,
        n_sil_frames: torch.Tensor,
        pop_embs: torch.Tensor,
        pop_preds: torch.Tensor,
        cache_preds_this_pass: torch.Tensor,
    ):
        mean_sil_emb, n_sil_frames = self.m._get_silence_profile(
            mean_sil_emb, n_sil_frames, pop_embs, pop_preds
        )
        spkcache = torch.cat([spkcache, pop_embs], dim=1)
        if has_preds:
            spkcache_preds = torch.cat([spkcache_preds, pop_preds], dim=1)
        if spkcache.shape[1] > self.m.spkcache_len:
            if not has_preds:
                spkcache_preds = torch.cat([cache_preds_this_pass, pop_preds], dim=1)
                has_preds = True
            spkcache, spkcache_preds = self.m._compress_spkcache(
                emb_seq=spkcache, preds=spkcache_preds, mean_sil_emb=mean_sil_emb
            )
        return spkcache, spkcache_preds, has_preds, mean_sil_emb, n_sil_frames


class _CoreEntryPoint(nn.Module):
    """Gives the traced core a plain `forward`, so the runner can just call it."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        return self.model.forward_for_export(
            chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths
        )


# --------------------------------------------------------------------------
# 3. driver
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nemo", help="path to diar_streaming_sortformer_4spk-v2.1.nemo")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(arguments.device)

    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(arguments.nemo, map_location=arguments.device)
    model.eval()
    modules = model.sortformer_modules
    featurizer = model.preprocessor.featurizer

    problems = check_featurizer_assumptions(featurizer)
    if problems:
        raise SystemExit("this checkpoint breaks what the copied front end bakes in:\n  - " + "\n  - ".join(problems))
    if getattr(model, "async_streaming", False):
        raise SystemExit("async_streaming is on; the exported cache update mirrors the synchronous path")

    print(f"config: feat_in={model.encoder._feat_in} emb_dim={modules.fc_d_model} "
          f"n_spk={modules.n_spk} spkcache={modules.spkcache_len} fifo={modules.fifo_len} "
          f"chunk={modules.chunk_len} subsampling={model.encoder.subsampling_factor}")

    # --- core: NVIDIA's own export boundary, traced ourselves.
    #
    # NOT `model.export()`: that traces and then calls `torch.jit.freeze`
    # (`nemo/core/classes/exportable.py`), which folds BatchNorm into the
    # convolutions -- mathematically equal, numerically ~1e-4 off. This
    # architecture cannot absorb that: the speaker cache keeps frames by a
    # discrete top-k, so one flipped selection diverges the run permanently.
    # Measured on 61.6 minutes: 0.35% of frames flip and one segment boundary
    # moves 156 s. Unfrozen, every shape the loop actually produces is bit-exact.
    #
    # Traced at the shape the loop spends ~99% of its calls in (a full window, a
    # full speaker cache, an empty FIFO); the first and last chunks differ in
    # shape and were verified exact against it anyway.
    subsampling = model.encoder.subsampling_factor
    window = (modules.chunk_len + modules.chunk_left_context + modules.chunk_right_context) * subsampling
    example = (
        torch.rand([1, window, model.encoder._feat_in], device=device),
        torch.tensor([window], device=device),
        torch.randn([1, modules.spkcache_len, modules.fc_d_model], device=device),
        torch.tensor([modules.spkcache_len], device=device),
        torch.zeros([1, modules.fifo_len, modules.fc_d_model], device=device),
        torch.tensor([0], device=device),
    )
    with torch.inference_mode():
        traced_core = torch.jit.trace(_CoreEntryPoint(model), example, check_trace=False)
    core_path = out_dir / "sortformer_core.ts"
    torch.jit.save(traced_core.eval(), str(core_path))
    print(f"  {core_path.name}: {core_path.stat().st_size / 1048576:.1f} MB")

    # --- mel front end
    if "forward" in featurizer.__dict__:
        del featurizer.__dict__["forward"]  # the no_grad rebind
    featurizer.__class__ = make_scriptable_featurizer(type(featurizer))
    scripted_featurizer = torch.jit.script(featurizer)
    pre_path = out_dir / "sortformer_preprocessor.ts"
    scripted_featurizer.save(str(pre_path))
    print(f"  {pre_path.name}: {pre_path.stat().st_size / 1024:.0f} KB")

    # --- cache update
    modules.__class__ = make_scriptable_modules(type(modules))
    scripted_update = torch.jit.script(CacheUpdate(modules).eval())
    update_path = out_dir / "sortformer_cache_update.ts"
    scripted_update.save(str(update_path))
    print(f"  {update_path.name}: {update_path.stat().st_size / 1024:.0f} KB")

    print("\nexported. Parity against the nemo path is not checked here -- run the "
          "G1-G4 harness (docs/archive/speaker-diarization-experiments.md 5a) before shipping these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
