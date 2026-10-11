"""Export DiariZen's segmentation model to TorchScript, so production imports neither it nor pyannote.

The same decision D1 made for Sortformer: the research path reaches the model through the
`diarizen` package and a vendored `pyannote.audio` fork, and production may not carry either. What
production needs is one forward pass -- `(batch, channel, samples) -> (batch, frames, speakers)` --
so that is what gets exported.

⚠ Trace, do not `torch.jit.script`: the model body calls into WavLM and a Conformer stack, and a
scripted export has to type-annotate its way through both. Tracing records the actual graph, which
is what we want, at the cost of baking in anything that depends on the input's shape. The check
below is therefore not a formality -- it runs the traced module against eager at several batch
sizes and demands bit-exact agreement, because a trace that silently fixed the batch would produce
plausible wrong answers rather than an error.

    python tools/diarizen_export/export_diarizen.py --out tmp/diarizen-export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: The research checkout the model class lives in. Export-only, like nemo for Sortformer.
SOURCE = REPO / "tmp" / "diarizen-source"
WEIGHTS = REPO / "tmp" / "diarization-models" / "diarizen-v2"

#: 16 s at 16 kHz -- `config.toml`'s `seg_duration`. The window the pipeline slides.
CHUNK_SAMPLES = 16 * 16000


def load_eager(device: str):
    """The segmentation model, taken out of the pipeline the research path builds.

    ⚠ Not constructed from `config.toml` directly. The class inherits from the VENDORED
    `pyannote.audio.Model`, which the research loader reaches by rewriting `pyannote.__path__`
    before importing anything -- build it without that surgery and it inherits the installed
    pyannote instead, whose `__init__` rejects the checkpoint's own hyper-parameters. Reaching
    through the research loader also means the exported module is, by construction, the one every
    cached reading came from, rather than a second model that merely agrees on its config.
    """
    sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
    sys.path.insert(0, str(REPO / "src"))
    import accuracy_diarizen as research

    pipeline = research.load_pipeline()
    model = pipeline._segmentation.model
    import tomllib

    config = tomllib.loads((WEIGHTS / "config.toml").read_text(encoding="utf-8"))
    return model.to(device).eval(), config


def _unmark_jit_exports(model) -> int:
    """Drop `@torch.jit.export` markers so tracing records the path instead of compiling the class.

    The WavLM module this model wraps carries `@torch.jit.export` on `extract_features_mc`, and one
    such marker makes `torch.jit.trace` script the whole class -- including a branch we never take
    (`normalize_waveform` with explicit lengths) whose `F.layer_norm(wave[:length], (length,))` the
    compiler rejects outright. Production always hands the model full-length chunks, so that branch
    is dead here.

    ⚠ This changes WHICH METHODS end up in the artifact, not what any of them computes: tracing
    records the ops that actually ran. `verify` is what turns that from a claim into a check.
    """
    import torch

    cleared = 0
    for module in model.modules():
        for name in dir(type(module)):
            attribute = getattr(type(module), name, None)
            if getattr(attribute, "_torchscript_modifier", None) is torch._jit_internal.FunctionModifiers.EXPORT:
                attribute._torchscript_modifier = torch._jit_internal.FunctionModifiers.DEFAULT
                cleared += 1
    return cleared


def _drop_grad_multiply(model) -> int:
    """Set `feature_grad_mult` to 1.0 so the trace stops calling `GradMultiply`.

    `GradMultiply` is a Python `autograd.Function` that scales the GRADIENT and returns its input
    untouched in forward -- so under inference it does nothing at all, and TorchScript cannot
    serialise it ("Could not export Python function call"). Setting the multiplier to 1.0 takes the
    `!= 1.0` branch out, which is the same arithmetic by definition.

    ⚠ "By definition" is still a claim, and `verify` runs after this, so the bit-exactness check
    covers it rather than this comment having to be believed.
    """
    changed = 0
    for module in model.modules():
        if getattr(module, "feature_grad_mult", 1.0) != 1.0:
            module.feature_grad_mult = 1.0
            changed += 1
    return changed


def verify(traced, eager, device: str, batches=(1, 2, 3), repeats: int = 4) -> None:
    """Bit-exact at several batch sizes AND across repeated calls at one size.

    Several sizes because a trace can freeze the batch. ⚠ And `repeats`, because the first version
    of this check ran exactly one call per size -- three different shapes, three first calls -- and
    TorchScript's profiling executor only recompiles with fused kernels AFTER a shape's first run.
    It therefore could not see the thing it existed to see: measured later on real audio, call 1
    matched and calls 2+ were off by 1.5e-3, which the hard powerset argmax turns into flipped
    labels. Production disables fusion (`diarizen_runtime.jit_fusion_disabled`); this check runs
    under the same condition and repeats the call so a regression there turns it red.

    ⚠ Gaussian noise alone was also too weak a probe. The repeat is what makes the difference
    observable at all, so it is not optional.
    """
    import torch

    from finesub.speech.diarization.diarizen_runtime import jit_fusion_disabled

    torch.manual_seed(20260919)
    for batch in batches:
        sample = torch.randn(batch, 1, CHUNK_SAMPLES, device=device)
        with jit_fusion_disabled(), torch.inference_mode():
            want = eager(sample)
            for call in range(1, repeats + 1):
                got = traced(sample)
                assert got.shape == want.shape, f"batch {batch}: {got.shape} against {want.shape}"
                if not torch.equal(got, want):
                    gap = (got - want).abs().max().item()
                    raise AssertionError(
                        f"batch {batch}, call {call}: traced differs from eager, "
                        f"max |delta| {gap:.3e}. Close is not equal, and the port's acceptance "
                        "asks for equality against the research path."
                    )
        print(f"   batch {batch}: bit-exact over {repeats} calls, shape {tuple(got.shape)}",
              flush=True)


class _NoTrainer:
    """Stands in for a Lightning Trainer during export. Nothing calls it; it only has to exist."""


def _silence_lightning_trainer(model) -> int:
    """Give every LightningModule a `_trainer` so `hasattr(module, "trainer")` stops raising.

    ⚠ `torch.jit.trace` calls `module_has_exports`, which walks `dir(module)` with `hasattr`.
    Lightning's `trainer` property raises RuntimeError -- not AttributeError -- when nothing is
    attached, so `hasattr` propagates it and the trace dies with "is not attached to a `Trainer`"
    before it has looked at a single tensor. Wrapping the model does not help: the scan recurses
    into submodules.

    Export-time only, and it touches no weight and no forward path.
    """
    touched = 0
    for module in model.modules():
        if hasattr(type(module), "trainer") and getattr(module, "_trainer", None) is None:
            module._trainer = _NoTrainer()
            touched += 1
    return touched


def _copy_plda(out: Path) -> int:
    """Put the PLDA tables beside the TorchScript, so production loads one directory.

    The clustering needs `plda.npz` and `xvec_transform.npz`, which ship inside the DiariZen repo
    rather than being derived from it. ⚠ They belong here and not behind `hf_weights.prepare()`:
    production never fetches the raw checkpoint -- it loads the EXPORTED artifacts, exactly as the
    Sortformer backend does. The manifest entries exist so this exporter can get the weights, not
    so a run can.
    """
    import shutil

    target = out / "plda"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in ("plda.npz", "xvec_transform.npz"):
        source = WEIGHTS / "plda" / name
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing; the clustering cannot run without it")
        shutil.copyfile(source, target / name)
        copied += 1
    print(f"copied {copied} PLDA tables to {target}")
    return copied


def _export_embedding(out: Path, device: str) -> None:
    """WeSpeaker, the model the clustering half runs on. Same rules as the segmentation half.

    The diarization path always calls it WITH per-frame weights -- the mask that says which frames
    of the chunk belong to the speaker being embedded -- so that is the signature traced. Tracing
    the unweighted call would produce an artifact that silently ignores the mask and returns one
    embedding per chunk instead of one per speaker.
    """
    import torch

    sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
    sys.path.insert(0, str(REPO / "src"))
    import accuracy_diarizen as research

    model = research.load_pipeline()._embedding.model_.to(device).eval()
    _unmark_jit_exports(model)
    _drop_grad_multiply(model)
    _silence_lightning_trainer(model)

    # 5 s at 16 kHz, and the frame count the model derives from it (probed, not guessed).
    class Embedder(torch.nn.Module):
        """`resnet(fbank, weights)[1]` -- the half of WeSpeaker that can be traced.

        ⚠ The waveform half CANNOT. `compute_fbank` runs `torch.vmap` over
        `torchaudio.compliance.kaldi.fbank`, and the tracer dies inside it with "Unsupported value
        kind: Tensor". The split is clean, though: `forward` is exactly `compute_fbank` followed by
        `resnet(...)[1]`, and fbank is torchaudio's own public function -- already a declared
        dependency -- so production recomputes that part and loads this one.

        The wrapper also keeps the tracer away from Lightning's `trainer` property and pins the
        `[1]`: the resnet returns a pair and only the second element is the embedding.
        """

        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet

        def forward(self, fbank, weights):
            return self.resnet(fbank, weights=weights)[1]

    wrapped = Embedder(model.resnet).eval()
    # 5 s of audio gives 249 fbank frames of 80 mel bins (probed from the model, not assumed).
    fbank = torch.randn(2, 249, 80, device=device)
    weights = torch.rand(2, 249, device=device)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, (fbank, weights), check_trace=False).eval()

    torch.manual_seed(20260919)
    for batch in (1, 2, 3):
        sample = torch.randn(batch, 249, 80, device=device)
        mask = torch.rand(batch, 249, device=device)
        with torch.inference_mode():
            want = model.resnet(sample, weights=mask)[1]
            got = traced(sample, mask)
        if not torch.equal(got, want):
            raise AssertionError(
                f"embedding batch {batch}: max |delta| {(got - want).abs().max().item():.3e}")
        print(f"   embedding batch {batch}: bit-exact, shape {tuple(got.shape)}", flush=True)

    _check_production_fbank(model, device)

    target = out / "diarizen_embedding.ts"
    traced.save(str(target))
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


def _check_production_fbank(model, device: str) -> None:
    """The half that is NOT in the artifact has to be checked too, or the split hides a bug.

    Exporting `resnet(fbank, weights)` and recomputing `compute_fbank` in production splits one
    forward pass across two implementations. The traced half is verified against eager above; this
    verifies the other half against the same source, so that "bit-exact" covers the whole model
    rather than the convenient part of it.
    """
    import torch

    sys.path.insert(0, str(REPO / "src"))
    from finesub.speech.diarization.diarizen_features import compute_fbank

    torch.manual_seed(20260919)
    for samples in (5 * 16000, 3 * 16000 + 137):
        waveform = torch.randn(2, 1, samples, device=device)
        with torch.inference_mode():
            want = model.compute_fbank(waveform)
            got = compute_fbank(waveform)
        if not torch.equal(got, want):
            raise AssertionError(
                f"fbank at {samples} samples differs from the research path: "
                f"max |delta| {(got - want).abs().max().item():.3e}")
        print(f"   fbank {samples} samples: bit-exact, shape {tuple(got.shape)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-export"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading eager model", flush=True)
    model, config = load_eager(args.device)

    print("tracing", flush=True)
    cleared = _unmark_jit_exports(model)
    print(f"   cleared {cleared} @torch.jit.export markers", flush=True)
    neutralised = _drop_grad_multiply(model)
    print(f"   neutralised {neutralised} GradMultiply call(s)", flush=True)
    example = torch.randn(2, 1, CHUNK_SAMPLES, device=args.device)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
    # ⚠ NO `torch.jit.freeze`. It folds BatchNorm into the preceding convolution and the answer
    # moves by 4.5e-4 to 7.3e-4 -- measured on three batch sizes, against an eager model that is
    # bit-exact with itself. Unfrozen tracing is bit-exact at every size. This repository already
    # knew the shape of this bug: `tools/sortformer_export/README.md` says the same about NeMo's
    # `model.export()`, which freezes. A folded export would have passed a "close enough" reading
    # and failed the port's actual acceptance.
    traced = traced.eval()

    print("verifying traced against eager", flush=True)
    verify(traced, model, args.device)

    target = out / "diarizen_segmentation.ts"
    traced.save(str(target))
    _export_embedding(out, args.device)
    _copy_plda(out)
    (out / "identity.json").write_text(json.dumps({
        "model": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        "chunk_samples": CHUNK_SAMPLES,
        "model_args": config["model"]["args"],
        "inference_args": config["inference"]["args"],
    }, indent=1), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
