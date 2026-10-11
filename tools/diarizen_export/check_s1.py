"""S1 of the export-equivalence pre-registration: our sliding window against the frozen cache.

`tmp/diarization-dz3/<id>/w16/diarizen-local.npz` is what the research path produced and every
published reading was scored on. The port's local segmentation has to reproduce it bit for bit --
that is what "semantic invariance" means here, and the pre-registration says so before this ran.

    python tools/diarizen_export/check_s1.py --ids hina-40 hina-14
    python tools/diarizen_export/check_s1.py            # every cached file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

CACHE = REPO / "tmp" / "diarization-dz3"
EXPORT = REPO / "tmp" / "diarizen-export" / "diarizen_segmentation.ts"


def cached_ids() -> list[str]:
    return sorted(p.parent.parent.name for p in CACHE.glob("*/w16/diarizen-local.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", default=None)
    # ⚠ 2, not 32. The research path's loader overrides config.toml's 32 with 2, so every frozen
    # cache came from batches of two -- and the batch size is NOT neutral here: at 8 and at 32 the
    # answer moves on 0.0117% of frames with max|delta| 1.0, because the hard powerset argmax turns
    # a last-bit float difference into a flipped label. Comparing at 32 would fail the equivalence
    # check for a reason that has nothing to do with the port.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    import soundfile
    import torch

    import accuracy_lab as lab
    from finesub.speech.diarization import diarizen_runner as runner

    rows = []
    for ident in (args.ids or cached_ids()):
        reference = np.load(CACHE / ident / "w16" / "diarizen-local.npz")
        want = reference["data"]
        signal, rate = soundfile.read(lab.source(ident), dtype="float32", always_2d=True)
        assert rate == runner.SAMPLE_RATE, rate
        waveform = torch.from_numpy(signal.mean(axis=1))[None]

        got = runner.segment(EXPORT, waveform, batch_size=args.batch_size).data
        same_shape = got.shape == want.shape
        exact = bool(same_shape and np.array_equal(got, want))
        gap = float(np.abs(got - want).max()) if same_shape else float("nan")
        # Where they differ at all, how much of the timeline is it?
        moved = float((got != want).any(axis=-1).mean()) if same_shape else float("nan")
        per_chunk = ((got != want).any(axis=-1).mean(axis=1) if same_shape
                     else np.zeros(1))
        offenders = np.flatnonzero(per_chunk)
        rows.append({"ident": ident, "shape": list(got.shape), "same_shape": same_shape,
                     "bit_exact": exact, "max_abs": gap, "frames_moved": moved,
                     "chunks_differing": int(offenders.size),
                     "first_offenders": offenders[:10].tolist(),
                     "last_chunk_differs": bool(offenders.size and offenders[-1] == len(per_chunk) - 1)})
        if offenders.size:
            print(f"   {offenders.size}/{len(per_chunk)} chunks differ; first {offenders[:8].tolist()}"
                  f" last {offenders[-3:].tolist()} (final chunk index {len(per_chunk) - 1})", flush=True)
        print(f"{ident:10} {'BIT-EXACT' if exact else 'DIFFERS ':10} shape {got.shape} "
              f"vs {want.shape}  max|delta| {gap:.3e}  frames moved {moved:.4%}", flush=True)

    exact = sum(1 for row in rows if row["bit_exact"])
    print(f"\nS1: {exact}/{len(rows)} bit-exact")
    moved = [row["frames_moved"] for row in rows if row["same_shape"]]
    if moved:
        print(f"    frames moved: median {float(np.median(moved)):.4%}, "
              f"max {max(moved):.4%}, over {len(moved)} files")
    if args.write:
        out = REPO / "tmp" / "diarizen-export" / "s1-results.json"
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    # ⚠ A reading, not a gate. Swapping eager for a traced module is an execution-form change in
    # `CLAUDE.md`'s A/B sense, and those are characterised rather than required to be identical --
    # equality is a free no-regression proof when you get it, not a bar to clear. Making this exit
    # non-zero was my own added constraint and it nearly pushed the port into vendoring the model
    # definition to buy an equality nobody had asked for.
    if exact != len(rows):
        print("    ⇒ not bit-exact, which is a number to carry rather than a failure. What decides "
              "anything is the end-to-end subtitle label change rate against DiariZen's own "
              "perturbation band -- see the pre-registration.")


if __name__ == "__main__":
    main()
