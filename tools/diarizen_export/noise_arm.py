"""DiariZen's own sensitivity band, measured on enough files to be a band.

The pre-registration (plan section 4, prerequisite 2) fixed the threshold for the port against a
reference band -- `noise` sigma=1e-7 -> 1.36%, `pcm16` -> 1.14%, one sample of shift -> 2.84% --
and then said, in the same breath, that all three are **n=1** and must be measured on >=10 files
before they are used as a gate. This is that measurement.

⚠ Both arms run the PORT, not the research path. The quantity wanted is "how far does this chain
move when the input is nudged by nothing", and mixing two implementations into it would measure
the port as well. One implementation, two inputs.

⚠ `micro` (a pure gain) is deliberately NOT an arm here: WavLM's frontend layer-norms a constant
gain away and the WeSpeaker side's fbank+CMN does the same, so it reads 0.00% by construction
(probe 7.7). An equivalence check built on it proves nothing.

    python tools/diarizen_export/noise_arm.py --ids hina-10 hina-11 ...
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_port import (  # noqa: E402
    EXPORT_DIR, PLDA, episode_for, label_change, source_for, to_lab_grid,
)


def run_port_on(waveform):
    """The port's four stages on an already-loaded `(1, samples)` tensor."""
    from finesub.speech.diarization import diarizen_assemble as assembler
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_embed as embedder
    from finesub.speech.diarization import diarizen_runner as runner

    local = runner.segment(EXPORT_DIR / "diarizen_segmentation.ts", waveform).data
    filtered = runner.median_filtered(local)
    embeddings = embedder.embed(EXPORT_DIR / "diarizen_embedding.ts", waveform, filtered)
    hard = clusterer.cluster(embeddings, filtered, PLDA).hard
    native = assembler.assemble(filtered, hard).astype(np.float32)
    return to_lab_grid(native, assembler.FRAME_STEP), native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--arm", default="noise", choices=("noise", "pcm16", "shift1"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[None],
                        help="repeat the arm with these draws. ⚠ Measured: the reading is "
                             "seed-INVARIANT -- three files at three seeds each returned the same "
                             "figure and the same flipped rows, while the timelines differed. So "
                             "extra seeds confirm that property; they do not reduce an error bar.")
    parser.add_argument("--out", default=str(EXPORT_DIR / "noise-arm.json"))
    args = parser.parse_args()

    import soundfile
    import torch

    import accuracy_lab as lab
    import backend_cost as bc
    from dz_probe import perturbations

    rows = []
    for ident in args.ids:
        loaded = episode_for(ident)
        if loaded is None:
            print(f"{ident}: no refined axis, skipped", flush=True)
            continue
        stable, grid, _, _, _ = loaded

        signal, rate = soundfile.read(source_for(ident), dtype="float32", always_2d=True)
        base = signal.mean(axis=1)
        clean, clean_native = run_port_on(torch.from_numpy(base)[None])

        # Kept because this GPU pass is the only place the port's own clean timeline exists:
        # pricing the acceptance set against the PORT rather than the research cache needs it,
        # and re-running the chain a second time to get it would be the wasteful way round.
        timelines = EXPORT_DIR / "port-timelines"
        timelines.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(timelines / f"{ident}.npz", probs=clean.astype(np.int8))
        # ⚠ The NATIVE 20 ms grid too. Without it, comparing the port against the research path
        # means crossing two rasterisation rules, and the systematic width difference that comes
        # with it cannot be separated from the port's own jitter (findings 5.1). With it,
        # the same-yardstick number is a free offline calculation later.
        native_dir = EXPORT_DIR / "port-timelines-native"
        native_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(native_dir / f"{ident}.npz", probs=clean_native.astype(np.int8))

        previous_nudged = previous_wave = None
        for seed in args.seeds:
            wave = perturbations(base, seed)[args.arm]
            nudged, _ = run_port_on(torch.from_numpy(wave)[None])
            # ⚠ Every seed on a file read exactly the same percentage, three files out of three.
            # A rate cannot say whether that is the same rows flipping or the same TIMELINE, so
            # the timelines are compared directly -- and the input delta is printed beside it,
            # because "the answer did not move" and "the input did not move" look identical from
            # downstream and only one of them is a finding.
            if previous_nudged is not None:
                same = (previous_nudged.shape == nudged.shape
                        and np.array_equal(previous_nudged, nudged))
                delta = float(np.abs(wave.astype(np.float64)
                                     - previous_wave.astype(np.float64)).max())
                print(f"    vs previous seed: timeline "
                      f"{'IDENTICAL' if same else 'differs'}; input max delta {delta:.3e}",
                      flush=True)
            previous_nudged, previous_wave = nudged, wave

            moved, rows_changed = label_change(nudged, clean, stable, grid, with_rows=True)
            rows.append({"ident": ident, "arm": args.arm, "seed": seed,
                         "label_change_pct": moved,
                         "speakers_clean": int(clean.shape[1]),
                         "speakers_nudged": int(nudged.shape[1]),
                         "rows_changed": rows_changed})
            # ⚠ Reported every file: the way this goes wrong on Windows is silent -- once the
            # dedicated side fills, the driver spills into shared memory and the run keeps going
            # over PCIe at a tenth of the speed, with nothing raised and nvidia-smi showing a flat
            # plateau (docs/bench-discipline.md, 第七条).
            reserved = torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
            print(f"{ident} seed={seed}: {args.arm} moves {moved:.3f}% of subtitle labels  "
                  f"(speakers {clean.shape[1]} vs {nudged.shape[1]}, "
                  f"rows {rows_changed[:6]})  [reserved {reserved:.2f} GiB]", flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")

    if rows:
        values = sorted(row["label_change_pct"] for row in rows)
        same = sum(r["speakers_clean"] == r["speakers_nudged"] for r in rows)
        print(f"\n{args.arm} arm, n={len(rows)}: median {np.median(values):.3f}%  "
              f"P90 {np.percentile(values, 90):.3f}%  max {max(values):.3f}%")
        print(f"  speaker count unchanged on {same}/{len(rows)}")
        print("  the n=1 reading this replaces: 1.36% (onetonton-14, probe 7.5)")


if __name__ == "__main__":
    main()
