"""S3 alone: the numpy rewrite of the clustering, on the research path's OWN embeddings.

⚠ `check_s4.py` feeds the cached local segmentation but OUR embeddings, so its 2.378% mixes two
things: the rewrite, and a 1e-3 embedding difference amplified through a hard argmax. It cannot
tell them apart. This one removes the neural half entirely -- cached segmentation AND cached
embeddings, both frozen -- so what is left is the clustering and assembly rewrite by itself.

That matters for the rule the owner set: a port goes through as A-class when no defect is found
*and the layered evidence shows each step faithful*. S1, S2 and the fbank have that evidence;
this is the step that did not.

⚠ CPU only, no GPU. Everything it needs is already on disk.

    python tools/diarizen_export/check_s3.py --ids hina-32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = REPO / "tmp" / "diarization-dz3"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"


def reference_style_raster(native, out_frames: int, *, lab_frame: float = 0.08):
    """Our native 20 ms output through the REFERENCE's own conversion, not ours.

    Not an approximation -- the two steps are transcribed from the source:

    * `Binarize.__call__` (pyannote/audio/utils/signal.py) walks the frames and, with
      `onset = offset = 0.5` on 0/1 data, emits `Segment(start, t)` where `start` is the timestamp
      of the FIRST ACTIVE frame and `t` is the timestamp of the FIRST INACTIVE one -- so the end
      lands one frame PAST the last active frame. Timestamps are frame MIDDLES
      (`frames[i].middle`). ⚠ That one-frame overshoot is the whole mechanism: it is why the
      reference is systematically wider, and it is not a rounding difference.
    * `raster()` (accuracy_diarizen.py) then writes `int(start / 0.08) : ceil(end / 0.08)`.

    `min_duration_on` / `min_duration_off` / padding are all 0 for this pipeline, so `support()`
    and the short-track removal are skipped; if that ever changes this stops being faithful.
    """
    import numpy as np

    from finesub.speech.diarization import diarizen_assemble as assembler

    out = np.zeros((out_frames, native.shape[1]), dtype=np.float32)
    last = native.shape[0] - 1
    for column in range(native.shape[1]):
        active = (native[:, column] > 0.5).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], active, [0]))))
        for begin, end in zip(edges[0::2], edges[1::2]):
            start_time = (assembler.FRAME_START + begin * assembler.FRAME_STEP
                          + 0.5 * assembler.FRAME_DURATION)
            stop_frame = end if end <= last else last
            stop_time = (assembler.FRAME_START + stop_frame * assembler.FRAME_STEP
                         + 0.5 * assembler.FRAME_DURATION)
            low = max(0, int(start_time / lab_frame))
            high = int(np.ceil(stop_time / lab_frame))
            out[low:high, column] = 1.0
    return out


def reference_from_rttm(path, frames: int, *, lab_frame: float = 0.08):
    """The reference grid for a corpus whose cache kept the RTTM but not the `.npz`.

    ⚠ Not a second rasterisation rule -- `raster()` writes BOTH files from the same `Annotation`,
    so rebuilding the grid from the RTTM reproduces the `.npz` exactly. That is what lets the
    meetings and the vox clips join this check at all: their caches have the segmentation and the
    embeddings, so only the reference grid was missing.

    Columns are ordered by label the way `raster` does (`annotation.labels()` is sorted).
    """
    import numpy as np

    turns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start, duration, who = float(fields[3]), float(fields[4]), fields[7]
        turns.append((start, start + duration, who))
    speakers = sorted({who for _, _, who in turns})
    out = np.zeros((frames, max(1, len(speakers))), dtype=np.float32)
    for start, stop, who in turns:
        low = max(0, int(start / lab_frame))
        out[low:int(np.ceil(stop / lab_frame)), speakers.index(who)] = 1.0
    return out


def discriminate(aligned, reference):
    """Where the differences sit -- the table the attribution was built on."""
    import numpy as np

    differing = aligned != reference
    only_reference = int((reference & ~aligned).sum())
    only_ours = int((aligned & ~reference).sum())
    edge = np.zeros_like(reference)
    edge[1:] |= reference[1:] != reference[:-1]
    edge[:-1] |= reference[:-1] != reference[1:]
    total = int(differing.sum())
    return {
        "differing": total,
        "reference_only": only_reference,
        "ours_only": only_ours,
        "at_edge": int((differing & edge).sum()),
        "active_ours": int(aligned.sum()),
        "active_reference": int(reference.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", default=["hina-32"])
    args = parser.parse_args()

    from check_port import to_lab_grid
    from finesub.speech.diarization import diarizen_assemble as assembler
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner

    for ident in args.ids:
        local_path = CACHE / ident / "w16" / "diarizen-local.npz"
        embed_path = CACHE / ident / "w16" / "embeddings.npy"
        want_path = CACHE / ident / "w16_default.npz"
        rttm_path = CACHE / ident / "w16_default.rttm"
        if not (local_path.exists() and embed_path.exists()):
            print(f"{ident}: cache incomplete, skipped")
            continue
        if not (want_path.exists() or rttm_path.exists()):
            print(f"{ident}: no reference grid, skipped")
            continue

        local = np.load(local_path)["data"]
        embeddings = np.load(embed_path)
        want = None if not want_path.exists() else np.load(want_path)["probs"]

        filtered = runner.median_filtered(local)
        hard = clusterer.cluster(embeddings, filtered, PLDA).hard
        native = assembler.assemble(filtered, hard).astype(np.float32)
        ours = to_lab_grid(native, assembler.FRAME_STEP)
        if want is None:
            want = reference_from_rttm(rttm_path, len(ours))
        mimicked = reference_style_raster(native, len(want))

        frames = min(len(ours), len(want))
        width = max(ours.shape[1], want.shape[1])

        def pad(grid):
            if grid.shape[1] >= width:
                return grid[:frames]
            return np.hstack([grid, np.zeros((len(grid), width - grid.shape[1]),
                                             grid.dtype)])[:frames]

        from scipy.optimize import linear_sum_assignment

        a, b = pad(ours).astype(bool), pad(want).astype(bool)
        table = a.T.astype(np.int64) @ b.astype(np.int64)
        rows, columns = linear_sum_assignment(-table)
        aligned = np.zeros_like(a)
        for source, target in zip(rows, columns):
            aligned[:, target] |= a[:, source]

        either = aligned.any(axis=1) | b.any(axis=1)
        moved = float((aligned[either] != b[either]).any(axis=1).mean()) if either.any() else 0.0
        print(f"{ident}: speakers {ours.shape[1]} vs {want.shape[1]}  "
              f"our-grid frames differing {moved:.4%}")

        stats = discriminate(aligned, b)
        print(f"   of {stats['differing']} differing cells: "
              f"reference-only {stats['reference_only']} "
              f"({stats['reference_only'] / max(stats['differing'], 1):.1%}), "
              f"ours-only {stats['ours_only']}, "
              f"at a reference edge {stats['at_edge'] / max(stats['differing'], 1):.1%}; "
              f"reference is {stats['active_reference'] / max(stats['active_ours'], 1):.4f}x wider")

        # The direct test: our own output through the reference's conversion rule.
        m = mimicked.astype(bool)[:frames]
        m = m if m.shape[1] >= width else np.hstack(
            [m, np.zeros((len(m), width - m.shape[1]), bool)])
        remapped = np.zeros_like(m)
        for source, target in zip(rows, columns):
            remapped[:, target] |= m[:, source]
        residual = int((remapped != b).sum())
        if residual == 0:
            print("   ⇒ our output through the REFERENCE's rule: BIT-IDENTICAL")
        else:
            share = residual / max(stats["differing"], 1)
            print(f"   ⇒ our output through the REFERENCE's rule: {residual} cells still differ "
                  f"({residual / max(b.size, 1):.4%} of all cells, "
                  f"{share:.1%} of the original gap)")
            # ⚠ The residual is the only part that is NOT the grid rule, so it is the part worth
            # looking at: list where it sits rather than averaging it away.
            rows_with = np.flatnonzero((remapped != b).any(axis=1))
            print(f"      on {len(rows_with)} frames; first few: {rows_with[:8].tolist()}")


if __name__ == "__main__":
    main()
