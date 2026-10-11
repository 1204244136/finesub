"""Does misattributed speech carry a confidence signal? (plan 5i.2)

onepass on the 5i development files. Every frame where exactly one reference
speaker talks is either right (its loudest track maps to that speaker under the
one-to-one Hungarian mapping) or wrong. Candidate signals, per frame:
  p_max          the loudest track's probability
  margin         p_max minus the runner-up
  p_max_smooth   p_max averaged over +-0.5 s
and AUC for telling wrong frames from right ones (0.5 = no signal).
"""

from __future__ import annotations

import numpy as np

import over4


def auc(scores_wrong, scores_right):
    """P(a random wrong frame scores lower than a random right one)."""
    both = np.concatenate([scores_wrong, scores_right])
    ranks = both.argsort().argsort() + 1.0
    wrong_ranks = ranks[: len(scores_wrong)]
    n_w, n_r = len(scores_wrong), len(scores_right)
    return 1.0 - (wrong_ranks.sum() - n_w * (n_w + 1) / 2) / (n_w * n_r)


def frames_for(runner, name):
    import soundfile
    from scipy.optimize import linear_sum_assignment

    waveform, rate = soundfile.read(str(over4.audio(name)), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    probs = over4.run(runner, waveform)
    grid = over4.truth(name, len(probs))
    frames = min(len(probs), len(grid))
    probs, grid = probs[:frames], grid[:frames]
    use = (grid.sum(axis=1) == 1) & ((probs > over4.THRESHOLD).any(axis=1))
    out, ref = probs.argmax(axis=1), grid.argmax(axis=1)
    table = np.zeros((probs.shape[1], grid.shape[1]))
    np.add.at(table, (out[use], ref[use]), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    right = np.array([mapping.get(o) == r for o, r in zip(out, ref)])
    ordered = np.sort(probs, axis=1)
    p_max, margin = ordered[:, -1], ordered[:, -1] - ordered[:, -2]
    width = int(0.5 / over4.FRAME)
    smooth = np.convolve(p_max, np.ones(2 * width + 1) / (2 * width + 1), mode="same")
    return {"right": right[use], "p_max": p_max[use], "margin": margin[use], "p_max_smooth": smooth[use]}


def main():
    runner = over4.make_runner()
    many, few = over4.select()
    for label, group in (("many", many), ("few", few)):
        pooled = {key: [] for key in ("right", "p_max", "margin", "p_max_smooth")}
        for name, _, _ in group:
            for key, values in frames_for(runner, name).items():
                pooled[key].append(values)
        data = {key: np.concatenate(values) for key, values in pooled.items()}
        right = data["right"]
        print(f"\n{label}: frames {len(right)}, wrong {(~right).mean():.1%}")
        for signal in ("p_max", "margin", "p_max_smooth"):
            print(f"  {signal:<13} AUC {auc(data[signal][~right], data[signal][right]):.3f}  "
                  f"median right {np.median(data[signal][right]):.3f} / wrong {np.median(data[signal][~right]):.3f}")
        for threshold in (0.6, 0.7, 0.8, 0.9):
            flagged = data["p_max"] < threshold
            caught = (flagged & ~right).sum() / max((~right).sum(), 1)
            precision = (flagged & ~right).sum() / max(flagged.sum(), 1)
            print(f"  flag p_max < {threshold}: flags {flagged.mean():.1%} of frames, catches {caught:.1%} of wrong ones, "
                  f"{precision:.1%} of flags are wrong")


if __name__ == "__main__":
    main()
