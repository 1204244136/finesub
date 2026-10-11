"""Render the frozen holdout comparison; requires matplotlib only for this figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "results" / "accuracy-20260916.json").read_text(encoding="utf-8"))["summaries"]["holdout"]
arms = ["ema", "high", "split", "community"]
labels = ["Cache EMA 0.25", "Sortformer high context", "Recursive split", "Community-1"]
colors = ["#8a94a6", "#16847e", "#d18b24", "#7153b6"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, metric, interval, title in zip(axes, ["delta", "bcubed_delta"], ["ci", "bcubed_delta_ci"],
                                     ["Subtitle attribution", "Identity B-cubed F1"]):
    ax.axvline(0, color="#8a94a6", linewidth=1, linestyle="--")
    for y, arm, color in zip(range(4), arms, colors):
        center = data[arm][metric]*100
        lo, hi = np.asarray(data[arm][interval])*100
        ax.errorbar(center, y, xerr=[[center-lo], [hi-center]], fmt="o", color=color, capsize=5, markersize=7)
        ax.annotate(f"{center:+.2f}", (center, y), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=10)
    ax.set_title(title, loc="left", fontsize=12, pad=16)
    ax.set_xlabel("Change vs. baseline (percentage points)")
    ax.set_yticks(range(4), labels)
    ax.set_ylim(3.6, -0.6)
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
fig.suptitle("67 held-out episodes: average gain is not a per-file guarantee", fontsize=14, x=0.04, ha="left")
fig.text(0.04, 0.02, "Paired episode bootstrap, 95% CI. Development episodes excluded. Attribution is not DER or name recognition.", fontsize=9, color="#596273")
fig.tight_layout(rect=[0, 0.07, 1, 0.93], w_pad=3)
destination = HERE.parents[1] / "docs" / "plans" / "assets" / "diarization-accuracy-20260916.png"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=160, facecolor="white")
print(destination)
