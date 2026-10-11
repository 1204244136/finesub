"""Standalone figure for the frozen second-round results."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import accuracy_lab as lab


def overlap_plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    arms = ["high", "community", "diarizen"]
    labels = ["Sortformer high", "Community-1 (native)", "DiariZen v2 (native)"]
    for ax, (corpus, title) in zip(axes, [("alimeeting", "AliMeeting: 3 meetings"), ("callhome_jpn", "CALLHOME Japanese: 10 calls")]):
        group = data["overlap_summary"][corpus]
        if group["n"] != group["expected"]:
            raise ValueError("Overlap figure requires the complete frozen comparison")
        y = np.arange(3)
        rows = [group["arms"][a]["metrics"] for a in arms]
        ax.barh(y-.18, [r["der"]["mean"]*100 for r in rows], height=.32, color="#b94848", label="DER (lower is better)")
        ax.barh(y+.18, [r["overlap_all_speakers_recall"]["mean"]*100 for r in rows], height=.32, color="#2878b5", label="All-speaker overlap recall")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("File-equal mean (%)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Full-audio UEM, zero collar, overlap included; CALLHOME uses utterance boundaries", fontsize=10)
    path = lab.WT/"docs/archive/assets/diarization-overlap-v2-20260916.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def main():
    data = json.loads((lab.HERE/"results/accuracy-v2-20260916.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    groups = [
        ("development", ["high", "long680", "long1360", "high_ema", "high_refresh", "high_cache376", "high_right80"],
         ["High (27.2 s)", "Chunk 54.4 s", "Chunk 108.8 s", "High + EMA", "High + refresh", "Cache x2", "Right context 6.4 s"], "Development: geometry (7 files)"),
        ("development", ["community", "diarizen", "dz_community", "local_vote"],
         ["Community-1", "DiariZen v2", "DZ local + community", "Vote (community anchor)"], "Development: model combinations (7 files)"),
        ("confirmation", ["base", "high", "high_right80", "community", "diarizen"],
         ["Sortformer base", "Sortformer high", "Right context 6.4 s", "Community-1", "DiariZen v2"], "New-file confirmation (9 episodes)"),
    ]
    for ax, (group, arms, labels, title) in zip(axes, groups):
        rows = [data[group][arm] for arm in arms]
        expected = 9 if group == "confirmation" else 7
        if any(r["n"] != expected for r in rows):
            raise ValueError("Do not plot incomplete or unequal samples as a final comparison")
        y = np.arange(len(rows))
        ax.barh(y-.17, [r["fixed"]*100 for r in rows], height=.32, color="#2878b5", label="Subtitle attribution")
        ax.barh(y+.17, [r["bcubed_f1"]*100 for r in rows], height=.32, color="#e39437", label="Identity B-cubed F1")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score (%)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Diarization experiments — frozen configurations, identical decoded audio", fontsize=14)
    path = lab.WT/"docs/archive/assets/diarization-accuracy-v2-20260916.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)
    overlap_plot(data)


if __name__ == "__main__":
    main()
