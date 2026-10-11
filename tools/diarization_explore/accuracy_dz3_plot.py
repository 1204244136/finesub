"""Standalone figure for DiariZen's higher-speaker-count validation."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import accuracy_lab as lab


def main():
    result = json.loads((lab.HERE / "results/diarizen-v3-20260917.json").read_text(encoding="utf-8"))
    labels = {"base": "Sortformer base", "high": "Sortformer high", "community": "community-1",
              "w16_default": "DiariZen 16 s", "w8_default": "DiariZen 8 s",
              "w16_median5": "DiariZen median 5", "w16_joint": "Joint / 16 s output",
              "w8_joint": "Joint / 8 s output"}
    colors = {"base": "#9aa4b2", "high": "#64748b", "community": "#e69f00",
              "w16_default": "#0072b2", "w8_default": "#009e73", "w16_median5": "#56b4e9",
              "w16_joint": "#cc79a7", "w8_joint": "#8e44ad"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, group, metric, title in (
        (axes[0, 0], "heldout_over4", "fixed", "5-8 speakers: subtitle attribution (higher is better)"),
        (axes[0, 1], "heldout_over4", "f1", "5-8 speakers: identity B-cubed F1 (higher is better)"),
        (axes[1, 0], "new_four_person_meetings", "der", "Four-person meetings: DER (lower is better)"),
        (axes[1, 1], "vox_5_to_13_people", "der", "VoxConverse, 5-13 speakers: DER (lower is better)"),
    ):
        rows = result["summaries"][group] if metric != "der" else result["public"]["summaries"][group]
        arms = [a for a in labels if a in rows]
        values = [100*(rows[a][metric] if metric != "der" else rows[a][metric]["mean"]) for a in arms]
        ax.barh(range(len(arms)), values, color=[colors[a] for a in arms])
        ax.set_yticks(range(len(arms)), [f"{labels[a]} (n={rows[a]['n']})" for a in arms])
        ax.invert_yaxis()
        ax.set_xlim(0, max(100 if metric != "der" else 10, max(values, default=0)*1.17))
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Percent; file-weighted mean")
        ax.spines[["top", "right"]].set_visible(False)
        for y, value in enumerate(values):
            ax.text(value+.5, y, f"{value:.2f}", va="center", fontsize=9)
    fig.suptitle("DiariZen: validation beyond the three-speaker confirmation set", fontsize=15)
    destination = lab.WT / "docs/archive/assets/diarizen-v3-20260917.png"
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=160)
    print(destination)


if __name__ == "__main__":
    main()
