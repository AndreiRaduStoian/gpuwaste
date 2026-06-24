from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("/home/andrei/final/gpuwaste/results/accelsim_results/")
OUT = ROOT / "combined_figures"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "key": "low_n",
        "title": "Initial low-N\nvectoradd N=1,048,576\ncompute/shared N=262,144",
        "path": ROOT / "low_n_final" / "low_n_sweep_outputs_final" / "low_n_sweep_summary.csv",
    },
    {
        "key": "large3",
        "title": "Main run\nall kernels N=1,048,576",
        "path": ROOT / "large3_final" / "large3_sweep_outputs_final" / "large3_sweep_summary.csv",
    },
    {
        "key": "large3_n2",
        "title": "Doubled-N run\nall kernels N=2,097,152",
        "path": ROOT / "large3_n2_final" / "large3_n2_sweep_outputs_final" / "large3_n2_sweep_summary.csv",
    },
    {
        "key": "large3_n4",
        "title": "Quadrupled-N run\nall kernels N=4,194,304",
        "path": ROOT / "large3_n4_final" / "large3_n4_sweep_outputs_final" / "large3_n4_sweep_summary.csv",
    },
]

PARAMS = [
    ("sm", "SM scaling"),
    ("core", "Core-clock scaling"),
    ("mem", "Memory-clock scaling"),
]

KERNELS = ["compute", "vectoradd", "shared"]

LINESTYLES = {
    "compute": "-",
    "vectoradd": "--",
    "shared": ":",
}

MARKERS = {
    "compute": "o",
    "vectoradd": "s",
    "shared": "^",
}

def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["scale"] = float(r["scale"])
        r["time_slowdown"] = float(r["time_slowdown"])

    return rows


def main() -> None:
    data = {
        run["key"]: read_rows(run["path"])
        for run in RUNS
    }

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(21, 12),
        sharex=True,
        sharey=False,
    )

    for col, run in enumerate(RUNS):
        axes[0][col].set_title(run["title"], fontsize=12)

    for row, (param, param_title) in enumerate(PARAMS):
        axes[row][0].set_ylabel(f"{param_title}\nTime slowdown", fontsize=11)

        for col, run in enumerate(RUNS):
            ax = axes[row][col]
            rows = [
                r for r in data[run["key"]]
                if r["param"] == param
            ]

            for kernel in KERNELS:
                kr = sorted(
                    [r for r in rows if r["kernel"] == kernel],
                    key=lambda x: x["scale"],
                )
                if not kr:
                    continue

                ax.plot(
                    [r["scale"] for r in kr],
                    [r["time_slowdown"] for r in kr],
                    marker=MARKERS[kernel],
                    linestyle=LINESTYLES[kernel],
                    linewidth=2.0,
                    markersize=5,
                    label=kernel,
                )

            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Resource scale")
            ax.set_xticks([0.25, 0.5, 0.75, 1.0])

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        "Time slowdown across problem sizes and resource-scaling experiments",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = OUT / "grid_3x4_time_slowdown_problem_size_vs_scaling.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
