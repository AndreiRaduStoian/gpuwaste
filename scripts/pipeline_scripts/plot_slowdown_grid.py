#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FAMILY_ORDER = ["SM", "SP", "SFU", "CORE", "MEM"]
KERNEL_ORDER = ["compute", "vectoradd", "shared", "pointer", "sfu"]


def ordered_present(values, preferred):
    vals = list(dict.fromkeys(values))
    return [x for x in preferred if x in vals] + [x for x in vals if x not in preferred]


def plot_set(df_set, out_path):
    df = df_set[df_set["family"] != "BASE"].copy()
    if df.empty:
        return

    kernels = ordered_present(df["kernel"].dropna().unique(), KERNEL_ORDER)
    families = ordered_present(df["family"].dropna().unique(), FAMILY_ORDER)

    fig, axes = plt.subplots(
        nrows=len(kernels),
        ncols=len(families),
        figsize=(3.2 * len(families), 2.5 * len(kernels)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for r, kernel in enumerate(kernels):
        for c, family in enumerate(families):
            ax = axes[r][c]
            sub = df[(df["kernel"] == kernel) & (df["family"] == family)].copy()
            sub = sub.sort_values("fraction", ascending=False)

            if not sub.empty:
                ax.plot(sub["fraction"], sub["slowdown_vs_base"], marker="o", label="model")
                if "ideal_slowdown_if_fully_bound" in sub.columns:
                    ax.plot(
                        sub["fraction"],
                        sub["ideal_slowdown_if_fully_bound"],
                        marker="x",
                        linestyle="--",
                        label="inverse",
                    )

            ax.set_title(f"{kernel} / {family}")
            ax.grid(True, alpha=0.3)
            ax.set_xticks([1.0, 0.75, 0.5, 0.25])
            ax.invert_xaxis()

            if r == len(kernels) - 1:
                ax.set_xlabel("resource fraction")
            if c == 0:
                ax.set_ylabel("slowdown vs base")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False)

    fig.suptitle(f"Pipeline slowdown grid", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="pipeline_scaling_results.csv")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for set_name, df_set in df.groupby("set", sort=False):
        safe = str(set_name).replace("/", "_")
        out_path = out_dir / f"slowdown_grid_{safe}.png"
        plot_set(df_set, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
