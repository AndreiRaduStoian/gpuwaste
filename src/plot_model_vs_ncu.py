import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = "pipeline_base3070_predictions.csv"
NCU_CSV = "ncu_native_comparison.csv"
OUT_DIR = "plots_model_vs_ncu"

os.makedirs(OUT_DIR, exist_ok=True)

pred = pd.read_csv(PRED_CSV)
ncu = pd.read_csv(NCU_CSV)

for df in (pred, ncu):
    for c in df.columns:
        if c not in ("kernel",):
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")

merged = pred.merge(ncu, on=["kernel", "n", "iterations"], how="inner")

merged["native_runtime_ms"] = merged["gpu_time_duration_us"] / 1000.0
merged["runtime_ratio_pred_over_ncu"] = merged["predicted_runtime_ms"] / merged["native_runtime_ms"]
merged["runtime_error_pct"] = 100.0 * (merged["predicted_runtime_ms"] - merged["native_runtime_ms"]) / merged["native_runtime_ms"]
merged["runtime_abs_error_pct"] = merged["runtime_error_pct"].abs()
merged["runtime_log2_error"] = np.log2(merged["runtime_ratio_pred_over_ncu"])

merged["gpu_cycle_ratio_pred_over_ncu"] = merged["gpu_cycles"] / merged["gpu_cycles_elapsed_avg"]
merged["gpu_cycle_abs_error_pct"] = 100.0 * (merged["gpu_cycles"] - merged["gpu_cycles_elapsed_avg"]).abs() / merged["gpu_cycles_elapsed_avg"]
merged["gpu_cycle_log2_error"] = np.log2(merged["gpu_cycle_ratio_pred_over_ncu"])

merged.to_csv(os.path.join(OUT_DIR, "merged_model_vs_ncu.csv"), index=False)

summary = (
    merged.groupby("kernel")
    .agg(
        points=("runtime_abs_error_pct", "count"),
        mape_runtime_pct=("runtime_abs_error_pct", "mean"),
        median_abs_runtime_pct=("runtime_abs_error_pct", "median"),
        mean_runtime_log2_error=("runtime_log2_error", "mean"),
        mape_gpu_cycles_pct=("gpu_cycle_abs_error_pct", "mean"),
    )
    .reset_index()
)

summary.to_csv(os.path.join(OUT_DIR, "error_summary.csv"), index=False)
summary.round(2).to_latex(os.path.join(OUT_DIR, "error_summary.tex"), index=False)

# Parity plot: predicted runtime vs NCU runtime.
fig, ax = plt.subplots(figsize=(6, 5))
for kernel, g in merged.groupby("kernel"):
    ax.scatter(g["native_runtime_ms"], g["predicted_runtime_ms"], label=kernel, s=28)

lo = min(merged["native_runtime_ms"].min(), merged["predicted_runtime_ms"].min())
hi = max(merged["native_runtime_ms"].max(), merged["predicted_runtime_ms"].max())
ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("NCU runtime (ms)")
ax.set_ylabel("Pipeline prediction (ms)")
ax.set_title("Predicted vs native runtime")
ax.legend(fontsize=8)
ax.grid(True, which="both", linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "runtime_parity.png"), dpi=200)
plt.close(fig)

# Error heatmap: signed log2(pred/native). One panel per N.
kernels = ["compute", "vectoradd", "pointer", "shared", "sfu"]
iterations = sorted(merged["iterations"].unique())
ns = sorted(merged["n"].unique())

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
axes = axes.ravel()

vmax = max(1.0, np.nanmax(np.abs(merged["runtime_log2_error"])))

for ax, n in zip(axes, ns):
    mat = []
    for k in kernels:
        row = []
        for it in iterations:
        # it = 2097152
            x = merged[(merged["kernel"] == k) & (merged["n"] == n) & (merged["iterations"] == it)]
            row.append(float(x["runtime_log2_error"].iloc[0]) if len(x) else np.nan)
        mat.append(row)

    im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_title(f"N={int(n)}")
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels([str(int(x)) for x in iterations])
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_xlabel("iterations")

    for r in range(len(kernels)):
        for c in range(len(iterations)):
        #for c in [0]:
            val = mat[r][c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:+.2f}", ha="center", va="center", fontsize=7)

for ax in axes[len(ns):]:
    ax.axis("off")

fig.suptitle("Runtime signed error: log2(predicted / NCU)")
fig.colorbar(im, ax=axes[:len(ns)], shrink=0.85, label="log2 error")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "runtime_error_heatmap_log2.png"), dpi=200)
plt.close(fig)

# Small bar plot for the LaTeX table values.
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(summary["kernel"], summary["mape_runtime_pct"])
ax.set_ylabel("MAPE runtime (%)")
ax.set_title("Mean absolute runtime error by kernel")
ax.grid(True, axis="y", linewidth=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mape_by_kernel.png"), dpi=200)
plt.close(fig)

print("wrote", OUT_DIR)
print(summary.round(2).to_string(index=False))


def plot_one_error_heatmap(merged, target_n, out_path=None):
    kernels = ["compute", "vectoradd", "pointer", "shared", "sfu"]
    iterations = sorted(merged["iterations"].unique())

    sub = merged[merged["n"] == target_n].copy()

    vmax = max(1.0, np.nanmax(np.abs(sub["runtime_log2_error"])))

    fig, ax = plt.subplots(figsize=(5, 5))

    mat = []
    for k in kernels:
        row = []
        for it in iterations:
            x = sub[(sub["kernel"] == k) & (sub["iterations"] == it)]
            row.append(float(x["runtime_log2_error"].iloc[0]) if len(x) else np.nan)
        mat.append(row)

    im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="coolwarm")

    ax.set_title(f"Runtime error, N={int(target_n)}")
    ax.set_xticks(range(len(iterations)))
    ax.set_xticklabels([str(int(x)) for x in iterations])
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_xlabel("iterations")

    for r in range(len(kernels)):
        for c in range(len(iterations)):
            val = mat[r][c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:+.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2(predicted / native)")

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200)

    return fig, ax

plot_one_error_heatmap(merged, target_n=2097152, out_path=os.path.join(OUT_DIR, "runtime_error_heatmap_log2_N2097152.png"))