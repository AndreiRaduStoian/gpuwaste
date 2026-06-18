#!/usr/bin/env python3
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path(".")
manifest = pd.read_csv(root / "manifest.csv")

def parse_cycles(stats_path):
    text = stats_path.read_text(errors="ignore").splitlines()
    in_cycles = False
    rows = []

    for line in text:
        line = line.strip()

        if "gpu_tot_sim_cycle" in line and line.endswith(","):
            in_cycles = True
            continue

        if in_cycles:
            if line.startswith("----") or not line:
                continue
            if line.startswith("APPS,"):
                continue
            if line.endswith(",") and "gpu_tot_sim_cycle" not in line:
                break

            parts = line.split(",")
            if len(parts) >= 2:
                app_field = parts[0]
                value = parts[1]
                kernel = app_field.split("/")[0]
                kernel = kernel.split("--")[0]
                rows.append((kernel, int(float(value))))

    return rows

all_rows = []

for _, row in manifest.iterrows():
    run_name = row["run_name"]
    scale = float(row["scale"])
    stats_path = root / "stats" / f"{run_name}_stats.csv"

    for kernel, cycles in parse_cycles(stats_path):
        all_rows.append({
            "kernel": kernel,
            "scale": scale,
            "cycles": cycles,
        })

df = pd.DataFrame(all_rows)

baseline = (
    df[df["scale"] == 1.0]
    .set_index("kernel")["cycles"]
    .to_dict()
)

df["slowdown"] = df.apply(
    lambda r: r["cycles"] / baseline[r["kernel"]],
    axis=1
)

df = df.sort_values(["kernel", "scale"])

df.to_csv(root / "sm_sweep_summary.csv", index=False)
print(df)

plt.figure(figsize=(8, 5))

for kernel, group in df.groupby("kernel"):
    group = group.sort_values("scale")
    plt.plot(group["scale"], group["slowdown"], marker="o", label=kernel)

plt.xlabel("SM scale")
plt.ylabel("Cycle slowdown vs baseline")
plt.title("Accel-Sim SM count scaling")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(root / "sm_sweep_slowdown.png", dpi=200)

print("Saved:")
print(root / "sm_sweep_summary.csv")
print(root / "sm_sweep_slowdown.png")
