from __future__ import annotations

import csv
from pathlib import Path
import math

ROOT = Path("/home/andrei/final/gpuwaste/results/accelsim_results/")
OUT = ROOT / "combined_figures"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "key": "low_n",
        "title": "Low-N",
        "path": ROOT / "low_n_final" / "low_n_sweep_outputs_final" / "low_n_sweep_summary.csv",
    },
    {
        "key": "large3",
        "title": "N=1,048,576",
        "path": ROOT / "large3_final" / "large3_sweep_outputs_final" / "large3_sweep_summary.csv",
    },
    {
        "key": "large3_n2",
        "title": "N=2,097,152",
        "path": ROOT / "large3_n2_final" / "large3_n2_sweep_outputs_final" / "large3_n2_sweep_summary.csv",
    },
    {
        "key": "large3_n4",
        "title": "N=4,194,304",
        "path": ROOT / "large3_n4_final" / "large3_n4_sweep_outputs_final" / "large3_n4_sweep_summary.csv",
    },
]

PARAMS = ["sm", "core", "mem"]
PARAM_LABELS = {
    "sm": "SM",
    "core": "Core",
    "mem": "Memory",
}

KERNELS = ["compute", "vectoradd", "shared"]
SCALES = [0.5, 0.25]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        r["scale"] = float(r["scale"])
        r["time_slowdown"] = float(r["time_slowdown"])
        r["proportional_slowdown"] = r["scale"] * r["time_slowdown"]

    return rows


def find_row(rows, kernel, param, scale):
    for r in rows:
        if (
            r["kernel"] == kernel
            and r["param"] == param
            and abs(r["scale"] - scale) < 1e-9
        ):
            return r
    return None


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.2f}"


def main():
    out_csv = OUT / "proportional_slowdown_table.csv"
    out_tex = OUT / "proportional_slowdown_table.tex"

    table_rows = []

    for run in RUNS:
        rows = read_rows(run["path"])

        for kernel in KERNELS:
            for param in PARAMS:
                out = {
                    "run": run["title"],
                    "kernel": kernel,
                    "resource": PARAM_LABELS[param],
                }

                for scale in SCALES:
                    r = find_row(rows, kernel, param, scale)
                    suffix = str(scale).replace(".", "p")

                    if r is None:
                        out[f"S_{suffix}"] = float("nan")
                        out[f"P_{suffix}"] = float("nan")
                    else:
                        out[f"S_{suffix}"] = r["time_slowdown"]
                        out[f"P_{suffix}"] = r["proportional_slowdown"]

                table_rows.append(out)

    fieldnames = [
        "run",
        "kernel",
        "resource",
        "S_0p5",
        "P_0p5",
        "S_0p25",
        "P_0p25",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    with out_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lllrrrr}\n")
        f.write("\\toprule\n")
        f.write("Run & Kernel & Resource & $S(0.5)$ & $P(0.5)$ & $S(0.25)$ & $P(0.25)$ \\\\\n")
        f.write("\\midrule\n")

        current_run = None
        for r in table_rows:
            if current_run is not None and r["run"] != current_run:
                f.write("\\midrule\n")
            current_run = r["run"]

            f.write(
                f"{r['run']} & {r['kernel']} & {r['resource']} & "
                f"{fmt(r['S_0p5'])} & {fmt(r['P_0p5'])} & "
                f"{fmt(r['S_0p25'])} & {fmt(r['P_0p25'])} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print("Saved:", out_csv)
    print("Saved:", out_tex)


if __name__ == "__main__":
    main()