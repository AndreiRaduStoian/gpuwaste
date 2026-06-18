from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

RESULTS_ROOT = Path("/workspace/gpuwaste/results")

def latest_dir(pattern: str) -> Path:
    dirs = sorted(RESULTS_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No directory matches {pattern}")
    return dirs[0]

RESULT_DIRS = {
    "sm": latest_dir("accelsim_sm_n4_*"),
    "core": latest_dir("accelsim_core_n4_*"),
    "mem": latest_dir("accelsim_mem_n4_*"),
}

OUT_ROOT = RESULTS_ROOT / f"large3_n4_sweep_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
KERNELS = ["compute", "vectoradd", "shared"]

CORE_MHZ = {
    "sm": {1.0: 1132.0, 0.75: 1132.0, 0.5: 1132.0, 0.25: 1132.0},
    "core": {1.0: 1132.0, 0.75: 849.0, 0.5: 566.0, 0.25: 283.0},
    "mem": {1.0: 1132.0, 0.75: 1132.0, 0.5: 1132.0, 0.25: 1132.0},
}


def read_manifest(d: Path):
    with (d / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_gpu_tot_sim_cycle(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.startswith("gpu_tot_sim_cycle"):
            start = i
            break

    if start is None:
        raise RuntimeError(f"No gpu_tot_sim_cycle block found in {path}")

    out: dict[str, int] = {}

    for line in lines[start + 1:]:
        line = line.strip()

        if not line:
            continue
        if line.startswith("-"):
            break
        if line.startswith("APPS,"):
            continue

        parts = line.split(",")
        if len(parts) < 2:
            continue

        app = parts[0].lower()
        value = parts[1].strip()

        for kernel in KERNELS:
            if f"/{kernel}_" in app or f"/{kernel}/" in app or kernel in app:
                out[kernel] = int(value)
                break

    if set(out) != set(KERNELS):
        raise RuntimeError(f"Expected {KERNELS}, parsed {out} from {path}")

    return out


def main() -> None:
    rows = []

    print("Using n4 result directories:")
    for param, d in RESULT_DIRS.items():
        print(f"  {param}: {d}")

    for param, d in RESULT_DIRS.items():
        for entry in read_manifest(d):
            run_name = entry["run_name"]
            config = entry["config"]
            scale = float(entry["scale"])

            stats_csv = d / "stats" / f"{run_name}_stats.csv"
            cycles_by_kernel = extract_gpu_tot_sim_cycle(stats_csv)

            for kernel, cycles in cycles_by_kernel.items():
                core_mhz = CORE_MHZ[param][scale]
                rows.append({
                    "param": param,
                    "kernel": kernel,
                    "scale": scale,
                    "cycles": cycles,
                    "core_mhz": core_mhz,
                    "time_proxy": cycles / core_mhz,
                    "run_name": run_name,
                    "config": config,
                })

    baselines = {
        (r["param"], r["kernel"]): r
        for r in rows
        if r["scale"] == 1.0
    }

    for r in rows:
        b = baselines[(r["param"], r["kernel"])]
        r["cycle_slowdown"] = r["cycles"] / b["cycles"]
        r["time_slowdown"] = r["time_proxy"] / b["time_proxy"]
        r["delta_time"] = r["time_slowdown"] - 1.0

    rows.sort(key=lambda r: (r["param"], r["kernel"], r["scale"]))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    fields = [
        "param", "kernel", "scale", "cycles", "core_mhz", "time_proxy",
        "cycle_slowdown", "time_slowdown", "delta_time", "run_name", "config",
    ]

    def write_csv(path: Path, subset: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    write_csv(OUT_ROOT / "large3_n4_sweep_summary.csv", rows)

    for param in ["sm", "core", "mem"]:
        subset = [r for r in rows if r["param"] == param]
        write_csv(OUT_ROOT / f"large3_n4_{param}_sweep_summary.csv", subset)

        for metric, ylabel, suffix in [
            ("cycle_slowdown", "Cycle slowdown", "cycle"),
            ("time_slowdown", "Simulated-time slowdown", "time"),
        ]:
            plt.figure(figsize=(7, 4.5))
            for kernel in KERNELS:
                kr = sorted([x for x in subset if x["kernel"] == kernel], key=lambda x: x["scale"])
                plt.plot([x["scale"] for x in kr], [x[metric] for x in kr], marker="o", label=kernel)
            plt.xlabel("Resource scale")
            plt.ylabel(ylabel)
            plt.title(f"{param.upper()} scaling slowdown, N=4,194,304")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT_ROOT / f"large3_n4_{param}_{suffix}_slowdown.png", dpi=200)
            plt.close()

    write_csv(
        OUT_ROOT / "large3_n4_alpha_05_summary.csv",
        [r for r in rows if abs(r["scale"] - 0.5) < 1e-9],
    )

    print()
    print(f"Saved outputs in: {OUT_ROOT}")
    print(f"Summary CSV: {OUT_ROOT / 'large3_n4_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
