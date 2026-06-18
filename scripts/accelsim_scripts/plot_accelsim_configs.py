from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import matplotlib.pyplot as plt


PARAMS = {
    "gpgpu_n_clusters": "num_clusters",
    "gpgpu_n_cores_per_cluster": "cores_per_cluster",
    "gpgpu_n_mem": "memory_channels",
    "gpgpu_n_sub_partition_per_mchannel": "subpartitions_per_channel",

    "gpgpu_shader_registers": "registers_per_sm",
    "gpgpu_registers_per_block": "registers_per_block_limit",
    "gpgpu_shader_cta": "max_ctas_per_sm",
    "gpgpu_shmem_size": "shared_mem_per_sm",
    "gpgpu_shmem_per_block": "shared_mem_per_block_limit",
    "gpgpu_unified_l1d_size": "unified_l1d_kb",

    "gpgpu_num_sched_per_core": "schedulers_per_sm",
    "gpgpu_max_insn_issue_per_warp": "max_issue_per_warp",
    "gpgpu_dual_issue_diff_exec_units": "dual_issue_diff_units",

    "gpgpu_num_sp_units": "sp_units",
    "gpgpu_num_sfu_units": "sfu_units",
    "gpgpu_num_dp_units": "dp_units",
    "gpgpu_num_int_units": "int_units",
    "gpgpu_num_tensor_core_units": "tensor_units",

    "gpgpu_l1_latency": "l1_latency",
    "gpgpu_smem_latency": "shared_mem_latency",
    "gpgpu_l2_rop_latency": "l2_latency",
    "dram_latency": "dram_latency",

    "gpgpu_inst_fetch_throughput": "inst_fetch_throughput",
}


LIST_PARAMS = {
    "ptx_opcode_latency_int": "latency_int",
    "ptx_opcode_initiation_int": "initiation_int",
    "ptx_opcode_latency_fp": "latency_fp",
    "ptx_opcode_initiation_fp": "initiation_fp",
    "ptx_opcode_latency_dp": "latency_dp",
    "ptx_opcode_initiation_dp": "initiation_dp",
    "ptx_opcode_latency_sfu": "latency_sfu",
    "ptx_opcode_initiation_sfu": "initiation_sfu",
    "ptx_opcode_latency_tensor": "latency_tensor",
    "ptx_opcode_initiation_tensor": "initiation_tensor",
}


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_scalar(value: str):
    value = value.strip()

    if not value:
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_number_list(value: str) -> List[float]:
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.append(float(part))
        except ValueError:
            pass
    return result


def parse_config_file(path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "config_name": path.stem,
        "config_path": str(path),
    }

    raw: Dict[str, str] = {}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_comment(line)

        if not line.startswith("-"):
            continue

        parts = line.split(None, 1)
        key = parts[0].lstrip("-")
        value = parts[1].strip() if len(parts) > 1 else ""

        raw[key] = value

    # Simple scalar parameters
    for config_key, output_key in PARAMS.items():
        if config_key in raw:
            row[output_key] = parse_scalar(raw[config_key])

    # Clock domains
    if "gpgpu_clock_domains" in raw:
        parts = raw["gpgpu_clock_domains"].split(":")
        if len(parts) == 4:
            row["core_clock_mhz"] = float(parts[0])
            row["interconnect_clock_mhz"] = float(parts[1])
            row["l2_clock_mhz"] = float(parts[2])
            row["dram_clock_mhz"] = float(parts[3])

    # Shader core pipeline: usually max_threads_per_sm:warp_size
    if "gpgpu_shader_core_pipeline" in raw:
        parts = raw["gpgpu_shader_core_pipeline"].split(":")
        if len(parts) >= 2:
            row["max_threads_per_sm"] = int(parts[0])
            row["warp_size"] = int(parts[1])
            row["max_warps_per_sm"] = int(parts[0]) // int(parts[1])

    # Pipeline widths
    if "gpgpu_pipeline_widths" in raw:
        widths = parse_number_list(raw["gpgpu_pipeline_widths"])
        names = [
            "width_id_oc_sp",
            "width_id_oc_dp",
            "width_id_oc_int",
            "width_id_oc_sfu",
            "width_id_oc_mem",
            "width_oc_ex_sp",
            "width_oc_ex_dp",
            "width_oc_ex_int",
            "width_oc_ex_sfu",
            "width_oc_ex_mem",
            "width_ex_wb",
            "width_id_oc_tensor",
            "width_oc_ex_tensor",
        ]

        for name, value in zip(names, widths):
            row[name] = value

    # Instruction latency/initiation lists
    opcode_names = ["add", "max", "mul", "mad", "div"]

    for config_key, output_prefix in LIST_PARAMS.items():
        if config_key not in raw:
            continue

        values = parse_number_list(raw[config_key])

        if len(values) == 1:
            row[output_prefix] = values[0]
        else:
            for op_name, value in zip(opcode_names, values):
                row[f"{output_prefix}_{op_name}"] = value

    # Derived parameters
    clusters = row.get("num_clusters")
    cores_per_cluster = row.get("cores_per_cluster")
    if isinstance(clusters, int) and isinstance(cores_per_cluster, int):
        row["num_sms"] = clusters * cores_per_cluster

    mem_channels = row.get("memory_channels")
    subparts = row.get("subpartitions_per_channel")
    if isinstance(mem_channels, int) and isinstance(subparts, int):
        row["memory_subpartitions"] = mem_channels * subparts

    schedulers = row.get("schedulers_per_sm")
    max_issue_per_warp = row.get("max_issue_per_warp")
    if isinstance(schedulers, int) and isinstance(max_issue_per_warp, int):
        row["approx_issue_limit"] = schedulers * max_issue_per_warp

    return row


def load_configs(config_dir: Path) -> pd.DataFrame:
    files = sorted(
        list(config_dir.glob("*.config"))
        + list(config_dir.glob("*.cfg"))
        + list(config_dir.glob("*.txt"))
    )

    if not files:
        raise FileNotFoundError(f"No .config, .cfg, or .txt files found in {config_dir}")

    rows = [parse_config_file(path) for path in files]
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, output_dir: Path):
    if y_col not in df.columns:
        return

    data = df[[x_col, y_col]].dropna()
    if data.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(data[x_col].astype(str), data[y_col])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_col)
    plt.title(y_col.replace("_", " ").title())
    plt.tight_layout()

    output_path = output_dir / f"{y_col}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_grouped_bars(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str, filename: str, output_dir: Path):
    cols = [col for col in y_cols if col in df.columns]
    if not cols:
        return

    data = df[[x_col] + cols].dropna(how="all", subset=cols)
    if data.empty:
        return

    ax = data.set_index(x_col)[cols].plot(kind="bar", figsize=(11, 5))
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def make_plots(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    x_col = "config_name"

    individual_plots = [
        "num_sms",
        "core_clock_mhz",
        "dram_clock_mhz",
        "max_warps_per_sm",
        "registers_per_sm",
        "shared_mem_per_sm",
        "unified_l1d_kb",
        "memory_channels",
        "memory_subpartitions",
        "schedulers_per_sm",
        "approx_issue_limit",
        "l1_latency",
        "shared_mem_latency",
        "l2_latency",
        "dram_latency",
    ]

    for col in individual_plots:
        plot_bar(df, x_col, col, output_dir)

    plot_grouped_bars(
        df,
        x_col,
        ["sp_units", "int_units", "sfu_units", "dp_units", "tensor_units"],
        "Functional Units per SM",
        "functional_units_per_sm.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        x_col,
        [
            "latency_fp_add",
            "latency_fp_mul",
            "latency_fp_mad",
            "latency_fp_div",
            "latency_int_add",
            "latency_sfu",
        ],
        "Instruction Latencies",
        "instruction_latencies.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        x_col,
        [
            "initiation_fp_add",
            "initiation_fp_mul",
            "initiation_fp_mad",
            "initiation_fp_div",
            "initiation_int_add",
            "initiation_sfu",
        ],
        "Instruction Initiation Intervals",
        "instruction_initiation_intervals.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        x_col,
        ["l1_latency", "shared_mem_latency", "l2_latency", "dram_latency"],
        "Memory Latencies",
        "memory_latencies.png",
        output_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Folder containing Accel-Sim config files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/accelsim_config_plots"),
        help="Folder for CSV and plots.",
    )

    args = parser.parse_args()

    df = load_configs(args.config_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "accelsim_config_summary.csv"
    df.to_csv(csv_path, index=False)

    make_plots(df, args.output_dir)

    print(f"Parsed {len(df)} config files.")
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()