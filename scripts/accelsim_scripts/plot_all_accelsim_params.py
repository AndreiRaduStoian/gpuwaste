from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt




def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def try_parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def find_config_files(config_dir: Path) -> List[Path]:
    files = (
        list(config_dir.glob("*.config"))
        + list(config_dir.glob("*.cfg"))
        + list(config_dir.glob("*.txt"))
    )

    return sorted(files)




def parse_raw_config_file(path: Path) -> List[Dict[str, str]]:
    rows = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        clean = strip_comment(line)

        if not clean.startswith("-"):
            continue

        parts = clean.split(None, 1)
        key = parts[0].lstrip("-")
        value = parts[1].strip() if len(parts) > 1 else ""

        rows.append(
            {
                "config_name": path.stem,
                "config_path": str(path),
                "line": line_number,
                "key": key,
                "value": value,
            }
        )

    return rows


def load_raw_configs(config_dir: Path) -> pd.DataFrame:
    files = find_config_files(config_dir)

    if not files:
        raise FileNotFoundError(
            f"No .config, .cfg, or .txt files found in {config_dir}"
        )

    rows = []
    for path in files:
        rows.extend(parse_raw_config_file(path))

    return pd.DataFrame(rows)




def split_numeric_list(value: str, separator: str) -> Optional[List[float]]:
    parts = value.split(separator)

    if len(parts) <= 1:
        return None

    parsed = []
    for part in parts:
        part = part.strip()

        if not part:
            return None

        number = try_parse_float(part)
        if number is None:
            return None

        parsed.append(number)

    return parsed


def extract_numeric_values(value: str) -> Dict[str, float]:

    value = value.strip()

    scalar = try_parse_float(value)
    if scalar is not None:
        return {"": scalar}

    comma_values = split_numeric_list(value, ",")
    if comma_values is not None:
        return {
            f"_{i}": number
            for i, number in enumerate(comma_values)
        }

    colon_values = split_numeric_list(value, ":")
    if colon_values is not None:
        return {
            f"_{i}": number
            for i, number in enumerate(colon_values)
        }

    return {}


def build_all_numeric_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for config_name, group in raw_df.groupby("config_name"):
        row: Dict[str, object] = {"config_name": config_name}

        for _, item in group.iterrows():
            key = str(item["key"]).strip()
            value = str(item["value"]).strip()

            extracted = extract_numeric_values(value)

            for suffix, number in extracted.items():
                column = f"{key}{suffix}"
                row[column] = number

        rows.append(row)

    return pd.DataFrame(rows)



def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {
        "gpgpu_n_clusters",
        "gpgpu_n_cores_per_cluster",
    }.issubset(df.columns):
        df["derived_num_sms"] = (
            df["gpgpu_n_clusters"] * df["gpgpu_n_cores_per_cluster"]
        )

    if {
        "gpgpu_n_mem",
        "gpgpu_n_sub_partition_per_mchannel",
    }.issubset(df.columns):
        df["derived_memory_subpartitions"] = (
            df["gpgpu_n_mem"] * df["gpgpu_n_sub_partition_per_mchannel"]
        )

    if {
        "gpgpu_num_sched_per_core",
        "gpgpu_max_insn_issue_per_warp",
    }.issubset(df.columns):
        df["derived_approx_issue_limit"] = (
            df["gpgpu_num_sched_per_core"]
            * df["gpgpu_max_insn_issue_per_warp"]
        )

    if {
        "gpgpu_shader_core_pipeline_0",
        "gpgpu_shader_core_pipeline_1",
    }.issubset(df.columns):
        df["derived_max_threads_per_sm"] = df["gpgpu_shader_core_pipeline_0"]
        df["derived_warp_size"] = df["gpgpu_shader_core_pipeline_1"]
        df["derived_max_warps_per_sm"] = (
            df["gpgpu_shader_core_pipeline_0"]
            / df["gpgpu_shader_core_pipeline_1"]
        )

    if {
        "gpgpu_clock_domains_0",
        "gpgpu_clock_domains_1",
        "gpgpu_clock_domains_2",
        "gpgpu_clock_domains_3",
    }.issubset(df.columns):
        df["derived_core_clock_mhz"] = df["gpgpu_clock_domains_0"]
        df["derived_interconnect_clock_mhz"] = df["gpgpu_clock_domains_1"]
        df["derived_l2_clock_mhz"] = df["gpgpu_clock_domains_2"]
        df["derived_dram_clock_mhz"] = df["gpgpu_clock_domains_3"]

    return df




def plot_single_numeric_column(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
    include_constant: bool,
):
    data = df[["config_name", column]].dropna()

    if data.empty:
        return

    if not include_constant and data[column].nunique() <= 1:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(data["config_name"].astype(str), data[column])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(column)
    plt.title(column.replace("_", " ").title())
    plt.tight_layout()

    output_path = output_dir / f"{safe_filename(column)}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_all_numeric_params(
    df: pd.DataFrame,
    output_dir: Path,
    include_constant: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        col for col in df.columns
        if col != "config_name" and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in numeric_cols:
        plot_single_numeric_column(
            df=df,
            column=col,
            output_dir=output_dir,
            include_constant=include_constant,
        )


def plot_grouped_bars(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    filename: str,
    output_dir: Path,
):
    available = [col for col in columns if col in df.columns]

    if not available:
        return

    data = df[["config_name"] + available].dropna(
        how="all",
        subset=available,
    )

    if data.empty:
        return

    ax = data.set_index("config_name")[available].plot(
        kind="bar",
        figsize=(11, 5),
    )

    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def plot_useful_groups(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_bars(
        df,
        [
            "derived_core_clock_mhz",
            "derived_interconnect_clock_mhz",
            "derived_l2_clock_mhz",
            "derived_dram_clock_mhz",
        ],
        "Clock Domains",
        "group_clock_domains.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        [
            "gpgpu_num_sp_units",
            "gpgpu_num_int_units",
            "gpgpu_num_sfu_units",
            "gpgpu_num_dp_units",
            "gpgpu_num_tensor_core_units",
        ],
        "Functional Units",
        "group_functional_units.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        [
            "gpgpu_l1_latency",
            "gpgpu_smem_latency",
            "gpgpu_l2_rop_latency",
            "dram_latency",
        ],
        "Memory Latencies",
        "group_memory_latencies.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        [
            "ptx_opcode_latency_fp_0",
            "ptx_opcode_latency_fp_1",
            "ptx_opcode_latency_fp_2",
            "ptx_opcode_latency_fp_3",
            "ptx_opcode_latency_fp_4",
        ],
        "FP Opcode Latencies: ADD, MAX, MUL, MAD, DIV",
        "group_fp_opcode_latencies.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        [
            "ptx_opcode_initiation_fp_0",
            "ptx_opcode_initiation_fp_1",
            "ptx_opcode_initiation_fp_2",
            "ptx_opcode_initiation_fp_3",
            "ptx_opcode_initiation_fp_4",
        ],
        "FP Opcode Initiation Intervals: ADD, MAX, MUL, MAD, DIV",
        "group_fp_opcode_initiation.png",
        output_dir,
    )

    plot_grouped_bars(
        df,
        [
            "derived_num_sms",
            "derived_max_warps_per_sm",
            "gpgpu_shader_cta",
            "derived_approx_issue_limit",
        ],
        "High-Level Model Parameters",
        "group_high_level_model_params.png",
        output_dir,
    )



def main():
    parser = argparse.ArgumentParser(
        description="Parse Accel-Sim configs and plot all numeric parameters.",
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing .config, .cfg, or .txt files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/accelsim_all_params"),
        help="Directory where CSVs and plots are written.",
    )

    parser.add_argument(
        "--include-constant",
        action="store_true",
        help="Also plot parameters that have the same value in every config.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_configs(args.config_dir)
    numeric_df = build_all_numeric_summary(raw_df)
    numeric_df = add_derived_fields(numeric_df)

    raw_csv = args.output_dir / "accelsim_raw_long.csv"
    numeric_csv = args.output_dir / "accelsim_numeric_wide.csv"

    raw_df.to_csv(raw_csv, index=False)
    numeric_df.to_csv(numeric_csv, index=False)

    all_plots_dir = args.output_dir / "all_numeric_plots"
    grouped_plots_dir = args.output_dir / "grouped_plots"

    plot_all_numeric_params(
        df=numeric_df,
        output_dir=all_plots_dir,
        include_constant=args.include_constant,
    )

    plot_useful_groups(
        df=numeric_df,
        output_dir=grouped_plots_dir,
    )

    print(f"Parsed {raw_df['config_name'].nunique()} config files.")
    print(f"Wrote raw long CSV: {raw_csv}")
    print(f"Wrote numeric wide CSV: {numeric_csv}")
    print(f"Wrote all numeric plots to: {all_plots_dir}")
    print(f"Wrote grouped plots to: {grouped_plots_dir}")


if __name__ == "__main__":
    main()