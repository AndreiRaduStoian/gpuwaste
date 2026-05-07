import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))


import csv
import sys

import matplotlib.pyplot as plt


def read_trace_csv(path):
    events = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            events.append({
                "group_id": int(row["group_id"]),
                "warp_id": int(row["warp_id"]),
                "instr_id": row["instr_id"],
                "op": row["op"],
                "subsystem": row["subsystem"],
                "issue_time": float(row["issue_time"]),
                "complete_time": float(row["complete_time"]),
                "raw": row["raw"],
            })

    return events


def plot_by_subsystem(events, output_path="trace_by_subsystem.png", max_events=None):
    if max_events is not None:
        events = events[:max_events]

    subsystems = sorted(set(event["subsystem"] for event in events))
    subsystem_to_y = {name: idx for idx, name in enumerate(subsystems)}

    fig, ax = plt.subplots(figsize=(14, 5))

    for event in events:
        y = subsystem_to_y[event["subsystem"]]
        start = event["issue_time"]
        duration = event["complete_time"] - event["issue_time"]

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=0.6,
            align="center",
        )

    ax.set_yticks(list(subsystem_to_y.values()))
    ax.set_yticklabels(list(subsystem_to_y.keys()))

    ax.set_xlabel("Cycles")
    ax.set_ylabel("Subsystem")
    ax.set_title("Pipeline trace by subsystem")

    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")


def plot_by_warp(events, output_path="trace_by_warp.png", max_events=None):
    if max_events is not None:
        events = events[:max_events]

    warp_ids = sorted(set(event["warp_id"] for event in events))
    warp_to_y = {warp_id: idx for idx, warp_id in enumerate(warp_ids)}

    fig, ax = plt.subplots(figsize=(14, 6))

    for event in events:
        y = warp_to_y[event["warp_id"]]
        start = event["issue_time"]
        duration = event["complete_time"] - event["issue_time"]

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=0.6,
            align="center",
        )

    ax.set_yticks(list(warp_to_y.values()))
    ax.set_yticklabels([f"warp {w}" for w in warp_ids])

    ax.set_xlabel("Cycles")
    ax.set_ylabel("Warp")
    ax.set_title("Pipeline trace by warp")

    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "hotspot_trace.csv"

    events = read_trace_csv(trace_path)

    print(f"Loaded {len(events)} trace events")

    # First 200 events is easier to inspect.
    plot_by_subsystem(events, "trace_by_subsystem.png", max_events=200)
    plot_by_warp(events, "trace_by_warp.png", max_events=200)
