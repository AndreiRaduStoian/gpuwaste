import csv
import json
import re
from pathlib import Path



def parse_lambdas_json_from_log(text):
    rows = []
    current_benchmark = ""
    current_mode = ""

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.rstrip()

        m = re.search(r"Message from master: (?P<bench>.+?)/(?P<mode>FAST|ACCURATE)$", line)
        if m:
            current_benchmark = m.group("bench")
            current_mode = m.group("mode")
            continue

        m = re.search(r"Sending benchmark (?P<bench>.+?) in (?P<mode>FAST|ACCURATE) mode", line)
        if m:
            current_benchmark = m.group("bench")
            current_mode = m.group("mode")
            continue

        if "LAMBDAS:" not in line:
            continue

        payload = line.split("LAMBDAS:", 1)[1].strip()

        try:
            obj = json.loads(payload)
            rows.append({
                "source": "log_json",
                "line": line_no,
                "benchmark": current_benchmark,
                "mode": current_mode,
                "bandwidth_gbs": obj.get("Bandwidth", ""),
                "issue_latency_c": obj.get("SmallLambda", ""),
                "completion_latency_c": obj.get("BigLambda", ""),
                "ridge_point_warps": obj.get("RidgePoint", ""),
                "memory_size_kb": obj.get("MemorySize", ""),
                "json_ok": True,
                "json_error": "",
                "raw_json": payload,
            })
        except Exception as e:
            rows.append({
                "source": "log_json",
                "line": line_no,
                "benchmark": current_benchmark,
                "mode": current_mode,
                "bandwidth_gbs": "",
                "issue_latency_c": "",
                "completion_latency_c": "",
                "ridge_point_warps": "",
                "memory_size_kb": "",
                "json_ok": False,
                "json_error": str(e),
                "raw_json": payload,
            })
        print("mode: ", rows[-1]["mode"], " benchmark: ", rows[-1]["benchmark"], "\t l: \t", rows[-1]["issue_latency_c"], " L: ", rows[-1]["completion_latency_c"])
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



if __name__ == "__main__":
    logfile = "../outputs/pipeline/bench_terminal_output.txt"
    log_text = Path(logfile).read_text(errors="replace")
    write_csv(Path("../results/pipeline_results/3070calibration.csv"), parse_lambdas_json_from_log(log_text))
