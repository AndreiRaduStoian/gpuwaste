#!/usr/bin/env bash
set -u

BIN="../benchmarks/cu/synthetic_kernels_nwu"
OUT="ncu_native_comparison.csv"
RAW_DIR="ncu_raw"
BLOCK=256

KERNELS=(compute vectoradd pointer shared sfu)
NS=(524288 1572864 2097152 4194304)
ITERS=(1 10 20)

mkdir -p "$RAW_DIR"

echo "kernel,n,iterations,block_size,sm_cycles_elapsed_avg,gpu_cycles_elapsed_avg,gpu_time_duration_us" > "$OUT"

for n in "${NS[@]}"; do
  for it in "${ITERS[@]}"; do
    for kernel in "${KERNELS[@]}"; do
      raw="$RAW_DIR/${kernel}_N${n}_I${it}_B${BLOCK}.txt"

      echo "Running $kernel N=$n I=$it B=$BLOCK" >&2

      sudo ncu \
        --target-processes all \
        --metrics sm__cycles_elapsed.avg,gpu__cycles_elapsed.avg,gpu__time_duration.sum \
        "$BIN" "$kernel" "$n" "$it" "$BLOCK" \
        > "$raw" 2>&1

      sm_cycles=$(awk '$1=="sm__cycles_elapsed.avg" {v=$3; gsub(",", "", v); print v; exit}' "$raw")
      gpu_cycles=$(awk '$1=="gpu__cycles_elapsed.avg" {v=$3; gsub(",", "", v); print v; exit}' "$raw")
      gpu_time_us=$(awk '$1=="gpu__time_duration.sum" {v=$3; gsub(",", "", v); print v; exit}' "$raw")

      echo "$kernel,$n,$it,$BLOCK,$sm_cycles,$gpu_cycles,$gpu_time_us" >> "$OUT"
    done
  done
done

echo "Wrote $OUT"
echo "Raw NCU logs in $RAW_DIR/"

