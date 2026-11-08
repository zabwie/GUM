#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-/home/zabwie/llama.cpp/build/bin/llama-cli}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/zabwie/llama.cpp/build/bin/llama-bench}"
MODEL="${MODEL:-$ROOT/models/Qwen3-0.6B-Q4_0.gguf}"
RUNS="${RUNS:-5}"
EXPECTED_SHA256="da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4"
PROMPT='What is 2 + 2? Answer with exactly one digit.'
RUN_TAG="$(date +%Y%m%d-%H%M%S)"

if [[ ! -x "$LLAMA_CLI" || ! -x "$LLAMA_BENCH" || ! -r "$MODEL" ]]; then
  echo "Refusing: set LLAMA_CLI and LLAMA_BENCH to existing llama.cpp binaries and MODEL to the local GGUF." >&2
  exit 2
fi
if [[ ! "$RUNS" =~ ^[1-5]$ ]]; then
  echo "Refusing: RUNS must be 1..5." >&2
  exit 2
fi
ACTUAL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "Refusing: the model hash does not match the pinned research artifact." >&2
  exit 2
fi

mapfile -t GPU_SNAPSHOTS < <(
  for ((sample = 0; sample < 5; ++sample)); do
    nvidia-smi --query-gpu=utilization.gpu,memory.free,temperature.gpu --format=csv,noheader,nounits
    sleep 1
  done
)
if (( ${#GPU_SNAPSHOTS[@]} != 5 )); then
  echo "Refusing: could not collect five complete GPU resource samples." >&2
  exit 2
fi
GPU_SUMMARY="$(printf '%s\n' "${GPU_SNAPSHOTS[@]}" | awk -F, '
  BEGIN { min_free = 1e9; max_temp = 0; max_util = 0; high_util = 0 }
  {
    for (i = 1; i <= 3; ++i) gsub(/[[:space:]]/, "", $i)
    util = $1 + 0; free_mib = $2 + 0; temp = $3 + 0
    utils = utils (NR > 1 ? "," : "") util
    if (free_mib < min_free) min_free = free_mib
    if (temp > max_temp) max_temp = temp
    if (util > max_util) max_util = util
    if (util >= 85) high_util++
  }
  END { printf "%s|%d|%d|%d|%d", utils, min_free, max_temp, max_util, high_util }
')"
IFS='|' read -r GPU_UTIL_SAMPLES GPU_MIN_FREE_MIB GPU_MAX_TEMP_C GPU_MAX_UTIL GPU_HIGH_UTIL_SAMPLES <<< "$GPU_SUMMARY"
RAM_AVAILABLE_MIB="$(awk '/MemAvailable:/ {print int($2 / 1024); exit}' /proc/meminfo)"
SWAP_USED_MIB="$(awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {print int((total-free)/1024)}' /proc/meminfo)"
PMON_OUTPUT="$(nvidia-smi pmon -c 5 -d 1)"
HEAVY_GPU_PIDS="$(printf '%s\n' "$PMON_OUTPUT" | awk '
  $1 !~ /^#/ && NF >= 9 && $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ && $4 >= 50 { hits[$2]++ }
  END { for (pid in hits) if (hits[pid] >= 3) printf "%s:%d ", pid, hits[pid] }
')"

printf 'Preflight: GPU utilization=[%s]%% (max %s%%), minimum free VRAM=%s MiB, available RAM=%s MiB, maximum temperature=%s C, swap used=%s MiB, sustained heavy GPU PIDs=%s\n' \
  "$GPU_UTIL_SAMPLES" "$GPU_MAX_UTIL" \
  "$GPU_MIN_FREE_MIB" "$RAM_AVAILABLE_MIB" "$GPU_MAX_TEMP_C" "$SWAP_USED_MIB" "${HEAVY_GPU_PIDS:-none}"
printf 'pmon_samples_begin\n%s\npmon_samples_end\n' "$PMON_OUTPUT"
if (( GPU_MIN_FREE_MIB < 4096 || RAM_AVAILABLE_MIB < 9216 || GPU_MAX_TEMP_C >= 80 || GPU_HIGH_UTIL_SAMPLES >= 4 || SWAP_USED_MIB > 0 )) || [[ -n "$HEAVY_GPU_PIDS" ]]; then
  echo "Deferred: inference needs 2 GiB VRAM headroom, 8 GiB RAM headroom, zero swap use, and a quiet GPU." >&2
  exit 3
fi

PCIE_LOG="$ROOT/results/llama-small-pcie-$(date +%Y%m%d-%H%M%S).txt"
GPU_LOG="$ROOT/results/llama-small-gpu-memory-$(date +%Y%m%d-%H%M%S).txt"
nvidia-smi dmon -s t -d 1 > "$PCIE_LOG" 2>&1 &
PCIE_MONITOR_PID=$!
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --loop-ms=500 > "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!
cleanup() {
  kill -INT "$PCIE_MONITOR_PID" "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$PCIE_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

printf 'experiment,llama_small_qwen3_0.6b_q4_0,date=%s,runs_per_arm=%s,model_sha256=%s,model_bytes=%s,llama_cli=%s\n' \
  "$(date -Is)" "$RUNS" "$ACTUAL_SHA256" "$(stat -c '%s' "$MODEL")" "$LLAMA_CLI"
printf 'settings,device=0,gpu_layers=all,context=512,batch=128,ubatch=128,bench_prompt_tokens=47,bench_generation_tokens=128,bench_repetitions=%s,temperature=0,top_k=1,seed=42,reasoning=off\n' "$RUNS"
printf 'correctness_prompt,%s\n' "$PROMPT"
printf 'pcie_log,%s\n' "$PCIE_LOG"
printf 'gpu_memory_log,%s\n' "$GPU_LOG"
"$LLAMA_CLI" --version

for mode in conventional_gpu unified_memory_gpu; do
  printf 'correctness_begin,mode=%s,time=%s\n' "$mode" "$(date -Is)"
  CLI_STDOUT="$ROOT/results/llama-small-${mode}-correctness-${RUN_TAG}-stdout.txt"
  CLI_STDERR="$ROOT/results/llama-small-${mode}-correctness-${RUN_TAG}-stderr.txt"
  if [[ "$mode" == "unified_memory_gpu" ]]; then
    if GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "$LLAMA_CLI" \
      --model "$MODEL" --gpu-layers all --ctx-size 512 --batch-size 128 --ubatch-size 128 \
      --predict 8 --temp 0 --top-k 1 --seed 42 --reasoning off --single-turn --simple-io \
      --verbosity 3 --log-colors off --no-display-prompt --prompt "$PROMPT" \
      >"$CLI_STDOUT" 2>"$CLI_STDERR"; then STATUS=0; else STATUS=$?; fi
  else
    if env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY "$LLAMA_CLI" \
      --model "$MODEL" --gpu-layers all --ctx-size 512 --batch-size 128 --ubatch-size 128 \
      --predict 8 --temp 0 --top-k 1 --seed 42 --reasoning off --single-turn --simple-io \
      --verbosity 3 --log-colors off --no-display-prompt --prompt "$PROMPT" \
      >"$CLI_STDOUT" 2>"$CLI_STDERR"; then STATUS=0; else STATUS=$?; fi
  fi
  printf 'correctness_artifacts,mode=%s,stdout=%s,stderr=%s,exit=%d\n' \
    "$mode" "$CLI_STDOUT" "$CLI_STDERR" "$STATUS"
  rg 'offloaded [0-9]+/[0-9]+ layers|CPU_Mapped model buffer|CUDA0 model buffer|eval time' "$CLI_STDERR" || true
  cat "$CLI_STDOUT"
  if (( STATUS != 0 )) || ! grep -Eq '^[[:space:]]*(2[[:space:]]*[+][[:space:]]*2[[:space:]]*=[[:space:]]*)?4[[:space:]]*$' "$CLI_STDOUT"; then
    echo "Inference correctness check failed for $mode; expected the answer 4, optionally as a full equation." >&2
    exit 1
  fi

  BENCH_STDOUT="$ROOT/results/llama-small-${mode}-bench-${RUN_TAG}.csv"
  BENCH_STDERR="$ROOT/results/llama-small-${mode}-bench-${RUN_TAG}-stderr.txt"
  printf 'throughput_begin,mode=%s,repetitions=%s,time=%s\n' "$mode" "$RUNS" "$(date -Is)"
  if [[ "$mode" == "unified_memory_gpu" ]]; then
    if GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "$LLAMA_BENCH" \
      --model "$MODEL" --n-gpu-layers 99 --n-prompt 47 --n-gen 128 \
      --repetitions "$RUNS" --batch-size 128 --ubatch-size 128 --output csv \
      >"$BENCH_STDOUT" 2>"$BENCH_STDERR"; then STATUS=0; else STATUS=$?; fi
  else
    if env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY "$LLAMA_BENCH" \
      --model "$MODEL" --n-gpu-layers 99 --n-prompt 47 --n-gen 128 \
      --repetitions "$RUNS" --batch-size 128 --ubatch-size 128 --output csv \
      >"$BENCH_STDOUT" 2>"$BENCH_STDERR"; then STATUS=0; else STATUS=$?; fi
  fi
  printf 'throughput_artifacts,mode=%s,csv=%s,stderr=%s,exit=%d\n' \
    "$mode" "$BENCH_STDOUT" "$BENCH_STDERR" "$STATUS"
  cat "$BENCH_STDOUT"
  if (( STATUS != 0 )); then exit "$STATUS"; fi
done

printf 'postrun,available_ram_MiB=%s,swap_used_MiB=%s\n' \
  "$(awk '/MemAvailable:/ {print int($2 / 1024); exit}' /proc/meminfo)" \
  "$(awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {print int((total-free)/1024)}' /proc/meminfo)"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
