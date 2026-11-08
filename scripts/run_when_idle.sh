#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIB="${MIB:-64}"
REPEATS="${REPEATS:-5}"
SAMPLES=5
MAX_TEMP_C=80

if (( MIB < 1 || MIB > 256 )); then
  echo "Refusing: MIB must be between 1 and 256 (the project resource cap)." >&2
  exit 2
fi
if (( REPEATS < 1 || REPEATS > 10 )); then
  echo "Refusing: REPEATS must be between 1 and 10." >&2
  exit 2
fi

mapfile -t GPU_SNAPSHOTS < <(
  for ((sample = 0; sample < SAMPLES; ++sample)); do
    nvidia-smi --query-gpu=utilization.gpu,memory.free,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
    sleep 1
  done
)
if (( ${#GPU_SNAPSHOTS[@]} != SAMPLES )); then
  echo "Refusing: could not collect five complete GPU resource samples." >&2
  exit 2
fi
GPU_SUMMARY="$(printf '%s\n' "${GPU_SNAPSHOTS[@]}" | awk -F, '
  BEGIN { min_free = 1e9; max_temp = 0; max_power = 0; max_power_limit = 0; max_util = 0; high_util = 0 }
  {
    for (i = 1; i <= 5; ++i) gsub(/[[:space:]]/, "", $i)
    util = $1 + 0; free_mib = $2 + 0; temp = $3 + 0; power = $4 + 0; power_limit = $5 + 0
    utils = utils (NR > 1 ? "," : "") util
    if (free_mib < min_free) min_free = free_mib
    if (temp > max_temp) max_temp = temp
    if (power > max_power) max_power = power
    if (power_limit > max_power_limit) max_power_limit = power_limit
    if (util > max_util) max_util = util
    if (util >= 85) high_util++
  }
  END { printf "%s|%d|%d|%.0f|%.0f|%d|%d", utils, min_free, max_temp, max_power, max_power_limit, max_util, high_util }
')"
IFS='|' read -r GPU_UTIL_SAMPLES GPU_MIN_FREE_MIB GPU_MAX_TEMP_C GPU_MAX_POWER_W GPU_POWER_LIMIT_W GPU_MAX_UTIL GPU_HIGH_UTIL_SAMPLES <<< "$GPU_SUMMARY"
RAM_AVAILABLE_MIB="$(awk '/MemAvailable:/ {print int($2 / 1024); exit}' /proc/meminfo)"
COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)"
PMON_OUTPUT="$(nvidia-smi pmon -c "$SAMPLES" -d 1)"
HEAVY_GPU_PIDS="$(printf '%s\n' "$PMON_OUTPUT" | awk '
  $1 !~ /^#/ && NF >= 9 && $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ && $4 >= 50 { hits[$2]++ }
  END { for (pid in hits) if (hits[pid] >= 3) printf "%s:%d ", pid, hits[pid] }
')"
MIN_GPU_FREE_MIB="$((2048 + MIB))"

printf 'Preflight: sampled GPU utilization=[%s]%% (max %s%%), minimum free VRAM=%s MiB, available RAM=%s MiB, max temperature=%s C, max power=%s/%s W, CUDA compute PIDs=%s, sustained heavy GPU PIDs=%s\n' \
  "$GPU_UTIL_SAMPLES" "$GPU_MAX_UTIL" "$GPU_MIN_FREE_MIB" "$RAM_AVAILABLE_MIB" "$GPU_MAX_TEMP_C" \
  "$GPU_MAX_POWER_W" "$GPU_POWER_LIMIT_W" "${COMPUTE_PIDS:-none}" "${HEAVY_GPU_PIDS:-none}"

if (( GPU_MIN_FREE_MIB < MIN_GPU_FREE_MIB )); then
  echo "Deferred: less than 2 GiB VRAM headroom remains after the benchmark buffer." >&2
  exit 3
fi
if (( RAM_AVAILABLE_MIB < 8192 )); then
  echo "Deferred: less than 8 GiB system RAM is currently available." >&2
  exit 3
fi
if (( GPU_MAX_TEMP_C >= MAX_TEMP_C )); then
  echo "Deferred: GPU temperature reached ${MAX_TEMP_C} C during sampling." >&2
  exit 3
fi
if (( GPU_HIGH_UTIL_SAMPLES >= 4 )); then
  echo "Deferred: GPU utilization remained at or above 85% in at least four of five samples." >&2
  exit 3
fi
if [[ -n "$COMPUTE_PIDS" ]]; then
  echo "Deferred: a CUDA compute process is already active." >&2
  exit 3
fi
if [[ -n "$HEAVY_GPU_PIDS" ]]; then
  echo "Deferred: a GPU process used at least 50% SM in three or more samples: $HEAVY_GPU_PIDS" >&2
  exit 3
fi

printf 'pmon_samples_begin\n%s\npmon_samples_end\n' "$PMON_OUTPUT"
printf 'Starting bounded benchmark at %s (host pinned MiB=%s, VRAM MiB=%s, measured repeats=%s).\n' "$(date -Is)" "$MIB" "$MIB" "$REPEATS"

exec "$ROOT/benchmarks/memory_paths" --mib "$MIB" --repeats "$REPEATS"
