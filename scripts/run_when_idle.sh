#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIB="${MIB:-64}"
REPEATS="${REPEATS:-5}"
BENCHMARK="${BENCHMARK:-memory_paths}"
PROFILE="${PROFILE:-none}"
SAMPLES=5
MAX_TEMP_C=80

case "$BENCHMARK" in
  memory_paths|managed_paths|reuse_paths|q4_matvec) ;;
  *) echo "Refusing: BENCHMARK must be memory_paths, managed_paths, reuse_paths, or q4_matvec." >&2; exit 2 ;;
esac
case "$PROFILE" in
  none|um_faults) ;;
  *) echo "Refusing: PROFILE must be none or um_faults." >&2; exit 2 ;;
esac
if [[ "$PROFILE" == "um_faults" && "$BENCHMARK" != "managed_paths" ]]; then
  echo "Refusing: PROFILE=um_faults requires BENCHMARK=managed_paths." >&2
  exit 2
fi

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
# A CUDA context by itself is not treated as active load: report its memory and
# defer only when sampled GPU/process activity shows substantial contention.
COMPUTE_APPS="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, '
  NF >= 2 { gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); printf "%s:%sMiB ", $1, $2 }
' || true)"
PMON_OUTPUT="$(nvidia-smi pmon -c "$SAMPLES" -d 1)"
HEAVY_GPU_PIDS="$(printf '%s\n' "$PMON_OUTPUT" | awk '
  $1 !~ /^#/ && NF >= 9 && $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ && $4 >= 50 { hits[$2]++ }
  END { for (pid in hits) if (hits[pid] >= 3) printf "%s:%d ", pid, hits[pid] }
')"
VRAM_BUFFER_COUNT=1
VRAM_FIXED_MIB=0
if [[ "$BENCHMARK" == "reuse_paths" ]]; then
  VRAM_BUFFER_COUNT=3
  VRAM_FIXED_MIB=16
elif [[ "$BENCHMARK" == "q4_matvec" ]]; then
  VRAM_BUFFER_COUNT=2
  VRAM_FIXED_MIB=48
fi
MIN_GPU_FREE_MIB="$((2048 + MIB * VRAM_BUFFER_COUNT + VRAM_FIXED_MIB))"
PLANNED_DEVICE_MIB="$((MIB * VRAM_BUFFER_COUNT + VRAM_FIXED_MIB))"
PLANNED_HOST_MIB=0
if [[ "$BENCHMARK" == "reuse_paths" || "$BENCHMARK" == "q4_matvec" ]]; then
  PLANNED_HOST_MIB="$((MIB * 2))"
elif [[ "$BENCHMARK" == "managed_paths" ]]; then
  PLANNED_HOST_MIB="$((MIB * 3))"
else
  PLANNED_HOST_MIB="$MIB"
fi

printf 'Preflight: sampled GPU utilization=[%s]%% (max %s%%), planned device buffers=%s MiB, planned host allocations~=%s MiB, minimum free VRAM=%s MiB, available RAM=%s MiB, max temperature=%s C, max power=%s/%s W, CUDA compute clients=%s, sustained heavy GPU PIDs=%s\n' \
  "$GPU_UTIL_SAMPLES" "$GPU_MAX_UTIL" "$PLANNED_DEVICE_MIB" "$PLANNED_HOST_MIB" "$GPU_MIN_FREE_MIB" "$RAM_AVAILABLE_MIB" "$GPU_MAX_TEMP_C" \
  "$GPU_MAX_POWER_W" "$GPU_POWER_LIMIT_W" "${COMPUTE_APPS:-none}" "${HEAVY_GPU_PIDS:-none}"

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
if [[ -n "$HEAVY_GPU_PIDS" ]]; then
  echo "Deferred: a GPU process used at least 50% SM in three or more samples: $HEAVY_GPU_PIDS" >&2
  exit 3
fi

printf 'pmon_samples_begin\n%s\npmon_samples_end\n' "$PMON_OUTPUT"
printf 'Starting bounded benchmark at %s (benchmark=%s, buffer MiB=%s, measured repeats=%s).\n' "$(date -Is)" "$BENCHMARK" "$MIB" "$REPEATS"

if [[ "$PROFILE" == "um_faults" ]]; then
  REPORT="$ROOT/results/managed-memory-faults-${MIB}mib-$(date +%Y%m%d-%H%M%S)"
  printf 'Profiling Unified Memory page faults and migrations to %s.nsys-rep.\n' "$REPORT"
  exec nsys profile --trace=cuda --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true --output="$REPORT" \
    "$ROOT/benchmarks/$BENCHMARK" --mib "$MIB" --repeats "$REPEATS" "$@"
fi

exec "$ROOT/benchmarks/$BENCHMARK" --mib "$MIB" --repeats "$REPEATS" "$@"
