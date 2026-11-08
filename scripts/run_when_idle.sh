#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIB="${MIB:-64}"
REPEATS="${REPEATS:-5}"
CHUNK_MIB="${CHUNK_MIB:-16}"
BENCHMARK="${BENCHMARK:-memory_paths}"
PROFILE="${PROFILE:-none}"
DEVICE_MIB="${DEVICE_MIB:-1024}"
HOST_MIB="${HOST_MIB:-2048}"
SAMPLES=5
MAX_TEMP_C=80
MAX_MIB=256

if [[ "$BENCHMARK" == "host_latency" ]]; then
  MAX_MIB=960
fi
if [[ "$BENCHMARK" == "q4_matvec" ]]; then
  MAX_MIB=960
fi
if [[ "$BENCHMARK" == "residency_manager" ]]; then
  MAX_MIB=512
fi

case "$BENCHMARK" in
  memory_paths|managed_paths|reuse_paths|q4_matvec|host_latency|mixed_residency|vmm_mixed_probe|residency_manager) ;;
  *) echo "Refusing: unknown BENCHMARK=$BENCHMARK." >&2; exit 2 ;;
esac
case "$PROFILE" in
  none|um_faults) ;;
  *) echo "Refusing: PROFILE must be none or um_faults." >&2; exit 2 ;;
esac
if [[ "$PROFILE" == "um_faults" && "$BENCHMARK" != "managed_paths" && "$BENCHMARK" != "host_latency" ]]; then
  echo "Refusing: PROFILE=um_faults requires BENCHMARK=managed_paths or host_latency." >&2
  exit 2
fi

if (( MIB < 1 || MIB > MAX_MIB )); then
  echo "Refusing: MIB must be between 1 and ${MAX_MIB} for this benchmark." >&2
  exit 2
fi
if (( REPEATS < 1 || REPEATS > 10 )); then
  echo "Refusing: REPEATS must be between 1 and 10." >&2
  exit 2
fi
if (( CHUNK_MIB < 4 || CHUNK_MIB > 64 )); then
  echo "Refusing: CHUNK_MIB must be between 4 and 64." >&2
  exit 2
fi
if [[ "$BENCHMARK" == "mixed_residency" || "$BENCHMARK" == "vmm_mixed_probe" ]]; then
  if [[ ! "$DEVICE_MIB" =~ ^[0-9]+$ || ! "$HOST_MIB" =~ ^[0-9]+$ ]] ||
     (( DEVICE_MIB < 1 || DEVICE_MIB > 2031 || HOST_MIB < 1 || HOST_MIB > 4096 )); then
    echo "Refusing: DEVICE_MIB must be 1..2031 and HOST_MIB must be 1..4096." >&2
    exit 2
  fi
  LOCKED_MEM_KIB="$(ulimit -l)"
  if [[ "$LOCKED_MEM_KIB" =~ ^[0-9]+$ ]] && (( LOCKED_MEM_KIB < HOST_MIB * 1024 )); then
    echo "Refusing: HOST_MIB exceeds this process's locked-memory limit." >&2
    exit 2
  fi
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
  # Add 8 MiB for input/output scratch beyond the eviction and two chunk buffers.
  VRAM_FIXED_MIB="$((24 + 2 * CHUNK_MIB))"
elif [[ "$BENCHMARK" == "host_latency" ]]; then
  VRAM_BUFFER_COUNT=2
  VRAM_FIXED_MIB=17
fi
if [[ "$BENCHMARK" == "mixed_residency" || "$BENCHMARK" == "vmm_mixed_probe" ]]; then
  PLANNED_DEVICE_MIB="$((DEVICE_MIB + 17))"
elif [[ "$BENCHMARK" == "residency_manager" ]]; then
  # Up to 128 MiB cache + 128 MiB pressure + 16 MiB requests + 32 MiB scratch.
  PLANNED_DEVICE_MIB=304
else
  PLANNED_DEVICE_MIB="$((MIB * VRAM_BUFFER_COUNT + VRAM_FIXED_MIB))"
fi
if (( PLANNED_DEVICE_MIB > 2048 )); then
  echo "Refusing: planned device buffers exceed the 2048 MiB project cap." >&2
  exit 2
fi
MIN_GPU_FREE_MIB="$((2048 + PLANNED_DEVICE_MIB))"
PLANNED_HOST_MIB=0
if [[ "$BENCHMARK" == "host_latency" ]]; then
  PLANNED_HOST_MIB="$((MIB * 3))"
elif [[ "$BENCHMARK" == "mixed_residency" || "$BENCHMARK" == "vmm_mixed_probe" ]]; then
  PLANNED_HOST_MIB="$HOST_MIB"
elif [[ "$BENCHMARK" == "residency_manager" ]]; then
  PLANNED_HOST_MIB="$((MIB + 32))"
elif [[ "$BENCHMARK" == "reuse_paths" || "$BENCHMARK" == "q4_matvec" ]]; then
  PLANNED_HOST_MIB="$((MIB * 2))"
elif [[ "$BENCHMARK" == "managed_paths" ]]; then
  PLANNED_HOST_MIB="$((MIB * 3))"
else
  PLANNED_HOST_MIB="$MIB"
fi
SWAP_USED_MIB="$(awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {print int((total-free)/1024)}' /proc/meminfo)"

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
if [[ "$BENCHMARK" == "mixed_residency" || "$BENCHMARK" == "vmm_mixed_probe" ]] && (( RAM_AVAILABLE_MIB < PLANNED_HOST_MIB + 8192 )); then
  echo "Deferred: less than the pinned host allocation plus 8 GiB RAM headroom is available." >&2
  exit 3
fi
if [[ "$BENCHMARK" == "mixed_residency" || "$BENCHMARK" == "vmm_mixed_probe" ]] && (( SWAP_USED_MIB > 0 )); then
  echo "Deferred: system swap is already in use; pinned-host runs require zero swap use." >&2
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
if [[ "$BENCHMARK" == "vmm_mixed_probe" ]]; then
  printf 'Starting bounded VMM probe at %s (device MiB=%s, host MiB=%s, swap used MiB=%s).\n' \
    "$(date -Is)" "$DEVICE_MIB" "$HOST_MIB" "$SWAP_USED_MIB"
elif [[ "$BENCHMARK" == "mixed_residency" ]]; then
  printf 'Starting bounded benchmark at %s (device MiB=%s, pinned host MiB=%s, repeats=%s, swap used MiB=%s).\n' \
    "$(date -Is)" "$DEVICE_MIB" "$HOST_MIB" "$REPEATS" "$SWAP_USED_MIB"
else
  printf 'Starting bounded benchmark at %s (benchmark=%s, buffer MiB=%s, measured repeats=%s).\n' "$(date -Is)" "$BENCHMARK" "$MIB" "$REPEATS"
fi

if [[ "$PROFILE" == "um_faults" ]]; then
  REPORT="$ROOT/results/unified-memory-faults-${BENCHMARK}-${MIB}mib-$(date +%Y%m%d-%H%M%S)"
  printf 'Profiling Unified Memory page faults and migrations to %s.nsys-rep.\n' "$REPORT"
  exec nsys profile --trace=cuda --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true --output="$REPORT" \
    "$ROOT/benchmarks/$BENCHMARK" --mib "$MIB" --repeats "$REPEATS" "$@"
fi

if [[ "$BENCHMARK" == "q4_matvec" ]]; then
  exec "$ROOT/benchmarks/$BENCHMARK" --mib "$MIB" --repeats "$REPEATS" \
    --chunk-mib "$CHUNK_MIB" "$@"
fi
if [[ "$BENCHMARK" == "mixed_residency" ]]; then
  exec "$ROOT/benchmarks/$BENCHMARK" --device-mib "$DEVICE_MIB" \
    --host-mib "$HOST_MIB" --repeats "$REPEATS"
fi
if [[ "$BENCHMARK" == "vmm_mixed_probe" ]]; then
  exec "$ROOT/benchmarks/$BENCHMARK" --device-mib "$DEVICE_MIB" \
    --host-mib "$HOST_MIB"
fi
if [[ "$BENCHMARK" == "residency_manager" ]]; then
  exec "$ROOT/benchmarks/$BENCHMARK" --host-mib "$MIB" --repeats "$REPEATS" "$@"
fi

exec "$ROOT/benchmarks/$BENCHMARK" --mib "$MIB" --repeats "$REPEATS" "$@"
