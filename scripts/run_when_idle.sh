#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIB="${MIB:-64}"
REPEATS="${REPEATS:-5}"
MAX_UTIL="${MAX_UTIL:-15}"

if (( MIB < 1 || MIB > 256 )); then
  echo "Refusing: MIB must be between 1 and 256 (the project resource cap)." >&2
  exit 2
fi
if (( REPEATS < 1 || REPEATS > 10 )); then
  echo "Refusing: REPEATS must be between 1 and 10." >&2
  exit 2
fi

read -r GPU_UTIL GPU_FREE_MIB < <(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2; exit}')
RAM_AVAILABLE_MIB="$(awk '/MemAvailable:/ {print int($2 / 1024); exit}' /proc/meminfo)"
COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)"
MIN_GPU_FREE_MIB="$((2048 + MIB))"

printf 'Preflight: GPU utilization=%s%%, free VRAM=%s MiB, available RAM=%s MiB, CUDA compute PIDs=%s\n' \
  "$GPU_UTIL" "$GPU_FREE_MIB" "$RAM_AVAILABLE_MIB" "${COMPUTE_PIDS:-none}"

if (( GPU_UTIL > MAX_UTIL )); then
  echo "Deferred: GPU utilization exceeds ${MAX_UTIL}%." >&2
  exit 3
fi
if (( GPU_FREE_MIB < MIN_GPU_FREE_MIB )); then
  echo "Deferred: less than 2 GiB VRAM headroom remains after the benchmark buffer." >&2
  exit 3
fi
if (( RAM_AVAILABLE_MIB < 8192 )); then
  echo "Deferred: less than 8 GiB system RAM is currently available." >&2
  exit 3
fi
if [[ -n "$COMPUTE_PIDS" ]]; then
  echo "Deferred: a CUDA compute process is already active." >&2
  exit 3
fi

exec "$ROOT/benchmarks/memory_paths" --mib "$MIB" --repeats "$REPEATS"
