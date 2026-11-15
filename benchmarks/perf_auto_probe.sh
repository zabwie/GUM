#!/usr/bin/env bash
# Quick single-turn diagnostic probes (NOT part of the frozen bench).
# Usage: perf_auto_probe.sh <binary> <extra flags...>
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BIN="$1"; shift
TARGET=${GUM_TARGET_MODEL:-"$HOME/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"}
MTP=${GUM_MTP_MODEL:-"$HOME/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf"}
IDS_FILE="$root/results/perf-auto/probe.ids"
PAYLOAD=$(python3 -c "
import sys
ids = open('$IDS_FILE').read().strip()
print('64:' + ids)")
stdbuf -oL -eL taskset -c 0-3 "$BIN" \
  --model "$TARGET" \
  --mtp "$MTP" \
  --session 1 --session-tail 1 --session-stream 1 \
  --prefill-chunks 8 --speculate 32 --packed-verify 1 --pld 0 --draft-max 31 \
  --ctx 196608 --gen-tokens 131072 --hot-mib 3456 --cache-mib 10240 \
  --kv-cache turboquant-ttkv --show-text 0 --measure-h2d 1 \
  --prompt-ids - \
  "$@" <<< "$PAYLOAD" 2>&1 | grep -E "^config,|^hot_load|spec_summary" | tail -4
