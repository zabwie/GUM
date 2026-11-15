#!/usr/bin/env bash
# E036: real-inference speculative decoding on the over-VRAM 27B model.
# Arm A: unified-memory target, no speculation. Arm B: same + native MTP draft head.
# KPI: effective PCIe bytes per accepted token; outputs must match at temp 0.
set -u
cd "$(dirname "$0")/.."
CLI="${CLI:-$HOME/llama-mtp-build/bin/llama-cli}"
TARGET=${GUM_TARGET_MODEL:-"$HOME/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"}
MTP=${GUM_MTP_MODEL:-"$HOME/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf"}
PROMPT='Explain what a graphics processing unit is, in one paragraph.'
TAG="$(date +%Y%m%d-%H%M%S)"
[[ -x "$CLI" ]] || { echo "missing $CLI (build first)"; exit 2; }

avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
swap0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
echo "preflight,ram_avail_MiB=$avail,swap_used_MiB=$swap0"
(( avail < 16384 )) && { echo "ABORT: need >=16 GiB available RAM for the managed target"; exit 4; }

for arm in baseline mtp; do
  out="results/e036-${arm}-${TAG}"
  extra=()
  [[ $arm == mtp ]] && extra=(-md "$MTP" -ngld 99 --spec-type draft-mtp --spec-draft-n-max 8)
  nvidia-smi dmon -s t -d 1 > "${out}-pcie.txt" 2>&1 &
  mon=$!
  GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 timeout 600 "$CLI" -m "$TARGET" -ngl 99 -c 1024 -n 128 \
    --temp 0 --reasoning off -p "$PROMPT" --single-turn --simple-io --no-display-prompt --verbosity 3 \
    "${extra[@]}" > "${out}.txt" 2>&1
  st=$?
  kill -INT $mon 2>/dev/null; wait $mon 2>/dev/null
  swap1=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)
  avail1=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  echo "== $arm exit=$st swap_used_MiB=$swap1 ram_avail_MiB=$avail1"
  grep -E "eval time|prompt eval|Generation:|draft|accept|spec\.|nextn" "${out}.txt" | tail -20
  (( avail1 < 4096 || swap1 > swap0 + 4096 )) && { echo "ABORT: resource floor breached"; exit 4; }
done
echo "outputs: results/e036-{baseline,mtp}-${TAG}.txt (compare generated text; must match at temp 0)"
