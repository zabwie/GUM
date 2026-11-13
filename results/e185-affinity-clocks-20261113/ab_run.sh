#!/usr/bin/env bash
cd /home/zabwie/gpu-unified-memory-research
RUN() {  # $1 label, $2 taskset-arg or empty
  local label=$1 ts=$2
  for i in 1 2; do
    echo "$PROMPT" | timeout 120 ${ts:+taskset -c $ts} ./benchmarks/real_decode --model /home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf \
      --mtp /home/zabwie/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf \
      --session 1 --session-tail 1 --session-stream 1 --prefill-chunks 8 \
      --speculate 32 --packed-verify 1 --pld 0 --draft-max 31 \
      --ctx 32768 --gen-tokens 40 --hot-mib 0 --cache-mib 15600 --kv-cache turboquant-ttkv --show-text 0 \
      --prompt-ids - 2>&1 | grep spec_summary \
      | python3 -c "
import sys,re
line=sys.stdin.read()
vals={k:(re.search(k+'=([0-9.]+)',line) or [None,'?'])[1] for k in ['mean_verify_s','mean_mtp_s','mean_cycle_s','tok_per_s']}
print('$label run$i', vals)"
  done
}
source /tmp/ab_prompt.sh
RUN "PIN0-3" "0-3"
echo "---"
RUN "PIN0-7" "0-7"
echo "---"
RUN "UNPIN" ""
