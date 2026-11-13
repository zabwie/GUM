#!/usr/bin/env bash
# One-command stack self-check for GUM.
# Verifies: service up, which profile is live, one real decode turn with throughput.
# Usage: scripts/verify_stream.sh   (API on 127.0.0.1:8765, key LOCAL_GUM_API_KEY)
set -euo pipefail
health=$(curl -s --max-time 5 http://127.0.0.1:8765/health)
echo "health: $health"
profile=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('profile','unknown'))" "$health")
if [ "$profile" != "stream" ]; then
    echo "NOTE: profile is '$profile', not the default 'stream' — interpret the band below"
    echo "from HANDOFF.md's table for THAT profile."
fi
start=$(python3 -c "import time; print(time.monotonic())")
completion=$(curl -s --max-time 600 -X POST http://127.0.0.1:8765/v1/chat/completions \
    -H "Authorization: Bearer ${LOCAL_GUM_API_KEY:-local-gum-key}" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.8-27b-gum","max_tokens":200,"stream":false,"messages":[{"role":"user","content":"In plain prose, explain why a pendulum clock kept in a steadily accelerating train runs slow, covering the effective gravity direction and what happens to the period around a curve."}]}')
end=$(python3 -c "import time; print(time.monotonic())")
wall=$(python3 -c "print($end-$start)")
echo "$completion" | python3 -c "
import json, sys
d = json.load(sys.stdin)
u = d['usage']; wall = float('$wall')
tps = u['completion_tokens'] / wall
print(f\"completion: {u['completion_tokens']} tokens in {wall:.1f}s = {tps:.2f} tok/s end-to-end (finish={d['choices'][0]['finish_reason']})\")
bands = {'stream': (4.0, 7.0), 'nano': (5.0, 7.6), 'lean': (5.8, 8.6), 'chat': (8.5, 12.5), '196k': (1.0, 30.0)}
lo, hi = bands.get('$profile', (0, 999))
print('IN BAND for profile $profile' if lo <= tps <= hi else f'OUT OF BAND for $profile (expected {lo}-{hi}) — check config before trusting any diagnosis')
"
