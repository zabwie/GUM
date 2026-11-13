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
health_hot=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('hot_mib') or 0)" "$health")
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
hot = $health_hot
hot_gib = hot / 1024
# E182: decode tok/s = 4.523 emitted / (0.6016 - 0.0429*hot_GiB) cycle seconds,
# measured 7.52/8.49/9.75/13.70 at hot 0/1600/3200/6400 (token-exact at all points).
# End-to-end adds prefill drag; the band below is deliberately wide (desktop load,
# adaptive respawn timing). GUM Adaptive may resize hot between health and now.
expected = 4.523 / (0.6016 - 0.0429 * hot_gib) if hot_gib < 14 else 14.0
lo, hi = expected * 0.65, expected * 1.35
print(f'hot_mib={hot} expected ~{expected:.1f} tok/s decode (e2e reads lower)')
print(f'IN BAND for $profile at hot={hot} MiB' if lo <= tps <= hi else f'OUT OF BAND for $profile at hot={hot} MiB (expected ~{expected:.1f} decode, measured {tps:.2f} e2e) — check config before trusting any diagnosis')
"
