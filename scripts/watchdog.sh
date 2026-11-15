#!/usr/bin/env bash
# GUM watchdog: detects silent failures in all background jobs and reports on CHANGE
# only (no spam). State file: logs/watchdog.state ; history: logs/watchdog.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STATE=logs/watchdog.state
LOG=logs/watchdog.log
touch "$STATE" "$LOG"
now=$(date +%s)
alert() { echo "$(date "+%F %T") ALERT $1" >> "$LOG"; echo "ALERT: $1"; }
okmsg() { echo "$(date "+%F %T") OK $1" >> "$LOG"; echo "OK: $1"; }
prev() { grep -F "$1" "$STATE" 2>/dev/null | head -1 | cut -d= -f2-; }
setst() { local kv="$1"; local key="${kv%%=*}"; grep -vF "${key}=" "$STATE" 2>/dev/null > "$STATE.tmp"; echo "$kv" >> "$STATE.tmp"; mv "$STATE.tmp" "$STATE"; }

# 1. disk
free_g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
[ "$free_g" -lt 10 ] && alert "disk: only ${free_g}G free" || setst disk=$free_g

# 2. codec ppl suite
PPL_PROC=$(pgrep -c llama-perplexity 2>/dev/null || echo 0)
SUITE_DONE=$(grep -c 'SUITE DONE' /tmp/codec_ppl2.log 2>/dev/null || echo 0)
PPL_ERR=$(grep -ilE 'error loading model|failed to load model' results/e194-codec-kl/*.log 2>/dev/null | tr '\n' ' ')
if [ "$SUITE_DONE" -ge 1 ]; then
  [ "$(prev ppl_state)" != "done" ] && { okmsg "ppl suite DONE: $(grep -hE 'Perplexity' results/e194-codec-kl/*.log | tr '\n' ' ')"; setst ppl_state=done; }
elif [ "$PPL_PROC" -ge 1 ]; then
  for f in results/e194-codec-kl/*.log; do
    [ -f "$f" ] || continue
    age=$(( now - $(stat -c %Y "$f") ))
    [ "$age" -gt 1800 ] && alert "ppl log $f stale ${age}s while suite active"
  done
  [ -n "$PPL_ERR" ] && alert "ppl load errors in: $PPL_ERR"
  setst ppl_state=running
else
  [ "$(prev ppl_state)" = "running" ] && alert "ppl suite died without SUITE DONE"
  setst ppl_state=died
fi

# 3. strip script
STRIP_OUT=${GUM_STRIPPED_MODEL:-"$HOME/Desktop/Chip/models/Qwen3.8-27B-BF16-nonextn.gguf"}
LLAMA_GGUF=${LLAMA_GGUF:-"$HOME/llama.cpp/build/bin/llama-gguf"}
if [ -s "$STRIP_OUT" ] && "$LLAMA_GGUF" "$STRIP_OUT" r n >/dev/null 2>&1; then
  [ "$(prev strip_state)" != "valid" ] && { okmsg "strip output valid"; setst strip_state=valid; }
elif pgrep -f strip_nextn_gguf >/dev/null; then
  setst strip_state=running
else
  [ "$(prev strip_state)" != "failed" ] && alert "strip not running and output missing/invalid"; setst strip_state=failed
fi

# 4. capture freshness (info only)
LAST=$(ls -t capture/group-*.bin 2>/dev/null | head -1)
if [ -n "$LAST" ]; then age=$(( (now - $(stat -c %Y "$LAST")) / 60 )); setst capture_age_min=$age; fi

# 5. server
curl -s --max-time 5 http://127.0.0.1:8765/health | grep -q '"status": "ok"' || alert "GUM server health FAIL"
setst checked=$now
