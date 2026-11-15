#!/usr/bin/env bash
# capture_when_idle.sh — EAGLE-scale capture pipeline wrapper (E167).
#
# Drives the in-tree decoder's batched hidden-state capture (E159/E169):
#   - exclusive GPU lock: flock -n on /tmp/gum-gpu.lock; skips (never waits)
#     when a foreground benchmark or another capture holds it
#   - preflight per cycle: GPU util/temp/VRAM, MemAvailable, heavy pids
#     (the decoder also monitors RAM and swap during execution)
#   - atomic publish: all outputs land as *.part, are validated, then renamed
#   - diversity ledger: per-prompt token hashes; groups whose prompts were
#     all seen before are recorded in the ledger without a residual file
#   - storage budget: checks space for the new capture before decoding
#   - quarantine: failed validations moved to capture/quarantine/ with logs
#   - explicit stop switch: touch capture/STOP  ->  every future run exits 0
#
# Usage:
#   scripts/capture_when_idle.sh --pilot 1   # foreground, one group, waits briefly for lock
#   scripts/capture_when_idle.sh             # one cycle for the idle scheduler (skip if busy)
#
# Storage math (measured, pilot): ~164 KB per captured position
# (7 mid layers 8..56 + final residual, fp32) -> 100k unique positions ~ 16.4 GB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPTURE_DIR="${CAPTURE_DIR:-$ROOT/capture}"
RIG_BIN="${RIG_BIN:-$ROOT/benchmarks/real_decode}"
TARGET="${GUM_TARGET_MODEL:-$HOME/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf}"
MTP="${GUM_MTP_MODEL:-$HOME/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf}"
TOKENIZER="$ROOT/scripts/tokenize_seed.py"
LOCK=/tmp/gum-gpu.lock
STOP="$CAPTURE_DIR/STOP"
LEDGER="$CAPTURE_DIR/LEDGER.md"
SEQHASHES="$CAPTURE_DIR/seq-hashes.txt"
CAPTURE_BUDGET_GB="${CAPTURE_BUDGET_GB:-40}"
GEN_TOKENS="${CAPTURE_GEN_TOKENS:-160}"
MAX_PROMPT_TOKENS=128
MIN_PROMPT_TOKENS=8
N_MEM=8
LAYERS=8,16,24,32,40,48,56
N_MID=7
WIDTH=5120

mkdir -p "$CAPTURE_DIR" "$CAPTURE_DIR/quarantine"

die_quiet() { echo "capture_when_idle: $1" >&2; exit "${2:-3}"; }

# --- stop switch ----------------------------------------------------------------
if [[ -e "$STOP" ]]; then
    echo "capture_when_idle: STOP sentinel present — exiting no-op."
    exit 0
fi

# --- exclusive GPU lock ----------------------------------------------------------
exec 9>"$LOCK"
if [[ "${1:-}" == "--pilot" ]]; then
    flock -w 60 9 || die_quiet "pilot: GPU lock busy for 60s" 3
else
    flock -n 9 || die_quiet "Deferred: GPU lock held by foreground work" 3
fi
[[ -e "$STOP" ]] && exit 0
[[ -x "$RIG_BIN" && -r "$TARGET" && -r "$MTP" ]] || die_quiet "decoder or model files are missing" 1
[[ "$GEN_TOKENS" =~ ^[1-9][0-9]*$ ]] && (( GEN_TOKENS >= 3 && GEN_TOKENS <= 512 - MAX_PROMPT_TOKENS )) ||
    die_quiet "CAPTURE_GEN_TOKENS must be in 3..$((512 - MAX_PROMPT_TOKENS))" 1
[[ "$CAPTURE_BUDGET_GB" =~ ^[1-9][0-9]*$ ]] || die_quiet "CAPTURE_BUDGET_GB must be a positive integer" 1

# Select the next index under the lock. Metadata-only duplicate groups also
# consume an index so that the authored seed cursor can advance.
GNUM=-1
for path in "$CAPTURE_DIR"/group-*.txt; do
    if [[ ${path##*/} =~ ^group-([0-9]+)\.txt$ ]]; then
        idx=$((10#${BASH_REMATCH[1]}))
        (( idx > GNUM )) && GNUM=$idx
    fi
done
GNUM=$((GNUM + 1))
printf -v GROUP '%03d' "$GNUM"

SEEDS="$ROOT/scripts/capture_seeds.txt"
LOCAL_SEEDS="$CAPTURE_DIR/seeds.txt"
# Older runs copied the default pool here. Let those exact snapshots pick up
# appended shipped seeds, while keeping edited local pools as overrides.
if [[ -s "$LOCAL_SEEDS" ]] && ! cmp -s "$LOCAL_SEEDS" <(head -n "$(wc -l < "$LOCAL_SEEDS")" "$SEEDS"); then
    SEEDS="$LOCAL_SEEDS"
fi
mapfile -t SEED_LINES < "$SEEDS"
N_SEEDS=${#SEED_LINES[@]}

# --- preflight (run_when_idle.sh style) ------------------------------------------
SAMPLES=5
mapfile -t SNAP < <(for ((i=0;i<SAMPLES;++i)); do
    nvidia-smi --query-gpu=utilization.gpu,memory.free,temperature.gpu --format=csv,noheader,nounits; sleep 1; done)
(( ${#SNAP[@]} == SAMPLES )) || die_quiet "Deferred: incomplete GPU samples" 3
MIN_FREE_VRAM=1000000; MAX_TEMP=0; HIGH_UTIL=0
for line in "${SNAP[@]}"; do
    IFS=, read -r util free temp <<< "$(tr -d ' ' <<< "$line")"
    (( free < MIN_FREE_VRAM )) && MIN_FREE_VRAM=$free
    (( temp > MAX_TEMP )) && MAX_TEMP=$temp
    (( util >= 85 )) && HIGH_UTIL=$((HIGH_UTIL+1))
done
HEAVY_PIDS="$(nvidia-smi pmon -c "$SAMPLES" -d 1 2>/dev/null | awk '
    $1 !~ /^#/ && NF >= 9 && $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ && $4 >= 50 { hits[$2]++ }
    END { for (pid in hits) if (hits[pid] >= 3) printf "%s ", pid }')"
RAM_AVAIL=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)

# Rig floor: E159 batch runs measured 856 MiB minimum free VRAM — 1500 MiB with margin.
(( MIN_FREE_VRAM < 1500 )) && die_quiet "Deferred: only ${MIN_FREE_VRAM} MiB free VRAM" 3
(( MAX_TEMP >= 80 )) && die_quiet "Deferred: GPU temp ${MAX_TEMP}C" 3
(( HIGH_UTIL >= 4 )) && die_quiet "Deferred: GPU busy ${HIGH_UTIL}/${SAMPLES} samples" 3
[[ -n "$HEAVY_PIDS" ]] && die_quiet "Deferred: heavy GPU pids: $HEAVY_PIDS" 3
(( RAM_AVAIL < 8192 )) && die_quiet "Deferred: only ${RAM_AVAIL} MiB RAM available" 3

# --- storage budget ---------------------------------------------------------------
REC_FLOATS=$(( 3 + (N_MID + 1) * N_MEM * WIDTH ))
EXPECTED_BYTES=$(( GEN_TOKENS * REC_FLOATS * 4 ))
USED_BYTES=$(du -sb "$CAPTURE_DIR" | cut -f1)
(( USED_BYTES + EXPECTED_BYTES <= CAPTURE_BUDGET_GB * 1024**3 )) ||
    die_quiet "Storage budget would exceed ${CAPTURE_BUDGET_GB} GiB" 4
FS_AVAIL=$(df -B1 --output=avail "$CAPTURE_DIR" | tail -1)
(( FS_AVAIL - EXPECTED_BYTES >= 20 * 1024**3 )) ||
    die_quiet "Deferred: capture would leave less than 20 GiB free" 3

# --- build one group: 8 lockstep prompt-id sequences -------------------------------
declare -a MEMBER_IDS=() MEMBER_SRC=() KEEP_TAIL=()
first=$(( GNUM * N_MEM ))
if (( first < N_SEEDS )); then
    seeds=("${SEED_LINES[@]:first:N_MEM}")
    tokenized=$(printf '%s\n' "${seeds[@]}" | python3 "$TOKENIZER" --model "$TARGET") ||
        die_quiet "seed tokenization failed" 1
    mapfile -t MEMBER_IDS <<< "$tokenized"
    (( ${#MEMBER_IDS[@]} == ${#seeds[@]} )) || die_quiet "tokenizer returned the wrong seed count" 1
    for i in "${!seeds[@]}"; do
        MEMBER_SRC+=("seed[$((first + i))]")
        KEEP_TAIL[i]=0
        [[ ${seeds[i]} == *'<|im_start|>'* ]] && KEEP_TAIL[i]=1
    done
fi
if (( ${#MEMBER_IDS[@]} < N_MEM )); then
    shopt -s nullglob
    prev=("$CAPTURE_DIR"/group-*.ids)
    shopt -u nullglob
    (( ${#prev[@]} )) || die_quiet "no previous streams for continuation seeds" 1
    # Sample across every member of prior groups. Only the seed slice needs to
    # fit the source; the decoder generates a new continuation.
    mapfile -t streams < <(awk -F, '{
        sub(/,$/, "")
        if (NF >= 40) print FILENAME ":" FNR "\t" $0
    }' "${prev[@]}" | shuf -r -n "$((N_MEM - ${#MEMBER_IDS[@]}))")
    for entry in "${streams[@]}"; do
        IFS=$'\t' read -r source ids <<< "$entry"
        IFS=, read -ra toks <<< "$ids"
        len=$(( 40 + RANDOM % 80 ))
        (( len > ${#toks[@]} )) && len=${#toks[@]}
        start=$(( RANDOM % (${#toks[@]} - len + 1) ))
        MEMBER_IDS+=("$(IFS=,; printf '%s' "${toks[*]:start:len}")")
        MEMBER_SRC+=("cont:${source##*/}#$start+$len")
        KEEP_TAIL+=(0)
    done
fi
(( ${#MEMBER_IDS[@]} == N_MEM )) || die_quiet "need $N_MEM seeds or reusable streams of 40 tokens" 1
# equalize: truncate every member to the shortest usable length (lockstep requirement)
minlen=$MAX_PROMPT_TOKENS
for ids in "${MEMBER_IDS[@]}"; do
    IFS=, read -ra toks <<< "$ids"
    (( ${#toks[@]} < minlen )) && minlen=${#toks[@]}
done
(( minlen < MIN_PROMPT_TOKENS )) && die_quiet "seed too short after truncation (min=$minlen)" 1
GROUP_BODY=""
for i in "${!MEMBER_IDS[@]}"; do
    IFS=, read -ra toks <<< "${MEMBER_IDS[i]}"
    start=0
    # Preserve the latest user turn and assistant handoff in chat seeds.
    (( KEEP_TAIL[i] )) && start=$(( ${#toks[@]} - minlen ))
    seq=$(IFS=,; printf '%s' "${toks[*]:start:minlen}")
    GROUP_BODY+="${GROUP_BODY:+;}$seq"
done
GP="$CAPTURE_DIR/group-$GROUP"
STAGE=$(mktemp -d "$CAPTURE_DIR/.group-$GROUP.XXXXXX")
PART="$STAGE/group-$GROUP"
printf '%s\n' "$GROUP_BODY" > "$PART.txt.part"
printf '{"members":[0,1,2,3,4,5,6,7],"pad_to":%d}\n' "$minlen" > "$PART-lens.json.part"
PROVENANCE="${MEMBER_SRC[*]}"

# --- run the batched capture rig (mirrors E159 batch-group invocation) --------------
# Canonical products live in capture/ (root ext4, survives reboots; /tmp is aged
# by systemd-tmpfiles). The rig is the in-tree decoder (env-gated capture).
echo "capture_when_idle: running group $GROUP (prompt_tokens=$minlen, gen=$GEN_TOKENS, sources: $PROVENANCE)"
if GUM_MID_CAPTURE="$LAYERS" GUM_MID_CAPTURE_FILE="$PART.bin.part" \
    taskset -c 0-3 "$RIG_BIN" \
    --model "$TARGET" --mtp "$MTP" --speculate 0 --batch "$N_MEM" \
    --prompt-ids "$GROUP_BODY" --ctx 512 --gen-tokens "$GEN_TOKENS" \
    --hot-mib 6880 --cache-mib 6144 --show-text 0 --out-ids "$PART.ids.part" \
    > "$PART.log.part" 2>&1 &&
    grep -q "^summary,PASS,batch=$N_MEM,gen_steps=$GEN_TOKENS," "$PART.log.part" &&
    python3 - "$PART" "$N_MEM" "$N_MID" "$minlen" "$GEN_TOKENS" "$REC_FLOATS" <<'PY'
import sys
from pathlib import Path

import numpy as np

prefix = sys.argv[1]
batch, layers, prompt_len, steps, record_floats = map(int, sys.argv[2:])
binary = Path(f"{prefix}.bin.part")
if binary.stat().st_size != steps * record_floats * 4:
    sys.exit("capture byte count does not match the requested records")
records = np.memmap(binary, dtype="<f4", mode="r", shape=(steps, record_floats))
if (not np.array_equal(records[:, 0], np.arange(prompt_len, prompt_len + steps))
        or not np.all(records[:, 1] == layers) or not np.all(records[:, 2] == batch)):
    sys.exit("capture record positions, layer count, or batch width are inconsistent")
if not all(np.isfinite(row).all() for row in records):
    sys.exit("capture has non-finite residual values")
lines = Path(f"{prefix}.ids.part").read_text().splitlines()
if len(lines) != batch:
    sys.exit("generated stream count does not match the batch")
for line in lines:
    ids = line.removesuffix(",").split(",")
    if len(ids) != steps or not all(token.isdecimal() for token in ids):
        sys.exit("generated IDs do not match the record count")
PY
then
    # --- diversity gate: keep residuals only for groups with new prompts -----------
    declare -A new_hashes=()
    while read -r seq; do
        h=$(printf '%s' "$seq" | sha256sum | cut -c1-16)
        if ! grep -qx "$h" "$SEQHASHES" 2>/dev/null; then
            new_hashes[$h]=1
        fi
    done < <(tr ';' '\n' < "$PART.txt.part")
    NEW_SEQS=${#new_hashes[@]}
    mv "$PART.ids.part" "$GP.ids"
    mv "$PART.log.part" "$GP.log"
    mv "$PART-lens.json.part" "$GP-lens.json"
    mv "$PART.txt.part" "$GP.txt"
    POS=$(( N_MEM * GEN_TOKENS ))
    BIN_SIZE=$EXPECTED_BYTES
    if (( NEW_SEQS == 0 )); then
        rm "$PART.bin.part"
        POS=0
        BIN_SIZE=0
    else
        # Readers discover a group via its .bin. Publish it after the sidecars.
        mv "$PART.bin.part" "$GP.bin"
        printf '%s\n' "${!new_hashes[@]}" >> "$SEQHASHES"
    fi
    rmdir "$STAGE"
    {
        echo "- **group-$GROUP** $(date -Is) | prompt=$minlen tok gen=$GEN_TOKENS | seqs_new=$NEW_SEQS/$N_MEM | pos=$POS | bin=$((BIN_SIZE/1048576))MiB | src: $PROVENANCE | rig=in-tree(real_decode)"
    } >> "$LEDGER"
    echo "capture_when_idle: group $GROUP published (new sequences $NEW_SEQS/$N_MEM, $POS positions, $((BIN_SIZE/1048576)) MiB)"
else
    Q="$CAPTURE_DIR/quarantine/${STAGE##*/}"
    mv "$STAGE" "$Q"
    die_quiet "group $GROUP failed decode or validation, quarantined at $Q" 1
fi
