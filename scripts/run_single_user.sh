#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
decoder=${GUM_DECODER:-"$root/build/real_decode"}
if (( $# == 0 )); then
    printf 'Usage: %s --prompt "Your prompt" [decoder options]\n' "$0" >&2
    exit 2
fi
if [[ ! -x "$decoder" ]]; then
    printf 'Decoder not built: run `make decoder` or set GUM_DECODER.\n' >&2
    exit 2
fi

target=${GUM_TARGET_MODEL:-"$HOME/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"}
case ${GUM_DRAFTER:-mtp} in
    mtp)
        export GUM_PACKED_DEFAULT=1
        draft=(--mtp "${GUM_MTP_MODEL:-"$HOME/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf"}")
        ;;
    eagle4)
        draft=(--eagle4 "${GUM_EAGLE_MODEL:-"$root/models/Qwen3.8-27B-EAGLE4-Q4_0.gguf"}")
        ;;
    *)
        printf 'GUM_DRAFTER must be mtp or eagle4\n' >&2
        exit 2
        ;;
esac

# E156a promoted short-sequence mode (GUM_RESIDENT=1): 8448 MiB resident
# target weights at context capacity 384. Validated faster on both chat
# prompts (+0.5-1.2 tok/s, exact-gated); requires prompt+generation <= 384.
# Default remains the long-context 8064/1024 configuration.
if [[ ${GUM_RESIDENT:-0} == 1 ]]; then
    exec "$decoder" \
        --model "$target" "${draft[@]}" --ctx 384 --gen-tokens 256 \
        --hot-mib 8448 --cache-mib 5600 "$@"
fi
exec "$decoder" \
    --model "$target" "${draft[@]}" --ctx 1024 --gen-tokens 256 \
    --hot-mib 8064 --cache-mib 5600 "$@"
