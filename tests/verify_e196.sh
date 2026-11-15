#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
decoder=${GUM_DECODER:-"$root/build/real_decode"}

: "${GUM_TARGET_MODEL:?Set GUM_TARGET_MODEL to the Qwen3.8-27B STRICT Q4_0 GGUF path}"
: "${GUM_MTP_MODEL:?Set GUM_MTP_MODEL to the matching native MTP GGUF path}"

for model in "$GUM_TARGET_MODEL" "$GUM_MTP_MODEL"; do
	if [[ ! -r "$model" ]]; then
		printf 'Model file is not readable: %s\n' "$model" >&2
		exit 2
	fi
done

if [[ -z "${GUM_DECODER:-}" ]]; then
	make -C "$root" decoder
elif [[ ! -x "$decoder" ]]; then
	printf 'GUM_DECODER is not executable: %s\n' "$decoder" >&2
	exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
	printf 'flock is required to avoid overlapping GPU validation runs.\n' >&2
	exit 2
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/gum-verify-e196.XXXXXX")
on_exit() {
	status=$?
	if (( status != 0 )) && [[ -f "$tmpdir/run.log" ]]; then
		printf '\nLast decoder output:\n' >&2
		tail -n 80 "$tmpdir/run.log" >&2
	fi
	rm -rf -- "$tmpdir"
	exit "$status"
}
trap on_exit EXIT

prompt=$(python3 - "$root/tests/fixtures/e196_depth_prompt.ids" <<'PY'
from pathlib import Path
import sys

ids = [int(value) for value in Path(sys.argv[1]).read_text().strip().split(",")]
if len(ids) != 7840:
	raise SystemExit(f"Expected the frozen 7,840-token prompt, found {len(ids)} IDs")
print("48:" + ",".join(map(str, ids)))
PY
)

exec 9>"${GUM_GPU_LOCK:-/tmp/gum-gpu.lock}"
if ! flock -n 9; then
	printf 'GPU validation lock is busy: %s\n' "${GUM_GPU_LOCK:-/tmp/gum-gpu.lock}" >&2
	exit 2
fi

printf '%s\n' "$prompt" |
	"$decoder" \
		--model "$GUM_TARGET_MODEL" \
		--mtp "$GUM_MTP_MODEL" \
		--session 1 --session-tail 1 --session-stream 1 \
		--prefill-chunks 8 --speculate 32 --packed-verify 1 \
		--pld 0 --draft-max 31 --kv-cache turboquant-ttkv \
		--ctx 33792 --hot-mib 0 --cache-mib 4096 \
		--gen-tokens 48 --show-text 0 --prompt-ids - \
		>"$tmpdir/run.log" 2>&1

python3 - "$tmpdir/run.log" "$root/tests/fixtures/e196_depth_expected.ids" <<'PY'
from pathlib import Path
import sys

log = Path(sys.argv[1]).read_text(errors="replace").splitlines()
actual_lines = [line.split(",", 1)[1] for line in log if line.startswith("session_ids,")]
if len(actual_lines) != 1:
	raise SystemExit(f"Expected one session_ids result, found {len(actual_lines)}")
actual = [int(value) for value in actual_lines[0].split(",")]
expected = [int(value) for value in Path(sys.argv[2]).read_text().strip().split(",")]
if actual != expected:
	first = next((i for i, pair in enumerate(zip(actual, expected), 1) if pair[0] != pair[1]), None)
	raise SystemExit(
		f"FAIL: {len(actual)} output IDs, expected {len(expected)}; first mismatch: {first}"
	)
print(f"PASS: all {len(expected)} output token IDs match the frozen E195 reference")
PY
