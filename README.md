# GUM — GPU Unified Memory

Run large quantized LLMs on small-GPU machines at usable speeds, by treating system RAM as a
designed tier of GPU memory — then making every generated token cost as few of those bytes
as possible. Built for the RTX 3060 12 GB class of machine; open source; single-user latency
is the product. Formerly "Vanta".

## Current configuration: pure streaming (2026-11-13)

The installed default is **`GUM_PROFILE=stream`**: no resident weight tier at all — every
weight byte streams from pinned system RAM over PCIe (24.7 GB/s measured) on every pass.
The GPU holds only the machinery (KV tier, drafter, buffers):

| profile | engine VRAM | free for your tools | decode tok/s |
|---|---|---|---|
| **stream** (default) | **~3.0 GiB** | **~9 GiB** | **~5.5** |
| nano | ~4.6 GiB | ~7.1 GiB | 6.27 |
| lean | ~6.1 GiB | ~5.3 GiB | 7.19 |
| chat | ~10.2 GiB | ~2.1 GiB | 10.79 |
| 196k | ~10.4 GiB | ~1.5 GiB | long-output profile (E181) |

Switch with `GUM_PROFILE=<name>`; the API runs as `gum-openai.service` (systemd,
`LimitMEMLOCK=infinity`). Lossless always: greedy token IDs identical to the exact
Q4_0 checkpoint — the streaming profile is gated 256/256 token-exact against the
baseline golden stream.

**Historical baselines (E149/E177, 2026-11-12: 16.99–23.87 tok/s) were different
configurations** — ~8 GiB resident hot tier and 2K bench contexts, not the current
pure-streaming stack. Do not compare numbers across eras without reading
[docs/SYSTEM_MAP.md](docs/SYSTEM_MAP.md) and [RESULTS.md](RESULTS.md).

## Navigating this repo

Start here:

1. **[AGENT.md](AGENT.md)** — operating mentality for any AI or human working on GUM: ambition
   + defiance, disciplined by measurement. Read first.
2. **[HANDOFF.md](HANDOFF.md)** — current machine state, baseline, validated components,
   correctness guards, reproduction commands.
3. **[docs/RESEARCH_FUNNEL.md](docs/RESEARCH_FUNNEL.md)** — the working protocol: every
   experiment enters through the decision-driven funnel; established bounds with citations.
4. **[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)** — strategy record.
5. **[NEXT_STEPS.md](NEXT_STEPS.md)** — task ladder. **[EXPERIMENTS.md](EXPERIMENTS.md)** —
   the evidence log, E000–E165, indexed against `results/`.

Reference: **[docs/DESIGN.md](docs/DESIGN.md)** (technical reference for `real_decode.cu`:
architecture, kernels, ggml semantics, buffer layouts, bug classes),
**[docs/HARDWARE.md](docs/HARDWARE.md)** (machine baseline).

Frozen study docs from the E001–E026 memory-mechanics era live in **[docs/archive/](docs/archive/)**
(AGENT_STATE, BENCHMARKS, FINDINGS, HYPOTHESES, OVERSUBSCRIPTION_PLAN, HANDOFF_HISTORY) —
historical record, superseded where they disagree with the docs above.

## Quickstart

```sh
scripts/run_single_user.sh --prompt "Your prompt"
# GUM_RESIDENT=1  → validated 8448/ctx-384 short-sequence mode (E156a)
# GUM_DRAFTER=eagle4 → optional EAGLE4 drafter
```

The launcher defaults to native MTP drafting with grouped trees, a packed verifier,
256 generated tokens, context capacity 1024, and 8064 MiB resident target weights.
It can verify up to 64 positions on supported GPUs. Model paths can be overridden with
`GUM_TARGET_MODEL` and `GUM_MTP_MODEL` (EAGLE4: `GUM_EAGLE_MODEL`; build it once with
`python3 scripts/prepare_eagle4.py --output models/Qwen3.8-27B-EAGLE4-Q4_0.gguf` — requires
NumPy and the `gguf` package; [EAGLE4 checkpoint and license](https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4)).

Direct invocation of the experiment decoder:

```sh
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/real_decode.cu \
  -o benchmarks/real_decode
# 27B on a 12 GB card (needs ~21 GB free RAM; adjust --prompt-ids to your chat template):
taskset -c 0-3 benchmarks/real_decode --model /path/to/Qwen3.8-27B-STRICT-Q4_0.gguf \
  --prompt-ids "<token ids>" --gen-tokens 32 --ctx 256 --hot-mib 7424 --cache-mib 6144 --readers 4
# lossless speculation (ALWAYS gate with --expect-ids against a dense --out-ids reference):
... --speculate 8 --expect-ids /tmp/expect.txt
# batch serving:
... --batch 8
```

`tok_per_s` includes drafting, verification, and state repair; `request_tok_per_s` also
includes prompt evaluation after model loading. `--prompt` uses the decoder's approximate
tokenizer — prefer passing the rendered chat template's exact token IDs with `--prompt-ids`.
Per-experiment commands live in [EXPERIMENTS.md](EXPERIMENTS.md).

For a multi-turn client, pass `--session 1` with the first rendered prompt
in `--prompt-ids`. After each `session_ids,<ids>` line on stdout, send the
next full rendered prompt as comma-separated token IDs on one stdin line.
The model stays loaded. When the next prompt starts with the saved prefix,
the decoder reuses its completed prefill chunks and evaluates the new
suffix. Closing stdin exits. Set `--ctx` to fit the longest turn plus
`--gen-tokens`. Session mode supports packed native MTP and EAGLE4.

The E171 single-user configuration on the RTX 3060 used
`--session 1 --session-tail 1 --prefill-chunks 4 --speculate 32 --packed-verify 1 --pld 0 --draft-max 31`
with `--ctx 2048 --gen-tokens 384`. It keeps the target state through the
previous answer's committed tokens while that full token prefix matches. A changed prefix
uses the last matching prompt checkpoint or starts a fresh prefill.

Use `--prefill-chunks 8` with that configuration to process up to 256 prompt
tokens per target weight pass. The RTX 3060 repository-job benchmark includes
startup and prompt processing. See [E173](EXPERIMENTS.md#e173--wider-exact-prefill).

E177 merges the final short prompt chunk with its preceding group when they
fit in the same pass. This is enabled for grouped native MTP sessions with
`--session-tail 1`. The original prompt checkpoint remains available when a
later request edits the suffix. Use `--fuse-prefill-tail 0` for the separate-pass
control. Explicit shared-prefix checkpoints and draft-hint sessions use their
existing prefill paths. See [E177](results/e177-general-20261112/README.md).

For separate requests that share an initial block of repository or tool
context, add `--session-prefix-tokens N` to the native MTP session command
above. Set `N` to a multiple of 32 within the shared rendered token prefix.
The decoder retains a checkpoint there and verifies all prefix IDs before
reusing it. An edit inside that prefix triggers fresh evaluation and rebuilds
the checkpoint when the new prompt reaches the boundary. See
[E174](EXPERIMENTS.md#e174--reuse-shared-context-across-separate-requests).

Clients that can propose an answer from tool output can use
`--session-draft-hints 1` with the E171 or E173 launcher settings. Append
`|<draft token IDs>` to the first `--prompt-ids` value or to any later
prompt line. Both lists are comma-separated. The draft starts at the first
generated token and excludes the prompt. A request without a draft uses
the ordinary decoder. The target checks each proposed token and confirms
EOS itself. See [E175](EXPERIMENTS.md#e175--verify-tool-drafts-during-prefill).

Correctness is the gate: every mode is checked against a CPU fp32 reference and/or the
validated token stream before its numbers count. See [docs/DESIGN.md](docs/DESIGN.md) §6.

## The short version of the science

RAM extends VRAM capacity, but not bandwidth (PCIe ≈ 1/14th of VRAM reads) — so the
architecture is a software-defined memory hierarchy (hot arena / RAM cache / NVMe) with
copy-compute overlap, and speed comes from **amortizing the cable**: batched decode shares
one weight streaming across sequences, and speculative decoding shares it across guessed
tokens (measured: 8 positions per crossing cost +7.4 ms over one). On hybrid
transformer/SSM models, speculation additionally requires recurrent-state snapshot/rollback —
rejected guesses corrupt cumulative SSM state in ways KV overwrite cannot fix.
