# GUM — GPU Unified Memory

Run large quantized LLMs on small-GPU machines at usable speeds, by treating system RAM as a
designed tier of GPU memory — then making every generated token cost as few of those bytes
as possible. Built for the RTX 3060 12 GB class of machine; open source; single-user latency
is the product. Formerly "Vanta".

## Baseline (RTX 3060 12 GB, Qwen3.8-27B Q4_0, 13.42 GiB, all lossless)

| Workload | Now |
|---|---|
| Normal chat | ~17–18 tok/s |
| Agent-style prompts (repetitive text) | ~24 tok/s |
| Dense decode (no speculation) | 3.59–3.73 tok/s |
| Ceiling with all remaining validated levers (better residency, tree tuning, sync trim) | ~20–25 chat / ~27 agent |

Measured six-prompt exact suite (E149 installed launcher): 16.99–23.87 generated tok/s,
every one of 1,024 token IDs matching the dense reference — agent trace 23.87, Rust merge
18.58, event-log design 18.57, science explanation 17.45, water-tower chat 17.59, async-queue
chat 16.99. For contrast, llama.cpp's own over-VRAM mode on this model/hardware: 0.63 tok/s
(1.04 with its MTP head).

Lossless means: greedy token IDs identical to the exact Q4_0 checkpoint, gated by
`--expect-ids` on every speculative run.

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

The launcher defaults to the native MTP draft (grouped trees, packed 24-position verifier,
256 generated tokens, context capacity 1024, 8064 MiB resident target weights) and verifies
up to 24 positions on Ampere-or-newer GPUs. Model paths can be overridden with
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
