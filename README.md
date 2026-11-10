# Vanta — RAM as an extension of VRAM

Run large quantized LLMs on small-GPU machines at usable speeds, by treating system RAM as a
designed tier of GPU memory — then making every generated token cost as few of those bytes
as possible. Built for the RTX 3060 12 GB class of machine; open source; single-user latency
is the product.

## Results (Qwen3.8-27B Q4, 13.42 GiB, on a 12 GiB RTX 3060 — all lossless)

| capability | tok/s | status |
|---|---|---|
| dense decode, single user | 3.39 | validated token-for-token vs llama.cpp |
| batch-8 serving (aggregate) | 13.2 | validated |
| speculative (PLD drafter), single user | 7.26 | first working cut, lossless |
| verify bound at 8 positions/crossing | 18.6 | measured ceiling |
| agent spans (deep γ, exact repeats) | 50–90 bursts | mechanical, pending drafter fix |

llama.cpp's own over-VRAM mode on this model/hardware: 0.63 tok/s (1.04 with its MTP head).

## Read in this order

1. **[AGENT.md](AGENT.md)** — operating mentality: ambition + defiance, disciplined by measurement.
2. **[HANDOFF.md](HANDOFF.md)** — current machine state, known issues, reproduction commands.
3. **[RESEARCH_PLAN.md](RESEARCH_PLAN.md)** — strategy, corrected projections, experiment ladder.
4. **[DESIGN.md](DESIGN.md)** — technical reference for real_decode.cu (architecture, kernels,
   ggml semantics, buffer layouts, bug classes).
5. **[EXPERIMENTS.md](EXPERIMENTS.md)** — E001–E042 raw evidence. **[NEXT_STEPS.md](NEXT_STEPS.md)** — task ladder.

Older study docs (memory mechanics, VMM, microbenchmarks): HARDWARE.md, FINDINGS.md,
BENCHMARKS.md, HYPOTHESES.md, OVERSUBSCRIPTION_PLAN.md, AGENT_STATE.md.

## Quickstart

```sh
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/real_decode.cu -o benchmarks/real_decode
# 27B on a 12 GB card (needs ~21 GB free RAM; adjust --prompt-ids to your chat template):
taskset -c 0-3 benchmarks/real_decode --model /path/to/Qwen3.8-27B-STRICT-Q4_0.gguf \
  --prompt-ids "<token ids>" --gen-tokens 32 --ctx 256 --hot-mib 7424 --cache-mib 6144 --readers 4
# lossless speculation:
... --speculate 8
# batch serving:
... --batch 8
```

Correctness is the gate: every mode is checked against a CPU fp32 reference and/or the
validated token stream before its numbers count. See DESIGN.md §6.

## The short version of the science

RAM extends VRAM capacity, but not bandwidth (PCIe ≈ 1/14th of VRAM reads) — so the
architecture is a software-defined memory hierarchy (hot arena / RAM cache / NVMe) with
copy-compute overlap, and speed comes from **amortizing the cable**: batched decode shares
one weight streaming across sequences, and speculative decoding shares it across guessed
tokens (measured: 8 positions per crossing cost +7.4 ms over one). On hybrid
transformer/SSM models, speculation additionally requires recurrent-state snapshot/rollback —
rejected guesses corrupt cumulative SSM state in ways KV overwrite cannot fix.
