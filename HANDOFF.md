# Vanta — Session Handoff (2026-11-10, post E042)

Repo: `/home/zabwie/gpu-unified-memory-research` (git; latest `a84c0e8`). Read AGENT.md first
for the operating mentality. RESEARCH_PLAN.md is the strategy; EXPERIMENTS.md the evidence;
this file is the machine state and the immediate next moves.

## Mission

RAM as an extension of VRAM for big-model inference on small GPUs — now focused on **single
user tok/s** for agent workloads, losslessly. Owner directives: no quality loss, open source,
system changes OK, ambitious (40 tok/s), CPU layer offload excluded, 22 GB card possibly
coming (don't optimize for it).

## Machine

RTX 3060 12 GiB (CC 8.6), i7-12700K, 31 GiB RAM, PCIe 4.0 x16 @ 24.6 GB/s H2D pinned
(`iommu=pt` applied — the one approved kernel change). RLIMIT_MEMLOCK 3978 MiB. CUDA 13.2 at
`/usr/local/cuda/bin`. Kernel 7.0.0-38. Desktop shares the GPU; benchmarks self-gate on
RAM/swap floors.

## What exists and is VALIDATED (trust it)

- **Real decode loop** `benchmarks/real_decode.cu` for qwen3 (Qwen3-0.6B) and qwen35
  (Qwen3.8-27B hybrid: 48 gated-delta-net + 16 GQA layers) — GGUF parsing, dp4a Q4_0 GEMV
  with int8 activation quant, per-head norms, RoPE (NEOX + IMROPE-as-half-split), GQA
  attention with KV, gated-delta-net with persistent SSM state, greedy sampling.
  **Token-for-token identical to llama.cpp CPU greedy** (24/24 on the 0.6B; the 27B likewise
  after per-op verification against a patched out-of-tree llama.cpp at /tmp/llama-dbg).
- **Tiered residency**: hot VRAM arena (7.25-8 GiB) + RAM cache (5.4-6.1 GiB, 3658 MiB pinned
  head) + NVMe only on token 0 + copy/compute overlap + slot events. Config hot 7424/cache
  6144 holds the 2 GiB VRAM floor at batch 8.
- **Batched decode** (`--batch B`): NB-template GEMV, per-seq states. 27B: batch 8 = 13.17
  aggregate tok/s, 0.772 GiB cold/token.
- **Multi-position verify** (`--multi B --feed-ids ...`): γ+1 positions of ONE sequence per
  cold-set pass. MEASURED (corrected, honest): 0.350 s (1 pos) → 0.430 s (8 pos), ~7.4
  ms/position marginal, 18.6 tok/s zero-draft bound at 8. 0.6B: 690 tok/s all-hot.
- **Speculative loop** (`--speculate γ`): PLD drafter (CPU n-grams, ~free) → one verify pass →
  accept prefix → correction. Lossless (greedy rejection sampling). **27B chat: 7.26 tok/s
  (2.3× dense).** SSM/conv state snapshot-rollback (~150 MB D2D/cycle) — REQUIRED because
  rejected positions corrupt the cumulative delta-net state (KV alone is not enough; this is
  a genuine contribution for hybrid models).
- **CPU fp32 reference decoder** (`--validate 1`) for correctness gates (0.6B-scale models).

## Performance ledger (27B, single-user, lossless)

| path | tok/s | vs dense |
|---|---|---|
| dense (E038) | 3.39 | 1× |
| spec chat (E041, PLD 33% accept) | 7.26 | 2.3× |
| spec, agent-shaped repeat (28% accept — KNOWN BUG caps it) | 6.8 | 2× |
| verify bound at 8 positions (E040 measured) | 18.6 | — |
| verify bound deep γ=32 (E040 curve extrapolation) | ~50 | — |

## THE HEADLINE OPPORTUNITY (owner's insight)

**Agent tool-wrapping is predictable.** Tool-call JSON schemas, fences, keys, echoed context
— the scaffolding around an agent's text is fixed or repeats from history. The model only
generates the novel middle. So: deep-γ PLD drafting (16–32 positions, ~0.1 ms/token draft
cost) verified in one cold-set pass → in-span bursts of 50–90 tok/s, sustained agent sessions
~20–30 tok/s, lossless. E040's measured +7.4 ms/position makes deep verification nearly free;
the ONLY thing between us and this is the alignment bug below.

## KNOWN ISSUES — fix these first (next session)

1. **E041 known/hist alignment drift (the acceptance killer).** The carried `known_next`
   occasionally disagrees with the history tail (dbg showed known=244 while hist[n0-1]=271 in
   a periodic stream), capping acceptance at 28-33% where the periodic stream should accept
   ~100%. Debug hooks exist: `--dbg-dump 1` prints `dbg,draft` lines (hist tail + known per
   cycle) and `dbg,e040row` (per-row verify argmaxes). Suspects: the prefill-chunk boundary
   (13-token prompt, chunks of 8+5 — the last partial chunk's row indexing), the rewind
   bookkeeping after partial accept, or hist/feed[0] duplication. Do NOT touch the emission
   accounting — it was painful to get right (see the off-by-one saga in EXPERIMENTS.md).
2. **Deep γ untested.** Buffers already scale; run --speculate 16/32 on an agent-shaped trace
   (tool-call JSON + echoes). Watch VRAM (states don't scale with γ, activations do — small).
3. **MTP head drafter (E041a) not integrated.** 1.68 GiB, fits in VRAM beside hot 7.25;
   measured 2.84 tokens/verify on this model (E036). Requires decoding the qwen35 nextn block
   (eh_proj/enorm/hnorm + dense block, own small KV) — the main graph machinery is reusable.
4. **Summary metric bug**: verify mode's `cold_GiB_per_step` prints the TOTAL over all steps
   (incl. the warm-up); per-step cold = 6.17 GiB at hot 7424. Cosmetic.

## Bug classes to check in ANY new batched/multi kernel

- **Flat-index RMW race**: grid (x, B) with `i = blockIdx.x*blockDim.x + threadIdx.x` makes
  every blockIdx.y block touch the same indices — nondeterministic lost updates, invisible to
  all sanitizers and to batch-1 tests. Use global index `(by*gridDim.x+bx)*blockDim.x+tid`.
- **Per-sequence stride errors**: dense [B][N] vs buffer-pitch confusion (max_rows vs N).
- **g_batch ordering**: g_batch must be set AFTER multi/speculate mode adjustments, or the
  GEMV template silently computes only row 0 (E040's regression).
- **Tie-excused validation vacuity**: a mismatch check with a tie threshold will excuse
  all-zero garbage rows. Always pair with a row-energy garbage detector.
- **Stream discipline**: KV appends and anything reading kernel output belong on the compute
  stream (stc). Copy stream (st) is for weight staging only.
- **SSM rollback**: any speculative scheme on hybrid models MUST snapshot/restore the
  delta-net + conv state per cycle. KV overwrite is not sufficient.

## Reproduce the headline numbers

```
CUDA=/usr/local/cuda/bin
$CUDA/nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/real_decode.cu -o benchmarks/real_decode
# dense single-user (3.4 tok/s):
taskset -c 0-3 benchmarks/real_decode --model /home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf \
  --prompt-ids "248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198" \
  --gen-tokens 16 --ctx 256 --hot-mib 7424 --cache-mib 6144 --readers 4
# speculative (7.3 tok/s chat; expect more on repetitive/agent text once issue 1 is fixed):
taskset -c 0-3 benchmarks/real_decode --model /home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf \
  --prompt-ids "248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198" \
  --speculate 8 --gen-tokens 32 --ctx 256 --hot-mib 7424 --cache-mib 6144 --readers 4
# verify ceiling (E040):
... --multi 8 --feed-ids "$(paste -sd, /tmp/feed27.txt)"
```

Model: `/home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf` (13.42 GiB). Do not
touch the user's llama.cpp checkout (an out-of-tree instrumented copy lives at
/tmp/llama-dbg, rebuildable per RESEARCH_PLAN.md).

## Session history (one line each)

E001-E030: memory mechanics (mapped/managed/VMM/tiered pool). E031: streamed proxy 1.0 t/s.
E032-E034: overlap + dp4a + batching → 43.8 agg. E035-E037: IOMMU removal → 2× bandwidth.
E036: real llama.cpp MTP baseline (1.04 t/s, thrashing diagnosis). E038: real decode loop,
3.39 tok/s, llama.cpp-validated. E039: batched decode, 13.17 agg, RMW-race class discovered.
E040: verify ceiling measured (corrected). E041/E042: speculative loop, 7.26 tok/s chat,
SSM-rollback contribution. Full detail: EXPERIMENTS.md.
