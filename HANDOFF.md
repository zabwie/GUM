# Vanta — Session Handoff (2026-11-10, post E049)

Repo: `/home/zabwie/gpu-unified-memory-research` (git; latest see `git log -1`). Read AGENT.md
first for the operating mentality. RESEARCH_PLAN.md is the strategy; EXPERIMENTS.md the
evidence; this file is the machine state and the immediate next moves.

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
  accept prefix → correction. Lossless (greedy rejection sampling) — **now gated**:
  `--expect-ids <file>` diffs the spec stream against a dense `--out-ids` reference and exits 1
  on drift. After E043's two bug fixes the loop is proven lossless at γ=8 and γ=16 across
  four residency configs and two prompts (27B + 0.6B). SSM/conv snapshot-rollback + re-feed
  carry (~150 MB D2D/cycle) — REQUIRED because rejected positions corrupt the cumulative
  delta-net state (KV alone is not enough; this is a genuine contribution for hybrid models).
- **CPU fp32 reference decoder** (`--validate 1`) for correctness gates (0.6B-scale models).

## Performance ledger (27B, single-user, lossless-gated — post E049)

| path | tok/s | vs dense |
|---|---|---|
| dense (8704/5150, no NVMe tier) | 3.53 | 1x |
| spec PLD-only g=8 | 2.9-3.3 | ~1x |
| spec MTP g=16, chat prompt (78% accept) | 9.3-10.0 | 3.4x |
| spec hybrid g=16, chat (PLD rarely fires) | 9.22 | 3.3x |
| spec hybrid g=16, agent trace (34-42% accept) | 5.79 | 1.7x |
| verify bound g=8, hot 8192 (E047 re-measure) | 20.4 | — |
| verify bound deep g=32 (curve extrapolation) | ~50 | — |

**⚠ The old "7.26 tok/s chat at 33% acceptance" was measured on a corrupted stream** (E043
bug 1: the state divergence made the model echo its own repeats, which the n-gram drafter
then "predicted"). With correct state, PLD drafting quality — not verification — is the
bottleneck. The verify machinery is now correct and gated; the upside is in drafter quality
(adaptive-γ, RESEARCH_PLAN).

## THE HEADLINE OPPORTUNITY (owner's insight)

**Agent tool-wrapping is predictable.** Tool-call JSON schemas, fences, keys, echoed context
— the scaffolding around an agent's text is fixed or repeats from history. The model only
generates the novel middle. So: deep-γ drafting (16–32 positions, ~0.1 ms/token draft cost)
verified in one cold-set pass → in-span bursts. E040's measured +7.4 ms/position makes deep
verification nearly free, and E043 made it correct and gated. What stands between us and the
burst numbers is now purely **drafter quality** (issue 2): the verify side is solved.

## KNOWN ISSUES — next levers, ranked (post E049)

1. **Copy throughput is THE bound.** nsys: the pass is H2D-copy-bound at 13.6 GB/s effective
   (median copy 16.6 MB) vs the 24.6 GB/s peak; ALL compute is hidden behind copies. Lifting
   copies to ~20 GB/s is a 1.5x on every path. Failed attempts (E048, reverted): 64 MiB
   items, write-combined host buffers. Untried: dual copy streams (check
   asyncEngineCount first — consumer GPUs may have 1 CE), per-stream copy batching.
2. **Drafter acceptance on high-entropy content.** MTP gets 78% (chat) but 34% (agent JSON);
   the hybrid picks PLD only on 6-token matches. Next: tree/beam drafting needs multi-branch
   verify (rows are one sequence — an architecture change); or p_min adaptive per depth.
3. **delta-net kernel is 2x off its bandwidth floor** (85 ms/pass vs ~40 ms) — currently
   FREE (hidden behind copies); only matters after lever 1 lands.
4. **k_attn_decode is O(context) serial per head** with half the threads idle (hd=128,
  256 threads) — same story: hidden until copies lift.
5. **Summary metric bug**: verify mode's cold_GiB_per_step prints the TOTAL (cosmetic).
6. **E040 soft check**: keep it honest — prompt-region rows are skipped, absolute gap,
   garbage detector. Never "simplify" it back.

## THE HEADLINE OPPORTUNITY (owner's insight)

**Agent tool-wrapping is predictable.** Tool-call JSON schemas, fences, keys, echoed context
— the scaffolding around an agent's text is fixed or repeats from history. The model only
generates the novel middle. So: deep-γ drafting (16–32 positions, ~0.1 ms/token draft cost)
verified in one cold-set pass → in-span bursts. E040's measured +7.4 ms/position makes deep
verification nearly free, and E043 made it correct and gated. What stands between us and the
burst numbers is now purely **drafter quality** (issue 2): the verify side is solved.

## KNOWN ISSUES — fix these first (next session)

1. ~~E041 known/hist alignment drift~~ **FIXED in E043 — it was two bugs**: (a) the partial-
   accept rollback discarded accepted rows' cumulative SSM/conv updates → re-feed carry fix;
   (b) the reader's cached-copy branch read `staged_cache` for pinned-head items whose bytes
   live in `pinned_cache` → garbage weights on any pass-0 re-entry (spec cycle 0 only; dense
   and multi never re-enter pass 0, which is why E038/E040 never saw it). One-line fix in
   `benchmarks/real_decode.cu` (reader memcpy source: `(in_pin ? pinned_cache :
   staged_cache) + it.res_off`). Full story + validation matrix: EXPERIMENTS.md E043.
2. **Drafter quality is the bottleneck (supersedes "deep γ untested").** Deep γ is now
   lossless (γ=16 validated), but PLD acceptance on honest greedy output is 3.8% (chat) /
   14.8% (agent trace) at γ=8 and *drops* at γ=16 — the drafter rarely proposes the true
   continuation, so deep passes amortize over rejects. Next lever: adaptive-γ / better
   draft selection (RESEARCH_PLAN deep-γ section), measured with `--expect-ids` ALWAYS ON.
3. **MTP head drafter (E041a) not integrated.** 1.68 GiB, fits in VRAM beside hot 7.25
   (γ=8); at γ=16 use hot ~6144. Measured 2.84 tokens/verify on this model (E036). Requires
   decoding the qwen35 nextn block (eh_proj/enorm/hnorm + dense block, own small KV).
4. **Summary metric bug**: verify mode's `cold_GiB_per_step` prints the TOTAL over all steps
   (incl. the warm-up); per-step cold = 6.17 GiB at hot 7424. Cosmetic.
5. **E040 soft-check regression risk**: the fixed check now skips prompt-region rows and uses
   an absolute gap — keep it honest (the old inverted-gap version excused everything, which
   is how bug-class "tie-excused validation vacuity" hid for two sessions).

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
  delta-net + conv state per cycle. KV overwrite is not sufficient. AND the rollback discards
  the accepted rows' own updates — re-feed them as the next pass's front rows (E043).
- **Cached-copy buffer aliasing**: the RAM cache is one logical region backed by TWO host
  buffers (pinned head + pageable tail); every read-back must select by `in_pin`, not
  hard-code one. The bug fires only on a pass-0 re-entry with populated caches — a path
  dense/multi never take (E043 bug 2).
- **Vacuous tie-excused checks**: `gap = lg[expected] - lg[argmax]` is ≤ 0 by construction —
  every mismatch becomes an excused "tie". Compute the absolute gap. Also: teacher-forced
  checks are only valid where the sequence is the model's own output — skip prompt rows.

## Reproduce the headline numbers

```
CUDA=/usr/local/cuda/bin
$CUDA/nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/real_decode.cu -o benchmarks/real_decode
M=/home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf
IDS="248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198"
# dense single-user (2.8 tok/s on this prompt) + reference ids for the lossless gate:
taskset -c 0-3 benchmarks/real_decode --model $M --prompt-ids "$IDS" \
  --gen-tokens 32 --ctx 256 --hot-mib 7424 --cache-mib 6144 --readers 4 --show-text 0 \
  --out-ids /tmp/expect27.txt
# speculative MTP+hybrid, LOSSLESS-GATED (exit 1 on any drift — ALWAYS use --expect-ids):
taskset -c 0-3 benchmarks/real_decode --model $M \
  --mtp /home/zabwie/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf --p-min 0.3 \
  --prompt-ids "$IDS" --speculate 16 --gen-tokens 32 --ctx 256 --hot-mib 8192 --cache-mib 5600 \
  --readers 4 --show-text 0 --expect-ids /tmp/expect27.txt
# agent-shaped trace (results/e043-agent-prompt.txt): dense reference first, then --speculate 8
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
E040: verify ceiling measured (corrected). E041/E042: speculative loop (7.26 tok/s claim —
RETRACTED in E043, measured on a corrupted stream). E043: two silent correctness bugs fixed
(SSM-rollback re-feed carry + pinned-cache aliasing), E040 soft check un-vacuumed,
`--expect-ids` lossless gate added; loop now proven lossless γ=8/16 × 4 configs. Full detail:
EXPERIMENTS.md.
