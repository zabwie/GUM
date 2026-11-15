# HANDOFF history — superseded status sections (E043–E149 era)

Extracted 2026-11-11 when HANDOFF.md was rewritten to current-state-only.
Each section was the top-of-file handoff at its time; EXPERIMENTS.md is the
authoritative evidence log. Names/paths reflect their era (pre GUM rebrand).

## Current status after E149 (superseded)

`scripts/run_single_user.sh` launches grouped native MTP with the packed
24-position verifier on the RTX 3060. The six-prompt exact suite reached
16.994–23.868 generated tok/s, up from the E127 15.201–21.367 range.
All 1,024 emitted token IDs matched. Including prefill, the suite reached
13.775–16.712 tok/s. It left at least 1032 MiB free during these runs.
Pass `--packed-verify 0` for the compact verifier or set
`VANTA_DRAFTER=eagle4` to use EAGLE4. Both paths retain their dense-reference
checks. See EXPERIMENTS.md E144–E149 and the matching raw logs.

The 40–60 generated tok/s goal remains open. At the current 8064 MiB of
resident target weights, E146 measured 5.242 GB moved per verifier pass at
24.42 GB/s. E149 cut average verification to 226.7 ms on async-queue chat
with 62 passes. Reaching 40 at the observed cycle time would require
about 9.7 generated tokens per pass. Keep the exact six-prompt gate and
compare candidate policy changes with the 24-position default.


## Historical status after E127

The active goal remains 40–60 generated tok/s in a normal single-user launch,
with exact target output. At E127, `scripts/run_single_user.sh` defaulted to the original
native MTP head at hot 8064/cache 5600, context capacity 1024. The six exact
checks covered 15.201–21.367 generated tok/s. The actual launcher repeated the
slower chat at 15.236 with 256/256 matching. Prefill took 1.950 s. Select the
E105 EAGLE4 proposal policy with `VANTA_DRAFTER=eagle4`. The unpinned
8064 MiB hot launch measured 22.013 agent, 19.408 Rust, 16.929 explanation tok/s,
all 128-token dense matches. The new event-log design prompt measured 16.619
tok/s and matched all 256 dense IDs at context capacity 1024. Reducing hot
residency to 7808 MiB gave 15.912 tok/s and left 1383 MiB free on that prompt.
See the E105 table in EXPERIMENTS.md for prompt-inclusive rates.

E102 returns exact full-vocabulary greedy IDs from the GPU and uses 16 MiB
weight slots. E089 introduced 32-position prompt chunks, final-row prompt
projection, deferred recurrent writes and best-first drafting. The verifier
streams about 5.242 GB at a measured 24.66 GB/s and takes about 223 ms. E105
uses a broader eight-candidate proposal list. The target model is unchanged.
The prepared draft GGUF SHA-256 is
`16387576a6254f86d31f83b44c1066d5532933876e99169a115d35a330f150d3`.

E124 installs best-first native MTP trees with the original head and the shared
history continuation logic. This gives 22.235 agent, 17.497 Rust and 16.954
explanation generated tok/s, all exact. Two true chat-template prompts returned
15.570 and 17.082, with 256/256 and 128/128 dense matches. `--mtp-tree 0`
reproduces the linear path. EAGLE4 matched the agent reference at 21.941 after
the tree builder refactor.

The E106–E123 trials are recorded in EXPERIMENTS.md and `results/`. E118's
exact early features and E123's fitted confidence improved offline metrics
without a throughput gain. E120's two-head tree reduced pass counts but had
mixed throughput. E121's recycled predictions and E122's early transfer had
no broad gain. Packed 24-position E116 reached 15.366 on the first chat prompt
and slowed several others. E126 measured width at every dense prefix. A wider
tree could recover nine passes on the first chat before proposals saturated.
E127 enabled wider native prompt evaluation and kept the CPU draft selector.
Keep the GPU workload serial and recheck the full exact-output gate.


## Historical acceptance result after E076

The E076 user request to raise agent-trace MTP acceptance above 90% is met for the checked
workload: the new MTP-only default (`p_min=0.97`, γ=8) accepted 61/63 proposals (96.8%) and
matched all 128 dense-reference tokens exactly in two runs. End-to-end throughput was 5.77–5.94
tok/s; dense was 3.59 tok/s. This result is workload-specific, not a guarantee for arbitrary
prompts.

The prior 34% metric combined drafter sources: a 64-token replay separated MTP at 7/15 (46.7%)
and PLD at 46/141 (32.6%). With `--pld 1 --p-min 0.3`, the full 128-token fast hybrid passed the
same dense gate at 7.397 end-to-end tok/s, but acceptance was 30.8%. Use this as the
throughput-oriented mode when the 90% acceptance target is not required. The corrected timer now
includes MTP state sync and draft generation; older headline rates measured only the verifier.

MTP draft generation is skipped when the next cycle will certainly use a long PLD match. This
preserves the MTP state sync and reduced MTP time from 33.9 to 16.9 ms/cycle in a matched 64-token
replay, improving end-to-end throughput from 8.518 to 8.826 tok/s. The main remaining verifier
cost is `k_mat_q4` (about 80% of profiled kernel time); the existing shared tile gave about 7.5%
on the matched E072/E073 profile. The 20–25 tok/s projection remains unachieved.

See EXPERIMENTS.md E076 and the `results/e076-*` files. Keep `--expect-ids` enabled. The default
MTP settings can be overridden with `--p-min` and `--pld 1`.


## Historical performance profile through E075 (verifier-only rate)

The user asked to fix the measured GEMV and delta-net bottlenecks and raise throughput to 20–25
tok/s. **The target is not met.** Current source retains four changes: transposed delta-net state
`[head][i][j]`; a persisting-L2 window over the activation arena; packed `int2` scale/correction
metadata; and a shared activation tile for NB=16, K>=5120. E071 passed the 27B greedy token gate
(32 compared, zero mismatches) at 9.629 reported tok/s (γ=16, p_min=0.3, agent trace, hot
8192/cache 5600). A repeated run under lower VRAM headroom measured 8.799 tok/s, so keep the
throughput claim tied to the individual run and see the matched E072/E073 profile for kernel
comparisons.

Delta-net integration reduced the cycle from 0.4877 to 0.3873 s and raised the decoder metric
from 6.990 to 8.803 tok/s; the L2 window raised it again to 9.283. The matched Nsight Systems
profile shows the shared GEMV tile lowered summed GEMV time about 7.5% and mean cycle from 0.3609
to 0.3534 s in its 32-token control comparison, while GEMV still accounted for about 80% of kernel
time. Single-run 9.629 is above the E064 9.350 run, but throughput is variable and the 20–25 target
remains open. Draft width and confidence sweeps stayed around 7.3–9.2 tok/s. See EXPERIMENTS.md
E051–E075 and the E072/E073 traces.


## Historical performance ledger (27B, single-user, lossless-gated — pre E050)

| path | tok/s | vs dense |
|---|---|---|
| dense (8704/5150, no NVMe tier) | 3.53 | 1x |
| spec PLD-only g=8 | 2.9-3.3 | ~1x |
| spec MTP g=16, chat prompt (78% accept) | 9.3-10.0 | 3.4x |
| spec hybrid g=16, chat (PLD rarely fires) | 9.22 | 3.3x |
| spec hybrid g=16, agent trace (34-42% accept) | 5.79 | 1.7x |
| verify bound g=8, hot 8192 (E047 re-measure) | 20.4 | — |
| verify bound deep g=32 (curve extrapolation) | ~50 | — |

Latest exact-gated run: E071 at 9.629 reported tok/s on the agent trace, with an 8.799 repeat under
lower VRAM headroom. Use E072/E073 for a same-config profiler comparison; see Current status after
E075 for the measured range and caveats.

**Warning: the old "7.26 tok/s chat at 33% acceptance" was measured on a corrupted stream** (E043
bug 1: the state divergence made the model echo its own repeats, which the n-gram drafter
then "predicted"). With correct state, PLD drafting quality — not verification — is the
bottleneck. The verify machinery is now correct and gated; the upside is in drafter quality
(adaptive-γ, RESEARCH_PLAN).


## The headline opportunity, as framed pre-E150

**Agent tool-wrapping can be predictable.** Tool-call JSON schemas, fences, keys, and echoed
context can create long exact history matches. Deep-γ drafting can exploit those spans, but the
current agent trace still reaches only 33.8% acceptance. E071 measured 9.629 tok/s in one run; the
NB=16 GEMV remains the largest kernel cost, so both GEMV efficiency and draft quality must improve
to reach the requested 20–25 tok/s.


## Current bottlenecks (status after E075)

1. **k_mat_q4 remains the main cost after a partial optimization:** E050 measured 271 ms/cycle at
   NB=16. The retained shared tile improves the matched profile's summed GEMV time by about 7.5%,
   and is bit-exact, but GEMV still accounts for about 80% of profiled kernel time. Other tile,
   vector-warps, cache, and WMMA candidates lost or were inconclusive. Do not treat E050's projected
   70–110 ms/cycle as achieved.
2. **k_delta_net_multi was changed and passed:** state is `[head][i][j]`, with adjacent lanes
   reading adjacent `j`. E054's exact-gated run improved 6.990→8.803 reported tok/s. Preserve
   each `j`'s serial accumulation order, the `h % ngroup` mapping, and E043 snapshot/rollback.
3. **Measured throughput remains below target:** E071 reached 9.629 tok/s in one run with 33.8%
   acceptance; a repeat under lower VRAM headroom reached 8.799.
   p_min 0.1/0.5/0.8 and γ=8/32 did not improve the result. E050's 13–15 tok/s projection
   assumed a much larger GEMV speedup; the matched profile shows about 7.5% lower summed GEMV time.
   The 20–25 tok/s goal remains open.
4. **After the GEMV cost falls, cold-copy time and accepted tokens/pass become the next limits.**
   E050 measured 25.5 GB/s copy traffic at the PCIe peak, but that copy-only estimate is not the
   current cycle bound until GEMV is fixed.
5. **k_attn_decode is O(context) serial per head** with half the threads idle (hd=128,
   256 threads) — 6.4 ms/cycle in E050; lower priority until GEMV is addressed.
6. **Keep the E040 soft check honest:** prompt-region rows are skipped, use the absolute gap and
   garbage detector. Never simplify it back.


## Session history (one line each, E001–E043)

E001-E030: memory mechanics (mapped/managed/VMM/tiered pool). E031: streamed proxy 1.0 t/s.
E032-E034: overlap + dp4a + batching → 43.8 agg. E035-E037: IOMMU removal → 2× bandwidth.
E036: real llama.cpp MTP baseline (1.04 t/s, thrashing diagnosis). E038: real decode loop,
3.39 tok/s, llama.cpp-validated. E039: batched decode, 13.17 agg, RMW-race class discovered.
E040: verify ceiling measured (corrected). E041/E042: speculative loop (7.26 tok/s claim —
RETRACTED in E043, measured on a corrupted stream). E043: two silent correctness bugs fixed
(SSM-rollback re-feed carry + pinned-cache aliasing), E040 soft check un-vacuumed,
`--expect-ids` lossless gate added; loop now proven lossless γ=8/16 × 4 configs. Full detail:
EXPERIMENTS.md.
