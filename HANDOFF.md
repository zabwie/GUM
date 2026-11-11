# GUM — Session Handoff (2026-11-11, post-E165)

Repo: `/home/zabwie/gpu-unified-memory-research` (git; latest see `git log -1`).
Read [AGENT.md](AGENT.md) first for the operating mentality. **[docs/RESEARCH_FUNNEL.md](docs/RESEARCH_FUNNEL.md)
is the working protocol** — decision-driven funnel, two tracks, scored backlog; every experiment
enters through it. [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) is the strategy record;
[EXPERIMENTS.md](EXPERIMENTS.md) the evidence; this file is machine state and immediate next moves.
Superseded status sections (E043–E149 era) live in [docs/archive/HANDOFF_HISTORY.md](docs/archive/HANDOFF_HISTORY.md).

## Mission

GUM (GPU Unified Memory): system RAM as a designed tier of VRAM for big-model inference on
small GPUs — focused on **single-user tok/s**, losslessly. Owner directives: no quality loss,
open source, system changes OK, ambitious (40 tok/s), CPU layer offload excluded, 22 GB card
possibly coming (don't optimize for it).

## Current baseline (this is "now")

| Workload | Now |
|---|---|
| Normal chat | ~17–18 tok/s |
| Agent-style prompts (repetitive text) | ~24 tok/s |
| Ceiling with all remaining validated levers (better residency, tree tuning, sync trim) | ~20–25 chat / ~27 agent |

Measured reference points behind those numbers: the installed six-prompt exact suite runs
16.99–23.87 generated tok/s (agent trace 23.87, chats 16.99–18.58; all 1,024 token IDs exact,
E149). Dense decode is 3.59–3.73 tok/s. llama.cpp's own over-VRAM mode on this model/hardware:
0.63 tok/s (1.04 with its MTP head). Everything is lossless — greedy token IDs match the exact
Qwen3.8-27B Q4_0 checkpoint.

## State after E165 — C-track paused (owner, 2026-11-11)

**Pause state:** C1 Stage S produced a validated screening harness (byte-identical identity
control, `scripts/c1_requant.py`) and two sensitivity points — (3,64): 27.8% byte cut at ~81%
greedy agreement; (2,64): 50% cut at ~54%. The owner reviewed the signal and paused the
TOLERANCE track, holding the no-quality-loss line for the production engine. Do not advance C1
to Stage V or knee-mapping without an explicit owner decision to reopen; the harness and data
are preserved, so reopening costs one session.

**The two execution contracts stand** (owner decision, E163): EXACT (the installed Q4_0
launcher) is the permanent correctness oracle and regression baseline; TOLERANCE is paused as
a research track. Gates and rules: [docs/RESEARCH_FUNNEL.md](docs/RESEARCH_FUNNEL.md) §7.

**Highest-scored open exact-compatible item: agent/PLD deep-γ (A3, ~27 tok/s ceiling).**
The 40 tok/s goal stands with the GUM Adaptive product direction (funnel §7).

## What remains, in priority order

1. **Agent-shaped workloads (the owner's original insight).** 40 tok/s on PLD-exploitable
   spans is closer than on novel chat: E149 measures 23.9 with a ~27 ceiling for the current
   drafter. Deep-γ PLD bursts on exact spans remain mechanically available.
2. **Transfer-side complement:** 8448/ctx-384 (E156a, validated short-sequence config),
   sync-stall trim (~13.6 ms/cycle, E160/E163), tree width (~13% of passes, E126 — partially
   falsified by E150/E151). Combined known-lever upper bound: ~21 chat / ~27 agent.
3. **EAGLE-scale drafter training** is the only measured-physical route to >5 emitted/cycle
   on novel chat and needs ~100× more capture volume than the batched pipeline
   (E159, ~12.8 positions/s) produces in reasonable time. The drafter family is otherwise
   exhausted: five independent probes (E153/E157/E158×2/E159) agree no smaller drafter beats
   the authored MTP head at our data volume.
4. **Closed paths — do not reopen without new physics:** compression (void, ~5.5% ceiling,
   E161), zero-copy mapping (10–14% slower cycles, E162), GPU-decompression codecs (not
   built), sparsity byte-skipping (B1, negative).

Established bounds with citations live in [docs/RESEARCH_FUNNEL.md](docs/RESEARCH_FUNNEL.md) §2 —
cite them, do not re-derive them.

## What exists and is VALIDATED (trust it)

- **Real decode loop** `benchmarks/real_decode.cu` for qwen3 (Qwen3-0.6B) and qwen35
  (Qwen3.8-27B hybrid: 48 gated-delta-net + 16 GQA layers) — GGUF parsing, dp4a Q4_0 GEMV
  with int8 activation quant, per-head norms, RoPE (NEOX + IMROPE-as-half-split), GQA
  attention with KV, gated-delta-net with persistent SSM state, greedy sampling.
  **Token-for-token identical to llama.cpp CPU greedy** (24/24 on the 0.6B; the 27B likewise
  after per-op verification against a patched out-of-tree llama.cpp at /tmp/llama-dbg).
- **Installed single-user launcher** `scripts/run_single_user.sh`: grouped native MTP with
  the packed 24-position verifier, context capacity 1024, 8064 MiB resident target weights.
  This is the E149 baseline — keep the six-prompt exact gate for any policy change.
  Optional EAGLE4 drafter via `GUM_DRAFTER=eagle4`. Short-sequence 8448/ctx-384 mode via
  `GUM_RESIDENT=1` (E156a-validated; caps prompt+generation at 384 tokens).
- **Tiered residency**: hot VRAM arena + RAM cache (pinned head) + NVMe only on token 0 +
  copy/compute overlap + slot events.
- **Speculative loop** (`--speculate γ`): PLD + native MTP drafter → one verify pass →
  accept prefix → correction. Lossless (greedy rejection sampling) — **gated by
  `--expect-ids <file>`**, which diffs the spec stream against a dense `--out-ids` reference
  and exits 1 on drift. SSM/conv snapshot-rollback + re-feed carry (~150 MB D2D/cycle) is
  REQUIRED: rejected positions corrupt cumulative delta-net state (KV overwrite alone is not
  enough — a genuine contribution for hybrid models).
- **Batched decode** (`--batch B`) and **multi-position verify** (`--multi B --feed-ids`),
  both exact-gated.
- **CPU fp32 reference decoder** (`--validate 1`) for correctness gates at 0.6B scale.
- Correctness is the gate: every mode is checked against a CPU fp32 reference and/or the
  validated token stream before its numbers count. See [docs/DESIGN.md](docs/DESIGN.md) §6.

## Historic correctness notes — keep these guards

1. E041's "known/hist alignment drift" was **two bugs, fixed in E043**: (a) the partial-accept
   rollback discarded accepted rows' cumulative SSM/conv updates → re-feed carry fix; (b) the
   reader's cached-copy branch read `staged_cache` for pinned-head items whose bytes live in
   `pinned_cache` → garbage weights on any pass-0 re-entry. One-line fix in
   `benchmarks/real_decode.cu` (reader memcpy source: `(in_pin ? pinned_cache : staged_cache) + it.res_off`).
2. **The old "7.26 tok/s chat at 33% acceptance" was measured on a corrupted stream** (E043
   bug 1: state divergence made the model echo its own repeats, which the n-gram drafter then
   "predicted"). With correct state, drafter quality — not verification — limits tokens/pass.
3. **E040 soft-check regression risk**: the fixed check skips prompt-region rows and uses an
   absolute gap — keep it honest (the old inverted-gap version excused everything, which is
   how "tie-excused validation vacuity" hid for two sessions).
4. Keep `--expect-ids` enabled on every drafter/kernel change.

## Bug classes to check in ANY new batched/multi kernel

- **Flat-index RMW race**: grid (x, B) with `i = blockIdx.x*blockDim.x + threadIdx.x` makes
  every blockIdx.y block touch the same indices — nondeterministic lost updates, invisible to
  all sanitizers and to batch-1 tests. Use global index `(by*gridDim.x+bx)*blockDim.x+tid`.
- **Per-sequence stride errors**: dense [B][N] vs buffer-pitch confusion (max_rows vs N).
- **g_batch ordering**: g_batch must be set AFTER multi/speculate mode adjustments, or the
  GEMV template silently computes only row 0 (E040's regression).
- **Tie-excused validation vacuity**: a mismatch check with a tie threshold will excuse
  all-zero garbage rows. Always pair with a row-energy garbage detector; compute the absolute
  logit gap, not `lg[expected] - lg[argmax]` (≤ 0 by construction).
- **Teacher-forced checks** are only valid where the sequence is the model's own output —
  skip prompt rows.
- **Stream discipline**: KV appends and anything reading kernel output belong on the compute
  stream (stc). Copy stream (st) is for weight staging only.
- **SSM rollback**: any speculative scheme on hybrid models MUST snapshot/restore the
  delta-net + conv state per cycle, and the rollback discards the accepted rows' own updates —
  re-feed them as the next pass's front rows (E043).
- **Cached-copy buffer aliasing**: the RAM cache is one logical region backed by TWO host
  buffers (pinned head + pageable tail); every read-back must select by `in_pin`, not
  hard-code one. Fires only on pass-0 re-entry with populated caches — a path dense/multi
  never take (E043 bug 2).

## Machine

RTX 3060 12 GiB (CC 8.6, 28 SMs), i7-12700K, 31 GiB RAM, PCIe 4.0 x16 @ 24.6 GB/s H2D pinned
(`iommu=pt` applied — the one approved kernel change). RLIMIT_MEMLOCK 3978 MiB. CUDA 13.2 at
`/usr/local/cuda/bin`. Kernel 7.0.0-38. Desktop shares the GPU; benchmarks self-gate on
RAM/swap floors. Details: [docs/HARDWARE.md](docs/HARDWARE.md).

## Reproduce

```sh
# installed launcher (the baseline numbers):
scripts/run_single_user.sh --prompt "Your prompt"
# GUM_RESIDENT=1 for the validated 8448/ctx-384 short-sequence mode
# GUM_DRAFTER=eagle4 for the optional EAGLE4 drafter

# experiment decoder build:
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/real_decode.cu \
  -o /tmp/gum-decode
```

Model: `/home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf` (13.42 GiB).
`--prompt-ids` takes the rendered chat template's exact token IDs (preferred over the
approximate `--prompt` tokenizer). Every speculative run MUST pass `--expect-ids` against a
dense `--out-ids` reference. Per-experiment commands and raw result paths are in
[EXPERIMENTS.md](EXPERIMENTS.md). Do not touch the user's llama.cpp checkout (an
out-of-tree instrumented copy lives at /tmp/llama-dbg, rebuildable per
[docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)).

## Repo map

- `benchmarks/real_decode.cu` — the decoder (all modes: dense/batch/multi/speculate/MTP/EAGLE4).
- `benchmarks/microbench/` — per-experiment kernel probes (named `eXXX-*`).
- `benchmarks/*.cu` — early memory-mechanics benchmarks (E001–E037 era).
- `results/` — raw evidence, one file per run, named `eXXX-*`; indexed by EXPERIMENTS.md.
- `scripts/` — launcher, capture/training/probe scripts, resource gate.
- `models/` — drafter checkpoints + EAGLE4 draft GGUF (all gitignored; see .gitignore).
- `docs/` — current strategy/protocol/design; `docs/archive/` — frozen study docs and old handoffs.
- Env vars were renamed `VANTA_*` → `GUM_*` on 2026-11-11 (GUM_TARGET_MODEL, GUM_MTP_MODEL,
  GUM_EAGLE_MODEL, GUM_DRAFTER, GUM_RESIDENT, GUM_PACKED_DEFAULT). Historical /tmp artifacts
  (e.g. /tmp/vanta-e158) keep their original names.
