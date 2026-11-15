# GUM — Session Handoff (2026-11-13, **pure-streaming era**)

> **STOP — READ THIS FIRST IF YOU ARE AN AI OR HUMAN PICKING UP THIS REPO.**
> The installed configuration is **pure streaming** (`GUM_PROFILE=stream`): hot weight tier
> **0 MiB**, every weight byte streams from pinned RAM over PCIe each pass. This is the
> owner's chosen daily driver and the baseline for ALL current work.
>
> **Self-check before quoting any tok/s:** run `curl -s http://127.0.0.1:8765/health` —
> `profile` must say `stream`. Measured reference points (frozen suite, mid_pond probe):
>
> | profile | engine VRAM | desktop free | decode tok/s | claim check |
> |---|---|---|---|---|
> | **stream** (default) | ~3.0 GiB | ~9 GiB | **7.5** (full-pin, calm desktop; frozen 5.5 predates full-pin) | if you measure 7–8, you are correct |
> | **GUM Adaptive (E183, live)** | 0–6.4 GiB hot, auto | ≥4.6 GiB reserved | **9.8–13.7** at equilibrium, 7.5 floor under pressure | /health reports live `hot_mib` |
> | nano | ~4.6 GiB | ~7.1 GiB | 6.27 | |
> | lean | ~6.1 GiB | ~5.3 GiB | 7.19 | |
> | chat | ~10.2 GiB | ~2.1 GiB | 10.79 | |
> | 196k (E181 long-output) | ~10.4 GiB | ~1.5 GiB | workload-dependent | |
>
> **E182/E183 (2026-11-13):** hot-tier residency is token-exact and linear at
> 42–43 ms verify time per hot GiB (+0.72 tok/s per GiB at the standard fixture). The
> stream profile is now link-saturated (verify 0.575 s ≈ cold bytes ÷ ~25 GB/s;
> non-fetch overhead 4.3% of the cycle — the sync-stall lever is spent). GUM Adaptive
> respawns the idle decoder with the hot tier the free VRAM affords and shrinks back
> under desktop pressure; see EXPERIMENTS.md E182/E183 before touching residency code.
>
> **Historical numbers (17–24 tok/s, E149/E177 era) were DIFFERENT configurations**
> (8 GiB hot tier, 2K ctx bench contexts). If you measure 2–6 tok/s while believing you
> should see 17+, you are on a mismatched profile — check `--hot-mib` and `--ctx` before
> "diagnosing a regression". The 21→5 tok/s incident happened this way.
>
> Measured walls (do not re-derive, see [docs/SYSTEM_MAP.md](docs/SYSTEM_MAP.md)):
> DMA 24.7 GB/s (link, triple-measured), acceptance chain p₁ 0.79 → emitted/cycle 3.7
> (= p₁/(1−p₁), closes exactly), prefill ~7 TFLOP/s effective. Anomaly ledger A1–A9 in
> SYSTEM_MAP; next experiments: A8 shortlist coverage, A6 L2 persisting window (1.31 MB
> vs 1.6 MiB cap), A7 drafter-KV audit.

> **Latest long-context result — E196 (2026-11-15):** on the RTX 3060 12 GiB,
> Qwen3.8-27B Q4_0 processed the same 261,888-token test prompt, assembled from repeated
> frozen prose segments, and produced the same 256 greedy token IDs as E195. At 262,144
> context, prefill was **46.41 tok/s** and decode was **3.258 tok/s**, up from 8.99
> and 0.761. Decoder peak remained **4,308 MiB**. Total
> device use peaked at 6,250 MiB, including 1,942 MiB outside the decoder.
> This custom 256K run used zero hot weights, a 10 GiB host cache, and NVMe for the
> uncached 2.76 GiB of target weights per verifier pass. It exercises the `stream`
> weight path at a larger context and is a separate benchmark from the live service's
> standard context rates. [Full E196 record and paging path](results/e196/README.md).

Repo: `$HOME/gpu-unified-memory-research` (git; latest see `git log -1`).
Read [AGENT.md](AGENT.md) first for the operating mentality. **[docs/RESEARCH_FUNNEL.md](docs/RESEARCH_FUNNEL.md)
is the working protocol** — decision-driven funnel, two tracks, scored backlog; every experiment
enters through it. [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) is the strategy record;
[EXPERIMENTS.md](EXPERIMENTS.md) the evidence; this file is machine state and immediate next moves.
Superseded status sections (E043–E149 era) live in [docs/archive/HANDOFF_HISTORY.md](docs/archive/HANDOFF_HISTORY.md).

## Historical context (pre-streaming era, for orientation only)

E177 measures unhinted native MTP on two frozen task suites. The 8064 MiB
resident current-source baseline delivered 17.56 and 18.33 committed decode
tok/s, including drafting and state repair. The merged final prefill pass
reduces first-token latency and preserves the exact output stream. At the
measured cycle cost, reaching 40 decode tok/s requires about 10–11 committed
tokens per weight pass against 4.73–4.93 observed. The raw logs, Nsight
trace, reproduction commands and rejected draft changes are in the
[E177 ledger](results/e177-general-20261112/README.md).

The installed decoder supports `--prefill-chunks 8` with the E171 packed
MTP session settings. It processes up to 256 prompt IDs per target weight
pass. Use `--session-prefix-tokens N` when separate requests share a
known 32-aligned initial context. The decoder checks those IDs before
reusing the saved recurrent state and live KV.

E175 lets a client append draft tokens to a packed native MTP request with
`--session-draft-hints 1`. The target can verify them during the final
prompt pass. On the E170 fixture follow-up, a draft constructed from tool
output reduced two prefill and five decode passes to a single target pass.
This path serves structured requests with a usable candidate. The ordinary
MTP generation rate remains the relevant baseline on open-ended requests.

See [E173](EXPERIMENTS.md#e173--wider-exact-prefill) and
[E174](EXPERIMENTS.md#e174--reuse-shared-context-across-separate-requests) and
[E175](EXPERIMENTS.md#e175--verify-tool-drafts-during-prefill)
for the current launch settings and exact-gated timing records.

## Mission

GUM (GPU Unified Memory): system RAM as a designed tier of VRAM for big-model inference on
small GPUs — focused on **single-user tok/s**, losslessly. Owner directives: no quality loss,
open source, system changes OK, ambitious (40 tok/s), CPU layer offload excluded, 22 GB card
possibly coming (don't optimize for it).

## E167 baseline for legacy workload comparisons

| Workload | Now |
|---|---|
| Normal chat | ~16–17 tok/s (chat-neutral vs E149 by same-state control) |
| Agent-style prompts (repetitive text) | ~21–23 tok/s |
| Verbatim-repeat / long exact spans | **36.9 tok/s measured, lossless (E167 deep-γ)** |
| Ceiling with all remaining validated levers | deep-γ extends the agent ceiling past ~27 on exact spans; ~20–25 chat |

E167 (deep-γ PLD) is installed: packed default γ=64 with an adaptive direct-chain
override (chains > tree budget verify as one linear pass, gated on direct-chain
acceptance ≥0.75, bounded cold probe). `--speculate 24 --packed-verify 1` preserves
the E149 configuration byte-for-byte. All six exact-gate prompts lossless; prefill is
pinned at 32 rows (64-row prefill flips near-tie argmaxes). `draft_hist` in every
spec summary logs the per-cycle draft-length distribution. Machine-state caveat:
desktop VRAM pressure inflated all E167 pass times ~5–15 ms at both γ=24 and γ=64 —
re-baseline absolute tok/s on a calm desktop. Measured reference points: six-prompt
exact suite E167 runs 15.78–20.86 tok/s lossless (see EXPERIMENTS.md E167 for the
E149 comparison and controls); dense decode 3.59–3.73 tok/s; llama.cpp over-VRAM mode
0.63 tok/s. Background: E167b capture pipeline (`scripts/capture_when_idle.sh`,
hourly idle-gated automation, `touch capture/STOP` to halt) is accumulating
EAGLE-scale drafter training data; pilot metrics in EXPERIMENTS.md E167b.

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

Model: `$HOME/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf` (13.42 GiB).
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
