# GUM Research Plan — single-user decode acceleration

> **2026-11-11: the working protocol is now RESEARCH_FUNNEL.md** (decision-
> driven funnel, two tracks, scored backlog). This document remains the
> strategy record and measured-constants reference; new experiments enter
> through the funnel.

Updated 2026-11-10 (E039). Written to be self-contained: a new engineer, session, or AI model
should be able to resume from this document alone. Cross-references: HANDOFF.md (current state),
EXPERIMENTS.md (E001-E039 raw evidence), NEXT_STEPS.md (task ladder). This supersedes the
original 2026-11-08 research plan ("find the best reproducible way for local inference to use
system RAM with the RTX 3060 while keeping neural-network computation on the GPU") — that goal
is achieved and validated; the goal is now single-user speed.

---

## 1. Mission (owner's directives, 2026-11-10)

- **Current delivery target.** The owner's latest direction is 40–60 generated tokens per
  second in the normal single-user launch, with the same target model and greedy token
  stream. More throughput is the reason to improve drafting. Acceptance percentage by
  itself is a diagnostic. Prefer leaving more VRAM free where throughput permits.
- **Proof for this target.** Time one complete generated response on this RTX 3060,
  including drafting, verification, and state repair. Check every emitted token against
  a dense run of the unchanged target and report prompt evaluation separately. Repeat on
  agent, coding, and non-repetitive prompts before claiming a normal-use rate. The
  current verified checkpoint is E149 in EXPERIMENTS.md. The six-prompt
  single-user baseline is 16.99–23.87 generated tok/s with 1,024 exact IDs.
- **Single-user latency is the product.** Batch throughput matters only as a building block.
- **Open source.** Others will run this on their own hardware; avoid anything requiring our
  specific machine beyond documented, owner-approved system config.
- **No quality loss.** The target model is already Q4; output must remain the exact target
  model's distribution. Techniques that alter logits are rejected *as replacements* but are
  acceptable *inside a draft path* (they only affect acceptance rate, never the output).
- **System changes approved** (memlock raise pending; iommu=pt already applied).
- **Optimize for the RTX 3060 in this machine.** E077 surpassed the older 13–15 tok/s
  estimate on one trace by overlapping PCIe transfers with computation. E089 averages about 0.25 s per cycle, so 40 tok/s requires about 10 committed
  tokens per cycle at that cost. Improve both
  terms and verify the resulting implementation on this machine.
- **CPU layer offload is out of scope** (founding directive): the math stays on the GPU.
  CPU execution of cold layers is documented as a fallback (§4.5) but not on the roadmap.

## 2. Measured baseline (all verified, see EXPERIMENTS.md)

| Quantity | Value | Source |
|---|---|---|
| Model | Qwen3.8-27B hybrid (48 gated-delta-net + 16 GQA layers), Q4_0, 13.42 GiB | E031 |
| VRAM hot arena | 7.25-8.0 GiB weights resident | E038/E039 |
| RAM cache | 5.4-6.1 GiB (3658 MiB pinned head, memlock-capped) | E034 |
| PCIe Gen4 x16 | 24.6 GB/s H2D pinned (iommu=pt) | E037 |
| Single-user decode | **3.39 tok/s** (0.295 s/token), token-validated vs llama.cpp | E038 |
| Cold set per token | 5.42 GiB (hot 8192) / 6.17 GiB (hot 7424) | E038/E039 |
| Batch-8 serving | 13.17 aggregate tok/s, 0.772 GiB cold/token, step 0.607 s | E039 |
| Batch-8 compute cost | ~0.35 s GPU compute for 8 rows (~35-44 ms/row) — **the number most external estimates get wrong** | E039 |
| MTP head acceptance on this model | 2.84 tokens/verify (23% draft acceptance) | E036 |
| MTP head size | 1.68 GiB Q4 — fits wholly in VRAM beside the hot arena | E036 |
| Correctness gate | every experiment must reproduce the validated token stream | E038 |

**The governing equation:** single-user tok/s ≈ 24.6 GB/s ÷ cold-GiB-per-token. Baseline
5.42 GiB → 3.4 tok/s (86% link saturation). Every speedup must reduce effective GiB per
*accepted* token or raise delivered bandwidth. Bandwidth itself has no headroom left.

## 3. Verified building blocks (what exists and is trusted)

- Real decode graph for qwen3 + qwen35 (attention/KV + gated-delta-net/SSM), greedy sampling,
  token-validated against llama.cpp (E038).
- Tiered residency: hot arena + pinned RAM cache + overlapped H2D + slot events (E031-E037).
- Batched decode machinery: NB-template GEMV, per-sequence states, global-index batched
  kernels (E039) — **the verify pass for γ+1 draft positions is this same machinery** with
  rows belonging to one sequence.
- CPU fp32 reference decoder, mmap-backed, for correctness gates (E038).
- Known-fixed bug classes: stream races, per-sequence stride errors, flat-index RMW races
  (see EXPERIMENTS.md E039 — check any new batched kernel against these).

## 4. Candidate directions, corrected projections (our constants, not paper constants)

### 4.1 Speculative decoding: VRAM-resident MTP head + batched verify — PRIMARY
- Draft γ tokens with the 1.68 GiB MTP head entirely in VRAM (~8-10 ms/draft token), then
  verify all γ+1 positions in ONE pass: cold set streamed once, GEMV→GEMM over γ+1 rows.
- Lossless (rejection sampling against exact Q4 logits). E[N] = 2.84 accepted/verify (measured).
- Corrected projection (measured compute ~35-44 ms/row, not the 8 ms external estimates):
  T_iter ≈ 35 ms draft + 250 ms copy + ~180-250 ms verify-compute + sync ≈ 0.50-0.55 s
  → **6.3-7.3 tok/s first cut**.
- Path to 10-13: cut per-row compute (tensor-core int8 MMQ kernel, fewer kernel launches,
  deeper copy/compute overlap) and raise acceptance (PLD + MTP draft fusion, adaptive γ).
- VRAM budget: hot 7.25 + head 1.68 + states ≈ 0.2 + floor 2.0 = 11.1 ≤ 11.9 ✓ (trim hot to
  7.0 if needed; cold set grows to ~6.5 GiB — accounting included).

### 4.2 Prompt Lookup Decoding (PLD) — zero-VRAM, lossless, immediate
- n-gram lookup over the context drafts repetitive spans (code, RAG, editing, structured
  output). No drafter weights at all. Verify with the same machinery as 4.1.
- Workload-dependent acceptance; general chat may see little gain. Implement alongside 4.1
  as the fallback drafter and for structured workloads.

### 4.3 SubSpec-style substitute drafting — training-free acceptance booster
- Keep ~2-bit substitutes of the COLD layers in VRAM (~6.2 GB × 2/4.5 bpw ≈ 1.4-1.6 GiB);
  draft runs the full layer structure through the substitutes at zero PCIe cost; the exact
  Q4 weights verify. Lossless (draft-only approximation). Training-free, which matters for
  open-source release (EAGLE heads need training runs on each model).
- VRAM budget conflicts with 4.1's head (7.25 + 1.6 + 2.0 ≈ 10.85 ✓ but both drafter and
  substitutes together don't fit). Choose after measuring 4.1's acceptance.
- Novelty: our delta-net layers make the substitute-draft design nontrivial (recurrent layers
  need state-compatible substitutes) — a genuine research contribution if it works.

### 4.4 Compute-side optimization — multiplies everything
- Measured compute is ~35-44 ms/row; the dp4a GEMV is far from tensor-core throughput.
- Roadmap: tensor-core int8 MMQ kernel (E033 item), kernel-launch reduction (CUDA Graphs or
  fused per-layer sequences), deeper copy/compute overlap (the pending-window is 3 items deep;
  E032 showed +22% from overlap alone). Target: halve per-row compute → verify pass ~0.35 s
  → 4.1 reaches ~8-10 tok/s at measured acceptance, ~13 at acceptance 3.5+.

### 4.5 CPU execution of cold layers — documented, deferred
- External estimate claims ~2× vs PCIe (115-125 ms). Corrected with real Q4 CPU GEMV
  throughput (~140 GOPS effective on 12700K): cold layers ≈ 180-220 ms — marginal vs 252 ms,
  and it violates the founding no-CPU-offload directive. Revisit only if 4.1-4.4 stall.

### 4.6 Rejected / conditional
- Dual-precision cold shelf replacing Q4 (alters logits) — rejected per owner directive;
  its 2-bit mechanism survives only inside 4.3 (draft-only).
- SpAx sparse transfers over PCIe — fragmentation kills effective bandwidth (<5 GB/s);
  sparse compute belongs on-CPU (see 4.5) or inside draft substitutes (4.3). Deferred.
- More bandwidth hunting — exhausted (E035-E037).

## 5. Honest ceilings on this hardware (lossless, single-user)

**E040 MEASURED the verification ceiling (2026-11-10, CORRECTED same day).**
Multi-position verify, one sequence, cold set streamed once per step, teacher-forced tokens.
First run had a silent regression: g_batch was set before the multi-mode adjustment, so the
GEMV template stayed <1> and rows 1..B-1 were never computed — their all-zero logits were
excused by the tie threshold (gap 0 ≤ 0.25), making the checks vacuous. Caught while building
E041's drafter (spec-mode row argmaxes printed 0); fixed by reordering g_batch and adding an
all-zero-logits garbage detector. CORRECTED sweep (GEMV computes every row, garbage detector
active, 130/130 teacher checks meaningful and passing):

| positions/step | mean step | tok/s (upper bound) | marginal cost/position |
|---|---|---|---|
| 1 | 0.350 s | 2.86 | — |
| 2 | 0.362 s | 5.52 | +13 ms |
| 4 | 0.386 s | 10.37 | +8 ms |
| 8 | 0.430 s | **18.61** | **+7.4 ms** |

Amortization confirmed with real compute: the cable transfer dominates; per-position marginal
cost ~7-13 ms. Lesson recorded: tie-excused validation checks must include a row-energy
garbage detector, and any new mode must be checked against g_batch ordering. Consequences:

| Configuration | Ceiling |
|---|---|
| Dense streaming (today) | ~4.4 tok/s (measured 3.39; verify-path 2.88 — the verify pass pays a small attention/sync tax) |
| Speculation at measured acceptance E[N]=2.84, γ=7 | **~6.2 tok/s** (T_iter ≈ 0.455 s) |
| Speculation at acceptance 4 (PLD fusion / better drafting) | ~9.4 tok/s |
| AGENT spans: deep γ=16-32 PLD, near-100% acceptance on exact repeats | in-span ~50-90 tok/s (measured curve: 0.348+0.0074·(B-1)) |
| AGENT sustained (50-70% repetitive spans) | ~15-25 tok/s |
| Zero-draft, 100%-acceptance bound at γ=8 | 18.6 tok/s (measured); γ=32 ≈ 50 |
| 40 tok/s | Still open on 12 GB. The measured E149 cycle needs about 9.7 generated tokens per pass at the current verifier cost. |

## 6. Experiment ladder (E040 onward)

- **E040 — COMPLETE (2026-11-10).** Multi-position verify implemented in real_decode.cu
  (`--multi B --feed-ids ...`): causal per-row attention boundaries, sequential delta-net and
  conv state updates inside the kernels, per-row RoPE/positions, teacher-forced argmax checks.
  Result: amortization near-perfect (table in §5); 0 soft mismatches; the verify path is the
  foundation for E041/E042. Note: the summary's cold_GiB_per_step field reports the total over
  all steps (incl. the skipped warm-up); true per-step cold = 6.17 GiB (hot 7424 config),
  0.77 GiB/position at B=8.
- **E041/E042 — fused drafters with ADAPTIVE DEEP γ (NEXT, rewritten after owner pushback).**
  The owner challenged the 15 tok/s "ceiling" — correctly: E040's marginal cost (+5 ms/position)
  means DEEP verification is nearly free; the real constraint is draft cost and acceptance decay.
  Design: PLD-first drafting (n-gram context lookup, ~0.1 ms/token — deep γ=16-32 is cheap),
  MTP head as fallback drafter, adaptive γ by match confidence. On exact-repeat spans
  (agent tool-JSON, echoed context, code) acceptance does NOT decay per position — PLD
  literature shows near-100% on exact spans. Projection from the MEASURED E040 curve:
  T(32 pos) ≈ 0.50 s → in-span ~64 tok/s at full acceptance; sustained agent sessions
  (50-70% repetitive spans) ~20-30 tok/s; bursts 100+. Lossless (rejection sampling).
  Benchmark must be an agent-shaped trace (tool JSON + echoes + code), not chat.
  Verify-side prerequisite: attention kernel for γ+1 positions already exists (E040);
  tree-attention (SpecInfer-style) deferred until linear-chain acceptance is measured.
- **E041a — MTP-head drafter in VRAM (fallback drafter for novel text).** Decode the qwen35 nextn head (eh_proj/enorm/hnorm +
  dense block, own small KV) from VRAM; greedy self-verification first (accept if draft token
  == target argmax), then full rejection sampling. Measure real α on our decode loop.
- **E042 — PLD drafter.** n-gram lookup drafts; same verify path. Compare acceptance by
  workload type.
- **E043 — compute reduction.** Tensor-core MMQ GEMV + launch reduction; re-run E040 matrix.
- **E044 — SubSpec substitutes.** 2-bit cold-layer substitutes in VRAM as the drafter; only if
  E041/E042 acceptance is the binding constraint.
- **E045 — full pinning** (memlock raise, owner-approved) + deeper double-buffering; profile
  before assuming gains.

## 7. Context for resuming sessions/AI models

Read HANDOFF.md for machine state, EXPERIMENTS.md for evidence, this file for direction.
Non-negotiables: correctness gates against the validated token stream (E038), the 2 GiB VRAM
floor, the 6 GiB RAM floor with swap abort, no CPU layer offload, no quality loss in the
verified output path. Watch for the documented bug classes in EXPERIMENTS.md E039 (flat-index
RMW races in batched elementwise kernels; per-sequence stride errors) before trusting any new
batched result. Do not confuse multi-user aggregate tok/s with single-user tok/s. External
papers' ms-level compute estimates do not transfer; re-measure everything on this machine.
