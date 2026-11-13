# BUILD ROADMAP — the everything-plan (2026-11-13, owner mandate: "apply every boost")

Owner directive: mix all proven levers, no quality loss, no VRAM sacrifice. This
document sequences the builds; EXPERIMENTS.md holds the evidence for every number.

## Current installed state (all exact-gated)

| lever | evidence | delivered |
|---|---|---|
| GUM Adaptive residency (auto hot tier) | E182/E183 | 7.5–13.7 tok/s adaptive |
| Wavefront ring 16/16/12 | E186 | +3% |
| E185 affinity 0–7 (P-cores, no SMT) | E185 | +0.5–2.3% |
| Closed negatives: prefetch-during-draft (E187), L2 window (E190), compression (E161), zero-copy (E162), small drafters (E153–E159) | — | do not reopen |

## Build 1 — CPU row-split second stream (E191: gate PASSED 3.96 TFLOP/s, 32.4 GiB/s)

The largest proven lever: ceiling +58% (realistic +30–45%).

Design sketch (do not start without reading E191 + the slot-pipeline notes in HANDOFF):
- Split EVERY weight tensor by output rows: GPU computes rows [0, g), CPU computes [g, N).
  f ≈ 58% of bytes should be CPU-side at measured rates — tune f per tensor class.
- CPU kernel: the E191 AVX-VNNI path (`benchmarks/microbench/e191-cpu-vnni/`), extended to
  the engine's real quant scheme (xq int8 + int2 sum-meta) — the microbench proved the
  rate with Q4_0-style d×Σ(q−8)a math; the engine's per-group meta folds into the same
  16·Σa-style correction (VERIFY the algebra against k_quant_x before trusting).
- Per-layer protocol: GPU finishes its half of a tensor → nonlinearity needs BOTH halves →
  CPU partials must arrive within the GPU's per-layer budget (~9 ms at hot 0 stream rate).
  CPU per-tensor budget at 32.4 GiB/s: ~5.2 GB CPU-side per pass ÷ 215 tensors ≈ 0.8 ms
  per tensor — fits with margin. Exchange via pinned ring buffers; D2H activations
  (K×f_rows... actually GPU sends activations to CPU once per tensor: B×K int8 ≈ 557 KB
  per tensor at B=32 — total ~120 MB/cycle D2H, fits idle link windows).
- Exactness: CPU computes half the rows of the SAME Q4_0 math — accumulation order
  changes → near-tie argmax flips possible. Gate: six-prompt E149 suite + --expect-ids
  dense reference on every change; add a near-tie monitor (report min logit gap).
- RAM contention: CPU reads ~8 GB + DMA ~5.7 GB per cycle ≈ 55 GB/s vs ~65 GB/s real —
  measure with the decode running before trusting the projection (E191 §RAM).
- Config: --cpu-rows F flag, default 0 (off) until gated; sweep f ∈ {0.3, 0.45, 0.58}.

## Build 2 — W4A8 int8 tensor-core MMA kernels (QServe/QQQ-class, research-confirmed)

Unifies three problems: prefill (78–330 s pain), tree-width knee (E150/E151 lost because
compute spills the 0.555 s DMA window at R≥64), and acceptance headroom.
- Target: 7 → 14–20 TFLOP/s via mma.m16n8k32.s8 with in-register W4 dequant
  (QServe progressive quantization; our E097 already measured 1.58× on down-projection;
  E166 built the current MMA path — extend, don't restart).
- Payoff 1: prefill 2–3× faster (24K prompt: 330 s → ~120 s).
- Payoff 2: width knee 64 → ~128 rows → wider trees become FREE in bytes → emitted/cycle
  scales with drafter quality up to the new knee.
- Payoff 3: prefill-width latency drop helps the agent loop (tool calls between turns).

## Build 3 — Finetune campaign (owner-approved, target-side + drafter)

- Corpus: capture/ automation live (every 20 min, idle-gated, 40 GB budget ≈ 150K+
  positions). Raise GEN_TOKENS/diversity if val curves say so.
- Reviewer-corrected recipe: **KL against the target's full distribution, teacher-forced**
  (not hard labels — E169's +0.08pp on 3.8K pairs was hard-label; KL gives ~10× signal per
  token). Teacher-forced prefill generates data at ~100+ tok/s, not 7.
- Drafter finetune add-ons (research-confirmed): EAGLE-3 multi-layer fusion (capture
  ALREADY has 7 mid-layer taps + final residual), training-time-testing rollouts,
  Clover-style depth-weighted loss (upweight depth 3–4 where survival decays to 0.519),
  acceptance-weighted loss (2025–26 SOTA).
- Target-side adaptation (owner-approved 2026-11-13): finetune final block(s) + final norm,
  self-distillation toward own argmax — MUST pass: (a) val loss down, (b) greedy argmax
  preserved ≥ suite gate (ideally exact → contract retained), (c) p₁ measured lift.
- Gate (E169 standard): +2pp val top-1 + token-exact bench.

## Build 4 — Drafter/tree upgrades (zero-training, exact)

- SAM-decoding suffix automaton as second drafter merged into the tree (supersedes PLD's
  prompt-only lookup; free in bytes since verify width is free).
- HeteroSpec/EAGLE-2 confidence-conditioned dynamic trees using our measured per-depth
  survival — only meaningful AFTER Build 2 moves the width knee.
- TreeWY (tree-structured WY transform for GDN-hybrid speculative verification) — study
  first; it targets exactly our SSM rollback cost (~150 MB D2D/cycle, E043).

## Open decisions (owner)

1. **RESOLVED-CLOSED (2026-11-13): codec lever (+28%) closed permanently.** Full-disk
   sweep confirms no full-precision Qwen3.8-27B exists anywhere on this machine (the
   F32 file an earlier agent cited is the 1.8 GB mmproj vision projector; the only
   safetensors anywhere are nvidia/Qwen3.6-35B — a different model), and no HF source
   is re-obtainable (owner searched). Q4_0 is therefore the ONLY ground truth that has
   ever existed: "agreement with Q4_0" IS the quality metric, the (3,64) 81% figure
   stands as the measured quality-loss signal, and the TOLERANCE track stays closed
   on its own terms. No quality ambiguity; the exactness contract is absolute.
2. CPU offload directive: E191 turns the excluded lever into +30–45%; owner verbally
   approved "apply every boost" — treat as reopened, keep the exactness gates.
3. Target-side adaptation changes the exactness baseline to the adapted model (owner
   approved 2026-11-13 with the improve-not-degrade gate).
