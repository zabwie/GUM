# GUM Research Funnel — decision-driven protocol (2026-11-11)

Owner directive: replace the endless experiment loop with a funnel. E161 is
the reference pattern — the decisive question was tested before hours of
kernel-building. This document is the working protocol; EXPERIMENTS.md holds
the evidence; HANDOFF.md the machine state.

## 1. The funnel

```
research question
  → hypothesis (specific, falsifiable) + predictions (measurable)
  → smallest decisive experiment that tests the underlying assumption
  → SUPPORTED  → integrated benchmark (correctness + performance gates)
  → REJECTED   → record the limitation, close the path, next question
```

Every proposed optimization answers five questions **before implementation**:

1. What exact bottleneck does it address?
2. What is the measured or theoretical upper bound on the improvement?
3. What is the cheapest experiment that can disprove the hypothesis?
4. What result would justify implementing it?
5. What previous experiment might already have answered this?

If the expected improvement is tiny, it goes to the backlog, not the
calendar. Priority score = **impact × confidence / effort** (1–5 each;
a ranking heuristic, not a speedup estimate).

### Value of Information (VOI)

Confidence that an idea will *work* is not confidence that an experiment
will *teach something useful*. A low-confidence, high-impact direction
whose cheapest falsification test is inexpensive belongs in the funnel even
though its priority score is low — disproving it in two hours can beat
spending the same two hours chasing another 1%. Track such items in the
**VOI lane**: they are judged by (answer cost × decision change), not by
their success probability. Schedule-level byte-skipping (B1) is the current
VOI exemplar.

## 2. Established bounds (do not re-derive; cite the experiment)

| Fact | Value | Source |
|---|---:|---|
| Verify cycle (8064/ctx1024) | ~227 ms; 8448/ctx384 ~215 ms | E155/E156a |
| Emitted per cycle (installed tree) | 4.2 | E149/E155 |
| Verify pass composition | span = **44.2 ms/GB × cold bytes** (residuals ±3.6 ms over 4.445–5.242 GB; ≈ 22.6 GB/s effective). **Local relationship** — proportionality below 4.445 GB is hypothesis, not measurement. Kernel durations stretch with streaming copies; whole-compute floor ≈ 150 ms (estimate) | E164 (corrects E163/E160) |
| Byte→span coupling | ~1:1 with DMA across the tested range; reconciled stakes: 25% cut @8448 → 24.6 tok/s (+33% transfer-component); saturation ~25.3 tok/s at ~33% crossover (floor estimate) | E164 + owner reconciliation |
| GEMV compute per pass | ~150 ms true compute floor (est.); measured kernel durations are streaming-stretched, not compute-bound | E164 |
| Lossless compression ceiling | ~5.5% (payload at entropy bound) | E161 (corrects E156) |
| Zero-copy mapped reads | 18.2–21.8 GB/s vs 24.8 DMA → 10–14% slower cycles; only ~128 MiB freeable | E162 |
| Tree-width remaining headroom | ~13% of passes; partially falsified (32-pos and adaptive widening both lost, E150/E151) | E126/E150/E151 |
| Small-drafter ceiling | one-pass MLP ~0.45; sequential 150M ~0.07 depth-1; exit heads ≤ 40% feasible depth | E153/E157/E158/E159 |
| Agent/PLD ceiling with current drafter | ~27 tok/s (23.9 measured) | E149 + audit |
| Oracle headroom (drafter at 10–12 exact) | 47–56 tok/s | E152 |
| Batched capture throughput | ~12.8 positions/s (5× dense) | E159 |

**Track A arithmetic:** the combined upper bound of all remaining known
levers (sync 6% × tree ≤13% × agent-only 13%) reaches ~21 chat / ~27 agent.

Precise, falsifiable statement of the 40 tok/s conclusion:

> Under GUM's current Q4_0 representation, this hardware, the current
> transfer schedule, and the measured speculative-verification efficiency,
> the identified incremental optimizations do not provide a credible path
> to sustained 40 tok/s on novel text.

This is conditional on those assumptions, not an unconditional
impossibility; an architectural discovery that changes one of them
invalidates the conclusion without invalidating the measurements.

**Owner correction (2026-11-11), accepted:** the bytes/token identity is a
*validated performance model of the current architecture*, not a universal
impossibility theorem — and the 116 tok/s figure is an idealized bandwidth
ceiling under a particular speculation depth and schedule, not a
demonstrated end-to-end limit. Innovation changes the model's terms; the
equation describes today's engine, not the physics of the problem.

## 7. The fork decision: two execution contracts (OWNER DECISION, 2026-11-11)

- **Contract EXACT — permanent reference mode.** The original Q4_0 model
  with identical intended semantics. Correctness oracle and regression
  baseline for every experiment. Preserved forever; the installed launcher
  remains EXACT.
- **Contract TOLERANCE — primary research track.** Numerical weights and
  intermediate computations may change, provided measured quality stays
  within explicit gates. Promotion into the production path requires
  evidence; EXACT remains the control.

Quality contract (initial experimental gates — starting thresholds, to be
refined with confidence intervals over multiple workloads):

| Metric | Gate |
|---|---|
| Reference-logit divergence | measured vs EXACT baseline; investigate significant distribution shifts |
| Held-out perplexity | ≤ 2% relative degradation |
| Task accuracy | ≤ 1 percentage point absolute degradation |
| Agent/tool-call validity | no statistically supported regression in schema validity or task success |
| Performance (always reported) | sustained tok/s, VRAM usage, PCIe bytes/token, task completion time |

Quality is measured at task level, not token agreement. Two models can
disagree token-by-token and still be equally correct; token-similar outputs
can silently break tool calls.

**Separation rule:** approximate verification is a separate experiment
family from mixed-precision inference — different correctness problem, own
quality evaluation. Start with quantization (well-defined model, consistent
evaluation).

### C0 — benefit ceiling BEFORE the quality harness (owner modification)

Resolve a live data conflict: E154's oracle runs imply bytes→span ≈ 1:1
(15% byte cut → 16% faster), while the E155 8448 numbers fit a
"kernels are the floor" model (byte cuts buy ≤5%). The models give C1
stakes of +45% vs +5%. Cheapest decisive test: oracle-mode verify sweeps
(`--oracle-ids`, no MTP, ctx 384) at 3–4 residency points; fit span(cold
bytes). If slope ≈ DMA slope (≈ 40 ms/GB), C1 proceeds with a
+30–45% ceiling at 25% byte cut; if flat, C1 is rejected before the
harness is built.

### C1 status: **PAUSED by owner decision (2026-11-11)**

State at pause: harness validated byte-identically ((4,32) control =
100% agreement); ladder points measured — (3,64) = 27.8% byte cut at
~81% greedy agreement (yellow flag), (2,64) = 50% cut at ~54% (fails).
The owner reviewed the sensitivity signal and elected to hold the
no-quality-loss line: **the C-track does not advance toward Stage V
without an explicit owner decision to reopen it.** The harness
(`scripts/c1_requant.py`) and the measured points are preserved —
reopening costs one session of knee-mapping ((3,32)/(3,128)/FFN-only
mixes), nothing needs rebuilding. Next-best exact-compatible levers:
agent/PLD push (24 → ~27) and, only via its own funnel entry, the
drafter-scale campaign. The two-contract split stays as designed:
EXACT is the only production contract.

### C1 — cold-weight mixed precision (staged, four gates; owner-approved)

**Critical harness warning (owner):** offline activation replay with
reference activations *underestimates accumulated error* — every layer
receives reference activations instead of the quantized model's own
outputs, and GDN recurrent state can accumulate approximation error over
time. Two distinct stages, four gates:

- **Stage S (screening):** rank tensors by quantization error, logit
  sensitivity, and actual byte savings on the E159 captures. Reference
  activations are acceptable *for ranking only*.
- **Stage V (validation):** assemble promising combinations and run FULL
  autoregressive generation — errors propagate through all layers and
  recurrent states. Held-out evaluation data must be disjoint from the
  calibration captures used to select configurations.

Gates, in order:
1. **Feasibility:** ≥ 25% lower *actual packed bytes* (scales, metadata,
   alignment included — not nominal bit widths). Q4_0 is 4.5 bpw effective;
   a whole-model 25% cut forces aggressive reduction of the cold FFN block
   specifically (63.6% of bytes) — e.g. FFN at ~2.6 bpw effective, or a
   residency-dependent mix.
2. **Quality screening:** passes preliminary sensitivity limits on held-out
   captures.
3. **Full-model:** perplexity ≤ 2% relative degradation, task accuracy
   ≤ 1 pp, agent/tool-call reliability — vs the EXACT baseline, on
   calibration-disjoint data.
4. **Performance:** end-to-end sustained speed improves after unpack and
   GEMV costs. Bytes saved but inference slowed is not a GUM optimization;
   perplexity preserved but tool calls damaged is not a GUM optimization.

**Two success definitions (owner, 2026-11-11 — do not conflate):**
- *C1 performance success:* enough compression to justify integrating a
  faster inference format (the ≥25% gate).
- *C1 memory-efficiency success:* enough compression (even 15–20%) to
  reduce VRAM residency at roughly constant throughput — advancing the
  desktop-first objective. GUM Adaptive needs BOTH; a result that misses
  25% but lands 15–20% safely still closes an uncertainty and feeds the
  Pareto frontier.
Also standing: no massive drafter-training campaign launches because C1
fails — that path re-enters only through its own funnel entry.

**Residency-dependent precision (owner's extension, the C-track's
distinctive direction):** VRAM-resident tensors use formats optimized for
local GPU execution; RAM-resident tensors use formats optimized for
transfer bytes even at higher decode cost. GUM Adaptive chooses per
hardware. The optimizer's problem: *find the best (residency, precision)
combination for a given GPU subject to minimum throughput and the quality
contract* — including whether moving more tensors from VRAM into compressed
RAM improves the desktop-first objective. E164's ~1:1 byte→span coupling is
what makes this trade computable.

### GUM Adaptive (owner vision, recorded as product direction)

Hardware-specific builds from tested building blocks: measure the machine →
identify its bottleneck → configure/specialize the execution strategy →
validate → promote. GUM Runtime (stable, hardware-tuned) is separate from
GUM Research Agent (proposes/tests under this funnel; never changes the
runtime without measured evidence and approval). Three optimization modes:
desktop-first (minimize VRAM at a user-selected minimum speed), VRAM-budget
(maximize speed within a budget), dedicated (aggressive). The optimizer
populates a measured VRAM↔throughput Pareto frontier per hardware/model.
Dynamic GPU reservation: reduce residency and yield resources when other
GPU work appears; restore when it exits. Cross-user database of validated
configurations keyed by hardware/interconnect/model/workload. Founding
line: *GUM maximizes the usefulness of the entire computer while AI is
running.*

**Standing owner directive: the 40 tok/s goal remains.** GUM Adaptive and
TOLERANCE are the paths investigated to reach it — not distractions from it.

## 3. Track A — optimize the existing GUM (target 20–25 tok/s)

Rules: measured bottlenecks only; every change gated on the exact six-prompt
suite (`--expect-ids`); separate branch; promote through
`scripts/run_single_user.sh` only after gates pass.

| ID | Experiment | Bottleneck | Upper bound | Cheapest disproof | Impact | Conf | Effort | Score |
|---|---|---|---:|---|:-:|:-:|:-:|:-:|
| A1 | Promote 8448/ctx-384 as documented short-sequence config; re-run suite where prompts fit | exposed transfer | +0.5–1.2 tok/s (E156a measured) | done — E156a | 2 | 4 | 1 | **8.0** |
| A2 | ~~Sync-stall trim~~ **CLOSED E163**: intra-range idle is 2.4 ms; the 13.6 ms was span-minus-copy, not a stall | — | — | done | — | — | — | closed |
| A3 | Agent/PLD deep-γ: extend bursts on exact-match spans (mechanism exists, E076 96.8% acceptance) | drafter yield on agent spans | ~27 tok/s agent ceiling | offline replay of agent trace with deep-γ policy | 3 | 3 | 2 | **4.5** |
| A4 | Tree reallocation: shift rows from deep chain to depth-1/2 fanout | drafter yield | ≤13%; E150/E151 already rejected width increase per se | offline tree-policy simulation on captured trees | 3 | 2 | 2 | 3.0 |
| A5 | Codec opt-in for long generations only (prefill cost amortizes) | exposed transfer | ~1–2% request-inclusive | E155 measured it — likely backlog | 1 | 4 | 1 | 4.0 |
| A6 | Cache-preparation speedup for the codec (0.728 s prefill cost, E155) | prefill | removes the codec's only downside | profile the prep path | 1 | 3 | 2 | 1.5 |

Backlog (below-threshold): A5, A6, sync-stall beyond what the trace shows,
any further GEMV micro-optimization (E051–E075 hill, all failed).

## 4. Track B — RAM-first inference architectures (separate branches)

Rules: never destabilize the reference build; new ideas live in branches and
are promoted only through correctness + performance gates. Track B must
consume a bounded share of every session.

Closed paths (do not reopen without a NEW hypothesis that changes an
assumption, not a parameter):

- Mapped/managed memory as primary path (E031–E037, E162: ≤87% of DMA)
- General lossless weight codecs (E161: payload at entropy bound)
- Mid-depth exit heads / one-pass drafters / small sequential drafters
  (E153/E157/E158/E159: five probes, capacity argument)
- Feature-drift adapters on the MTP head (E132/E133)
- Deeper/wider verify as primary lever (E150/E151; rows are cheap, proposals saturate)

Open Track B questions (each needs its funnel entry before work):

| ID | Question | Why not closed |
|---|---|---|
| B1 | Is there a schedule where some weight bytes are *not* needed for some passes (execution-schedule reuse beyond the tree)? | VOI lane. Cheapest falsification: a dependency analysis of the verifier — enumerate mechanisms by which a cold-byte subset could be skipped while preserving exact greedy semantics; if none survives, close without code. Hypothesis: none exists (rejection is only observable at the final logits, so every row needs every layer). |
| B2 | Does the 22 GB card change the Track B calculus (full residency → oracle speeds)? | Hardware, out of scope by directive — but the measured 8448-path suggests the answer is trivially yes; document, don't build. |
| B3 | EAGLE-scale drafter campaign (Track A/B boundary) | Only measured-physical route to 40 on novel chat. Gate: owner commits to long unattended capture (infrastructure exists, E159). |

## 5. Session protocol

1. Pick the highest-scored open item; write its five answers in the session log.
2. Run the smallest decisive experiment; record raw results in `results/`.
3. SUPPORTED → integrated benchmark with `--expect-ids` on the exact suite;
   REJECTED → ledger entry with the limitation and the closed question.
4. Update this backlog's scores after every experiment.
5. Keep HANDOFF.md current; every new fact lands in the §2 bounds table.

**Session outcome rule:** every research session must end with exactly one
of — (a) a measurable improvement, (b) eliminated uncertainty, or (c) a
reproducible discovery that changes the research plan. A session ending
with none of these triggers a process review, not another experiment.

## 6. Assumption audit & the lower bound (Track B core, 2026-11-11)

Owner directive: change the level of the questions — identify the
fundamental assumptions, the evidence for each, and the cheapest way to
test relaxation. No code first.

**The central question (owner §5): what is the minimum information that
must cross PCIe to generate one correct token?** Under exact-output
semantics with residency V, row budget R, and emitted tokens per pass E:

    bytes/token = (B_model − V) / E      →  tok/s = BW × E / (B_model − V)

This two-term model *validates against measurement*: predicted 1.291 GB/token
at V=8448 MiB/E=4.113 vs measured 1.274 (E156 audit). Consequences:

- The measured system sits at 1.274 GB/token. The **perfect-drafter floor**
  (E = R+1) is **0.212 GB/token** at R=24. **Label (owner, 2026-11-11):
  conditional bandwidth-only bound — requires ~25 emitted tokens per 5.24 GB
  transfer AND verification capacity for that many rows. NOT an
  attainable-throughput prediction.** The packed 24-position tree emits at
  most one accepted path; if the max accepted path is 8–10 tokens, this
  bound is unreachable at that topology regardless of drafter accuracy.
  Three objectives stay separate: 20–25 chat (near-term), 40–60 chat
  (research), ~116 (conditional ideal). Traversal Verification (NeurIPS
  2025) noted for the backlog: tree topology and acceptance rules remain
  genuine variables — quantify expected benefit before any implementation.
- With V=0 (pure RAM), 3.27 GB/token → ~7.5 tok/s: **VRAM capacity is
  arithmetically equivalent to throughput** at fixed bandwidth (A-04 is an
  identity, not an assumption).
- Byte cuts c map to tok/s ≈ 1/(1−c) only while copies ≥ kernels (E163:
  crossover ≈ 17–20% byte cut; beyond it compute binds).

### The five assumptions

| # | Assumption | Evidence | Cheapest relaxation test | Verdict |
|---|---|---|---|---|
| A-01 | Every pass fetches all cold weights | B1 dependency closure (rejection observable only at final logits); rows are the only reuse mechanism and all rows share the byte stream | none found that preserves exact semantics; two-phase/adaptive schedules fail the causal lock (acceptance requires full depth) | **closed under exact semantics** |
| A-02 | Dense weights need substantial transferred information | E161 (payload at order-0 entropy bound, 5.5% ceiling); E163 byte→span coupling | model-aware representation — **requires the mission fork** (below) | closed within fixed Q4_0; open under tolerance |
| A-03 | Tokens per pass depend on prediction accuracy | five probes (E153/E157/E158/E159) + oracle 47–56 tok/s headroom (E152) + E148 (history-reuse saved zero passes) | EAGLE-scale training campaign (data now 5× cheaper, E159); or tolerance unlocks approximate verification | **the load-bearing assumption** — walls are capacity, not ideas |
| A-04 | VRAM must hold the fast weight tier | E162 (mapped 73–87% of DMA; hot arena at RAM speed = +390 ms/pass); the V=0 arithmetic above | none at fixed bandwidth — arithmetic identity | **identity, not assumption** |
| A-05 | Rows are nearly free (row budget expandable) | E150 marginal ~2.3 ms/row; E163: compute co-critical so rows cost real span | expand R from 24 → 32/48 when the drafter improves | drafter-gated; mechanism cheap, yield blocked by A-03 |

### The fork the audit forces (owner decision required)

Every closed path converges on one structural choice:

- **Contract EXACT (current):** the output is bit-identical to the fixed
  Q4_0 model's greedy stream. Under this contract the audit says: Track A
  tops out at ~21 chat / ~27 agent; 40+ requires attacking A-03 at EAGLE
  scale (the only measured-physical route) or hardware.
- **Contract TOLERANCE:** the output preserves model *quality* within a
  measured bound (e.g., argmax agreement ≥ X% on held-out fixtures, plus a
  perplexity/sampled-quality gate). This unlocks Breakthrough C directly:
  lower-bit cold weights (traffic ↓ 25–40%), approximate layer-skipped
  verification (rows no longer drafter-gated), mixed-precision residency.
  Each has a cheap decisive test (offline quality-vs-traffic curves on the
  E159 captures before any engine work).

The exact contract was the owner's founding directive; relaxing it is the
owner's call alone. Until decided, Track B's open frontier is the A-03
campaign (B3) and nothing else.
