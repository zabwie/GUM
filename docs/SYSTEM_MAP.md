# SYSTEM MAP — the whole machine, measured (2026-11-13)

Everything below was measured on this box today unless marked inferred. Companion to
[RESULTS.md](../RESULTS.md). Purpose: a researcher's map of where every byte goes, what
each path costs, and which assumptions the field (and we) never checked.

## 1. The physical machine

| component | fact | consequence for GUM |
|---|---|---|
| CPU | i7-12700K, 8P+4E cores, 25 MB L3, governor `performance` | reader threads + staging memcpys ride P-cores (taskset 0-3 ✓); CPU is never the decode bottleneck |
| RAM | 32 GiB, dual-channel; ~16-18 GiB available; swap 43 GiB (4-5 GiB used) | host tier (10.24 GiB staged) fits; swap-growth trips the decoder's own guard (seen once) |
| memlock rlimit | **3.9 GiB hard** (no root to raise) | only ~3.8 GiB of the staged host tier can be pinned; the rest is CPU-memcpy'd through the pinned ring each pass |
| PCIe | **gen4 x16 to GPU** (CPU-attached, not chipset); **ReBAR active: BAR1 = 16 GiB, ~9.9 GiB currently mapped** | link ceiling measured **24.7–24.9 GB/s** (1-stream = 2-stream: no engine-level gain); theoretical 31.5 GB/s — we run at ~78% of spec, which is normal for this topology |
| GPU | RTX 3060 12 GiB (GA106), 28 SMs, 2 copy engines, L2 = 3 MiB with **persisting window capped at 1.6 MiB** | decode is DMA-bound; L2 window the decoder wants (5.57 MB) is disabled for capacity — but the xq tensor it needs resident is **1.31 MB, under the cap** (see anomalies) |
| NVMe | WD Black SN850X, CPU-attached | out of the steady-state loop entirely (`nvme_per_token_GiB=0.00`); cold-start only |
| iGPU | none enabled (3060 is the only display adapter) | no offload/home for the drafter elsewhere |
| No root | passwordless sudo unavailable | memlock, swappiness, hugepages reserved pools, swapoff: all out of reach |

## 2. Where every byte goes (the lifecycle)

**Weights (15.1 GB Q4_0 file):**
- **VRAM hot tier** (config, 3.2–6.4 GiB): resident, zero fetch cost, packed into MMA tile layout on load.
- **Host staged cache** (up to 10.24 GiB in RAM): of which **~3.8 GiB pinned** (memlock cap) → direct DMA source; the unpinned remainder is CPU-memcpy'd into the pinned staging ring (8 × 16 MiB) by 4 reader threads each pass.
- **Device slot ring** (8 × 16 MiB): landing zone; each item: wait-staging → `cudaMemcpyAsync` (pinned → device) → GEMV/MMA kernel on the compute stream → slot released behind a 6-item window.
- **NVMe**: only if the working set exceeds RAM (never at these configs).

**Per decode cycle (the verify pass):**
- Cold bytes = 15.1 GB − hot tier. At the 196K baseline (hot 3.45 GiB): **10.6 GB @ 24.7 GB/s ≈ 0.47 s**. At the 32K chat profile (hot 6.4 GiB): **~8.7 GB ≈ 0.32–0.35 s**.
- Plus: MTP draft (22 ms), state repair (2 ms), ~15 ms scheduling gaps → cycle 0.48–0.57 s.
- The GPU's SMs are ~90% idle inside every verify pass (fetch-bound); the copy engine is idle during the draft.

**Per prefill pass (256–512 prompt IDs):**
- Same weight fetch, but **compute-dominated**: the batched Q4 kernels run ~7 TFLOP/s effective (~15% of int8 peak). The dominant term inside that is xq re-read traffic — every weight-row tile re-reads the quantized activations, ~314 GB per 256-row pass from L2/DRAM.

**KV (tiered, TurboQuant-TTKV):**
- 4,335 MiB at 196,608 ctx (recent 512 fp32, middle tier pooled, old rows protected). 723 MiB at 32,768 ctx. Allocated upfront for full ctx — **this is the VRAM the chat profile reclaimed**.

**RAM partition at runtime (32 GiB):** 15.1 GB file (mostly page cache) + 10.2 GiB staged cache + pinned 3.8 GiB (overlapping the staged figure) + desktop ~7 GiB anon + ~15 GB reclaimable cache. The decoder's own 13 GiB allocation pushes desktop anon pages into swap (the guard incident) — allocation order matters.

## 3. The measured walls (all falsifiable, all currently holding)

1. **DMA wall: 24.7 GB/s.** Three independent measurements (single stream, dual stream, live pipeline at 24.6) converge. ~78% of gen4 x16 spec. Not engine-limited, not staging-limited — link-limited.
2. **Acceptance wall: emitted/cycle = p₁/(1−p₁) ≈ 3.7.** Measured head top-1 = 0.79 (E169); chain math closes to two decimals. Published EAGLE-2/3 ≈ p₁ 0.82–0.85. The entropy wall caps any drafter near the model's own top-1.
3. **Prefill compute wall: ~7 TFLOP/s effective** on bit-exact MMA kernels (E166). The known inefficiency is xq re-read traffic; the known fix (L2 persisting window) is disabled by a 1.6 MiB capacity cap while the needed window is 1.31 MB — *a config problem, not a physics problem*.

## 4. Anomaly ledger (chase list)

| # | anomaly | status | potential |
|---|---|---|---|
| A1 | dual copy streams might beat one | **falsified today** (24.7 vs 24.6) | none — link-bound |
| A2 | verify 0.519 s at 4k vs 0.459 s at 128 | KV attention reads grow with context | matters more under a 6 GiB cap; quantify before profiling attention |
| A3 | prefill per-row cost drops 8.6→6.7 µs from B=256→512 | kernel underutilization at 256 rows | needs VRAM for bigger B; re-test under the 6 GiB budget with `compact_target_logits` |
| A4 | KV 4,335 MiB @ 196K vs owner's 1.5 GiB @ 1M claim | scaling ours linearly gives ~22 GiB — the "done correctly" regime must pool old tiers far harder | **the biggest lever inside the 6 GiB cap** — owner's own research territory |
| A5 | zero-copy (UVM fault) 10–14% slower than staging (E162) | explained: page-granularity faults vs 16 MiB bulk DMA | lesson: transfer size matters; supports staging architecture |
| A6 | `activation_l2_window,disabled=capacity` (5.57 MB requested, 1.6 MB allowed, xq is 1.31 MB) | **untested** — shrink the window to exactly xq per tensor-group | prefill xq re-reads become L2 hits; prefill is the one wall with software headroom |
| A7 | drafter 22 ms/cycle over an 8,192-token window | unknown whether the window's KV is recomputed each cycle | incremental drafter-KV → ~2 ms draft → wider prefetch window |
| A8 | drafter shortlist = 32,768 tokens of 248,320 | untested coverage rate on novel prose vs agent captures | if out-of-shortlist rate on prose is 5–10%, widening is a training-free p₁ gain |
| A9 | ReBAR 16 GiB active | explains why E162's zero-copy was even possible | no direct use found; noted for completeness |

## 5. The research agenda (owner framework applied)

**Cheap experiments (hours):** A6 L2 window sizing for prefill; A8 shortlist coverage from
existing captures; A7 drafter-KV audit; A2 attention-growth quantification; B-budget
sweep of `compact_target_logits`.

**Training-light:** capture-fed gentle finetune of the 230 MB MTP head (E169 gate: +2pp
val top-1 + token-exact bench) on the accumulating corpus; multi-layer feature
conditioning retraining only `eh_proj`.

**Novel/risky:** exactness-preserving bounded verification — greedy needs only the argmax;
any scheme that certifies the argmax from partial weight reads cuts bytes per cycle while
keeping the token stream identical. Nobody has built this on a fetch-bound UVM stack.
It is the only idea that attacks the 24.7 GB/s constant itself.

**Workload-level:** instrument the live server's per-request acceptance; route (PLD /
deep-γ / drafter) by span type; optimize the blended user-experienced tok/s.

## 6. The 6 GiB engine budget (owner directive)

Fixed non-weight overhead at 32K ctx (E189-corrected): KV 723 MiB + scratch 151 + slots 256
(E186) + MTP 978 — of which 715 MiB is the SHARED TARGET OUTPUT HEAD (device-resident,
already excluded from the cold stream) and the drafter itself is Q4_0-native (~260 MiB
specific) + logits ~32 (`compact_target_logits` already default-on) + activations ~500 +
CUDA context ~300 ≈ **2.0 GiB true overhead** → hot tier ≈ 3.0–3.6 GiB → cold 11.5–12.1 GB/pass → verify ≈ 0.47–0.49 s →
decode ≈ **7.1–7.5 tok/s** — i.e., the 6 GiB cap costs roughly the 196K baseline's
throughput, and the claw-backs (compact logits, KV regime A4) are worth ~0.4–0.8 tok/s
of it. Everything beyond that must come from the anomaly ledger and the agenda above.
