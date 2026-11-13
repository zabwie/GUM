# RESULTS.md — perf-auto optimization run (2026-11-13)

Contract: owner template + live addenda (see LOG.md). Frozen bench:
`benchmarks/perf_auto_bench.py` + prompt modules. 20 hand-authored diverse natural-language
prompts (7 × 94–115 tok, 7 × 677–865 tok, 6 × 3011–3657 tok; nominal 128/1k/4k buckets,
actual counts recorded per row), 256 greedy tokens each, baseline = the installed
`gum_openai_server.py` configuration on the RTX 3060 12 GiB. Prompt authoring froze after
four extension rounds; the mid bucket's counts are below nominal — flagged honestly here
and in every row (`prompt_tokens`).

Reproduce:
```
python3 benchmarks/perf_auto_bench.py <tag> --golden            # goldens from baseline binary
python3 benchmarks/perf_auto_bench.py <tag> --sweeps 5          # correctness-gated bench
GUM_DECODER=<candidate> python3 benchmarks/perf_auto_bench.py <tag> --sweeps N --flags "..."
```
Every measured turn is gated token-for-token against baseline goldens; the run exits
non-zero on any drift. Rows: results/perf-auto/baseline/rows.jsonl (aggregated from the
bench log — the run was stopped after sweep 2 per deviation D9; sweeps 0–2 complete,
60 turns).

## Baseline (unmodified binary; 60/60 turns token-exact vs goldens; 3 sweeps)

decode = committed decode tok/s; prefill = prompt_tokens / prefill_s (decoder CUDA-event
timing, wall-clock cross-checked). Median [min–max] across sweeps and prompts.

| bucket | decode tok/s | prefill tok/s | verify s/cycle | cycle s | emitted/cycle | H2D GB/s | min free VRAM |
|---|---|---|---|---|---|---|---|
| 128 | **7.75** [6.66–8.92] | **118** [116–126] | 0.459 | 0.484 | 3.75 | 24.6 | 1521 MiB |
| 1k | **7.42** [6.62–7.76] | **121** [119–123] | 0.462 | 0.493 | 3.64 | 24.6 | 1520 MiB |
| 4k | **6.57** [5.98–7.30] | **103** [100–105] | 0.519 | 0.569 | 3.75 | 24.6 | 1424 MiB |

All numbers are general-purpose (20 diverse prose prompts; the JSON-repeat workload that
produced 51 tok/s in an earlier probe is excluded by construction from the frozen bench).

## Kept changes

1. **Source recovery (integrity, not throughput).** Uncommitted E177–E181 decoder source
   (session-stream, turboquant-ttkv KV, kv-recent, fused prefill tail, draft hints) was
   lost to a `git checkout` during probe reverts and recovered in full from dangling git
   blob `9d0fcda`; committed on perf-auto; verified byte-identical token streams and
   re-synced the shipped binary (`benchmarks/real_decode` rebuilt from recovered source,
   hash `10fc035373933100` matches the pre-incident binary on the probe workload).
2. **Frozen benchmark + correctness harness** (`benchmarks/perf_auto_bench.py`,
   `perf_auto_prompts_{short,mid,long}.py`, `perf_auto_probe.sh`, `perf_auto_h2d_bw.cu`).
3. **Baseline ledger** (this file + rows.jsonl + logs) — the run's most durable output.

## Reverted / rejected ideas (with measured or arithmetic reasons)

| idea | result | reason |
|---|---|---|
| `--readers 8/16` | decode unchanged (6.73 vs 6.82, noise) | staging already keeps ahead; DMA-bound |
| `--o-direct 0` | unchanged | staged host cache serves warm; NVMe tier ~0 GB/token |
| `--mtp-tree 1` | unchanged (6.85 vs 6.82) | tree drafting engages on too few cycles at this acceptance; consistent with E126/E150/E151 |
| `--prefill-chunks 16` | prefill +13% on 1k-class, −4% on 4k-class | per-pass cost is compute-dominated; wider chunks trade fetch against compute unevenly by length; rejected as not a general gain (and would be a length-tuned special case if split) |
| `--prefill-chunks 32` | VRAM preflight abort | B=1024 logits exceed the 2 GiB free-VRAM floor |
| `--speculate 64` | rejected by validation | session-tail invariant caps speculate ≤ 32; relaxing risks the server's turn protocol |
| copy-merge / larger copy transactions | not attempted to keep | live pipeline already at **24.6 GB/s vs 24.8–24.9 GB/s measured hardware ceiling** (microbenchmark, decoder-realistic 4–16 MiB chunks) — ≤1% headroom |
| hot-tier growth (4608 MiB) | VRAM preflight abort | conflicts with owner's VRAM-usability addendum anyway |
| hot-tier reduction (2048 MiB, frees ~1.5 GiB) | decode −30–38% (probe: 4.8 tok/s) | documented trade, not a keep: see VRAM trade curve below |
| EAGLE4 drafter | not re-attempted | drafter family exhausted by five prior probes (E153/E157/E158×2/E159); acceptance on novel prose is the binding constraint |

## The physical ledger — why 4× on ordinary chat is not reachable on this card

- Decode cycle = verify (fetch 10.6 GB non-resident weights @ **24.6 GB/s ≈ measured
  hardware ceiling**) + MTP draft (22 ms) + repair (2 ms).
- Committed tokens per cycle = 3.6–3.8 on diverse prose (MTP acceptance ~0.10 average,
  ~0.6 at shallow chain positions; deeper drafting saturates; tree width previously
  falsified). Decode tok/s = emitted ÷ cycle ≈ 7.4 median.
- Raising throughput therefore needs *more bytes resident* (blocked by 12 GB VRAM minus
  4.3 GiB KV at 196K ctx, and by the owner's VRAM-usability addendum) or *a better drafter*
  (needs EAGLE-scale training data; capture pipeline accumulating at ~12.8 positions/s).
- 4× decode ⇒ ~15 committed tokens per 0.47 s pass ⇒ acceptance ≈ 0.85 per chain position
  on novel text — no exact speculative mechanism reaches that on general prose.
- 4× prefill ⇒ ~30+ TFLOP/s effective from the batched Q4 kernels (measured ~7 TFLOP/s,
  MMA path already bit-exact-validated E166). Requires a GEMM rewrite — out of scope for a
  correctness-contracted 6-hour session; recommended as the next experiment.

## VRAM trade curve (owner addendum: "free as much VRAM", measured, not claimed)

| hot tier | engine VRAM | free VRAM for desktop | decode tok/s (probe) | Δ vs baseline |
|---|---|---|---|---|
| 2048 MiB | ~8.9 GiB | +1.5 GiB | ~4.8–5.0 | ≈ −35% |
| 3456 MiB (installed) | ~10.4 GiB | 1521 MiB | 6.8–7.7 | — |
| 4608 MiB | rejected | — | — | VRAM preflight (2 GiB floor) |

The installed configuration is the measured sweet spot of the trade; further VRAM freedom
requires shrinking the 4.3 GiB KV allocation (context budget), not the weight tier.

## Stop condition

Plateau rule fired: five consecutive attempts under 2% general-purpose gain (readers ×2,
o-direct, tree, chunk-width, copy-side bounded by hardware). The 6-hour wall was not
reached; remaining time was spent on verification, documentation and server restoration.
Baseline and final are therefore identical tables — the run's contribution is the ledger,
the frozen bench, the correctness gate, the recovery, and the two measured walls.

## Honest bottom line

The GUM decoder on this 12 GiB card is already operating at its physical envelope for
lossless single-user inference under the no-downgrade, VRAM-sharing constraints: decode is
dual-bounded by PCIe weight refetch (at silicon-measured peak) and MTP drafter acceptance
(family exhausted), prefill by batched-Q4 compute efficiency. The credible breakthrough
paths, in order: (1) EAGLE-scale drafter training on the accumulating capture corpus —
raises emitted/cycle on ordinary text; (2) a rewritten batched Q4 GEMM for prefill (the
one wall with real software headroom); (3) the 22 GB card, which relaxes the residency
wall entirely. None of these was completable inside the correctness contract of this
session, and the numbers above are the evidence.
