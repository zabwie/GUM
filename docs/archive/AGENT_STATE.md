# Agent state

## Objective

Measure and improve local LLM inference throughput when model state exceeds the RTX 3060's 12 GiB VRAM, using supported system-memory mechanisms and reproducible evidence.

## Active user task — maximize real single-user throughput

- The active goal is 40–60 generated tok/s, exact target output, and more
  available VRAM where possible.
- E149 is installed. `scripts/run_single_user.sh` defaults to grouped
  native MTP with a packed 24-position verifier on the RTX 3060. It measured
  16.99–23.87 generated tok/s across the six prompts that measured
  15.20–21.37 in E127. Every one of the 1,024 token IDs matched the dense
  reference. At least 1032 MiB of VRAM remained free in these six runs.
- The optional EAGLE4 path reached 22.01 agent and 19.41 Rust tok/s. On the
  event-log prompt, EAGLE4 at 7808 MiB hot reached 15.91 and left 1383 MiB free.
  See `GUM_DRAFTER=eagle4` in README.md.
- See HANDOFF.md, EXPERIMENTS.md and README.md for evidence and launch.

## Historical E076 acceptance profile

- MTP acceptance target is met on the fixed 128-token agent trace: with default MTP-only `p_min=0.97`, γ=8, 61/63 drafts were accepted (96.8%) and all 128 tokens matched the dense reference in two runs. Throughput was 5.77–5.94 end-to-end tok/s versus 3.59 dense. This does not establish a 90% rate on arbitrary prompts.
- The old 34% metric combined MTP and PLD. A 64-token replay separated MTP 7/15 (46.7%) from PLD 46/141 (32.6%). The 128-token fast hybrid (`--pld 1 --p-min 0.3`, γ=16) passed the full dense gate at 7.397 tok/s and 30.8% total acceptance.
- `benchmarks/real_decode.cu` now reports source-specific proposal acceptance and end-to-end cycle timing; historical rates excluded MTP drafting. It skips an MTP chain when the next cycle will select PLD while keeping state sync, cutting measured MTP time 33.9→16.9 ms/cycle in a 64-token paired replay.
- E072/E073 still identify `k_mat_q4` as the main verifier bottleneck (~80% of profiled kernel time); their matched shared tile win was ~7.5%. The 20–25 tok/s projection remains unachieved. Nsight Compute counters remain blocked by `ERR_NVGPUCTRPERM`.
- Full evidence and reproduction state: [EXPERIMENTS.md](../../EXPERIMENTS.md) E076 and [HANDOFF.md](../../HANDOFF.md) post E076. Retain `--expect-ids` for every drafter/kernel change.

## Historical benchmark state — through E075

- The requested 20–25 tok/s target is not met. E071 reached 9.629 tok/s in one 64-token exact-gated
  run; another under lower free VRAM measured 8.799. Both compared 32 expected IDs with zero
  mismatches. Treat throughput as variable, not a guaranteed floor.
- Retained in `benchmarks/real_decode.cu`: delta-net state `[head][i][j]`, the activation L2 window,
  packed `int2` scale/sum metadata, and a bit-exact shared activation GEMV tile (`NB=16`, `K>=5120`,
  CB=64, 16 output rows/CTA). Matched E072/E073 profiling showed ~7.5% lower summed GEMV time and
  a 0.3534 vs 0.3609 s mean cycle; GEMV still accounts for ~80% of kernel time.
- E069 microbench tiled gains were 1.2–1.4x across tested production shapes; those do not predict
  end-to-end throughput. E074 vector-warps plus shared tiles, E068 scalar shared tiles, E056 WMMA,
  and E058/E059 dispatches lost or were inconclusive. E075 favored CB=64; row tile 32 did not win.
- Nsight Compute hardware counters are blocked by `ERR_NVGPUCTRPERM`. Nsight Systems reports are
  saved as `results/e072-wide-gemv-timeline-20261110.nsys-rep` and
  `results/e073-control-gemv-timeline-20261110.nsys-rep`.
- Immediate next step: continue profiling/shaping the GEMV path to reduce its 80% share, then run
  the 64-token `--expect-ids` gate. Do not claim the target or E050's 13–15 tok/s projection.
- Full measurements: [EXPERIMENTS.md](../../EXPERIMENTS.md) E051–E075. Current interpretation and
  reproduction state: [HANDOFF.md](../../HANDOFF.md) post E075.

## Verified platform

- RTX 3060 12 GiB, CC 8.6 / 28 SMs; Intel i7-12700K; 31 GiB RAM.
- PCIe 4.0 x16 at 16 GT/s; Linux kernel 7.0.0-38; NVIDIA driver 595.99.02; CUDA 12.0 and 13.2 installed.
- `nvidia-smi -q` reports HMM. CUDA 13.2 reports mapped-host support, UVA, concurrent managed access, pageable memory access, and software-coherent pageable HMM. Host-native atomics are not supported.
- See [HARDWARE.md](../HARDWARE.md) for evidence and remaining hardware questions.

## Completed measurements

- E003/E004/E011 confirm mapped pinned host reads, staged H2D, and VRAM scans. The resumed 64 MiB run measured 12.465 GB/s mapped, 14.601 GB/s H2D-only, and 331.566 GB/s VRAM. Staging was nearly tied at 16 MiB and faster from 24 to 128 MiB for the tested float4 scan.
- E005/E006 measured 64 MiB managed memory. Five fresh CPU-initialized first GPU reads: 5.201 ms median; five repeated scans: 0.210 ms; explicit GPU prefetch plus scan: 3.108 ms.
- E008 Nsight trace confirmed 640 MiB H2D and 320 MiB D2H migration, plus 960 CPU and 2,305 GPU page faults. Profiled timings are not used.
- E009/E010 found that mapped, full prefetch, and compact gather/cache preferences vary with selectivity and reuse. At 1–10% selectivity and four passes, the compact cache beat repeated mapped reads and full prefetch in the synthetic gather.
- E012 measured synthetic Q4_0 matvecs at 32/64/128 MiB. Managed prefetch beat mapped and full H2D at every tested size and pass count; all outputs passed.
- E015 validated a 47.812 MiB slice of the actual `blk.0.ffn_down.weight` Q4_0 tensor against a scale-aware CPU reference. With synthetic vectors, one-pass medians were 4.168 ms mapped, 3.833 ms staged, 2.532 ms managed-prefetch, and 0.351 ms VRAM-resident. Four-pass chunk streaming was slower than loading the tensor once.
- E014/E016 measured serialized versus double-buffered chunking. Chunk copies helped versus serialized chunks, but the 4–64 MiB size sweep was noisy; 8 MiB had the lowest observed medians, not a settled optimum.
- E017 found `cudaDevAttrHostRegisterReadOnlySupported=0`; read-only file-backed registration was skipped before mapping model pages.
- E018 measured dependent random-load latency: mapped pinned-host access was about 2.7–2.8× the VRAM path at 1–64 MiB and about 4.0× at 960 MiB.
- E019 measured independent-chain scaling and managed-memory crossovers. At 960 MiB, full prefetch crossed mapped access between 896 and 1,024 chains; demand migration crossed between 1,024 and 1,536. Nsight migration volume approached repeated full-allocation movement at 256+ chains.
- E021 extended synthetic Q4_0 GEMV to 960 MiB. At four passes, managed prefetch took 76.531 ms, versus 103.612 ms H2D-staged, 440.832 ms mapped, and 38.655 ms VRAM-resident. Every CPU-reference check passed.
- E022 compared 960 MiB migration traces at 1, 256, and 4,096 chains. Higher fanout was associated with near-full-allocation aggregate migration and fewer GPU page faults; the traces combine demand and prefetch paths.
- E023 verified a 1 GiB device plus 2 GiB mapped-host mixed kernel; all checks passed. Mixed one-pass latency was 394.184 ms versus 393.268 ms host-only, with 10.896 GB/s logical host bandwidth.
- E024 verified a single CUDA VMM address range backed by 64 MiB device and 128 MiB host memory; both areas passed GPU validation with zero mismatches.
- E025 compared Qwen3-0.6B Q4_0 actual inference in ordinary and cudaMallocManaged modes. Both outputs were correct and 29/29 layers were offloaded. Five-sample decode means were 383.054 and 381.812 tokens/s; this fitting model does not test oversubscription.
- E013 found no standalone model small enough to verify inference within the 4 GiB additional-host-RAM cap. The 15.15 GB (14.1 GiB) base GGUF was not loaded.
- Raw outputs, commands, source revisions, preflight conditions, and interpretation are in [EXPERIMENTS.md](../../EXPERIMENTS.md) and `results/`.

## Implementation

- `benchmarks/memory_paths.cu`: mapped-host, H2D-only, H2D-plus-scan, VRAM scan; 1–256 MiB input.
- `benchmarks/managed_paths.cu`: managed CPU-first reads, independent cold allocations, repeated scans, CPU/GPU prefetch, checksum validation, and device-free-memory snapshots.
- `benchmarks/reuse_paths.cu`: mapped, full H2D, or managed-prefetch paths across 1/10/25/100% strided access and 1/2/4 reuse passes, with a pre-trial L2 eviction scan.
- `benchmarks/q4_matvec.cu`: mapped, staged, managed-prefetch, and VRAM Q4_0 GEMV-style paths; optional read-only model-file input, CPU reference validation, and serialized/double-buffered chunk copies.
- `benchmarks/host_latency.cu`: deterministic dependent random-load chains across mapped host, managed demand/prefetch, and VRAM; supports 1–960 MiB under the project allocation limits.
- `benchmarks/oversubscribe.cu`: single large cudaMallocManaged allocation with GPU init/validation passes (E026/E027).
- `benchmarks/tiered_stream.cu`: NVMe (O_DIRECT) -> pinned staging ring -> RAM cache -> VRAM hot arena streaming pool with GPU validation (E028-E030).
- `benchmarks/streamed_decode.cu`: per-token Q4_0 GEMV over every 2-D Q4_0 tensor of a real GGUF model (embedding excluded), tiered VRAM/RAM/NVMe residency, CPU-reference validation (E031). `scripts/gguf_manifest.py` emits the tensor manifest; `scripts/inspect_gguf.py` inspects model layout.
- `scripts/run_when_idle.sh`: sampled resource gate; q4 chunk size is selectable with `CHUNK_MIB=4..64`. The gate accounts for chunk buffers and scratch, rejects planned device buffers above 2 GiB, and records CUDA clients, GPU activity, temperature, and RAM/VRAM headroom. It never stops unrelated processes.
- Q4_0 and host_latency inputs are capped at 960 MiB; the Q4 executable also rejects exact device-buffer plans above 2 GiB. The maximum measured Q4_0 device footprint was 1,974.728 MiB.
- Prior implementation milestones: Q4 transfer isolation `554861f`; synthetic chunk pipeline `e5534ce`; GGUF tensor support `c6f8bc0`; real-scale CPU reference `898e1df`; chunk-size parameter `893db0c`. The current work extends Q4 and random-load inputs to 960 MiB.

## Active conclusions and limits

- H1 is supported: mapped pinned host memory is directly readable by a GPU kernel on this machine.
- H2 is supported for the tested contiguous stream at 24 MiB and above; 16 MiB was near a tie. This is not an inference threshold.
- H4 is supported in this microbenchmark: explicit prefetch plus scan was about 40% lower in median elapsed time than demand first access. The arms differ in allocation/reuse pattern, so repeat with inference-like access before selecting an architecture.
- H5/H6 are supported for the tested synthetic gather: sparse low-reuse reads favor mapped access; compact gather/cache amortizes repeated sparse reads; full prefetch is better for denser accesses.
- Managed prefetch is the best tested path for reusable Q4_0 matrices that fit in VRAM. Chunk double-buffering reduces serialized chunk time but repeats transfers each pass and has not beaten one-time staging/prefetch for reused weights.
- Synthetic Q4_0 GEMV tests through 960 MiB still favor managed prefetch over mapped and full H2D host-backed paths; the 960 MiB matrix fits in VRAM, so no oversubscription conclusion follows.
- Random dependent loads show mapped-host aggregate throughput plateaus around 120–125 million loads/s at 64 MiB and around 76–87 million loads/s at 960 MiB. The managed crossover chain counts are specific to this test.
- A real Q4_0 tensor GEMV slice and a small quantized LLM are validated. E038 also runs the 13.42 GiB Qwen3.8-27B through the project's tiered residency decoder; this does not validate a general-purpose dynamic residency manager.
- `cudaMemGetInfo` changed by 128 MiB after first access and did not return after CPU prefetch; this global counter does not prove current placement.
- llama.cpp's managed-memory switch completed a small fitting-model baseline; E036 tested the 27B with MTP but exposed unified-memory thrashing. Read-only CUDA registration of file-backed weights remains unsupported.

## Historical implementation checkpoints (E038-E039)

E038 (real decode loop with tiered residency) is complete and validated: `benchmarks/real_decode.cu`
runs the actual 13.42 GiB Qwen3.8-27B (qwen35 hybrid: 48 gated-delta-net layers + 16 GQA attention
layers, KV + persistent SSM state, greedy sampling) with the E031-E037 tiered residency. Steady
3.39 tok/s = 5.4x llama.cpp's 0.63 t/s unified-memory baseline (E036), token-for-token identical
greedy output vs llama.cpp CPU (validated on both the 0.6B, 24/24 tokens, and the 27B via per-op
node dumps from a patched out-of-tree llama.cpp copy). The decisive bug was the delta-net k-head
mapping (h % n_group, not h/3); full list in EXPERIMENTS.md E038.

E039 (batched real decode) is complete: --batch B amortizes the cold set across sequences.
27B measured: batch 4 = 10.70 aggregate tok/s (1.36 GiB cold/token), batch 8 = 13.17 (0.772
GiB/token), both token-equivalent to the validated batch-1 sequence, 2 GiB VRAM floor held
(batch 8 needs hot 7424/cache 6144; batch 16 busts the floor on SSM states). The decisive fix
was a flat-index read-modify-write race in six elementwise kernels (grid (x, B) blocks all
touching the same indices) plus the delta-net qkv stride; full list in EXPERIMENTS.md E039.

The old E039 follow-up list proposed tensor-core MMQ, approximate-layer drafting, and fp16 SSM
states. E056 later measured a simple INT8 WMMA Q4 GEMV candidate at 0.415x the reference, so that
specific kernel was rejected. The active E054-E075 bottleneck work and current next step are
recorded at the top of this file.
