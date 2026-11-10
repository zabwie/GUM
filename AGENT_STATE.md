# Agent state

## Objective

Measure and improve local LLM inference throughput when model state exceeds the RTX 3060's 12 GiB VRAM, using supported system-memory mechanisms and reproducible evidence.

## Verified platform

- RTX 3060 12 GiB, CC 8.6 / 28 SMs; Intel i7-12700K; 31 GiB RAM.
- PCIe 4.0 x16 at 16 GT/s; Linux kernel 7.0.0-38; NVIDIA driver 595.99.02; CUDA 12.0 and 13.2 installed.
- `nvidia-smi -q` reports HMM. CUDA 13.2 reports mapped-host support, UVA, concurrent managed access, pageable memory access, and software-coherent pageable HMM. Host-native atomics are not supported.
- See [HARDWARE.md](HARDWARE.md) for evidence and remaining hardware questions.

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
- Raw outputs, commands, source revisions, preflight conditions, and interpretation are in [EXPERIMENTS.md](EXPERIMENTS.md) and `results/`.

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
- A real Q4_0 tensor GEMV slice works, and a small quantized LLM has now run end to end. No model larger than available VRAM or dynamic residency manager has been validated.
- `cudaMemGetInfo` changed by 128 MiB after first access and did not return after CPU prefetch; this global counter does not prove current placement.
- llama.cpp's managed-memory switch completed a small fitting-model baseline. Read-only CUDA registration of file-backed weights remains unsupported; over-VRAM inference remains untested.

## Next executable experiments

The 13 GiB managed-allocation experiment (E026) and the page-aware residency manager (E025) are complete. The 24 GiB attempt (E027) aborted at the RAM floor; the tiered NVMe/RAM/VRAM pool (E028-E030) streams 24-48 GiB at 6-7 GB/s with flat RAM/swap. E031 ran a streamed decode proxy on the 13.42 GiB Qwen3.8-27B STRICT Q4_0 model: 1.021 tok/s with a full 5.55 GiB RAM cache (zero NVMe in steady state), 0.721 tok/s with all cold weights host-resident.

- E032 (overlap fix): Fixed slot-reuse event bug in streamed_decode.cu (one event per slot, primed at startup). Overlap build now PASSES verification (0 mismatches, 3976 rows). Steady-state 1.247 tok/s (up from 1.021), effective 17.9 GB/s. Raw: results/streamed-decode-27b-hot8-cache5.6-overlap-20261108.txt.
- E033/E034 (speed): dp4a int8 GEMV (~275 GB/s), decode batching, hot-arena third stream, pinned RAM-cache head (3658 MiB; memlock hard cap 3978 MiB cannot be raised without root). Final binary all PASS zero mismatches: batch 1 = 1.951 tok/s, batch 16 = 31.8, batch 32 = 43.8 aggregate tok/s (127,232 rows), batch 64 = 37.9 (96 regs, occupancy loss). 43x E031; meets the 20-40 tok/s target as aggregate. Single-stream ceiling ~2.5 tok/s is PCIe physics (5.82 GB cold/token). Packed-float2 scales were 27% slower, reverted. Raw: results/streamed-decode-27b-hot8-cache5.6-final-batch{1,16,32}-20261108.txt.

- E036 (real speculative decoding, measured): MTP-capable llama-cli built out-of-source (/home/zabwie/llama-mtp-build, d775b8967; user checkout untouched). Real 27B over-VRAM via unified memory: baseline 0.63 t/s, native MTP head 1.04 t/s = 1.65x, acceptance 23% (mean len 2.84). KPI verdict: >=3x bytes/accepted-token NOT met — H2D ~7 GiB/token in both arms (managed-memory thrashing, weights+KV+draft > 12 GiB). Speedup is overhead amortization, not traffic reduction; temp-0 outputs diverge via batched-verification FP nondeterminism. Real inference lags the staged proxy because llama.cpp unified memory has no intelligent residency. Raw: results/e036-{baseline,mtp}-20261109-171251*.txt.

- E037 (IOMMU passthrough, measured; user-approved): `iommu=pt` + reboot flipped all 20 IOMMU groups DMA-FQ->identity. H2D 14.33->24.61 GB/s (1.72x), D2H 12.67->25.87 (2.04x) — the IOMMU was a ~42% config bandwidth tax, not silicon. Proxy throughput doubled: single-stream 1.951->4.009 tok/s, batch-16 31.8->64.1 aggregate (both PASS zero mismatches). Single-stream now near the Gen4-passthrough ceiling (4.23 theoretical). Persistent GRUB change. Raw: results/h2d-d2h-iommu-pt-*.txt, results/streamed-decode-27b-iommu-pt-batch{1,16}-20261109-172744.txt.

Next: (1) highest leverage — port the staged proxy's intelligent residency into a REAL decode loop (attention/SSM + KV) to close the 4 tok/s proxy vs 0.6-1.0 t/s llama.cpp unified-memory thrashing gap (E036); the IOMMU bandwidth lever is exhausted and single-stream is at the Gen4 ceiling, so further single-stream gains need fewer bytes/token (approximate-layer drafting + exact verification done right); (2) optional tensor-core int8 MMQ kernel to close the batch-32 compute gap (ceiling ~64 tok/s).
