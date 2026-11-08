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
- E013 found no standalone model small enough to verify inference within the 4 GiB additional-host-RAM cap. The 15.15 GB (14.1 GiB) base GGUF was not loaded.
- Raw outputs, commands, source revisions, preflight conditions, and interpretation are in [EXPERIMENTS.md](EXPERIMENTS.md) and `results/`.

## Implementation

- `benchmarks/memory_paths.cu`: mapped-host, H2D-only, H2D-plus-scan, VRAM scan; 1–256 MiB input.
- `benchmarks/managed_paths.cu`: managed CPU-first reads, independent cold allocations, repeated scans, CPU/GPU prefetch, checksum validation, and device-free-memory snapshots.
- `benchmarks/reuse_paths.cu`: mapped, full H2D, or managed-prefetch paths across 1/10/25/100% strided access and 1/2/4 reuse passes, with a pre-trial L2 eviction scan.
- `benchmarks/q4_matvec.cu`: mapped, staged, managed-prefetch, and VRAM Q4_0 GEMV-style paths; optional read-only model-file input, CPU reference validation, and serialized/double-buffered chunk copies.
- `benchmarks/host_latency.cu`: deterministic dependent random-load chains across mapped host, managed demand/prefetch, and VRAM; supports 1–960 MiB under the project allocation limits.
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
- A real Q4_0 tensor GEMV slice works, but model inputs and neural-network execution were not tested.
- `cudaMemGetInfo` changed by 128 MiB after first access and did not return after CPU prefetch; this global counter does not prove current placement.
- llama.cpp's existing `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` path remains the inference baseline to compare. No model was run because no standalone model fit the current RAM cap; read-only CUDA registration of file-backed weights is unsupported by the device.

## Next executable experiments

Implement page-selective managed prefetch for the nodes reached by the randomized chains and compare it with mapped access, demand migration, and full-allocation prefetch at sparse and dense chain counts. Nsight showed that low fanout has many faults while higher fanout moves nearly the full allocation; test whether selective prefetch reduces both costs.

Compare llama.cpp's ordinary offload and `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` using a standalone model whose measured host-memory use stays within the existing 4 GiB additional-RAM cap. Keep context, batch, prompt, output length, and GPU-layer settings matched; record prefill/decode speed, first-token latency, correctness, and memory use. No such base model was found locally, so do not load the 14.1 GiB GGUF under the current cap. If no suitable model becomes available, continue with bounded tensor-level experiments rather than claiming end-to-end inference results.
