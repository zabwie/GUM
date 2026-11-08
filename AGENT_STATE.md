# Agent state

## Objective

Measure and improve local LLM inference throughput when model state exceeds the RTX 3060's 12 GiB VRAM, using supported system-memory mechanisms and reproducible evidence.

## Verified platform

- RTX 3060 12 GiB, CC 8.6 / 28 SMs; Intel i7-12700K; 31 GiB RAM.
- PCIe 4.0 x16 at 16 GT/s; Linux kernel 7.0.0-38; NVIDIA driver 595.99.02; CUDA 12.0 and 13.2 installed.
- `nvidia-smi -q` reports HMM. CUDA 13.2 reports mapped-host support, UVA, concurrent managed access, pageable memory access, and software-coherent pageable HMM. Host-native atomics are not supported.
- See [HARDWARE.md](HARDWARE.md) for evidence and remaining hardware questions.

## Completed measurements

- E003/E004 prove mapped pinned host memory is kernel-readable and checksum-correct. For a sequential float4 scan, pinned H2D copy is about 14.7 GB/s; staging is near mapped access at 16 MiB and faster from 24 to 128 MiB in the tested sizes. VRAM reads are about 312–337 GB/s after warm-up. Results and ranges are in [BENCHMARKS.md](BENCHMARKS.md).
- E005/E006 measured 64 MiB managed memory. Five fresh CPU-initialized first GPU reads: 5.201 ms median; five repeated scans: 0.210 ms; explicit GPU prefetch plus scan: 3.108 ms.
- E008 Nsight trace confirmed 640 MiB H2D and 320 MiB D2H migration, plus 960 CPU and 2,305 GPU page faults, across five cold first-touch allocations and five prefetch round trips. Profiled timings are not used.
- Raw outputs are in `results/`; chronology, source revisions, procedures, and interpretation are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Implementation

- `benchmarks/memory_paths.cu`: mapped-host, H2D-only, H2D-plus-scan, VRAM scan; 1–256 MiB input.
- `benchmarks/managed_paths.cu`: managed CPU-first reads, independent cold allocations, repeated scans, CPU/GPU prefetch, checksum validation, and device-free-memory snapshots.
- `scripts/run_when_idle.sh`: sampled resource gate. It records CUDA clients and their memory but judges contention using sustained per-process SM activity, global utilization, temperature, and RAM/VRAM headroom. `PROFILE=um_faults` enables bounded Nsight Unified Memory tracing for `managed_paths`.
- Source revisions: H2D measurement `d47260b`; managed benchmark `2a4d9a9`; latest guard/profile support `67da456`.

## Active conclusions and limits

- H1 is supported: mapped pinned host memory is directly readable by a GPU kernel on this machine.
- H2 is supported for the tested contiguous stream at 24 MiB and above; 16 MiB was near a tie. This is not an inference threshold.
- H4 is supported in this microbenchmark: explicit prefetch plus scan was about 40% lower in median elapsed time than demand first access. The arms differ in allocation/reuse pattern, so repeat with inference-like access before selecting an architecture.
- Managed pages demonstrably migrate, but model-scale residency, local LLM throughput, cache hit rate, and sparse accesses remain unmeasured.
- `cudaMemGetInfo` changed by 128 MiB after first access and did not return after CPU prefetch; this global counter does not prove current placement.
- llama.cpp's existing `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` path is the future inference baseline; no model was run.

## Next executable experiment

Build a bounded reuse/selectivity test over a 64 MiB host tensor: compare mapped-host reads against explicit prefetch followed by 1, 2, and 4 GPU passes, with full scan and sparse 1%, 10%, and 25% access patterns. Report useful bytes, migration bytes, and total elapsed time. This will test when direct access or staging is worthwhile before adding a cache or attempting model inference.
