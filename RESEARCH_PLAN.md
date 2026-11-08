# Research plan

## Goal

Find the best reproducible way for local inference to use system RAM with the RTX 3060 while keeping neural-network computation on the GPU. Establish measured memory limits before building inference-specific machinery.

## Work order

1. Discover hardware, driver, CUDA toolkits, Linux HMM status, and current resource pressure.
2. Query CUDA device attributes at runtime; distinguish mapped access, managed-memory migration, and pageable HMM access.
3. Measure a bounded streaming read from mapped pinned host RAM against pinned H2D staging plus a VRAM scan and a VRAM-only scan.
4. Add separate managed-memory cold-touch and steady-state experiments, with migration behavior recorded.
5. Measure latency and bandwidth at useful tensor sizes and access patterns; compare with PCIe and DRAM ceilings.
   - **Completed:** Add randomized dependent-load tests from 1 to 960 MiB and measure independent-chain scaling, managed demand faults, full prefetch, and a migration trace.
6. **Completed:** Measure a compact GPU-side cache, synthetic Q4_0 matvec paths, and a bounded Q4_0 tensor GEMV. Add serialized and double-buffered chunk pipelines; chunk reuse remains slower than one-time staging/prefetch for the tested working sets.
   - **Completed:** Extend the synthetic Q4_0 matrix benchmark through 960 MiB. Managed prefetch remains the fastest tested host-backed path for this matrix workload.
7. **Completed for the small-model baseline:** Compare llama.cpp ordinary CUDA allocation and GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 on Qwen3-0.6B Q4_0 with matched generation settings. Both outputs passed; no speed difference was established.
8. **Pending explicit budget approval:** Run a 13 GiB managed working set and then an over-VRAM quantized LLM comparison only after the user approves the revised caps. See OVERSUBSCRIPTION_PLAN.md.

9. **Next safe stage:** Implement and measure dynamic hot/cold residency and eviction using a bounded GPU/host allocation within the existing limits.
10. **Completed:** CUDA VMM on this device can map host and device physical allocations into adjacent segments of one GPU-visible virtual range.

## Guardrails

- One GPU experiment at a time.
- Maximum incremental allocation: 2 GiB VRAM and 4 GiB system RAM; initial benchmark uses 64 MiB of each.
- No sustained GPU run while the desktop or another workload is materially using the GPU.
- No global installs, driver changes, elevated commands, system configuration changes, or edits outside this project.
- Do not use swap as working memory.
- Record raw observations, exact command, source revision, CUDA/driver versions, and interpretation separately.
