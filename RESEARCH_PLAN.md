# Research plan

## Goal

Find the best reproducible way for local inference to use system RAM with the RTX 3060 while keeping neural-network computation on the GPU. Establish measured memory limits before building inference-specific machinery.

## Work order

1. Discover hardware, driver, CUDA toolkits, Linux HMM status, and current resource pressure.
2. Query CUDA device attributes at runtime; distinguish mapped access, managed-memory migration, and pageable HMM access.
3. Measure a bounded streaming read from mapped pinned host RAM against pinned H2D staging plus a VRAM scan and a VRAM-only scan.
4. Add separate managed-memory cold-touch and steady-state experiments, with migration behavior recorded.
5. Measure latency and bandwidth at useful tensor sizes and access patterns; compare with PCIe and DRAM ceilings.
6. Prototype a bounded GPU-side cache and a simple prefetch/staging pipeline only after measurements identify a useful reuse opportunity.
7. Before inventing an allocator, evaluate llama.cpp's existing Linux CUDA path (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, which switches CUDA allocations to `cudaMallocManaged`) against its ordinary CUDA allocation/offload baseline. Do this only after microbenchmarks establish the transfer behavior and the run can respect resource limits.
8. Integrate an available local inference workload only after the preceding baseline is reproducible; add a GPU-side cache/prefetch prototype only if the evidence identifies a useful reuse opportunity.

## Guardrails

- One GPU experiment at a time.
- Maximum incremental allocation: 2 GiB VRAM and 4 GiB system RAM; initial benchmark uses 64 MiB of each.
- No sustained GPU run while the desktop or another workload is materially using the GPU.
- No global installs, driver changes, elevated commands, system configuration changes, or edits outside this project.
- Do not use swap as working memory.
- Record raw observations, exact command, source revision, CUDA/driver versions, and interpretation separately.
