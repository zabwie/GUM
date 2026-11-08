# Findings and source notes

## Confirmed on this machine

- `nvidia-smi -q` reports **Addressing Mode: HMM** on Linux kernel 7.0.0-38.
- CUDA 13.2 reports CC 8.6, `canMapHostMemory=1`, `unifiedAddressing=1`, `concurrentManagedAccess=1`, `pageableMemoryAccess=1`, `pageableMemoryAccessUsesHostPageTables=0`, and `hostNativeAtomicSupported=0`.
- A CUDA kernel successfully read a mapped pinned-host allocation and returned the expected checksum. H1 is functionally supported.
- For this sequential float4 scan, five-run H2D copy-only throughput stayed near 14.7 GB/s from 16 to 128 MiB. Mapped-host reads were about 14.1 GB/s at 16 MiB, 13.3 at 24 MiB, 13.0 at 32 MiB, 11.4 at 64 MiB on the repeat, and 12.1 at 128 MiB. H2D plus the same scan was about 13.7–14.2 GB/s over 16–128 MiB. Staging was nearly tied at 16 MiB and faster at 24 MiB and above in the tested runs. This is workload-specific, not an inference break-even point.
- The warmed VRAM scan measured about 312–337 GB/s at 16–128 MiB after three warm-ups. The first 64 MiB run had two outliers, but later samples were stable.
- On a CPU-initialized 64 MiB managed allocation, five independent first GPU scans took a 5.201 ms median. Repeated scans after the first took 0.210 ms median. Five explicit GPU-prefetch-plus-scan trials took 3.108 ms median. Checksums passed in all trials.
- Nsight Systems Unified Memory tracing recorded 640 MiB host-to-device and 320 MiB device-to-host migration across five CPU-first allocations and five explicit CPU↔GPU prefetch cycles, plus 960 CPU and 2,305 GPU page faults. The migration total matches the experiment's designed transfers. It establishes that migration occurs; it does not establish inference performance.
- A cache-evicted strided-gather test found that the preferred path changes with selectivity and reuse: mapped access was fastest at 1% for 1–4 passes and at 10% for one pass; full managed prefetch was fastest at 10% for 2+ passes and at 25% or 100% even for one pass. This supports selective data movement in this synthetic workload, not an inference policy.
- A `cudaMemGetInfo` snapshot showed 128 MiB less free VRAM after the first GPU access to the 64 MiB managed buffer, but free memory did not return after CPU prefetch. This global counter is inconclusive about current page placement and is not used to claim physical residency.
- Existing upstream llama.cpp CUDA support has a unified-memory mode. Its build guide documents `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; source inspection shows it selects `cudaMallocManaged` in place of `cudaMalloc`. This is an existing inference baseline to evaluate before a custom allocator/cache.
- The GPU link is PCIe 4.0 x16 at 16 GT/s. The approximately 31.5 GB/s one-way figure is a theoretical line-rate ceiling before overhead.

## Interpretation and limits

- The 64 MiB first-touch timing and Nsight migration records are consistent with demand-driven page migration. Explicit prefetch reduced the measured prefetch-plus-scan median from 5.201 ms to 3.108 ms in this synthetic workload, about 40% lower latency. The comparison is preliminary: the no-prefetch arm uses independent fresh allocations, the prefetch arm repeats on one allocation, and the benchmark is not a transformer kernel.
- The near-VRAM-speed repeat scan and observed migration totals are consistent with pages becoming GPU-local after first touch. Nsight confirms transfer direction and byte totals, but this benchmark does not query exact residency for every page.
- A compact partial-data cache, chunked prefetch, page-level locality policy, and real quantized tensor access remain untested.
- No local LLM inference throughput, model larger than VRAM, cache hit rate, expert-routing benefit, or end-to-end system-RAM extension has been measured.
- No novelty claim is made.

## NVIDIA sources

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [CUDA 13.2 Programming Guide: Unified Memory and prefetching](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/04-special-topics/unified-memory.html)
- [CUDA Runtime API: `cudaMemPrefetchAsync` and memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA Runtime API: `cudaHostGetDevicePointer`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

## Existing inference implementation

The upstream llama.cpp CUDA backend's unified-memory flag is a path to test, not evidence of speed or correctness on this machine. Compare it against ordinary offload with the same model, quantization, context, batch, prompt, and output length. Do not modify an existing user inference project or model files.

Sources inspected 2026-11-08:

- [llama.cpp build guide: unified-memory runtime flag](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp CUDA allocator: managed vs device allocation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu#L175-L200)
