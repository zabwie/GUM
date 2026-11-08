# Findings and source notes

## Confirmed on this machine

- `nvidia-smi -q` reports **Addressing Mode: HMM** on Linux kernel 7.0.0-38. This is evidence that the active NVIDIA stack selected HMM addressing; the CUDA runtime attributes still need a direct query.
- The CUDA 13.2 runtime query reports `canMapHostMemory=1`, `unifiedAddressing=1`, `concurrentManagedAccess=1`, `pageableMemoryAccess=1`, and `pageableMemoryAccessUsesHostPageTables=0`. Per NVIDIA's capability classification, this combination indicates full unified-memory access with software coherence on this Linux setup. The query did not allocate sample buffers or launch a kernel.
- `hostNativeAtomicSupported=0`; the initial prototype uses no host atomics.
- Existing upstream llama.cpp CUDA support already has a unified-memory mode. Its build guide documents `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; current CUDA backend source selects `cudaMallocManaged` for CUDA allocations when enabled and `cudaMalloc` otherwise. This is an existing managed-allocation baseline to evaluate before developing a separate allocator/cache.
- The GPU is linked at 16.0 GT/s x16, corresponding to PCIe 4.0 x16 signaling. About 31.5 GB/s one-way is a theoretical link-rate ceiling before overhead.
- CUDA 13.2 is installed alongside CUDA 12.0. The project benchmark is built with the installed 13.2 compiler at `/usr/local/cuda/bin/nvcc`, matching the driver-reported CUDA level.
- At discovery, GPU use was too high for a clean benchmark; no GPU result exists yet.

## Supported by NVIDIA documentation

- Mapped pinned host memory can be exposed to a GPU kernel through `cudaHostAllocMapped` and `cudaHostGetDevicePointer`; this is direct access to host memory over the CPU-GPU interconnect. NVIDIA describes its latency as higher and bandwidth as lower than device memory.
- On Linux with HMM, ordinary system allocations may participate in CUDA unified memory. Managed-memory residency, page migration, and access behavior depend on hardware and runtime attributes. Unified addressing by itself does not mean equal physical bandwidth or a single pool of VRAM.
- A user-visible HMM mode and successful CUDA attribute queries are separate evidence. The benchmark will print `concurrentManagedAccess`, `pageableMemoryAccess`, and `pageableMemoryAccessUsesHostPageTables` instead of inferring all behavior from the kernel version.

Sources:

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [CUDA 12.0 Programming Guide: mapped page-locked memory](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/)
- [CUDA Runtime API: `cudaHostGetDevicePointer`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

## Not established

- HMM does not establish that the RTX 3060 can keep an arbitrarily large model in host RAM and access it at VRAM-like speed.
- The kernel has not yet successfully read the mapped pinned allocation; `canMapHostMemory=1` only establishes that the device advertises the capability.
- No mapped-host or managed-memory bandwidth, page-fault count, inference throughput, or cache hit rate has been measured.
- No novelty claim is made.

## Existing inference implementation

The upstream llama.cpp CUDA backend has a runtime flag for unified memory. Its documented purpose is to let allocation spill into system RAM when VRAM is exhausted. Source inspection shows the switch changes the CUDA backend allocator from `cudaMalloc` to `cudaMallocManaged`; this confirms an implementation path exists, not that the current model/runtime is fast or correct on this machine. Use it as a future inference baseline after the microbenchmark, and compare enabled/disabled behavior with an identical model/workload.

Sources inspected 2026-11-08:

- [llama.cpp build guide: unified-memory runtime flag](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp CUDA allocator: managed vs device allocation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu#L175-L200)
