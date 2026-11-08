# Findings and source notes

## Confirmed on this machine

- `nvidia-smi -q` reports **Addressing Mode: HMM** on Linux kernel 7.0.0-38.
- CUDA 13.2 reports CC 8.6, `canMapHostMemory=1`, `unifiedAddressing=1`, `concurrentManagedAccess=1`, `pageableMemoryAccess=1`, `pageableMemoryAccessUsesHostPageTables=0`, and `hostNativeAtomicSupported=0`.
- A CUDA kernel successfully read a mapped pinned-host allocation and returned the expected checksum. H1 is functionally supported.
- For this sequential float4 scan, five-run H2D copy-only throughput stayed near 14.7 GB/s from 16 to 128 MiB. Mapped-host reads were about 14.1 GB/s at 16 MiB, 13.3 at 24 MiB, 13.0 at 32 MiB, 11.4 at 64 MiB on the repeat, and 12.1 at 128 MiB. H2D plus the same scan was about 13.7–14.2 GB/s over 16–128 MiB. Staging was nearly tied at 16 MiB and faster at 24 MiB and above in the tested runs. This is workload-specific, not an inference break-even point.
- In a randomized dependent pointer chase, mapped pinned-host access measured 586–754 ns/load across 1–64 MiB, versus 218–274 ns/load from VRAM. At 960 MiB, mapped was 1,108 ns/load versus 279 ns/load in VRAM. This is latency for this synthetic pointer chase, not streaming bandwidth.
- With many independent chains at 64 MiB, mapped-host aggregate throughput plateaued around 120–125 million 32-bit loads/s by 256–4,096 chains. At 960 MiB it was about 76–87 million loads/s across 256–4,096 chains. Managed full-prefetch became faster than mapped between 896 and 1,024 chains; demand migration crossed between 1,024 and 1,536 chains in this workload.
- Nsight traces of the 960 MiB managed test changed with chain count. At 1, 256, and 4,096 chains, aggregate H2D/D2H migration totals were 3,823.829/1,817.969 MB, 5,833.306/5,033.165 MB, and 5,932.446/5,033.165 MB respectively. GPU page-fault counts fell from 10,674 to 965 to 458. Each report includes both managed paths and warmups, so the totals do not assign exact bytes to the demand or prefetch arm.
- The warmed VRAM scan measured about 312–337 GB/s at 16–128 MiB after three warm-ups. The first 64 MiB run had two outliers, but later samples were stable.
- On a CPU-initialized 64 MiB managed allocation, five independent first GPU scans took a 5.201 ms median. Repeated scans after the first took 0.210 ms median. Five explicit GPU-prefetch-plus-scan trials took 3.108 ms median. Checksums passed in all trials.
- Nsight Systems Unified Memory tracing recorded 640 MiB host-to-device and 320 MiB device-to-host migration across five CPU-first allocations and five explicit CPU↔GPU prefetch cycles, plus 960 CPU and 2,305 GPU page faults. The migration total matches the experiment's designed transfers. It establishes that migration occurs; it does not establish inference performance.
- A cache-evicted strided-gather test found that the preferred path changes with selectivity and reuse: mapped access was fastest at 1% for 1–4 passes and at 10% for one pass; full managed prefetch was fastest at 10% for 2+ passes and at 25% or 100% even for one pass. This supports selective data movement in this synthetic workload, not an inference policy.
- A compact GPU-side cache prototype gathered selected mapped-host float4 values into VRAM and reused them. At 1% with four passes it took 0.302 ms vs 1.172 ms mapped and 2.957 ms full managed prefetch; at 10%/four passes it took 2.123 ms vs 8.167 ms mapped and 2.985 ms managed prefetch. One-pass cache gather was tied with direct mapping at 1% and 10%; at 25–100% full managed prefetch remained faster.
- A synthetic Q4_0 GEMV benchmark at 32, 64, and 128 MiB found managed prefetch faster than mapped and full H2D paths for the tested sizes and reuse counts. At 64 MiB/four passes, medians were 25.603 ms mapped, 6.462 ms staged, 4.778 ms managed-prefetch, and 1.860 ms with weights already resident in VRAM. This is still a synthetic kernel.
- Extending the synthetic Q4_0 GEMV to 256 and 960 MiB preserved that ordering. At 960 MiB/four passes, medians were 440.832 ms mapped, 103.612 ms staged, 76.531 ms managed-prefetch, and 38.655 ms VRAM-resident. All CPU-reference outputs passed; the entire matrix fit in VRAM.
- A bounded GEMV on the real `blk.0.ffn_down.weight` Q4_0 tensor (5,120 × 17,408; 47.812 MiB) passed CPU-reference validation after the reference decoded per-block half scales. With synthetic input vectors, one-pass medians were 4.168 ms mapped, 3.833 ms staged, 2.532 ms managed-prefetch, and 0.351 ms VRAM-resident; four-pass managed-prefetch was 3.473 ms versus 14.286 ms for 16 MiB double-buffered chunks, which reloaded weights each pass. This does not measure model inference.
- A double-buffered copy/compute pipeline improved chunked medians versus serialized chunks in several measured runs. On the same real tensor, 8 MiB chunks measured 3.472 ms for one pass and 13.788 ms for four passes; the chunk-size sweep was noisy across separate preflights and does not establish an optimum. Full staging and managed prefetch remained faster for reused weights because they moved the tensor only once.
- The RTX 3060 reports host registration support but not read-only host-registration support. Since this system does not use host page tables for pageable memory, the CUDA Runtime API's read-only registration requirement prevents registering the model's read-only file mapping. The test skipped registration before mapping or touching model pages.
- A `cudaMemGetInfo` snapshot showed 128 MiB less free VRAM after the first GPU access to the 64 MiB managed buffer, but free memory did not return after CPU prefetch. This global counter is inconclusive about current page placement and is not used to claim physical residency.
- Existing upstream llama.cpp CUDA support has a unified-memory mode. Its build guide documents `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; source inspection shows it selects `cudaMallocManaged` in place of `cudaMalloc`. This is an existing inference baseline to evaluate before a custom allocator/cache.
- The GPU link is PCIe 4.0 x16 at 16 GT/s. The approximately 31.5 GB/s one-way figure is a theoretical line-rate ceiling before overhead.

## New results from the continuation

- E023 verified one GPU kernel accessing a 1 GiB device allocation and 2 GiB mapped pinned-host allocation. All initialization, transforms, and validation ran on GPU. Five-sample medians were 6.620 ms for VRAM-only, 393.268 ms for mapped-host-only, and 394.184 ms for mixed. This confirms combined access, while host traffic dominates the stream.
- E024 demonstrated one contiguous GPU virtual address range with separate device and host physical backing through CUDA VMM. The RTX 3060 reported both VMM capabilities; minimum allocation granularity was 2 MiB. One pointer kernel correctly touched the 64 MiB device and 128 MiB host regions.
- E025 ran Qwen3-0.6B Q4_0 through llama.cpp in ordinary and managed-allocation modes. Both returned the correct answer and offloaded 29/29 layers. Five-repetition decode means were 383.054 and 381.812 tokens/s. The model fits in VRAM, so the result does not establish oversubscription or faster inference.
- CUDA VMM explicitly provides create, reserve, map, and access-right operations; its allocation granularity must be queried and respected. The device minimum measured 2 MiB here. See the [CUDA Driver API VMM](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html) and [CUDA Programming Guide VMM](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html).
- CUDA managed allocations can oversubscribe device memory when concurrent managed access is supported; migration and prefetch are controlled by the runtime and guided by usage hints. This machine reports concurrent managed access, but an over-VRAM working set has not yet been run. See the [CUDA Runtime memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) and [CUDA Unified Memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
- Linux HMM is driver infrastructure for address-space mirroring and page migration. Its presence in nvidia-smi does not specify a user-space mixed-memory policy. See the [Linux HMM documentation](https://docs.kernel.org/mm/hmm.html).

## Interpretation and limits

- The 64 MiB first-touch timing and Nsight migration records are consistent with demand-driven page migration. Explicit prefetch reduced the measured prefetch-plus-scan median from 5.201 ms to 3.108 ms in this synthetic workload, about 40% lower latency. The comparison is preliminary: the no-prefetch arm uses independent fresh allocations, the prefetch arm repeats on one allocation, and the benchmark is not a transformer kernel.
- The near-VRAM-speed repeat scan and observed migration totals are consistent with pages becoming GPU-local after first touch. Nsight confirms transfer direction and byte totals, but this benchmark does not query exact residency for every page.
- The cache prototype still only caches a synthetic strided float4 set. The Q4_0 tensor test uses real layer weights but synthetic vectors and a single GEMV, so it is not end-to-end model execution.
- A small Qwen3 LLM has now been measured end to end, but no model larger than available VRAM, dynamic cache hit rate, expert-routing benefit, or full system-RAM extension has been measured. The historical 14.1 GiB base GGUF was not loaded under the prior 4 GiB host cap.
- The 960 MiB random-load set still fits in the RTX 3060 VRAM; the latest tests do not establish oversubscription behavior. Chain-count crossover values are workload-specific and should not be used as inference thresholds.
- No novelty claim is made.

## NVIDIA sources

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [CUDA 13.2 Programming Guide: Unified Memory and prefetching](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/04-special-topics/unified-memory.html)
- [CUDA Runtime API: `cudaMemPrefetchAsync` and memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
- [CUDA Runtime API: `cudaHostGetDevicePointer`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

## Existing inference implementation

The upstream llama.cpp CUDA backend's unified-memory flag has now been measured against ordinary allocation for a fitting 0.6B model. The run validates the small-model path but not host-backed weights or oversubscription. Use the same model and decode settings in the next approved over-VRAM comparison; do not modify the existing llama.cpp checkout.

Sources inspected 2026-11-08:

- [llama.cpp build guide: unified-memory runtime flag](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp CUDA allocator: managed vs device allocation](https://github.com/ggml-org/llama.cpp/blob/d775b8967a46d8beb110d444aa3b8938179e0dd8/ggml/src/ggml-cuda/ggml-cuda.cu#L135-L155)
