# Initial hypotheses

## H1 — CUDA can run a kernel against system-resident pinned memory

**Claim to falsify:** This RTX 3060 and current CUDA stack can map a pinned host buffer and let a kernel read it correctly without first copying the full buffer into VRAM.

**Prediction:** Runtime reports mapped-host support; allocation and device-pointer lookup succeed; the kernel's checksum agrees with the CPU reference. Failure at any of those stages is recorded with the exact CUDA error and capability values.

**Smallest experiment:** `benchmarks/memory_paths.cu`, 64 MiB pinned buffer, one streaming reduction, checksum validation.

## H2 — Direct mapped reads are less effective for bulk streaming than explicit staging

**Claim to falsify:** For a one-pass contiguous scan, mapped-host access has lower end-to-end effective input bandwidth than an explicit pinned H2D copy followed by the same VRAM kernel.

**Prediction:** Staging will likely win for a 64 MiB streaming pass, while direct access avoids the explicit copy and can remain useful for sparse or low-reuse data.

**Metrics:** Per-run elapsed time, effective input GB/s, checksum validity, runtime/device attributes, GPU/host memory before and after. The VRAM scan is a device-local bandwidth reference. Report min and median, not just the best sample.

**Limitations:** This synthetic contiguous reduction is not transformer inference, quantized GEMV, attention, or MoE. It cannot establish end-to-end token throughput or cache benefit.

**Results:** Five-run size checks were close at 16 MiB (mapped 14.091 vs staged 14.173 GB/s), but staging led at 24 MiB (14.036 vs 13.293), 32 MiB (13.952 vs 12.977), 64 MiB on the repeat (13.656 vs 11.388), and 128 MiB (14.141 vs 12.121). E003's first 64 MiB run was noisier (mapped 10.009, staged 13.693 GB/s); E004 repeated it with longer warm-up and isolated H2D timing. H2 is supported for this contiguous scan at 24 MiB and above in the tested runs; 16 MiB was effectively tied. This does not establish a model-inference threshold.

## H3 — an existing managed-memory inference path is a stronger baseline than a new allocator

**Observation:** The upstream llama.cpp CUDA backend already switches its allocations from `cudaMalloc` to `cudaMallocManaged` when `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is set. The local CUDA runtime advertises concurrent managed access and HMM pageable-memory access.

**Claim to falsify:** With a model whose working set exceeds free VRAM, llama.cpp's existing managed-memory path completes GPU inference with correct outputs and lower host-side layer-compute cost than ordinary CPU/GPU layer offload, even if token throughput is slower than a VRAM-resident run.

**Prediction:** It should be a usable oversubscription baseline on this Linux/HMM setup. Decode throughput may be constrained by page movement and access locality. This is untested.

**Experiment:** After the memory microbenchmark, build or use a project-local llama.cpp checkout and compare the same model, quantization, context, batch, prompt, output length, and GPU layer configuration with the flag off/on. Record output correctness, prefill/decode rate, first-token latency, GPU/RAM use, and any page-fault/migration evidence. Do not touch an existing user inference project or model files.

## H4 — explicit managed-memory prefetch reduces first-use cost

**Claim to falsify:** For a CPU-initialized managed buffer, GPU-prefetching before a sequential GPU scan lowers total first-use time compared with letting the GPU demand the data.

**Results:** At 64 MiB, five independent CPU-first GPU scans had a 5.201 ms median; five GPU-prefetch-plus-scan trials had a 3.108 ms median. The latter is about 40% lower in this synthetic test. Nsight Systems observed 640 MiB H2D and 320 MiB D2H migrations across the five fresh first-touch trials and five explicit CPU↔GPU prefetch cycles. Repeated scans after first access were much faster (0.210 ms median), consistent with GPU-local reuse after migration.

**Limitations:** The no-prefetch and prefetch arms use different allocation/reuse patterns, profiling overhead invalidates profiled timing comparisons, and no transformer or inference workload was tested. Repeat with matched fresh allocations and measure realistic tensor access/reuse before using this result to guide an inference design.
