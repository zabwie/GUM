# Initial hypotheses

## H1 — CUDA can run a kernel against system-resident pinned memory

**Claim to falsify:** This RTX 3060 and current CUDA stack can map a pinned host buffer and let a kernel read it correctly without first copying the full buffer into VRAM.

**Prediction:** Runtime reports mapped-host support; allocation and device-pointer lookup succeed; the kernel's checksum agrees with the CPU reference. Failure at any of those stages is recorded with the exact CUDA error and capability values.

**Smallest experiment:** `benchmarks/memory_paths.cu`, 64 MiB pinned buffer, one streaming reduction, checksum validation.

## H2 — Direct mapped reads are less effective for bulk streaming than explicit staging

**Claim to falsify:** For a one-pass contiguous scan, mapped-host access has lower end-to-end effective input bandwidth than an explicit pinned H2D copy followed by the same VRAM kernel.

**Prediction:** Staging will likely win for a 64 MiB streaming pass, while direct access avoids the explicit copy and can remain useful for sparse or low-reuse data. This is a prediction, not a result. Compare five measured repetitions after a warm-up, with identical source values and the same scan kernel.

**Metrics:** Per-run elapsed time, effective input GB/s, checksum validity, runtime/device attributes, GPU/host memory before and after. The VRAM scan is a device-local bandwidth reference. Report min and median, not just the best sample.

**Limitations:** This synthetic contiguous reduction is not transformer inference, quantized GEMV, attention, or MoE. It cannot establish end-to-end token throughput or cache benefit.

## H3 — an existing managed-memory inference path is a stronger baseline than a new allocator

**Observation:** The upstream llama.cpp CUDA backend already switches its allocations from `cudaMalloc` to `cudaMallocManaged` when `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is set. The local CUDA runtime advertises concurrent managed access and HMM pageable-memory access.

**Claim to falsify:** With a model whose working set exceeds free VRAM, llama.cpp's existing managed-memory path completes GPU inference with correct outputs and lower host-side layer-compute cost than ordinary CPU/GPU layer offload, even if token throughput is slower than a VRAM-resident run.

**Prediction:** It should be a usable oversubscription baseline on this Linux/HMM setup. Decode throughput may be constrained by page movement and access locality. This is untested.

**Experiment:** After the memory microbenchmark, build or use a project-local llama.cpp checkout and compare the same model, quantization, context, batch, prompt, output length, and GPU layer configuration with the flag off/on. Record output correctness, prefill/decode rate, first-token latency, GPU/RAM use, and any page-fault/migration evidence. Do not touch an existing user inference project or model files.
