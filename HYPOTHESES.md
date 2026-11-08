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

## H5 — access selectivity and reuse determine whether full-buffer migration pays

**Claim to falsify:** Direct mapped reads are preferable for a small one-pass fraction of a large host buffer, while moving the full buffer to VRAM pays when enough data is used or reused.

**Experiment:** E009 sampled 100%, 25%, 10%, or 1% of float4 values at uniform strides across a 64 MiB buffer. It compared mapped reads, full pinned H2D staging, and full managed-memory prefetch for 1, 2, and 4 scans. Five samples per configuration used a 16 MiB L2 eviction scan before each timed trial; reuse within the passes of one trial remained available.

**Results:** Mapped access was fastest at 1% for 1–4 passes (0.380–1.260 ms) and at 10% for one pass (2.059 ms). Managed prefetch became fastest at 10% with two or four passes (2.978 and 2.998 ms), and at 25% or 100% even for one pass (2.974 and 3.130 ms). Every checksum passed.

**Limitations:** The gather is synthetic and uniformly strided; full-buffer staging/prefetch always moves 64 MiB. Before E010, no compact/selective cache path had been tested. These timings do not model quantized GEMV, attention, MoE routing, or end-to-end inference.

## H6 — compact GPU caching amortizes remote reads for repeated sparse access

**Claim to falsify:** Gathering only the selected mapped-host values into a compact VRAM buffer costs less than repeatedly reading them remotely once reuse is high enough.

**Results:** E010 adds a GPU gather into a compact buffer, then scans that buffer 1, 2, or 4 times. At 1% selectivity, the four-pass cache path took 0.302 ms versus 1.172 ms for repeated mapped reads and 2.957 ms for full managed prefetch. At 10% and four passes it took 2.123 ms versus 8.167 ms mapped and 2.985 ms prefetch. At 1–10% one-pass, cache gather and direct mapped access were roughly tied; at 25–100%, managed prefetch was faster.

**Limitations:** The gathered data are synthetic float4 values and the kernel does not model quantized weight formats or neural-network compute. `gathered_payload_MiB` counts useful values only; physical PCIe transaction bytes can be larger. No real inference benefit is established.

## H7 — managed prefetch benefits quantized matrix-vector reuse

**Claim to falsify:** For a Q4_0 matrix small enough to fit in available VRAM, moving it once with managed prefetch and reusing it for multiple matrix-vector passes is faster than mapped host reads or reloading chunks.

**Results:** On synthetic Q4_0 matrices from 32 to 128 MiB, managed prefetch beat mapped and full H2D staging at every tested size and pass count. On a real 47.812 MiB Q4_0 layer tensor with synthetic inputs, four-pass medians were 3.473 ms managed-prefetch, 4.765 ms full H2D, 17.324 ms mapped, and 14.286 ms double-buffered 16 MiB chunks. Every output passed the scale-aware CPU reference.

**Limitations:** The real-tensor test is a single GEMV with synthetic inputs; all tested matrices fit in VRAM. It does not establish an inference policy for weights that cannot remain resident.

## H8 — double-buffered chunk transfers hide copy latency

**Claim to falsify:** Copying the next Q4_0 chunk on one stream while computing on the previous chunk on another reduces total time versus serialized chunk-copy plus matvec.

**Results:** On the real tensor, observed double-buffered medians were lower than serialized medians for 4, 8, and 16 MiB chunks; at 8 MiB the one-pass median was 3.472 ms versus 3.869 ms serialized, and the four-pass median was 13.788 ms versus 15.698 ms. A synthetic 128 MiB run showed a larger 29% reduction, but global GPU utilization was 67–69% during preflight.

**Limitations:** The chunk-size runs had varying desktop GPU activity, so the apparent 8 MiB optimum is not established. Chunk streaming reloads the matrix on every pass; full staging and managed prefetch moved it once and were faster for reuse.

## H9 — CUDA can register a read-only model-file mapping for direct GPU access

**Claim to falsify:** A bounded read-only mmap of GGUF tensor pages can be registered and accessed directly by a CUDA kernel without first copying them into a CUDA-pinned allocation.

**Result:** The RTX 3060 reports `cudaDevAttrHostRegisterSupported=1` but `cudaDevAttrHostRegisterReadOnlySupported=0`. Because this system does not use host page tables for pageable memory, the CUDA Runtime API requires the read-only registration flag for this mapping; the API route is unavailable on this device. The benchmark exited before mapping or registering model pages.

**Alternative:** Keep using a bounded copy into mapped pinned host memory, which was measured successfully on the same real tensor.


## H10 — mapped-host dependent latency stays near a small multiple of VRAM

**Claim to falsify:** For the randomized dependent pointer chase, mapped pinned-host reads remain within four times the VRAM-resident latency as the working set grows to 960 MiB.

**Results:** Five-sample medians were 586.47 vs 217.94 ns/load at 1 MiB, 753.81 vs 273.78 ns/load at 64 MiB, and 1,108.27 vs 278.59 ns/load at 960 MiB for mapped host and VRAM respectively. The measured ratio rose from about 2.7–2.8× at 1–64 MiB to 4.0× at 960 MiB.

**Limitations:** One synthetic pointer-cycle kernel, one chain, and fixed 32,768-step walks. The 960 MiB set fits in VRAM. This is not a general remote-memory latency or inference estimate.

## H11 — independent chains expose a mapped-host throughput ceiling

**Claim to falsify:** Independent dependent-load chains hide much of the mapped-host latency up to a finite throughput ceiling, which falls as the randomized working set grows.

**Results:** At 64 MiB, mapped throughput rose from 1.316 million loads/s with one chain to about 120–125 million loads/s with 256–4,096 chains, then plateaued. At 960 MiB it stayed around 76–87 million loads/s for 256–4,096 chains. With 960 MiB, full managed prefetch crossed mapped-host time between 896 and 1,024 chains; managed demand migration crossed between 1,024 and 1,536 chains.

**Limitations:** These crossover brackets depend on the kernel, allocation, and GPU. The 896-chain prefetch comparison is close, and desktop utilization varied between preflights. These are synthetic load paths, not an inference scheduler.

## H12 — distributed random access can make managed demand migration nearly full-sized

**Claim to falsify:** When many disjoint random chains touch pages throughout a 960 MiB managed allocation, demand faults move a large fraction of the allocation and bulk prefetch becomes faster than fault-driven demand migration.

**Results:** At 4,096 chains, managed demand took 66.44 ms and full-allocation prefetch plus kernel took 47.31 ms. An Nsight trace containing two warmups and one measured sample of each managed arm recorded 5,932.446 MB H2D and 5,033.165 MB D2H migration, close to six and five full 960 MiB movements. The trace totals are aggregate and do not attribute each transfer to one arm.

**Limitations:** All data fit in VRAM; the trace includes both paths and warmups. It does not measure a model exceeding VRAM or prove a per-fault migration size.
## H13 — managed prefetch remains effective for large Q4_0 matvecs that fit in VRAM

**Claim to falsify:** For a large synthetic Q4_0 matrix-vector workload that fits in VRAM, full managed prefetch plus matvec remains faster than mapped reads, pinned H2D staging, and chunk reloading.

**Results:** With 4,096 rows and a 959.98 MiB Q4_0 matrix, all CPU-reference validations passed. One-pass medians were 108.887 ms mapped, 77.317 ms pinned staging, 51.965 ms managed prefetch, and 8.604 ms VRAM resident. Four-pass medians were 440.832, 103.612, 76.531, and 38.655 ms. Managed prefetch transfer-only measured 23.980 GB/s versus 14.738 GB/s pinned H2D for this run. At four passes, the double-buffered chunk path took 278.928 ms because it moved the matrix on each pass.

**Limitations:** Weights and vectors were synthetic; the matrix fit in VRAM. This extends the Q4_0 microbenchmark but does not measure inference or model oversubscription.


## H14 — managed migration depends on chain fanout

**Claim to falsify:** For a fixed 960 MiB managed allocation, increasing independent random-chain fanout raises aggregate migration volume toward the full allocation while reducing Nsight-reported GPU page-fault counts.

**Results:** Across traces with two warmups and one measured sample of both managed paths, aggregate H2D/D2H migration rose from 3,823.829/1,817.969 decimal MB at one chain to 5,833.306/5,033.165 MB at 256 chains and 5,932.446/5,033.165 MB at 4,096 chains. Reported GPU page faults fell from 10,674 to 965 to 458. Timing crossovers also changed: mapped access won at 256 chains, explicit full prefetch was near a tie at 896 and faster at 1,024, and demand migration became faster between 1,024 and 1,536 chains.

**Limitations:** Nsight totals aggregate both managed paths and warmups. They do not attribute bytes to individual arms, query exact per-page placement, or represent a model inference workload.


## H15 — CUDA VMM can combine host and device backing in one GPU address range

**Claim to falsify:** On this RTX 3060, a single contiguous GPU virtual range can contain adjacent host-backed and device-backed regions, both accessible through one pointer by a GPU kernel.

**Result:** E024 queried VMM and host-VMM support, reserved one range, mapped 64 MiB of device physical backing and 128 MiB of host physical backing, and used one pointer for GPU initialization, transform, and validation. Both capability attributes were enabled, the minimum granularity was 2 MiB, and the error count was zero.

**Limitations:** The mapping is static. It does not show automatic page migration, oversubscription, or a high-performance cache policy. Host-backed accesses still traverse the GPU interconnect.

## H16 — llama.cpp managed allocations preserve small-model inference

**Claim to falsify:** The existing llama.cpp CUDA allocation switch to cudaMallocManaged can run a small fitting quantized model with output matching ordinary CUDA allocation.

**Result:** E025 ran Qwen3-0.6B Q4_0 in both modes. Both offloaded 29/29 layers and returned “2 + 2 = 4”. Five-repetition decode means were 383.054 tokens/s ordinary and 381.812 tokens/s managed.

**Limitations:** The model fits in VRAM; the result does not validate host backing, page migration under pressure, or oversubscription. The small decode difference does not establish a performance change.
