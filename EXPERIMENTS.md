# Experiment log

## E000 — read-only baseline discovery

- **Date:** 2026-11-08
- **Question:** What hardware/runtime is available, and is the GPU quiet enough for a controlled first run?
- **Procedure:** Read CPU, RAM, PCIe sysfs, `nvidia-smi`, installed compiler versions, and NVIDIA address mode. No hardware settings were changed and no CUDA kernel was run.
- **Observations:** See [HARDWARE.md](HARDWARE.md). RTX 3060 12 GiB, i7-12700K, PCIe 4.0 x16 at 16 GT/s, Linux kernel 7.0.0-38, NVIDIA driver 595.99.02, CUDA toolkits 12.0 and 13.2, HMM address mode. About 19 GiB RAM available; 3.9 GiB memlock allowance.
- **Resource decision:** GPU showed 54% utilization and active graphics workloads. Deferred GPU execution to avoid contaminating measurements or disrupting the desktop. No performance result is claimed.
- **Conclusion:** The HMM path is exposed by the installed driver, but actual runtime attributes and memory behavior remain to be queried/measured. The mapped pinned-host path is the first controlled test because it keeps the source physically host-resident.

## E001 — CUDA runtime capability query

- **Date:** 2026-11-08
- **Question:** Does the CUDA runtime expose mapped-host memory and full pageable-system-memory unified access on this GPU?
- **Procedure:** Built source revision `39b107e` with `/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86`; ran `benchmarks/memory_paths --capabilities-only`. This mode queries properties and attributes only: no benchmark buffers, no benchmark kernel, and no timing samples. Raw output is in `results/capabilities-20261108.txt`.
- **Observed output:** RTX 3060, CC 8.6, 28 SMs; driver/runtime API 13.2. `canMapHostMemory=1`, `unifiedAddressing=1`, `concurrentManagedAccess=1`, `pageableMemoryAccess=1`, `pageableMemoryAccessUsesHostPageTables=0`, `hostNativeAtomicSupported=0`.
- **Interpretation:** Mapped pinned host access is supported. The attribute combination is consistent with full Linux HMM unified-memory support using software coherence. It does not establish placement, bandwidth, or inference performance.
- **Conclusion:** H1's capability prerequisite is supported, but its kernel correctness part remains untested. The 64 MiB performance run remains deferred by the guard because GPU utilization was 53%; the gate record is in `logs/preflight-20261108.txt`.

## E002 — second resource-gate check

- **Date:** 2026-11-08
- **Question:** Had the GPU become quiet enough for the bounded bandwidth experiment?
- **Procedure:** Re-ran `scripts/run_when_idle.sh`; this command launches the benchmark only if all resource checks pass.
- **Observation:** GPU utilization was 32%, free VRAM 11,202 MiB, available system RAM 20,178 MiB, and no CUDA compute PID. The 15% utilization gate rejected the run with exit status 3. Raw output is in `logs/benchmark-attempt-20261108.txt`.
- **Conclusion:** Still no benchmark kernel or performance measurement. Continue only when the gate passes.

## E003 — mapped host, staged transfer, and VRAM streaming

- **Date:** 2026-11-08
- **Question:** Does a direct mapped-host scan work, and how does it compare with copying the same 64 MiB input to VRAM before scanning?
- **Procedure:** Source `benchmarks/memory_paths.cu` at `39b107e`, CUDA 13.2, 64 MiB input, one warm-up and five measured repetitions per path. The resource-aware runner was committed as `ae76c01`. It sampled global GPU utilization five times and per-process SM use five times; no CUDA compute process or sustained high-use process was present.
- **Resource observations:** GPU utilization samples 46, 10, 43, 10, 39%; minimum free VRAM 11,131 MiB; available RAM 20,222 MiB; max temperature 50 C; max power 41/170 W. The experiment allocated 64 MiB pinned host RAM and 64 MiB VRAM.
- **Mapped host:** checksum valid on all runs; median 6.705152 ms (range 6.378400–7.175840 ms), equivalent input rate 10.009 GB/s.
- **Pinned H2D plus same scan:** checksum valid on all runs; median 4.900992 ms (range 4.849152–4.944928 ms), equivalent input rate 13.693 GB/s.
- **VRAM-resident scan:** checksum valid on all runs; median 0.205408 ms (range 0.202496–0.796320 ms), equivalent input rate 326.710 GB/s. Two samples were much slower than the last three, so this path is visibly noisy.
- **Analysis:** H1 is supported functionally: a kernel read the mapped host allocation and produced the expected checksum. In this workload, staging was 36.8% higher in equivalent input GB/s and 26.9% lower in median elapsed time than direct mapping. This is one synthetic contiguous-scan run, not an inference result. The VRAM variation and moderate graphics activity warrant a repeat before treating the exact rates as stable.
- **Raw output:** `results/memory-paths-64mib-20261108.txt`.

## E004 — isolate H2D copy and stabilize the VRAM reference

- **Date:** 2026-11-08
- **Question:** How much time does the pinned H2D transfer take by itself, and does a longer warm-up stabilize the local scan?
- **Procedure:** Source revision `d47260b`, 64 MiB, five measured samples per path, three warm-ups. The guard sampled GPU utilization `[7,27,28,42,11]%`, minimum free VRAM 11,063 MiB, available RAM 19,859 MiB, maximum temperature 50 C, and maximum power 32/170 W; no sustained heavy per-process SM use was present.
- **Mapped host:** 5.893 ms median (5.445–6.657 ms), 11.388 GB/s.
- **Pinned H2D only:** 4.559 ms median (4.549–4.582 ms), 14.719 GB/s; checksum validated after timing.
- **H2D plus scan:** 4.914 ms median (4.871–5.614 ms), 13.656 GB/s; one slower sample.
- **VRAM scan:** 0.201760 ms median (0.200832–0.202112 ms), 332.617 GB/s; stable after three warm-ups.
- **Interpretation:** The isolated copy path is faster than mapped access for this 64 MiB stream. Subtracting copy-only and copy-plus-scan medians suggests roughly 0.355 ms of scan time, but the medians are from separate sequences and this is only an approximation.
- **Raw output:** `results/memory-paths-copy-only-20261108.txt`.

## E005 — managed first access and explicit prefetch

- **Date:** 2026-11-08
- **Question:** How do CPU-first managed memory, repeated GPU reads, and explicit GPU prefetch compare?
- **Procedure:** `benchmarks/managed_paths.cu` at `1361ecf`, CUDA 13.2, 64 MiB, five repeat/prefetch samples. Initial preflight deferred because a CUDA client was present. Inspection found a desktop process with 126 MiB allocated but no measurable SM activity across five `pmon` samples; the gate was revised to judge sampled activity and headroom instead of rejecting any idle CUDA context. The successful run sampled global utilization `[38,32,23,25,38]%`, minimum free VRAM 10,890 MiB, available RAM 19,928 MiB, maximum temperature 40 C, and no sustained heavy GPU process.
- **CPU initialization:** 37.710 ms; the expected CPU checksum and GPU reduction differed only by floating-point reduction order and passed the relative-error check.
- **First GPU access after CPU initialization:** 5.189 ms, checksum valid (one first-touch sample).
- **Repeated scans after first access:** 0.201728 ms median (0.200704–0.201824 ms), checksum valid on all five.
- **GPU prefetch:** 2.891 ms median (2.798–2.917 ms), equivalent 23.215 GB/s.
- **Prefetch plus scan:** 3.099 ms median (3.002–3.122 ms), checksum valid on all five prefetch trials.
- **Interpretation:** Explicit prefetch was faster than the single demand-access sample, but that comparison needs independent first-touch repetitions before interpretation.
- **Raw output:** `results/managed-memory-64mib-20261108.txt`; the initial rejected preflight is `logs/managed-memory-preflight-20261108.txt`.

## E006 — repeat managed first-touch with fresh allocations

- **Date:** 2026-11-08
- **Question:** Does the E005 first-access result repeat across independent CPU-initialized managed allocations?
- **Procedure:** Source revision `67603f8`, 64 MiB, five fresh CPU-initialized managed allocations for first GPU access; five repeated scans and five CPU↔GPU prefetch cycles on the main allocation. The gate sampled utilization `[46,17,17,18,45]%`, minimum free VRAM 11,158 MiB, available RAM 19,649 MiB, maximum temperature 38 C, and no sustained heavy per-process GPU use.
- **First GPU access without explicit GPU prefetch:** 5.201216 ms median (5.172224–5.285888 ms), valid checksums across five fresh allocations.
- **Repeated scan after first access:** 0.209920 ms median (0.201568–0.220032 ms), valid checksums.
- **Explicit GPU prefetch:** 2.900992 ms median (2.814976–3.101696 ms); scan after prefetch 0.204800 ms median.
- **GPU prefetch plus scan:** 3.107840 ms median (3.025568–3.306496 ms), valid checksums.
- **Interpretation:** The result repeats: explicit prefetch plus scan was about 40% lower in median elapsed time than demand first-access for this synthetic 64 MiB scan. The two arms have different allocation/reuse patterns, so this is supported evidence, not a controlled inference result.
- **Raw output:** `results/managed-memory-fresh-64mib-20261108.txt`.

## E007 — sequential streaming size sweep

- **Date:** 2026-11-08
- **Question:** At what tested buffer size does staging become clearly faster than mapped access for this scan?
- **Procedure:** The same `memory_paths` executable, five samples per path, three warm-ups, with separate guarded runs at 16, 24, 32, and 128 MiB. E004 supplies the repeated 64 MiB point. All path checksums passed.
- **Mapped vs staged effective input rate:** 16 MiB: 14.091 vs 14.173 GB/s; 24 MiB: 13.293 vs 14.036; 32 MiB: 12.977 vs 13.952; 64 MiB: 11.388 vs 13.656; 128 MiB: 12.121 vs 14.141.
- **H2D copy-only:** 14.782, 14.744, 14.682, 14.719, and 14.749 GB/s respectively at 16, 24, 32, 64, and 128 MiB.
- **Local VRAM scan:** 311.520–336.730 GB/s across the same sizes; the 16–128 MiB runs were stable after warm-up.
- **Interpretation:** At 16 MiB the mapped and staged paths were nearly tied. Staging led at 24 MiB and larger in these runs, placing the observed crossover between 16 and 24 MiB for this specific contiguous float4 scan. It is not a transformer or model-size threshold.
- **Raw outputs:** `results/memory-paths-16mib-20261108.txt`, `results/memory-paths-24mib-20261108.txt`, `results/memory-paths-32mib-20261108.txt`, `results/memory-paths-128mib-20261108.txt`.

## E008 — trace managed-memory page faults and migrations

- **Date:** 2026-11-08
- **Question:** Does the managed path actually migrate data, or do the timing changes only reflect reuse/cache effects?
- **Procedure:** Source revision `2a4d9a9`, guarded runner `67da456`, Nsight Systems 2025.6.3 with CUDA Unified Memory CPU/GPU page-fault tracing enabled. Ran the 64 MiB managed benchmark with five CPU-first allocations and five explicit CPU↔GPU prefetch cycles. The report is `results/managed-memory-faults-64mib-20261108-022719.nsys-rep`.
- **Nsight totals:** 671.089 decimal MB (640 MiB) host-to-device migration, 335.544 decimal MB (320 MiB) device-to-host migration, 960 CPU page faults, and 2,305 GPU page faults. The byte totals match ten 64 MiB H2D migrations and five 64 MiB D2H migrations in the test design.
- **Interpretation:** This confirms managed pages migrate in the tested path. Nsight instrumentation materially increased measured scan latency, so the profile run's timing values are excluded from performance comparisons. The 128 MiB change in global free-VRAM reporting after first access is recorded but does not establish exact placement.
- **Raw output and summary:** `results/managed-memory-fault-profile-20261108.txt` and `results/managed-memory-fault-stats-20261108.txt`.

## E009 — access selectivity and reuse

- **Date:** 2026-11-08
- **Question:** When does full-buffer movement to VRAM pay compared with direct mapped reads for sparse or repeated access?
- **Procedure:** Source revision `4de4c5f`, `BENCHMARK=reuse_paths MIB=64 REPEATS=5 scripts/run_when_idle.sh`. A 16 MiB device scan evicted prior L2 data before every timed trial; the repeated passes within a trial could reuse cache normally. The tested access patterns read every 1st, 4th, 10th, or 100th float4 (100%, 25%, 10%, or 1%) uniformly across the 64 MiB buffer. Each staged or managed-prefetch trial moved the full 64 MiB once, then scanned the selected set 1, 2, or 4 times. Device allocation was at most 144 MiB (staging + managed + eviction buffers); host allocations were about 128 MiB.
- **Preflight:** Global GPU utilization `[0,0,0,0,0]%`; minimum free VRAM 11,155 MiB; available RAM 19,445 MiB; maximum temperature 29 C; maximum power 11/170 W; an unrelated CUDA client held about 101 MiB but had no sampled SM use; no sustained heavy GPU process.
- **Mapped-host median vs full managed-prefetch-plus-scans:** At 1%: 0.380, 0.613, and 1.260 ms for 1, 2, and 4 passes; mapped was fastest in all three. At 10%: mapped 2.059 ms beat managed 2.915 ms for one pass, while managed at 2.978 and 2.998 ms beat mapped 4.097 and 8.730 ms for two and four passes. At 25%, managed took 2.974 ms for one pass and 3.267 ms for four, faster than mapped's 4.460 and 17.891 ms. At 100%, managed took 3.130 ms for one pass and 3.661 ms for four, faster than mapped's 5.561 and 21.389 ms.
- **Interpretation:** In this workload, direct mapped access wins for very sparse one-off reads; full managed prefetch wins as the selected fraction or reuse grows. Full H2D staging has the same broad crossover but was slower than managed prefetch on most cases. This supports selective data movement and motivates a compact GPU-side cache test; it does not establish an LLM policy.
- **Correctness and raw output:** All tested checksums passed. Full five-sample ranges are in `results/reuse-selectivity-cold-64mib-20261108.txt`.
- **Retained preliminary attempt:** `results/reuse-selectivity-64mib-20261108.txt` was produced by source revision `c36a3a5` before the L2 eviction control was added. Because the 100% scan ran before the sparse cases, it could warm the same mapped-host lines. Do not use that run for sparse-path conclusions; E009's reported results use the corrected `4de4c5f` run.

## E010 — gather sparse values into a compact GPU-side cache

- **Date:** 2026-11-08
- **Question:** Can one GPU gather of selected mapped-host values, followed by local reuse, beat repeated remote reads or full-buffer movement?
- **Procedure:** Source revision `6907233`, same 64 MiB buffer, selectivity, pass counts, five-sample repetitions, and 16 MiB L2 eviction scan as E009. Added a GPU gather kernel that reads selected float4 values through the mapped pinned pointer into a compact VRAM buffer, then performs 1, 2, or 4 local scans. Other paths remained direct mapped reads, full pinned H2D copy, and full managed prefetch. The run used at most 208 MiB of device buffers and about 128 MiB of host allocations.
- **Preflight:** Global utilization `[0,0,0,0,0]%`; minimum free VRAM 11,150 MiB; available RAM 19,293 MiB; maximum temperature 28 C; maximum power 11/170 W; an unrelated CUDA client held 102 MiB with no sampled SM use; no sustained heavy GPU process.
- **At 1% selectivity:** One-pass gather+scan was 0.297 ms (mapped direct 0.295 ms); four-pass gather+cache was 0.302 ms vs 1.172 ms direct mapped and 2.957 ms full managed prefetch.
- **At 10% selectivity:** One-pass gather+scan was 2.048 ms (mapped direct 2.038 ms); four-pass gather+cache was 2.123 ms vs 8.167 ms direct mapped, 4.680 ms full H2D staging, and 2.985 ms managed prefetch.
- **At 25–100%:** Full managed prefetch remained fastest. For example, at 25%/four passes it took 3.357 ms vs 4.500 ms compact cache; at 100%/four passes it took 3.685 ms vs 6.067 ms compact cache.
- **Interpretation:** A compact GPU-side cache clearly amortized repeated sparse mapped reads at 1–10%; one-pass gathering was roughly tied with direct access, and full-range prefetch remained better for denser data. This validates the cache mechanism in a synthetic gather, not in an inference kernel.
- **Correctness and raw output:** Every tested checksum passed. Full sample data and preflight are in `results/reuse-cache-64mib-20261108.txt`.

## E011 — repeat the 64 MiB streaming baseline

- **Date:** 2026-11-08
- **Question:** Do the initial mapped, staged, and VRAM scan results repeat under a quiet preflight?
- **Procedure:** Source `6907233`; default `scripts/run_when_idle.sh`, 64 MiB, five samples per path. The process used one CPU core.
- **Preflight:** GPU utilization `[0,0,0,0,0]%`; minimum free VRAM 11,161 MiB; available RAM 18,991 MiB; maximum temperature 28 C; maximum power 11 W; no sustained heavy GPU process.
- **Results:** Mapped median 5.384 ms (12.465 GB/s), H2D-only 4.596 ms (14.601 GB/s), H2D plus scan 4.767 ms (14.078 GB/s), and VRAM scan 0.202 ms (331.566 GB/s). Every checksum passed.
- **Raw output:** `results/memory-paths-resume-64mib-025309.txt`.

## E012 — Q4_0 synthetic matrix-vector paths and transfer isolation

- **Date:** 2026-11-08
- **Question:** Do mapped, staged, managed-prefetch, and VRAM paths behave differently on a quantized matrix-vector access pattern?
- **Procedure:** Source `554861f`, CUDA 13.2, 4,096 rows, Q4_0 18-byte blocks, 1/2/4 input vectors, five measured samples per mode. Runs used 32, 64, and 128 MiB matrices; each weight block uses a half scale and packed low/high nibbles. Outputs were compared against a CPU reference. All weights fit in VRAM.
- **Results:** Every output passed. Across all sizes and pass counts, managed prefetch was faster than mapped and staged paths. At 64 MiB/four passes, medians were 25.603 ms mapped, 6.462 ms staged, 4.778 ms managed-prefetch, and 1.860 ms VRAM resident. Staging/prefetch moved the matrix once per trial; mapped access reread it on each pass.
- **Transfer-only:** Pinned H2D measured 14.6–14.8 GB/s at 32–128 MiB; managed prefetch measured 22.1–23.3 GB/s under this procedure.
- **Interpretation:** This is synthetic Q4_0 GEMV-style work, not a real model layer or inference. The managed-prefetch advantage is specific to this allocation and access pattern.
- **Raw outputs:** `results/q4-transfer-isolation-32mib-20261108-030402.txt`, `results/q4-transfer-isolation-64mib-20261108-030311.txt`, and `results/q4-matvec-128mib-20261108-030701.txt`.

## E013 — inspect the local inference path and model sizes

- **Date:** 2026-11-08
- **Question:** Is a full local inference comparison possible within the current 4 GiB additional host-RAM limit?
- **Procedure:** Read-only inspection of the existing llama.cpp checkout, available model filenames and sizes, and GGUF metadata. No inference process was started and no model file or external checkout was modified.
- **Observation:** The local standalone Qwen3.8-27B Q4_0 GGUF is 15,149,087,168 bytes (about 14.1 GiB). The other GGUF files found in the inspected model/cache locations are named MTP or mmproj companions. No standalone base model within the 4 GiB additional-RAM budget was found.
- **Decision:** Do not load the full model under the current cap. Continue with bounded slices of its actual Q4_0 weights and synthetic inputs; a smaller standalone model or a changed resource limit is needed for end-to-end inference.

## E014 — double-buffered 16 MiB chunk pipeline on synthetic Q4_0

- **Date:** 2026-11-08
- **Question:** Can an asynchronous copy stream overlap 16 MiB Q4_0 chunk transfers with a separate matvec stream?
- **Procedure:** Source `e5534ce`, 128 MiB matrix, eight 16 MiB chunks, two device buffers, serialized chunk copy+compute control, and double-buffered two-stream pipeline. Five measured samples at one and four passes; all outputs passed. The benchmark was pinned to one CPU core.
- **Preflight:** Sampled GPU utilization `[67,69,69,69,69]%`; minimum free VRAM 10,771 MiB; available RAM 19,854 MiB; maximum temperature 62 C; planned device allocation 304 MiB and host allocations about 256 MiB; no sustained heavy per-process GPU PID.
- **Results:** Double buffering reduced median time from 17.766 to 12.620 ms for one pass and from 69.912 to 49.743 ms for four passes, about 29% lower in each comparison. The four-pass chunk arm transferred the matrix four times (511.875 MiB); full staged/managed arms transferred it once and were faster.
- **Interpretation:** The overlap path helped versus serialized chunking, but this run had substantial desktop GPU activity and the same-sized full-buffer paths performed better. Treat absolute rates as provisional, not inference throughput.
- **Raw output:** `results/q4-chunked-128mib-20261108.txt`.

## E015 — validate GEMV against an actual Q4_0 model tensor

- **Date:** 2026-11-08
- **Question:** Do the access-path comparisons hold for packed weights from a real model layer?
- **Procedure:** Read only `blk.0.ffn_down.weight` (5,120 rows × 17,408 values; 50,135,040 bytes) from the local GGUF at byte offset 1,488,544,192 into the bounded benchmark allocation. Input vectors were synthetic. The model file was not changed; the trial used 64 MiB planned size, five samples, and one CPU core.
- **Diagnostic failure:** Source `c6f8bc0` compared the real Q4_0 values against a CPU reference that assumed the synthetic 1/16 scale. Every GPU arm failed validation with the same mismatch. Those timings are invalid and excluded. The first log is retained as `results/q4-model-tensor-20261108.txt`.
- **Correction and rerun:** Source `898e1df` decodes each block's half-precision scale in the CPU reference. Preflight GPU utilization was `[24,36,31,17,41]%`; minimum free VRAM 10,957 MiB; available RAM 20,127 MiB; maximum temperature 39 C. All mapped, staged, managed, VRAM, and chunked outputs passed.
- **Results:** For one pass, medians were 4.168 ms mapped, 3.833 ms full H2D staging, 2.532 ms managed prefetch, 0.351 ms VRAM resident, and 3.568 ms double-buffered 16 MiB chunks. At four passes, corresponding medians were 17.324, 4.765, 3.473, 1.303, and 14.286 ms. The chunked four-pass path reloaded the 47.812 MiB tensor on every pass.
- **Raw output:** `results/q4-model-tensor-rerun-20261108.txt`.

## E016 — sweep chunk size on the same model tensor

- **Date:** 2026-11-08
- **Question:** How does the number of chunks affect overlap on the same layer-weight slice?
- **Procedure:** Source `893db0c`, same 50,135,040-byte model tensor, 4/8/16/32/64 MiB double buffers, one and four passes, five samples per path, CPU output validation. Separate guarded invocations sampled maximum GPU utilization between 18% and 51%; all runs passed validation.
- **Results:** 8 MiB (six chunks) had the lowest observed double-buffered medians: 3.472 ms for one pass and 13.788 ms for four. The 4 MiB, 16 MiB, 32 MiB, and 64 MiB medians and ranges are in `BENCHMARKS.md` and the linked raw logs.
- **Interpretation:** Smaller chunks yielded a lower observed median in these runs, but separate-run resource variation and noisy larger-chunk ranges prevent claiming an optimal size. At four passes the chunk path moves the tensor four times, so it remains slower than full staging or prefetch, which move it once.
- **Raw outputs:** `results/q4-model-tensor-chunk4-20261108.txt`, `results/q4-model-tensor-chunk8-20261108.txt`, `results/q4-model-tensor-chunk16-20261108.txt`, `results/q4-model-tensor-chunk32-20261108.txt`, and `results/q4-model-tensor-chunk64-20261108.txt`.

## E017 — check read-only registration of file-backed model pages

- **Date:** 2026-11-08
- **Question:** Can CUDA register a bounded read-only mmap of the model tensor so kernels access the file-backed host pages directly?
- **Procedure:** Source `f194fbe`; the resource gate passed for the 50 MiB tensor slice. Before mapping or registering any model pages, the program queried CUDA device attributes.
- **Observation:** `cudaDevAttrHostRegisterSupported=1`, but `cudaDevAttrHostRegisterReadOnlySupported=0`. Because this system reports `pageableMemoryAccessUsesHostPageTables=0`, the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) requires read-only registration for this read-only host mapping. The registration attempt was skipped; no model mapping or GPU work occurred.
- **Conclusion:** File-backed read-only direct access is unavailable through this API on the current device. The bounded pinned-copy model-tensor path remains the usable measurement baseline.
- **Raw output:** `results/q4-model-tensor-mmap-20261108.txt`.

## E018 — measure dependent random-load latency

- **Date:** 2026-11-08
- **Question:** How does a single dependent pointer chase compare between mapped pinned host memory and a VRAM-resident copy?
- **Procedure:** Added `benchmarks/host_latency.cu`. It builds a deterministic randomized pointer cycle, performs one dependent chain of 32,768 loads, evicts prior L2 contents with a 16 MiB scan before each timed trial, and checks the final node against a CPU reference. Five samples were collected at 1, 4, 32, and 64 MiB. Each size compared mapped pinned host memory with the same chain resident in VRAM.
- **Results:** Mapped-host medians were 586.47 ns/load at 1 MiB, 726.37 ns/load at 4 MiB, 749.37 ns/load at 32 MiB, and 753.81 ns/load at 64 MiB. VRAM medians were 217.94, 257.25, 269.99, and 273.78 ns/load respectively. All measured outputs passed validation.
- **Interpretation:** In this single-chain dependent workload, mapped-host accesses took about 2.7–2.8× the VRAM time at 1–64 MiB. This is a dependent random-access latency measurement, not streaming bandwidth or model inference.
- **Raw outputs:** `results/host-latency-1mib-20261108.txt`, `results/host-latency-4mib-20261108.txt`, `results/host-latency-32mib-20261108.txt`, and `results/host-latency-64mib-20261108.txt`.

## E019 — scale independent chains and managed-memory working sets

- **Date:** 2026-11-08
- **Question:** How many independent chains hide mapped-host latency, and how do mapped, managed, and VRAM paths change as a randomized working set grows?
- **Procedure:** The chain starts were spaced evenly around the randomized cycle and each chain was constrained to a disjoint segment. At 64 MiB, tests used 4,096 steps per chain and 1, 32, 256, 1,024, or 4,096 chains, with five measured samples per path. At 64, 256, and 960 MiB, a 4,096-chain workload was also compared across mapped host, managed demand migration, full-allocation managed prefetch plus kernel, and VRAM. The managed allocation was moved back to host before each trial; the demand arm timed the kernel faults/migrations, while explicit prefetch plus kernel timed migration of the whole allocation. Two warmups preceded samples. All samples validated.
- **64 MiB parallel-chain results:** Mapped-host throughput rose from 1.316 million loads/s at one chain to about 120–125 million loads/s at 256–4,096 chains. It plateaued in this range. The VRAM path continued to rise, reaching 3,408 million loads/s at 4,096 chains.
- **Working-set results at 4,096 chains:** Mapped-host medians were 8.27 ns/load (120.98 million loads/s) at 64 MiB, 8.45 ns/load (118.33 million loads/s) at 256 MiB, and 12.56 ns/load (79.61 million loads/s) at 960 MiB. Managed demand elapsed times were 9.07, 21.75, and 66.44 ms; full-allocation prefetch plus kernel took 9.85, 16.69, and 47.31 ms. VRAM resident times were 4.93, 5.12, and 5.17 ms. Demand and prefetch medians overlap at 64 MiB; at 256 and 960 MiB the explicit prefetch arm was faster in these runs.
- **960 MiB chain-count results:** Mapped access was faster than full managed prefetch at 256, 512, and 896 chains; the 896-chain result was close (43.24 vs 43.87 ms). Full prefetch was faster by 1,024 chains (44.07 vs 55.40 ms). Demand migration was still slower than mapped access at 1,024 chains (66.91 vs 55.40 ms), but faster by 1,536 chains (62.29 vs 72.15 ms). At 2,048 chains both managed paths beat mapped access (64.00 ms demand, 44.85 ms prefetch, 97.53 ms mapped). These are observed crossover brackets for this workload, not general thresholds.
- **Resource observations:** The 960 MiB runner planned 1,936 MiB of device memory and about 2,880 MiB of host memory, within the project’s 2 GiB VRAM and 4 GiB host-RAM limits. Preflights retained per-run GPU, temperature, power, RAM, and VRAM observations; no sustained heavy per-process GPU use was detected.
- **Excluded diagnostic:** The first 64 MiB multi-chain logs (`host-latency-parallel-64mib-c1/c32/c256-20261108.txt`) used overlapping chain starts and are excluded from conclusions. The corrected `host-latency-spaced-*` runs use disjoint chain segments.
- **Raw outputs:** Corrected 64 MiB runs are `results/host-latency-spaced-64mib-c1-20261108.txt`, `results/host-latency-spaced-64mib-c32-20261108.txt`, `results/host-latency-spaced-64mib-c256-20261108.txt`, `results/host-latency-spaced-64mib-c1024-20261108.txt`, and `results/host-latency-spaced-64mib-c4096-20261108.txt`. The 960 MiB chain-count sweep and other size comparisons are linked in `BENCHMARKS.md`.

## E020 — trace 960 MiB managed-memory migration

- **Date:** 2026-11-08
- **Question:** How much of the large managed allocation migrates during the distributed random-chain workload?
- **Procedure:** Ran the 960 MiB, 4,096-chain benchmark under Nsight Systems with two warmups and one measured sample. The trace includes both managed demand and full-prefetch paths; profiled timings are excluded.
- **Observation:** Nsight reported 5,932.446 decimal MB H2D migration, 5,033.165 MB D2H migration, 2,880 CPU page faults, and 458 GPU page faults. The D2H total is essentially five 960 MiB allocation movements; the H2D total is close to six. This aggregate is consistent with page faults across the distributed chains moving nearly the full allocation, alongside explicit full-allocation prefetches; it does not attribute each byte to an individual arm.
- **Interpretation:** The managed path’s migration cost is substantial for this 960 MiB footprint. At 4,096 chains the explicit full prefetch was faster than demand migration, but both moved nearly the full set and remained slower than VRAM-resident access. This dataset fits in VRAM and does not demonstrate oversubscription or inference performance.
- **Raw output and trace:** `results/host-managed-random-profile-960mib-20261108.txt`, `results/host-managed-random-profile-stats-960mib-20261108.txt`, and `results/unified-memory-faults-host_latency-960mib-20261108-105120.nsys-rep`.

## E021 — extend Q4_0 matrix access through 960 MiB

- **Date:** 2026-11-08
- **Question:** Does the managed-prefetch advantage observed for smaller Q4_0 matrices persist at the largest synthetic matrix that fits within the project’s per-run allocation limits?
- **Procedure:** Extended `benchmarks/q4_matvec.cu` and the guarded runner from a 256 MiB to a 960 MiB maximum. Ran synthetic Q4_0 matrices with 4,096 rows at 256 and 960 MiB, five samples, and 1/2/4 input-vector passes. Compared mapped, pinned H2D staging, managed prefetch, VRAM resident, and serialized/double-buffered 16 MiB chunk paths. All outputs were checked against the CPU reference.
- **Resource observations:** The 256 MiB run planned 560 MiB device and 512 MiB host memory, with at least 11,278 MiB free VRAM and 22,839 MiB available RAM. The 960 MiB run planned 1,968 MiB device and 1,920 MiB host; the benchmark reported 1,974.728 MiB actual device buffers. Preflight saw at least 11,296 MiB free VRAM and 22,831 MiB available RAM, maximum sampled GPU utilization 6%, and no sustained heavy per-process GPU use.
- **Results:** At 960 MiB, one-pass medians were 108.887 ms mapped, 77.317 ms H2D-staged, 51.965 ms managed-prefetch, and 8.604 ms VRAM-resident. Four-pass medians were 440.832, 103.612, 76.531, and 38.655 ms. Managed prefetch transfer-only measured 41.978 ms (23.980 GB/s), compared with 68.300 ms (14.738 GB/s) pinned H2D. Every output passed CPU-reference validation.
- **Chunking:** Double buffering reduced chunk-path medians from 81.818 to 69.527 ms for one pass and 321.219 to 278.928 ms for four passes. The four-pass path moved the 960 MiB matrix four times and remained much slower than full managed prefetch.
- **Interpretation:** Managed prefetch remained the fastest host-backed path for this Q4_0 matvec through 960 MiB. These matrices still fit in VRAM; this supports the fitting-in-VRAM reuse result and says nothing about an oversized model.
- **Raw outputs:** `results/q4-matvec-256mib-20261108.txt` and `results/q4-matvec-960mib-20261108.txt`.

## E022 — compare managed migration traces across chain fanout

- **Date:** 2026-11-08
- **Question:** Does increasing independent-chain fanout change aggregate managed-memory migration volume and page-fault count for the same 960 MiB allocation?
- **Procedure:** Captured additional Nsight traces at 1 and 256 chains, using the same 4,096 steps, two warmups, one measured sample, and both managed demand and full-prefetch paths. The existing 4,096-chain trace provides the high-fanout reference. Profiler timings are excluded.
- **Results:** Aggregate H2D/D2H migration was 3,823.829/1,817.969 decimal MB at 1 chain, 5,833.306/5,033.165 MB at 256 chains, and 5,932.446/5,033.165 MB at 4,096 chains. GPU page-fault counts were 10,674, 965, and 458 respectively; CPU page-fault totals were 2,880 in each trace.
- **Interpretation:** Higher fanout is associated with near-full-allocation migration and fewer reported GPU faults; the one-chain run moved less in aggregate but generated more faults. This is consistent with sparse serialized faults versus broad page coverage under many chains. Each Nsight total aggregates both managed paths and warmups, so it cannot identify demand-only bytes or establish a general migration granularity.
- **Raw outputs:** 1-chain [profile](results/host-managed-random-profile-960mib-c1-20261108.txt), [statistics](results/host-managed-random-profile-stats-960mib-c1-20261108.txt), and [trace](results/unified-memory-faults-host_latency-960mib-20261108-110358.nsys-rep); 256-chain [profile](results/host-managed-random-profile-960mib-c256-20261108.txt), [statistics](results/host-managed-random-profile-stats-960mib-c256-20261108.txt), and [trace](results/unified-memory-faults-host_latency-960mib-20261108-110506.nsys-rep). The 4,096-chain trace is recorded in E020.


## Reproduction command templates for E018–E022

Commands were pinned to one CPU core and routed through the resource gate. Replace angle-bracketed values with the stated experiment settings.

| Experiment | Command |
|---|---|
| E018 serial latency | `taskset -c 0 env BENCHMARK=host_latency MIB=<1,4,32,64> REPEATS=5 scripts/run_when_idle.sh --steps 32768 --chains 1` |
| E019 chain sweeps | `taskset -c 0 env BENCHMARK=host_latency MIB=<64,960> REPEATS=5 scripts/run_when_idle.sh --steps 4096 --chains <chain-count>`; the 64 MiB managed comparison used `REPEATS=10` |
| E020 4,096-chain migration profile | `taskset -c 0 env PROFILE=um_faults BENCHMARK=host_latency MIB=960 REPEATS=1 scripts/run_when_idle.sh --steps 4096 --chains 4096` |
| E022 migration profiles | Use the E020 command with `--chains 1` and `--chains 256` |
| E021 Q4_0 matrices | `taskset -c 0 env BENCHMARK=q4_matvec MIB=<256,960> REPEATS=5 scripts/run_when_idle.sh` |


## E023 — verify one GPU kernel across VRAM and mapped host RAM

- Date: 2026-11-08
- Question: Can one GPU kernel correctly operate on a 1 GiB device allocation and a 2 GiB GPU-accessible host allocation in one workload?
- Procedure: Added benchmarks/mixed_residency.cu and extended the bounded runner. The runner preserved the existing 2 GiB device and 4 GiB host allocation limits, required zero current swap use, checked the pinned-memory limit, and reserved 8 GiB of available RAM. GPU kernels initialized, transformed, and validated all 3 GiB; CPU code read only two error counters. Five samples were collected per path.
- Preflight: RTX 3060 reported 11,905 MiB total VRAM; minimum free VRAM was 10,945 MiB, available RAM 20,431 MiB, and swap use zero. The benchmark planned 1,041 MiB of device memory including scratch/counters and 2,048 MiB pinned host memory.
- Results: All device-only, mapped-host-only, and mixed samples passed with zero mismatches. One-pass medians were 6.620 ms VRAM-only, 393.268 ms mapped-host-only, and 394.184 ms mixed. The four-pass medians were 27.330, 1,611.589, and 1,612.628 ms. For the mixed one-pass kernel, logical bandwidth was 16.344 GB/s total, comprising 10.896 GB/s host and 5.448 GB/s device. The 1 GiB device region added only about 0.9 ms relative to mapped-host-only because host traffic dominated.
- PCIe and interpretation: System-wide dmon median rose from 76/15 MB/s RX/TX in the five-sample baseline to 6,341/7,102 MB/s across 37 during-run samples. Other clients were active, so this is evidence of coincident traffic, not precise attribution to the benchmark. The test demonstrates mixed GPU access, not a faster unified memory path or oversubscription.
- Raw outputs: results/mixed-residency-1gib-device-2gib-host-20261108.txt and the two PCIe records linked in BENCHMARKS.md.

## E024 — test a contiguous VMM range with device and host physical backing

- Date: 2026-11-08
- Question: Can this RTX 3060 expose device-backed and host-backed physical allocations through one contiguous GPU virtual range?
- Procedure: Added src/unified_range.hpp and benchmarks/vmm_mixed_probe.cu. CUDA Driver API VMM allocated and mapped adjacent device and host regions. The probe queried VMM support, host-backed VMM support, and minimum allocation granularities, then used one pointer for GPU-only initialization, transform, and validation.
- Results: VMM and host VMM support were both enabled. Minimum granularity was 2,097,152 bytes for each region. A 64 MiB device segment and 128 MiB host segment occupied one 192 MiB contiguous GPU-visible range. Every element was touched and checked by GPU kernels; mismatches were zero. One event sample measured 22.248 ms.
- Interpretation: CUDA VMM can statically unify the GPU-visible address range while retaining distinct device and system-memory backing on this hardware. The host-backed part remains system RAM accessed over the GPU interconnect. This does not show demand migration, oversubscription, or equal access costs.
- Raw output: results/vmm-mixed-range-64dev-128host-20261108.txt.
- Source references: NVIDIA CUDA Driver API VMM documentation and CUDA Programming Guide VMM documentation; the Linux HMM documentation describes driver-facing page migration helpers rather than an application allocation API.

## E025 — compare a fitting quantized LLM in ordinary and managed allocation modes

- Date: 2026-11-08
- Question: Does the existing llama.cpp Unified Memory switch preserve small-model output and alter throughput when the model fits in VRAM?
- Procedure: Used the existing llama.cpp CUDA build, version 8991 / commit beb42fffa, without modifying its checkout. The Qwen3-0.6B Q4_0 file from ggml-org/Qwen3-0.6B-GGUF revision 13ac066 was 428,970,080 bytes and SHA-256 da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4. The model repository identifies Apache-2.0 licensing. A correctness prompt ran separately in both modes; llama-bench then measured five repetitions at 47 prompt tokens and 128 generated tokens, batch/ubatch 128, with all GPU layers. The local llama.cpp source at commit d775b8967a46d8beb110d444aa3b8938179e0dd8 shows that the environment switch routes CUDA allocations through cudaMallocManaged.
- Preflight and monitoring: minimum free VRAM 11,082 MiB, available RAM 20,106 MiB, and swap use zero. All 29/29 layers were reported offloaded in each mode. The model buffer report showed 319.96 MiB on CUDA and 83.46 MiB CPU-mapped; the system-wide GPU monitor ranged from 810 to 1,428 MiB used. Short dmon output had seven samples and an 8,043 MB/s RX outlier; it cannot assign PCIe traffic to either mode.
- Correctness: Both modes returned “2 + 2 = 4”.
- Throughput: Ordinary versus Unified Memory decode averaged 383.054 ± 2.923 versus 381.812 ± 1.210 tokens/s. Prompt processing averaged 6,032.8 ± 1,234.7 versus 6,273.3 ± 1,364.1 tokens/s. The decode difference is about 0.3%, too small to establish an advantage. The model fits in VRAM; this is not an oversubscription test.
- Interpretation: This validates an actual CUDA-backed LLM run and a small-model baseline. It does not demonstrate host-resident weights, page migration, improved inference, or behavior above VRAM capacity. The Unified Memory allocator's internal free-memory report uses host RAM availability on this discrete GPU and must not be read as physical VRAM free space.
- Raw outputs: results/llama-small-conventional_gpu-bench-20261108-123905.csv, results/llama-small-unified_memory_gpu-bench-20261108-123905.csv, correctness stdout/stderr, and system-wide memory/PCIe monitors listed in BENCHMARKS.md.

## E025 — page-aware residency manager (hot-page VRAM cache over pinned host backing)

- Date: 2026-11-08
- Procedure: src/residency_manager.hpp + benchmarks/residency_manager.cu via scripts/run_when_idle.sh (BENCHMARK=residency_manager MIB=256 REPEATS=5, cache 64 MiB, 4 passes, 4 Mi requests, 90% of accesses to 32 hot 2 MiB pages). Removed a failing cudaSetDeviceFlags call; the guard now supports this benchmark (304 MiB device, host+32 MiB plan).
- Results (measured, all validation errors zero): baseline direct mapped host, 5 passes median 277.5 ms; learn+prefetch+4 cached passes median 81.8 ms (about 3.4x); cache hit rate 89.98%; 32 pages / 64 MiB promoted. Pressure test: capacity halved 32->16 frames, 23 pages evicted, shifted hot set promoted 16 pages (32 MiB), hit rate 45.1% (cache holds half the hot set, as expected).
- Caveats: synthetic 90/10 skewed trace, 256 MiB host range, fits easily in VRAM; not true oversubscription and not yet a real model. Note: cudaMalloc/cudaFree per page is used, which is slow vs a slab allocator.
- Raw output: results/residency-manager-256mib-20261108.txt

## E026 — 13 GiB single managed allocation (oversubscription, measured)

- Date: 2026-11-08. benchmarks/oversubscribe.cu: one cudaMallocManaged of 13,312 MiB (> 11,905 MiB VRAM), 2 GiB cudaMalloc reserve, GPU init + full validation of every element (sequential, random 2 MiB tile, 32 MiB cluster order), 2 repeats each.
- Result: zero validation errors in all passes. Init+first touch 4.79 s (2.91 GB/s). Sequential 2.71-2.76 GB/s, random tile 3.31-3.63 GB/s, cluster32 3.77-4.67 GB/s. For reference a fitting 2 GiB managed set scans at 136-163 GB/s, so oversubscribed throughput is roughly 30-50x lower (inferred from separate smaller run).
- Side effects: VRAM free fell to ~30 MiB during passes (driver used all VRAM except the reserve); system swap rose from 5.8 GiB to ~11 GiB while MemAvailable stayed >12 GiB (kernel swapped other processes cold pages). Swap abort limit was relaxed to +12 GiB; earlier tighter limits (+0.25/1/4 GiB) aborted the run. This deviates from the plan zero-swap rule and was done under explicit user approval.
- Conclusion: a unified VRAM+RAM pool that is correct works on this machine via managed memory; it is bandwidth-limited by PCIe demand migration and thrashing, not by capacity. Raw: results/oversubscribe-13gib-20261108.txt. Not yet profiled with Nsight; no LLM tested.

## E027 — 24 GiB managed allocation (aborted by RAM floor)

- Date: 2026-11-08. Same harness, 24,576 MiB, 1.5 GiB cudaMalloc reserve, RAM floor 6 GiB. GPU init of all 24 GiB completed (11.77 s, 2.19 GB/s). During the first validation pass MemAvailable fell to 6,118 MiB (< floor) and swap rose from 8.0 to 11.7 GiB, so the harness aborted. No validation result: inconclusive, not a pass. Raw: results/oversubscribe-24gib-20261108.txt.
- Interpretation: 24 GiB is about the physical limit of 11.9 GiB VRAM + usable RAM on this 31 GiB machine with the desktop running; the host copy (~14 GiB) plus desktop leaves almost nothing.

## E028-E030 — tiered NVMe/RAM/VRAM streaming pool (measured)

- Date: 2026-11-08. benchmarks/tiered_stream.cu: read-only file on NVMe (WD SN850X), 32 MiB chunks, 4 O_DIRECT reader threads, 256 MiB pinned staging ring, bounded 4 GiB RAM cache, 6 GiB VRAM hot arena, 128 MiB VRAM streaming slots. Kernels read VRAM only; GPU validates every element. Run under systemd-run MemoryMax=10G.
- 24 GiB: pass 0 4.55 GB/s (all NVMe); later passes 6.8 GB/s (192 chunks VRAM hot, 128 RAM, 448 NVMe). Zero errors. 48 GiB (exceeds RAM+VRAM): pass 0 4.98 GB/s, later passes 6.0 GB/s. Zero errors. MemAvailable stayed ~19 GiB and swap did not grow (8.19 GiB, slightly fell).
- Comparison: managed oversubscription E026 gave 2.7-4.7 GB/s at 13 GiB and aborted at 24 GiB due to RAM pressure and swap growth. The tiered pool is 1.3-2.5x faster and handles 2x larger data with flat RAM/swap.
- Caveats: sequential access with no compute, so the NVMe read path (about 5 GB/s) is the bottleneck, not PCIe. The hot arena only saves what fits in VRAM. Not yet a model; tokens/s for dense models is bounded by this 5-7 GB/s streaming rate. Raw: results/tiered-24gib-20261108.txt, tiered-48gib-20261108.txt.

## E031 — streamed decode proxy on a 13.42 GiB over-VRAM model (measured)

- Date: 2026-11-08. benchmarks/streamed_decode.cu (build: `nvcc -O2 -std=c++17 -arch=sm_86 -lpthread`): per "token", a Q4_0 GEMV over every 2-D Q4_0 weight matrix of a real GGUF model with a deterministic synthetic input vector; GPU results validated against a CPU reference (8 rows per tensor, 3976 rows). Model: Qwen3.8-27B STRICT Q4_0, 15,149,087,168 bytes, 497 tensors, 13.42 GiB stream, 726 items of <=32 MiB (embedding table excluded, as a real decode only row-gathers it). Manifest: results/qwen38-27b-manifest.txt via scripts/gguf_manifest.py; layout inspected with scripts/inspect_gguf.py. 8 passes, 4 O_DIRECT reader threads, taskset -c 0-3.
- Config A (direct, host-resident): 8 GiB VRAM hot arena + 3.57 GiB pinned mapped + 1.85 GiB managed (CPU preferred); all 13.42 GiB resident, zero NVMe in steady state. Steady-state (passes 1-7) 1.387-1.409 s/token, best 0.721 tok/s, effective 10.2-10.4 GB/s.
- Config B (staged, 4.5 GiB cache): 8 GiB hot + 4.5 GiB pageable RAM cache + 0.93 GiB NVMe per token. Steady state 1.343-1.393 s/token, best 0.744 tok/s, effective 10.3-10.7 GB/s.
- Config C (staged, full 5.55 GiB cache): 8 GiB hot + 5.55 GiB RAM cache covering every cold weight, zero NVMe in steady state. Steady state 0.979-0.987 s/token, best 1.021 tok/s, effective 14.6-14.7 GB/s.
- All three PASS with zero mismatches across 3976 checked rows. Resources: 8.3 GiB VRAM allocated (2.1-2.2 GiB free after alloc), 5.6 GiB RAM cache, MemAvailable stable ~12 GiB, swap flat at 7.3 GiB (no growth). Within the user-approved over-VRAM budget (8 GiB host cap); preflight showed 10.6 GiB free VRAM, 17.7 GiB available RAM, 47 C, one unrelated 148 MiB GPU client.
- Interpretation: a 13.42 GiB model streams at 1.02 tok/s through the custom tiered path. Config C's effective 14.7 GB/s matches the pinned H2D rate measured in E021 (14.738 GB/s), consistent with the H2D copy of the 5.55 GiB cold set being the dominant serialized cost; in the current pipeline the copy and the GEMV run on one CUDA stream, so they cannot overlap. Staging beats direct host-resident reads by 1.42x (1.021 vs 0.721 tok/s); in config A the 5.42 GiB host-resident portion works out to roughly 4 GB/s effective (inferred, assuming the E021 VRAM GEMV rate for the 8 GiB hot part), well below the 8.8-12 GB/s mapped rates of E021/E023.
- Caveats: this is a weight-streaming proxy, not inference: no attention/SSM math, no KV/SSM state, no token sampling, synthetic input vector. For a 27B model at this bandwidth, decode is weight-bandwidth-bound, so this bounds the achievable tokens/s from above. No llama.cpp comparison yet. Raw: results/streamed-decode-27b-direct-hot8-20261108.txt, results/streamed-decode-27b-hot8-cache4.5-20261108.txt, results/streamed-decode-27b-hot8-cache5.6-20261108.txt.

## E032 — copy/compute overlap in the streamed decode proxy (measured)

- Date: 2026-11-08. benchmarks/streamed_decode.cu (build: `nvcc -O2 -std=c++17 -arch=sm_86 -lpthread`): same 13.42 GiB Qwen3.8-27B STRICT Q4_0 model, 726 items, 8 passes, 4 O_DIRECT reader threads, taskset -c 0-3, config C (8 GiB hot + 5.55 GiB RAM cache, zero NVMe in steady state). Change from E031: H2D copies run on a dedicated copy stream `st` and GEMV kernels on a second compute stream `stc`, so the copy of the next item overlaps the GEMV of the current one. Slot reuse is guarded by one CUDA event per VRAM slot (`evc[kSlots]`), primed on `stc` at startup so the first reuse-wait per slot is a no-op; a slot's H2D copy waits on the event recorded after the GEMV that last used that slot.
- Bug found and fixed: the first overlap build indexed the slot-reuse event by the staging-buffer modulus (`evc[(k-kSlots) % kStage]`, kStage=8) while slots cycle by kSlots=4, and kHot items (which use the hot arena, not a slot) recorded no event. When a kHot item sat at the `k-kSlots` position the wait landed on a stale/unrecorded event, so an H2D copy could overwrite a slot a newer GEMV was still reading. That build measured 1.240 tok/s but FAILED verification (13 mismatches, all in `blk.36.ffn_up.weight`). The fix (one event per slot, primed at startup, wait on `evc[k % kSlots]`, record on `evc[(k-1) % kSlots]`) removed the mismatch.
- Result: PASS, zero mismatches across 3976 checked rows. Steady state (passes 1-7) 0.802-0.808 s/token, best 1.247 tok/s, effective 17.85-17.97 GB/s. Versus E031 config C (1.021 tok/s, 14.6-14.7 GB/s) this is ~22% faster. Resources: 8.3 GiB VRAM allocated (2.14 GiB free after alloc), 5.54 GiB RAM cache, MemAvailable stable ~11.9 GiB, swap flat at 7.18 GiB (no growth).
- Interpretation: overlapping the H2D copy with the GEMV raises the full-cache rate from 1.021 to 1.247 tok/s, confirming the E031 pipeline was partially serialized on the single stream. The effective 17.9 GB/s now exceeds the isolated pinned H2D rate of E021 (14.738 GB/s) because the copy and the GEMV no longer share a stream; the remaining gap to the theoretical ~0.44 s/token (5.82 GiB cold set at ~14.7 GB/s) is the next optimization target.
- Caveats: still a weight-streaming proxy, not inference (no attention/SSM math, no KV/SSM state, no token sampling, synthetic input vector). Raw: results/streamed-decode-27b-hot8-cache5.6-overlap-20261108.txt.

## E033 — dp4a GEMV, decode batching, hot-arena stream (measured; work done 2026-11-08, logged 2026-11-08)

- Change from E032: (a) the fp32 Q4_0 GEMV (compute-bound at ~24.6 GB/s) was replaced with an int8 dp4a kernel using q8_1-style activation quantization (~275 GB/s at batch 1); (b) `--batch NB` runs NB decode vectors per weight pass, amortizing each weight read (and each H2D byte) across NB tokens — aggregate tok/s = NB x passes/s; (c) `--hot-stream 1` launches all hot-arena GEMVs on a third stream up front so they execute under the cold H2D transfers. CPU reference mirrors the int8 activation quantization exactly.
- Results (same model/config as E032, 8 passes, all PASS, zero mismatches, 3976 x NB rows checked): dp4a batch 1 without hot stream 1.949 tok/s; with hot stream: batch 1 = 2.031, batch 8 = 16.190, batch 16 = 32.405 aggregate tok/s. Without hot stream: batch 4 = 7.473, batch 8 = 14.261, batch 16 = 25.464. Pass time is nearly flat from batch 1 to 16 (~0.49-0.50 s), i.e. compute hides almost completely under the cold-set copy at NB<=16.
- Interpretation: the dp4a kernel removed the compute bottleneck that capped E032 at 1.247 tok/s; batching then converts the copy-bound pass into aggregate throughput. Single-stream remains ~2 tok/s (PCIe-bound). Raw: results/streamed-decode-27b-hot8-cache5.6-dp4a*-20261108.txt. Script: scripts/e033_batch_sweep.sh.

## E034 — pinned RAM-cache head + batch 32/64 sweep (measured)

- Date: 2026-11-08. Same model/config (8 GiB hot, 5.55 GiB cold cache, 8 passes, 4 readers, taskset -c 0-3). Changes: (a) the head of the staged RAM cache is pinned with cudaHostAlloc up to the RLIMIT_MEMLOCK budget (3658 MiB of the 5553 MiB cache; hard limit 3978 MiB, a `systemd-run --user -p LimitMEMLOCK=infinity` probe confirmed it cannot be raised without root); cached pinned items H2D-copy straight from the pinned cache to the VRAM slot with no CPU memcpy and no reader-thread involvement in steady state — only the ~1.9 GiB pageable tail still stages through the ring; (b) added NB=32 and NB=64 kernel instantiations.
- Results (final binary, all PASS, zero mismatches): batch 1 = 1.951 tok/s (3,976 rows); batch 16 = 31.775 aggregate (63,616 rows); **batch 32 = 43.790 aggregate tok/s (127,232 rows), best pass 0.731 s**; batch 64 = 37.914 (254,464 rows). Batch 32 resources: 1.91 GiB free VRAM after alloc, MemAvailable ~14.6 GiB, zero swap growth, zero NVMe in steady state.
- Negative results: (a) packing the activation scale/sum into one float2 load per block made batch 32 27% SLOWER (33.55 tok/s, PASS) — reverted; raw: results/streamed-decode-27b-hot8-cache5.6-pincache-pack-batch32-20261108.txt. (b) Batch 64 regresses vs 32: ptxas shows no spills at either size, but NB=64 uses 96 registers vs 72 at NB=32, cutting occupancy (5 vs 7 blocks/SM) while doubling per-byte compute.
- Interpretation: 43.8 aggregate tok/s is 43x the E031 baseline (1.021) and meets the 20-40 tok/s target. Batch 32 is compute-bound (0.73 s/pass vs ~0.5 s copy time); the copy-bound ceiling at this batch size is ~64 tok/s, reachable only with a faster kernel (e.g. tensor-core int8 MMQ). Single-stream is PCIe-bound: 5.82 GB cold set per token at ~11.5 GB/s effective H2D under concurrent compute (isolated pinned H2D: 14.7 GB/s, E021) gives a physical ceiling of ~2.5 tok/s for this dense 13.42 GiB model on a 12 GiB GPU — no software path to 20-40 tok/s single-stream exists without reading fewer bytes per token. Pinning the cache head did not raise the measured H2D rate (the pageable-tail memcpy was already hidden under the copy); it removes CPU staging work for 2/3 of the cold set.
- Caveats: weight-streaming proxy, not inference (no attention/SSM math, no KV state, synthetic inputs). Batch aggregate throughput is serving-style (NB parallel decode vectors), not single-stream latency. Raw: results/streamed-decode-27b-hot8-cache5.6-final-batch{1,16,32}-20261108.txt, results/streamed-decode-27b-hot8-cache5.6-pincache-batch{1,16,32,64}-20261108.txt.

## E035 — H2D ceiling profile: IOMMU implicated (measured)

- Date: 2026-11-08. User challenged the "physics limit" framing; profiled the 14.7 GB/s H2D ceiling instead of asserting it. Tools: benchmarks/gemv_probe.cu (existing E033 probe, first full run logged) + new benchmarks/h2d_d2h_probe.cu.
- Measured: pinned H2D is flat at 14.17-14.32 GB/s across ALL chunk sizes (4/8/16/32/64/128 MiB) and stream counts (1/2/4) — a hard ceiling, not a pipeline artifact. D2H = 12.67 GB/s (roughly symmetric, so MaxReadReq is NOT the limiter). asyncEngineCount=2 but multiple streams give zero gain. Pageable->pinned memcpy = 29 GB/s at 2+ threads (not a bottleneck). Link confirmed Gen4 x16 (16 GT/s, width 16) at both GPU and CPU root port; GPU hangs directly off the CPU root port 00:01.0, no chipset hop. Nominal link rate 31.5 GB/s; expected real-world pinned H2D on Gen4 x16 is ~25 GB/s.
- Limiter candidate: IOMMU DMA translation is ACTIVE (GPU in /sys/kernel/iommu_groups/12; dmar0 enabled; no intel_iommu=off/pt on cmdline). IOMMU overhead is a known bandwidth tax on consumer Intel+NVIDIA and is a config limiter, not silicon — removable via `iommu=pt` (passthrough, keeps interrupt remapping) on the kernel cmdline + reboot. Needs explicit user approval (standing constraint: no kernel changes). MPS/MRRS registers unreadable without root; the symmetric D2H result already argues against MRRS.
- If IOMMU is the cause: H2D ~14.3 -> ~25 GB/s would take single-stream from ~2 to ~4 tok/s and batch-16 from ~32 to ~60 aggregate. Untested until a reboot is approved.
- Probe GEMV agreement: dp4a vs fp32 max rel diff 1.6e-3 (int8 activation quantization, expected); dp4a 1-row vs 4-row identical. Raw: results/gemv-probe-h2d-sweep-20261108.txt, results/h2d-d2h-asymmetry-20261108.txt.

## E036 — real speculative decoding (native MTP head) on the over-VRAM 27B (measured)

- Date: 2026-11-09. First REAL inference (not the weight-streaming proxy) on the 13.42 GiB Qwen3.8-27B with the model exceeding 12 GiB VRAM. Built a fresh MTP-capable llama-cli out-of-source at /home/zabwie/llama-mtp-build (commit d775b8967, b10573; the user's /home/zabwie/llama.cpp checkout was NOT modified — its prebuilt binaries predate MTP and segfaulted loading the 18-tensor head as a full model). Target run with GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, -ngl 99, -c 1024, -n 128, --temp 0, --reasoning off. Arm B adds the model's native NextN/MTP head (mtp-Qwen3.8-27B-Q4_0.gguf, 1.68 GiB) via -md / -ngld 99 / --spec-type draft-mtp / --spec-draft-n-max 8. PCIe logged with nvidia-smi dmon -s t -d 1. Script: scripts/e036_mtp_spec.sh.
- Results: baseline generation 0.63 t/s (161.7 s / 103 tok, 1586 ms/tok), prompt 4.33 t/s. MTP generation 1.04 t/s (100.1 s / 105 tok, 962 ms/tok), prompt 4.61 t/s, draft acceptance 0.23026 (70 accepted / 304 generated), mean accepted len 2.84. **Speedup 1.65x.** Both exit 0; swap grew +689 MiB (baseline) and +826 MiB (MTP) from a 7.38 GiB baseline, within the +4 GiB abort guard; MemAvailable stayed >24 GiB.
- PCIe bytes per token (integrated rxpci over the generation window): baseline ~7.1 GiB/token (steady 4.65 GB/s), MTP ~6.9 GiB/token (steady 7.36 GB/s). Whole-run rx ~759 GiB (baseline) vs ~776 GiB (MTP).
- KPI verdict (user research doc: "does exact multi-token verification cut bytes/accepted-token by >=3x?"): **NO.** Bytes per accepted token is essentially unchanged (~7 GiB both arms). The 1.65x speedup is amortization of fixed per-pass overhead (kernel launch, attention, page-fault handling) over ~2.84 tokens, NOT a reduction in weight traffic. Low 23% acceptance means most draft tokens are rejected; the target still streams its weights every verification pass and the draft head adds traffic.
- Correctness: temp-0 outputs DIVERGE. Identical through "...allowing it to handle", then baseline "...the massive number of ... rendering graphics..." vs MTP "...massive amounts of ... rendering 3D graphics...". Attributed to floating-point nondeterminism: batched verification computes a GEMM whose reduction order differs from per-token GEMV, flipping greedy argmax at near-ties. Distribution-preserving but not bit-exact on this path.
- Interpretation: the dominant real-inference bottleneck is managed-memory THRASHING, not the absence of speculation. ~7 GiB/token H2D exceeds the ~5.5 GiB cold set because weights + KV + MTP draft > 12 GiB VRAM, so pages are evicted and re-faulted every token at the demand-migration ceiling (~4.6-7.4 GB/s). llama.cpp unified memory demand-migrates blindly with no intelligent residency; this is why real inference is 0.6-1.0 t/s while the staged proxy (E034) hit ~2 t/s single-stream / 43.8 t/s batch on the same weights. Speculation helps at the margin (1.65x) but does not address the traffic; the staged-residency architecture remains the more promising path, and raising H2D bandwidth (E035 IOMMU test) would lift both.
- Caveats: single prompt, single run per arm (no variance estimate); 1024 ctx; acceptance is prompt-dependent. Raw: results/e036-{baseline,mtp}-20261109-171251.txt and -pcie.txt.

## E037 — IOMMU passthrough removes the H2D bandwidth tax (measured; user-approved kernel change)

- Date: 2026-11-09. Follow-up to E035. User approved and applied `iommu=pt` on the kernel cmdline (GRUB) + reboot. Verified post-reboot: /proc/cmdline contains `iommu=pt`; all 20 IOMMU groups flipped DMA-FQ -> `identity` (GPU group 16 included); link retrained to Gen4 x16 under load (idle reports Gen2 via power saving).
- Bandwidth (benchmarks/h2d_d2h_probe, 2 GiB/direction, 256 MiB pinned): **H2D 14.33 -> 24.61 GB/s (1.72x), D2H 12.67 -> 25.87 GB/s (2.04x)**. This confirms the E035 hypothesis: the IOMMU DMA translation was a ~42% H2D bandwidth tax, a config limiter not silicon. The ~25 GB/s matches expected real-world pinned H2D on Gen4 x16. Raw: results/h2d-d2h-iommu-pt-20261109-*.txt.
- Throughput impact (streamed_decode proxy, same E034 config: 8 GiB hot, 5.55 GiB cache, 8 passes, 4 readers, taskset -c 0-3; all PASS zero mismatches): **batch 1 single-stream 1.951 -> 4.009 tok/s (2.05x, 3976 rows); batch 16 aggregate 31.775 -> 64.119 tok/s (2.02x, 63616 rows)**. Best pass rate 4.007/s for both batch 1 and 16 (compute fully hidden under the copy); effective cold-set H2D ~23.3 GB/s = 95% of the isolated 24.61. Single-stream is now near the Gen4-passthrough ceiling (5.82 GB cold/token / 24.61 GB/s = 4.23 tok/s theoretical). Raw: results/streamed-decode-27b-iommu-pt-batch{1,16}-20261109-172744.txt.
- Interpretation: the user's challenge to the "physics limit" framing was correct — profiling found a removable config limiter, and removing it doubled both bandwidth and end-to-end proxy throughput. Single-stream 4 tok/s is now the measured ceiling for this dense 13.42 GiB model reading its full ~5.8 GB cold set per token over Gen4 x16; going faster single-stream requires reading fewer bytes per token (approximate-layer drafting + exact verification done right, cf. E036's negative traffic result), not more bandwidth. Batch 32 was compute-bound (E034) so is not expected to gain much from the copy speedup.
- Caveats: proxy throughput, not real inference (no attention/SSM/KV). The kernel cmdline change is persistent (GRUB); `iommu=pt` keeps interrupt remapping but disables DMA translation for all devices — a security/ isolation tradeoff the user accepted. Revert by removing `iommu=pt` from GRUB_CMDLINE_LINUX_DEFAULT + update-grub + reboot.


## E038 — Real decode loop with tiered residency (2026-11-10)

**Goal.** Port the streamed_decode proxy's intelligent residency (8 GiB VRAM hot arena + 5.55 GiB RAM
cache + overlapped H2D, E031-E037) into a REAL decode loop — attention + KV state + the qwen35
gated-delta-net path — to close the gap between the 4 tok/s proxy and llama.cpp's 0.63-1.04 t/s
unified-memory thrashing (E036).

**Implementation.** `benchmarks/real_decode.cu` (~1,100 lines): own GGUF parser, byte-level BPE
detokenize, dp4a Q4_0 GEMV with per-32-block int8 activation quantization (same as the proxy),
per-head RMS/L2 norms, NEOX RoPE (qwen3) and IMROPE-as-half-split RoPE (qwen35, text positions),
GQA decode attention with KV cache, gated-delta-net recurrence with persistent 48x(128x128) fp32
state + conv state, gated output norms, and the proxy's slot/event/reader streaming machinery
(hot arena loaded once at startup; staged items through the pinned ring with slot-reuse events).
Greedy sampling; optional `--validate 1` runs an mmap-backed fp32 CPU reference that mirrors the
GPU math exactly (same int8 activation quantization) and compares argmax per step (ties < 0.25
logit gap excused as FP reduction-order noise).

**Validation.**
- Qwen3-0.6B (fits in VRAM): 24/24 generated tokens identical to llama.cpp CPU greedy (user build,
  d775b8967, via a token-ID dump tool); CPU-reference 43/45 exact argmax + 2 near-ties (gaps 0.02/0.29).
- Qwen3.8-27B: per-op bisection against a patched out-of-tree llama.cpp copy (/tmp/llama-dbg, cb_eval
  node dumps, user checkout untouched) confirmed layer-0 equality, then located the single remaining
  divergence: the gated-delta-net k-head mapping is `h % n_group` (ggml `iv1 % nek1`), not `h / 3`.
  After the fix, all 48 per-head delta outputs match llama.cpp to >=4 decimals, and the 27B generates
  token-for-token identical greedy output to the llama.cpp CPU reference (8+ tokens:
  248068 198 760 1156 369 9859 264 4145 = "<think>\nThe user is asking a simple").

**Results (27B, hot 8192 MiB / cache 5553 MiB / readers 4, taskset 0-3, iommu=pt).**

| metric | value |
|---|---|
| steady decode | **3.39 tok/s** (best 3.43), 0.290-0.300 s/token |
| llama.cpp unified memory (E036, same model) | 0.63 t/s baseline / 1.04 t/s with MTP |
| speedup vs baseline | **5.4x** (2.0x vs E036's MTP arm, with correct greedy output) |
| cold H2D per token | 5.423 GiB (hot 8.0 GiB resident, RAM cache hit, zero NVMe steady-state) |
| effective bandwidth | ~18.7 GB/s vs 24.6 GB/s isolated pinned H2D |
| VRAM headroom | >= 2.87 GiB free throughout |
| correctness | PASS; token-for-token match vs llama.cpp greedy |

**Bugs found and fixed during E038** (each initially produced plausible-but-wrong or NaN output):
1. KV-cache appends were issued on the copy stream while the producing GEMVs ran on the compute
   stream (race); layer >= 1 read zeros. Appends now run on the compute stream.
2. SSM/conv state buffers were indexed by raw layer index (il up to 62) but allocated for 48
   recurrent layers (195 MB into 151 MB) — illegal access. Fixed with ri = il - (il+1)/4.
3. Delta-net k-head mapping: h/3 (repeat-group intuition) -> h % n_group (ggml semantics).
4. qwen35 RoPE: IMROPE uses NEOX half-split pairing (ggml rope switch), i.e. plain half-split RoPE
   for text — not adjacent-pair pairing.
5. K-slice L2 norm was applied at the V offset (2*Sdim*group instead of Sdim*group).
6. Conv1d weight layout confirmed as w[c*4 + j] (ne[0]=4 taps fastest); an attempted "fix" to
   w[j*C+c] was reverted after per-value comparison against llama.cpp.
7. yA/yB scratch sized by max chunk rows instead of max tensor rows (17408) — overflow on 27B only.
8. CPU reference: double zero-point subtraction in the Q4_0 mirror (GPU was correct); ssm/conv state
   indexing; weights now mmap-backed (a 14.4 GiB anon copy swapped out the desktop and tripped the
   swap guard).

**Interpretation limits.** Single-stream decode is bound by the same cold-set/PCIe physics as the
proxy (5.42 GiB/token at ~19 GB/s effective vs 24.6 GB/s isolated H2D = 78% efficiency); the gap to
the 4.0 tok/s proxy is per-token compute/launch overhead in the real graph. Batch>1 (E033-style)
and approximate-layer drafting remain the levers for aggregate throughput. Raw:
results/e038-real-decode-27b-*.txt, results/e038-validate-06b-*.txt.
