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
