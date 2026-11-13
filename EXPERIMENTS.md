# Experiment log

## E000 — read-only baseline discovery

- **Date:** 2026-11-08
- **Question:** What hardware/runtime is available, and is the GPU quiet enough for a controlled first run?
- **Procedure:** Read CPU, RAM, PCIe sysfs, `nvidia-smi`, installed compiler versions, and NVIDIA address mode. No hardware settings were changed and no CUDA kernel was run.
- **Observations:** See [HARDWARE.md](docs/HARDWARE.md). RTX 3060 12 GiB, i7-12700K, PCIe 4.0 x16 at 16 GT/s, Linux kernel 7.0.0-38, NVIDIA driver 595.99.02, CUDA toolkits 12.0 and 13.2, HMM address mode. About 19 GiB RAM available; 3.9 GiB memlock allowance.
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


## E039 — Batched real decode: amortizing the cold set (2026-11-10)

**Goal.** Cut the 5.42 GiB/token cold-set cost of E038 by sharing each weight read across B
decode sequences (E033-style amortization on the real graph).

**Implementation.** `benchmarks/real_decode.cu` gained a `--batch B` dimension (power of two,
1..64): the dp4a Q4_0 GEMV is templated on NB (ported from streamed_decode, one weight read
shared by NB activation vectors), every activation buffer is [B][...], KV cache and SSM/conv
states are per-sequence ([B][layer][...]), and all non-GEMV kernels take a gridDim.y = B
sequence axis. Prompts: ';'-separated equal-length sequences, replicated to B for equivalence
testing. Lockstep decode; per-sequence greedy sampling and EOS.

**Bugs found (each verified by per-layer/per-op dumps before fixing):**
1. **Flat-index RMW race (the big one).** The six elementwise kernels (k_add, k_silu_mul,
   k_sigmoid, k_gate_ssm, k_sigmoid_mul, k_attn_gate_apply) used grid (x, B) with a flat
   `i = blockIdx.x*blockDim.x + threadIdx.x` index — every blockIdx.y block processed the SAME
   index range, so `y[i] += x[i]` became a non-atomic read-modify-write race across B blocks.
   Lost/duplicated updates produced rare nondeterministic token divergence (batch 4: mostly
   correct; batch 8: always wrong). Fixed with the global index
   `(blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x`. Isolated in a standalone
   kernel: 4-block `+=` on one address yields 1-4 effective adds depending on timing.
2. **k_delta_net q/k/v stride.** The kernel advanced q16/k16 by vs*ngroup*S and v by vs*H*S,
   but the conv output is dense [B][qkv_out=10240] — v>=1 read seq0's K slice as its Q.
   Fixed with an explicit qkv_seq_stride = m.qkv_out.
3. **KV-cache stride units.** kv_seq_stride was bytes, added to a float* (4x overshoot,
   illegal access at batch 8). Fixed to float units.
4. **Logits scatter.** The output-head GEMV used the max_rows buffer pitch instead of n_vocab,
   scattering batch vectors across the logits buffer.
5. **Prefill handoff.** The first sampled token was dropped (argmax taken before prefill end).
6. **k_l2_heads not batched** (all sequences normalized seq0's data).

**Validation.** Batch-8 replicated-prompt runs are token-for-token identical to the validated
batch-1/llama.cpp sequence: 0.6B 6/6 runs, 27B equivalent=yes (all 8 sequences), seq0 =
"<think>\nThe user is asking a simple factual question about the capital of France,". Per-op
dumps confirm qkv/beta/conv/dnout/gatednorm are bit-identical across sequences at batch 8.
compute-sanitizer memcheck/initcheck/racecheck: 0 errors (note: racecheck cannot catch the
global-memory RMW race; the standalone reproducer did).

**Results (27B, iommu=pt, taskset 0-3, 16 gen steps, all PASS, equivalence=yes).**

| batch | hot/cache MiB | step time | aggregate tok/s | cold GiB/token | min free VRAM |
|---|---|---|---|---|---|
| 1 (E038) | 8192/5553 | 0.295 s | 3.39 | 5.42 | 2876 MiB |
| 4 | 8192/5553 | 0.374 s | **10.70** | 1.356 | 2189 MiB |
| 8 | 7424/6144 | 0.607 s | **13.17** | **0.772** | 2223 MiB |

Batch 8 is the compliant maximum: batch 16 needs 2.36 GiB of SSM states and busts the 2 GiB
VRAM floor even with the hot arena trimmed (the first batch-8 attempt at hot 8192 ran at
14.8 tok/s but 1254 MiB free — below floor — and was re-run at hot 7424). The cold set is now
0.772 GiB/token — 7x less than single-stream — and aggregate throughput is 3.9x single-stream
and 12.7x llama.cpp's MTP arm (1.04 t/s). Step time is now compute+copy mixed: the copy grew
5.42->6.17 GiB/step (smaller hot arena) while batched compute scales with B, so marginal
efficiency declines beyond batch 8 without further kernel work (tensor-core MMQ, E028 item).

**Interpretation limits.** Aggregate tok/s is serving-style (8 independent sequences), not
single-user latency; single-stream remains ~3.4 tok/s and is PCIe-bound. All batch sequences
here are replicated for equivalence; diverse prompts are supported (equal lengths) and expected
to show the same throughput. Raw: results/e039-real-decode-27b-batch8*.txt,
results/e039-real-decode-27b-batch4-*.txt.

## E043 — the acceptance killer: two silent correctness bugs in the speculative loop (2026-11-10)

**Question.** HANDOFF known-issue #1: PLD acceptance capped at 28–33% where a periodic stream
should accept ~100%; `known` drifted from the history tail. Find and fix the alignment drift.

**What was actually wrong — two independent bugs, both silent:**

1. **Rollback discarded accepted rows' recurrent state (issue #1, the designed-in bug).**
   On a partial accept the code restored the SSM/conv snapshot taken *before* the verify
   pass. The snapshot covers positions 0..base-1, but verify rows 0..mt had processed the
   carried token AND the accepted drafts — cumulative gated-delta-net/conv updates the
   restore throws away, while `cur_pos = base+mt+1` never reprocessed them. From the first
   partial accept, 48/64 layers ran on a state missing the accepted positions' contributions:
   logits drifted, `known` diverged from the history tail, acceptance collapsed, and — worse —
   the emitted stream silently stopped being the greedy decode. The old "7.26 tok/s chat at
   33% acceptance" was measured on this corrupted stream (the drift made the model echo its
   own repeats, which the n-gram drafter then "predicted"). **Fix = re-feed carry:** a carry
   buffer holds all confirmed-but-unprocessed tokens; after a rollback they are reprocessed
   as the front rows of the next pass — deterministic recomputation (same tokens, order,
   kernels, bit-exact restored state) rebuilds the state exactly and re-verifies the carry
   inside the same pass. `dbg` logits traces confirm cycles 1+ recompute identical positions
   to the correct values.

2. **Pinned-cache read-back aliasing (found because the fix exposed it; pre-existing).**
   The reader's cached-copy branch did `memcpy(stage, staged_cache + it.res_off, bytes)` for
   *all* cached staged-tier items — but pinned-head items' bytes live in `pinned_cache`, and
   the `[0, pin_bytes)` region of `staged_cache` is never written. This branch only executes
   on a **pass-0 re-entry with populated caches** — a code path only speculative decoding
   reaches (spec cycle 0; dense/multi never re-enter pass 0). Every cycle-0 pass therefore
   computed on garbage weights for the pinned band (~3.6 GiB). Detection signature: cycle 0
   produced correct argmax at the tuned 13-token operating point but with *different logits*
   (23.69 vs 26.90 warm) — it flipped the argmax on every other config/prompt tried
   (6144/4608, 6656/5120, 6144/6144, γ=16, 153-token agent prompt). The old "hist[n0-1]=271"
   dbg observation was this bug plus bug 1 compounding. **Fix = one line:** read
   `(in_pin ? pinned_cache : staged_cache) + it.res_off`.

3. **E040's teacher-forced soft check was vacuous (pre-existing).** It compared row p's argmax
   against `seq[base+Bp+p]` (wrong index — comment said base+p+1) and computed
   `gap = lg[expected] - lg[pred]`, which is ≤ 0 for an argmax — every mismatch was excused as
   a tie. The multi-position math itself is CORRECT (dense per-position argmaxes match verify
   rows exactly); only the check was dead. Fixed: correct index, absolute gap, and skip
   prompt-region rows (teacher forcing does not imply argmax inside the prompt — the earlier
   "step-0 garbage" scare was this distinction, not a bug).

**New tooling.** `--expect-ids <file>`: lossless gate — diffs the speculative stream against a
dense greedy reference saved with `--out-ids` (same prompt, --gen-tokens); exit 1 on drift.
This is the correctness gate the spec loop never had.

**Validation (all `spec_lossless ... match=yes`, zero mismatches):** 27B chat prompt γ=8 at
7424/6144, 6144/4608, 6656/5120, 6144/6144; γ=16 at 6144/4608 (deep γ now lossless — it
previously needed the low-hot configs that bug 2 corrupted); 153-token agent trace γ=8; 0.6B
γ=8 32/32. Dense streams are config-independent (7424 vs 6144 identical), multi-verify shows
0 real mismatches at both configs.

**Honest re-measurement (27B, single-user, lossless, hot 7424):** chat γ=8: 2.46 tok/s, 3.8%
draft acceptance. Agent-shaped trace (synthetic tool-call JSON, results/e043-agent-prompt.txt)
γ=8: 3.63 tok/s, 14.8% acceptance; γ=16: 1.91 tok/s, 7.6%. Dense reference: 2.81 tok/s.

**Interpretation.** The known-issues list predicted the fix would unlock acceptance; it did
not, because the drift itself was *inflating* the old numbers (corrupted state → predictable
echoes). With correct state, PLD drafting quality is the bottleneck: the drafter rarely
proposes what a greedy 27B actually continues with on these traces, and deep γ amortizes a
pass over mostly-rejected rows. The verify machinery (one-pass multi-position, state
rollback, re-feed) is now correct and gated; the next lever is drafter quality (adaptive-γ,
RESEARCH_PLAN), not verification depth. Raw: results/e043-*.

## E044-E049 — the drafter sprint: 2.8 -> 9.2 tok/s chat, 5.8 agent, all lossless-gated (2026-11-10)

Owner directive: "2 t/s is too slow — fix everything, chase the ceiling." Sequence, one change
at a time, every run gated by --expect-ids vs the dense greedy stream:

- **E044 (VRAM right-sizing):** ssm/conv/kv were allocated per batch slice but multi/spec rows
  share slice 0 — 1.27 GiB dead VRAM at batch 8. Sized to 1 slice: gamma=16 now runs at the
  fast hot config. RAM cache sized to cover the whole cold set (nvme_per_token 0.17 -> 0.00
  GiB/token). Dense 2.85 -> 3.53 tok/s (8704/5150) or 3.35 desktop-safe (8448/5400).
- **E045 (no dummy rows):** the verify pass width is exactly carry+drafts. Padded dummies
  cost ~7.4 ms/row for guaranteed rejects. Low-draft cycles now cost dense-equivalent.
- **E046 (MTP head drafter — the big one):** the qwen3.5 nextn block (blk.64) integrated as a
  model-guided drafter. token_embd + output.weight are byte-identical to the main model
  (sha256-verified): ~230 MiB unique weights + 682 MiB head in VRAM; embeddings reuse the
  main pread path. Scheme mirrors llama.cpp speculative.cpp: per-position h_tgt (= the
  final-normed hidden the main pass already computes — zero extra cost), MTP sync over
  accepted rows with the h input shifted one position back, recursive draft chain (h = the
  MTP's own output), MTP KV synced over the whole prefill. Two bugs on the way:
  (1) k_add must use the 2D flat grid ((n*B+255)/256, B) — dim3(1,B) truncated the MTP
  residual to the first 256*B flat elements; teacher-forced accuracy was 6-8% until fixed.
  Ground truth: llama.cpp reference (E036 build, unified memory) gets 50% acceptance / mean
  accepted len 5.0 on the same prompt. (2) MTP KV empty over the prefill — invisible on a
  13-token prompt, fatal on a 153-token agent prompt (draft steps attended zero KV).
  **Chat: 2.9 -> 9.3-10.0 tok/s (3.4x dense), 78% acceptance at p_min 0.3.**
- **E047 (copy pipeline):** slot ring 4->8, release window 3->6. nsys profile: the pass is
  H2D-copy-bound — 0.40 s/pass at 13.6 GB/s effective (median copy 16.6 MB) vs the 24.6 GB/s
  peak; all compute (GEMV 140 ms, delta-net 85 ms) is fully hidden behind copies.
- **E048 (negative):** 64 MiB items + write-combined host buffers both REGRESSED (0.45 ->
  0.99 s/cycles). Reverted; recorded so nobody tries it blind again.
- **E049 (hybrid drafter):** PLD overrides the MTP chain when it finds a max-length (6-token)
  history match — periodic/echo spans are near-certain and free to draft; MTP handles novel
  content. **Agent trace 4.6 -> 5.79 tok/s; chat 9.22 (PLD rarely fires there).** Synthetic
  "periodic" traces do NOT produce periodic greedy continuations (the model helpfully
  comments on the pattern instead) — bursts are bounded by real echo content, and the
  zero-draft verify bound is 20.4 tok/s at gamma=8.

**Ledger (27B, single-user, lossless-gated, hot 8192/cache 5600 + MTP):** dense 2.7-3.5 |
PLD-only 2.9-3.3 | MTP 9.4 chat / 4.6 agent | hybrid 9.22 chat / 5.79 agent (1.7-3.4x dense)
| verify bound gamma=8: 20.4. Raw: results/e04[4-9]-*.

## E050 — profile of the 6.99 tok/s path: it is GEMV-bound, not copy-bound (2026-11-10)

Reproduced the current best config ungated first: hybrid drafter γ=16, agent trace, 64 tokens,
hot 8192/cache 5600 → **6.990 tok/s, lossless PASS** (22 cycles, 3.41 tok/cycle, 34% accept).
Then nsys (results/e050-spec-agent-g16.nsys-rep) + isolated microbenches
(benchmarks/microbench/e050-gemv-load-decomposition.cu). Full analysis: results/e050-profile-20261110.md.

**The E047 "copy-bound" diagnosis is superseded at this config.** Copies run at 25.5 GB/s
effective (139 GB in decode) — at the PCIe peak — and hide under compute; GPU is 98.1% busy.
The 488 ms cycle is: k_mat_q4 271 ms (55%) + k_delta_net_multi 116 ms (24%) + small kernels
17 ms + exposed copies 82 ms.

- **k_mat_q4 collapses 5.4x at NB=16**: measured 273 GB/s at NB=1 → 50.6 GB/s at NB=16.
  E034's "275 GB/s dp4a" was an NB=1 microbench; the pipeline always ran at ~50 GB/s, hidden
  behind copies until now. Decomposition: xq re-reads ~0.25 ms + xs/xsum 32x-overfetch loads
  ~0.22 ms + weight loads ~0.11 ms + ALU 0.05 ms per 29.8 MB tensor. Negative results:
  repacked layout 0.85x, per-row smem tile 0.15x, once-per-block smem tile ≈ current.
  Fix (designed, unimplemented): smem xq+xs/xsum tile per K-half, grid-stride rows,
  deterministic partial sums (no atomics), optional weight repack. Est. 2.5-4x → 271→70-110 ms.
- **k_delta_net_multi at ~0.3% fp32 peak**: thread-per-(h,j) serial 128-loops ×4 ×17 positions,
  512 B lane stride on state, 48 blocks/28 SMs. Fix: transposed state + warp-split S-loops.
  Est. 116→5-15 ms.
- **Next wall after both**: the copy chain (222 ms/cycle = 5.42 GiB / 25.5 GB/s) bounds the
  cycle → ~13-15 tok/s at today's 34% acceptance; beyond that, acceptance × cold-bytes are
  the terms (consistent with the 20.4 tok/s zero-draft verify bound).
- Rollback D2D (151 MB × 39) = 35.6 ms TOTAL — the E043 hybrid-state tax is negligible.

## E051-E053 — follow-up kernel probes (2026-11-10)

- **E051 GEMV tile/multi-row probe:** baseline 0.539 ms. Tiled activation variants were 0.594–0.674
  ms; multi-row amortization R=2/4/8 was 0.730/0.529/0.655 ms. The R=4 result is within noise
  and does not establish a useful win. All real variants were bit-exact. Raw:
  `results/e051-kernel-probes-20261110.txt`.
- **E051 delta-net probe:** transposed state and split-S loops reduced the isolated H48/S128/B17
  kernel from 3.301 ms to 0.817 ms; the fused variant took 0.747 ms. Output and updated state
  were bit-exact. Scale-only shared-memory variants in E051b were slower (0.578–0.659 ms vs
  0.546 ms baseline). Raw: `results/e051b-kernel-probes-20261110.txt`.
- **E052 GEMV mapping/cache sweep:** shared-weight vector mappings, warp-count/block-size changes,
  and `__ldg`/`.cg` cache modifiers did not produce a stable GEMV improvement. The CUDA
  persisting-L2 window did: median 0.5087 → 0.4415 ms (1.152×) on the 11,652×5,120 NB=16
  microbench, bit-exact. Raw E052a-e are in `results/e052*-20261110.txt`.
- **E053 weight cache policy:** `.cg` weight loads and `__ldg` activation loads regressed or tied
  the default path; not integrated. Raw E053a-c are in `results/e053*-20261110.txt`.

## E054-E055 — integrate delta-net layout and persist activation L2 (2026-11-10)

- **E054 delta-net layout:** production state is now `[head][i][j]`; adjacent lanes cover adjacent
  `j`, while each `j` keeps its original serial `i` accumulation order. Updated the one-token
  path, multi-token path, CPU reference, and multi-token launch grid. Snapshot/rollback remains a
  byte copy. The real 27B MTP+PLD run improved 6.990 → 8.803 reported tok/s, cycle 0.4877 →
  0.3873 s, and passed `--expect-ids` (32 compared, zero mismatches).
- **E055 activation L2 window:** `xq`, `xs`, and `xsum` share an aligned arena and the compute
  stream marks it persisting when it fits the device's L2 persisting limit. The same real run
  reached 9.283 tok/s and 0.3672 s/cycle (5.5% over E054), still lossless. On this GPU the arena
  was 348,160 bytes and the persisting limit 1,622,016 bytes. Raw:
  `results/e054-delta-layout-agent-g16-20261110.txt`,
  `results/e055-l2-window-agent-g16-20261110.txt`.

## E056-E060 — GEMV alternatives tested and rejected (2026-11-10)

- **E056 INT8 WMMA/Tensor Core:** 1.331 ms vs 0.552 ms reference (0.415×); accumulation order
  also changed. Not integrated. `results/e056-int8-wmma-q4-gemv-20261110.txt`.
- **E057 shared activation tile with grid-stride rows:** all tested row groups were slower
  (0.65–0.79×); numerical differences stayed below 1e-6 but were not bit-exact. The direct
  experiment rules out that implementation. `results/e057-gemv-gridstride-tile-20261110.txt`
  and `results/e057-rg*-20261110.txt`.
- **E058 vector-warps:** the four-vector-per-warp variant was bit-exact and 1.13–1.21× faster on
  selected synthetic shapes, but a broad real-decoder dispatch regressed to 8.629 tok/s. Gating it
  to K=5120 gave 9.098 tok/s, still below the 9.316 tok/s control. Raw E058 shape and full-run
  records are `results/e058*-20261110.txt`.
- **E059 four-row vector-warp tile:** bit-exact, 1.04–1.07× on the tested K=5120 microbench
  shapes; the full path measured 9.104 tok/s vs a 9.316 tok/s control. Not integrated.
  `results/e059*-20261110.txt`.
- **E060 shared activation + vector-warps + grid-stride rows:** 0.45–0.60× reference with a
  53.75 KiB tile; reducing the chunk to 40 blocks still measured 0.44–0.49×. These candidates
  changed FP grouping but were deterministic; neither is in production. Raw E060 records are in
  `results/e060*-20261110.txt`.

The production `k_mat_q4<16>` remains unchanged. No GEMV replacement in this series met the
end-to-end acceptance bar; the E050 estimate of a 2.5–4× GEMV speedup remains unproven.

## E061-E064 — draft tuning and final lossless run (2026-11-10)

- **E061 confidence cutoff sweep, γ=16:** `p_min=0.1` gave 7.636 tok/s / 27.6% acceptance;
  `0.5` gave 8.948 / 31.3%; `0.8` gave 8.980 / 29.2%. The existing 0.3 cutoff remains best.
- **E062 draft width sweep, p_min=0.3:** γ=8 gave 9.171 tok/s / 47.9% acceptance; γ=32 gave
  7.291 / 17.6%. The γ=16 control was about 9.3 tok/s / 33.8%. All passed the exact-token gate.
- **E063 PLD lookback 6→12:** no change in draft count, acceptance, or throughput; reverted.
- **E064 final source:** rebuilt `benchmarks/real_decode.cu` after removing the losing GEMV
  dispatches. With γ=16, p_min=0.3, hot 8192/cache 5600, the agent trace produced 9.350 reported
  tok/s, 3.409 emitted tokens/cycle, 0.3646 s/cycle, and 33.8% acceptance. `--expect-ids` passed
  (32 compared, zero mismatches). Raw: `results/e064-final-state-layout-l2-agent-g16-20261110.txt`.

The requested 20–25 tok/s was **not reached**. Delta-net and activation-cache changes are retained;
the later E065-E075 work adds a partial GEMV improvement, but GEMV remains the dominant compute
cost. Wider draft windows and confidence cutoffs did not bridge the remaining gap on this trace.
The 13–15 tok/s post-fix estimate in E050 assumed a larger, unimplemented GEMV speedup and remains
an unmeasured projection.

## E065-E075 — packed GEMV metadata and bank-friendly shared tile (2026-11-10)

- **E065 L1 carveout / profiler access:** the `PreferL1` E058 sweep was inconclusive under
  variable GPU timing and is not integrated. Nsight Compute rejected hardware counters with
  `ERR_NVGPUCTRPERM`; Nsight Systems remains available for kernel timelines.
- **E066 packed activation metadata:** combine each activation block's scale bits and integer
  correction into one aligned `int2`. Across 11,652x5,120, 17,408x5,120, 5,120x17,408, and
  17,408x17,408 synthetic GEMVs, it was bit-exact and measured 1.10-1.21x vs separate scale/sum
  arrays. Integrated as `xmeta` in `real_decode.cu`. Three lossless full-path runs had median
  9.442 tok/s vs 9.337 for the prior binary; the gain is small (~1.1%). Raw shape results:
  `results/e066a-gemv-packed-meta-20261110.txt` and
  `results/e066-gemv-packed-meta-shapes-20261110.txt`. The three candidate runs are
  `results/e067-packed-meta-agent-g16-20261110.txt`, `results/e067b-packed-meta-agent-g16-20261110.txt`,
  and `results/e067c-packed-meta-agent-g16-20261110.txt`.
- **E068 shared-tile chunk sweep:** the earlier scalar shared layout remained slower (CB=64
  0.573 ms vs 0.508 ms baseline); bit-exact, not retained. Raw:
  `results/e068-gemv-smem-chunk-sweep-20261110.txt`.
- **E069 bank-friendly packed tile:** stage each 32-value activation block and its `{scale,sum}`
  metadata into a 40-byte shared record. Pair loads use 8-byte bank mode and an `int2` layout;
  16 output-row warps reuse one activation tile. CB=64 preserved the original lane accumulation
  order and was bit-exact. Microbench results: 0.388 vs 0.507 ms at 11,652x5,120; 0.180 vs
  0.223 ms at 5,120x5,120; 0.532 vs 0.710 ms at 17,408x5,120; 1.675 vs 2.415 ms at
  17,408x17,408; 0.329 vs 0.433 ms at 5,120x10,240. Raw E069a-d are in `results/`.
- **E070/E071 integration:** dispatch the shared tile for NB=16 and K>=5120; smaller K keeps the
  regular kernel. The first 5,120-only run was 9.573 tok/s; the generalized run was 9.629 tok/s,
  both exact-gated (32 compared, zero mismatches). A subsequent run under lower VRAM headroom was
  8.799 tok/s, so single-run throughput is variable. The earlier un-profiled control median was
  9.337 tok/s.
- **E072/E073 matched timeline:** under the same 32-token profile config, candidate/control mean
  cycle was 0.3534/0.3609 s and both exact gates passed. Nsight Systems counted 20,242 GEMV
  launches: summed GEMV time was 4.789 s vs 5.174 s (~7.5% lower); GEMV still used about 80% of
  kernel time. Delta-net time was essentially unchanged between arms. These profiler runs explain
  why the isolated 1.2-1.4x GEMV gains translate to only a few percent end-to-end. The timeline
  reports are `results/e072-wide-gemv-timeline-20261110.nsys-rep` and
  `results/e073-control-gemv-timeline-20261110.nsys-rep`; stdout is in matching `*-run-*` files.
- **E074 vector-warps plus shared tile:** output was bit-exact but took 0.983 ms vs 0.508 ms;
  rejected. **E075 chunk/row-tile sweep:** CB=64 was the best of 32/64/96/128 in the first run
  (0.389 ms); a later run under GPU activity was noisy and not faster. Row-tile 32 did not beat
  the CB=64/16-row configuration. The production dispatch remains CB=64, 16 rows/CTA.

Current source therefore retains the transposed delta-net layout, L2 activation window, packed
activation metadata, and exact-order shared GEMV tile. The requested 20–25 tok/s remains open;
E071 reached 9.629 tok/s in one run, and the full-path timeline still identifies GEMV as dominant.

## E076 — attribute draft acceptance and time MTP end to end (2026-11-10)

**Question.** The reported 34% “MTP acceptance” combined two drafter sources and timed only the
target verification pass. Separate MTP from PLD, measure the whole decode cycle, and check whether
high-confidence MTP can meet the requested 90% acceptance without changing greedy output.

**Measurement fixes and optimization.** `spec_summary` now reports MTP and PLD proposal counts and
accepted counts separately. Per-cycle timing starts before draft selection and ends after MTP state
sync and recursive drafting; the prior `tok_per_s` omitted that work. The hybrid path also skips
building a recursive MTP chain when the next history lookup is already certain to replace it with
PLD. The MTP KV/state sync still runs, so a later MTP draft starts from the correct history.

**Baseline attribution.** On the same 64-token agent trace, the default hybrid run reported 34.0%
combined acceptance, but source counters separated it into MTP **7/15 (46.7%)** and PLD **46/141
(32.6%)**. The small MTP denominator is why the combined statistic was a poor description of MTP
quality. Corrected end-to-end timing was 8.826 tok/s versus 9.287 verifier-only; after skipping an
unused MTP chain, MTP time fell from 33.9 to 16.9 ms/cycle in these runs and end-to-end throughput
rose from 8.518 to 8.826 tok/s. These are single-run comparisons under varying desktop GPU load.

**Acceptance-first setting.** A confidence sweep led to MTP-only with `--p-min 0.97`, γ=8. On the
128-token agent trace it accepted **61/63 MTP proposals (96.8%)** in two runs and matched all 128
tokens from the dense reference in both runs. End-to-end throughput was 5.770 and 5.937 tok/s;
dense was 3.593 tok/s on the same trace. The 0.98 run accepted 57/60 (95.0%) at 5.417 tok/s; 0.99
accepted 55/56 (98.2%) at 5.357 tok/s. A shorter 0.95 run was 27/32 (84.4%), so 0.97 leaves
measured margin over the 90% criterion while drafting more often. The default with `--mtp` is now
`p_min=0.97` and PLD disabled; `--pld 1 --p-min 0.3` restores the faster historical hybrid.

**Fast hybrid comparison.** With `--pld 1 --p-min 0.3`, γ=16, and the same 128-token trace, the
full output also matched dense exactly. It measured 7.397 end-to-end tok/s, with 30.8% combined
acceptance (MTP 35/79 = 44.3%; PLD 50/197 = 25.4%). This is the throughput-oriented option; it
does not meet the 90% acceptance goal on this trace.

**Current bottleneck and limits.** The high-confidence setting favors acceptance over speed. Its
mean verifier pass was 304 ms/cycle and its MTP work was 16.6 ms/cycle. The matched E072/E073
timeline still attributes about 80% of kernel time to `k_mat_q4`; the existing CB=64 activation
tile reduced summed GEMV time about 7.5%, but did not remove the main wall. The 90% result is
demonstrated on this fixed agent trace and model, not established across arbitrary prompts. Keep
the exact dense gate for every drafter or kernel change.

Raw evidence: `results/e076-source-split-agent-g16-20261110.txt`,
`results/e076-e2e-hybrid-agent-g16-20261110.txt`,
`results/e076-skip-wasted-mtp-agent-g16-20261110.txt`,
`results/e076-dense-agent-n128-20261110.txt`,
`results/e076-dense-agent-n128-ids-20261110.txt`,
`results/e076-default-mtp-agent-g8-n128-20261110.txt`,
`results/e076-mtp-highconf-agent-g8-p097-n128-20261110.txt`,
`results/e076-mtp-highconf-agent-g8-p098-n128-20261110.txt`,
`results/e076-mtp-highconf-agent-g8-n128-fullgate-20261110.txt`, and
`results/e076-hybrid-fast-agent-g16-n128-fullgate-20261110.txt`.

## E077 — Single-user throughput and balanced residency

The E076 confidence cutoff improved acceptance by withholding proposals. A fresh replay
with that default measured 6.10 tok/s. The user clarified that throughput is the objective,
with exact single-sequence output, and set a follow-up target of 40–60 tok/s.

- Fixed MTP context rows overwriting one another and sized GPU matrix dispatch by active
  positions. KV sync now computes only the projections it needs. The verifier commits an
  accepted prefix by replaying saved recurrent inputs after rejection. Reprocessing the
  prefix through the model would require another pass over cold weights.
- Byte comparison at load verified that the 715,161,600-byte output tensor in both GGUFs
  is identical. Target and draft share the GPU copy. The target still scores the entire
  vocabulary. Only recursive draft steps use a shortlist of 32,768 IDs and seen tokens.
- Batched KV writes, reused attention exponentials, and register-resident delta state
  shortened the verifier and state repair. MTP can extend a draft with a matching
  continuation from the actual request history when it agrees on the next token.
- With the same resident bytes, spreading weights across layers reduced mean verification
  time from 281.6 to 220.1 ms on the agent trace. Both variants ran 32 cycles with the same
  proposed and emitted tokens. Generation increased from 13.380 to 16.862 tok/s.
- The first spread-residency run hit the existing swap guard. A retry after host memory
  stabilized passed with the same guard settings.

| Normal launch, RTX 3060, one sequence | Generated tok/s | Including prefill | Exact match |
|---|---:|---:|---:|
| Agent trace | 16.438 | 11.537 | 128/128 |
| Rust merge prompt | 13.700 | 11.215 | 128/128 |

The timer starts after prefill for generation rate and includes all drafting, verification
and state repair. The first token is counted in the prompt-inclusive rate. The gate now
requires equal lengths as well as equal IDs. The binary installed at
`benchmarks/real_decode` was built from this source.

The alternate MTP branch was exact but improved the unbalanced-layout comparison only from
11.527 to 11.628 tok/s and added complexity. Reusing predictions from rejected target rows
reduced speed to 8.72–9.19 tok/s. Neither is in the launch. An exact packed INT8 Tensor
Core probe improved FFN down projection at width 8 from 0.377 to 0.236 ms, with little
change on the common K=5120 shapes. This is isolated kernel evidence.

Raw logs: `results/e077-{agent,coding,coding-dense,balanced-residency,front-loaded-residency,e076-baseline,packed-mma-probe}-20261110.txt`.
The Rust prompt and its dense IDs are in `results/e077-coding-{prompt,dense-ids}.txt`.

## E078–E085 — Draft depth, verification cost, and GPU memory

All generation runs here used one sequence and checked the full 128-token output
against the same dense target. Rates are generation after the first token and include
drafting, verification, and recurrent-state repair. Raw runs use E078–E085 prefixes.

| Experiment | Agent tok/s | Rust tok/s | Agent / Rust tokens per pass |
|---|---:|---:|---:|
| E077 native MTP | 16.438 | 13.700 | 3.969 / 3.256 |
| E078 DFlash 2 with native Q4_K, Q8-converted V/down, and host selectors | 12.782 | 15.639 | 3.629 / 4.233 |
| E079 linear EAGLE4 | 19.065 | 15.078 | 4.885 / 3.969 |
| E080 EAGLE4 with one alternate continuation | 20.366 | 17.119 | 5.080 / 4.379 |
| E082 GPU top-128 candidate selection for E080 | 20.701 | 17.458 | 5.080 / 4.379 |
| E083 Q8 draft weights and less resident target weight | 17.177 | 15.531 | 4.885 / 4.379 |
| E084 deferred recurrent writes, reduced scratch | 19.437 | 16.929 | 5.080 / 4.379 |
| E085 best-first EAGLE4, 16 positions | 20.531 | 18.764 | 5.080 / 4.704 |
| E085 best-first EAGLE4, 32 positions | 12.908 | 12.210 | 5.292 / 5.080 |

E080 is available via `--eagle4`. The draft has 308.6 MiB of device weights and keeps
its 682 MiB full output head in a file-backed host mapping. It gathers 128 draft rows
after scoring the low-rank head. The exact verifier still scores the target's full
vocabulary. Draft branches have separate attention ancestry and recurrent inputs.
Only the target-selected path becomes the next context.

At 8064 MiB of resident target weights, E080 took about 229 ms per verifier pass.
A 48-token, exact-gated Nsight Systems run measured 110.074 GB of streamed H2D
traffic at 24.66 GB/s during copies. Per pass, 5.242 GB at that rate accounts for
roughly 213 ms. CUDA kernels ran concurrently with the transfers. The summed
time in the 16-position Q4 shared-matrix kernel was 2.942 s across the 21 prompt
and generation passes. The profile is `results/e080-eagle4-timeline-20261111.nsys-rep`.

E081 was a bounded feasibility probe of AVX-VNNI on part of the FFN gate projection.
The smaller split took 1.061 GB out of the transfer stream. Agent generation reached
20.524 tok/s with all 128 IDs matching. The larger split moved 2.147 GB and reached
20.352 tok/s. The dependency waits absorbed most of the copy savings. Neither CPU
path is installed. E084 reclaimed 180 MiB in matched desktop conditions. Its verifier
time stayed within 1 ms of E080. E085's 32-position verifier took about 379 ms, making
the extra candidates too expensive with the original wide kernel.

## E086–E094 — Best-first drafting and prompt prefill

The installed E089 path uses up to 16 target verification positions from a
best-first EAGLE4 tree. It commits recurrent state after the verifier chooses
the accepted path. This removes the 150 MiB snapshot and avoids allocating two
full-vocabulary arrays as layer scratch. The CUDA 13 top-k path reduces draft
work by transferring only candidate IDs. The output projection runs on the final
prompt row. A wider, exact-order shared Q4 kernel lets prompt evaluation use
32 positions per pass while generation keeps its 16-position budget.

| E089, one sequence, 8064 MiB hot | Generated tok/s | Including prefill | Prefill s | Free VRAM MiB | Exact |
|---|---:|---:|---:|---:|---:|
| Agent trace | 20.340 | 14.615 | 2.514 | 1130 | 128/128 |
| Rust merge | 18.784 | 14.826 | 1.872 | 1132 | 128/128 |
| Science explanation | 15.467 | 12.285 | 2.208 | 1132 | 128/128 |

The plain-language prompt and dense IDs are `results/e080-explanation-{prompt,dense-ids}.txt`.
The native MTP path still matched the 128-token Rust reference and reached 13.983
tok/s. Qwen3-0.6B passed its CPU-reference argmax checks (17 positions, zero
mismatches). E086 compared each wide kernel output to the baseline float output
exactly. At 32 positions, the selected shared tile took 1.050 vs 1.483 ms on the
common K=5120, N=17408 shape. Packed INT8 MMA took 0.742 vs 1.013 ms on FFN down.

E091 tested a 24-position exact kernel and wider tree. Mean verification was
258–261 ms. End-to-end generation fell to 18.506 agent, 17.555 Rust, and 14.866
explanation tok/s despite an increase in tokens per pass. E092 added the native
MTP root candidate to EAGLE4 while reducing hot target residency by 256 MiB.
Results were 20.248 / 17.700 / 14.648 tok/s. E094 used the original BF16 draft
embeddings from a host mapping and recovered no verifier cycles on these prompts.
E089 remains installed.

E090 measured a 41.9 MB sample of the original Q4 weights. Planar ANS compressed
it to 92.4% of its size and decompressed byte-for-byte at about 128 GB/s in
isolation. The overlapped copy and decompression microbenchmark took 1.575 ms
versus 1.693 ms for raw copy. E093 integrated compression and a matrix kernel
that consumes the planar representation directly. It transferred 4.859 GB vs
5.242 GB per pass, yet mean verification rose from 230 to 252 ms because
decompression shared the verifier stream. It remained exact and slower
(18.787 vs 20.478 generated tok/s on the agent trace). Cache preparation
added 3.184 s before prompt evaluation.

## E095–E102 — Fewer transfer buffers and exact GPU argmax

E102 keeps the unchanged target and a 16-position draft tree. It computes
the full target vocabulary on the GPU and reduces each verifier row to its
greedy token before returning it to the CPU. Nonfinite and all-zero logits
fail the verifier. The smaller 16 MiB transfer buffers save 128 MiB of device
memory at the same 8064 MiB target residency. The normal launch was tested
without CPU affinity.

| E102, one sequence | Generated tok/s | Including prefill | Free VRAM MiB | Exact |
|---|---:|---:|---:|---:|
| Agent trace | 21.135 | 15.321 | 1280 | 128/128 |
| Rust merge | 19.334 | 15.701 | 1281 | 128/128 |
| Science explanation | 15.915 | 12.942 | 1281 | 128/128 |

E095 DSpine ran 18.563 / 15.450 / 12.123 tok/s at 7424 MiB hot with exact output.
E096 split planar ANS decompression into a separate stream. At 8064 MiB hot,
it measured 19.699 / 18.828 / 15.563 tok/s and passed the exact gate. Compression
still added startup preparation and consumed more VRAM. E098's separate copy
scheduler tied its matched single-thread scheduling control at 20.86 tok/s.
E099 measured about 24.9 GB/s with one, two, or four H2D streams. Write-combined
memory gave no gain.

E097 split exact INT8 Tensor Core work across groups of draft positions.
On FFN down projection at 32 positions, it took 0.640 ms against 1.012 ms
for the shared DP4A tile. E100 moved the 24-position verifier to about
229 ms from E091's 260 ms. With the wider tree, it reached 20.796 / 19.620 /
16.582 generated tok/s with exact output. Packing added startup work and
prefill was slower. E102 remains the installed launch.

## E103–E105 — Broader proposals on a separate 256-token prompt

E103 tested 64-position prefill with packed INT8 MMA. It reduced that path's
prefill time compared with its own 32-position packed run, but still had
startup packing and wider draft costs. E104 broadened the proposal list
from four to eight and calibrated proposal probabilities at temperature 1.2.
The target still uses its original full-vocabulary greedy choice. A sharper
proposal temperature and a separate history branch did not beat the broader
proposal policy across the measured prompts.

| Installed E105, one sequence | Generated tok/s | Including prefill | Free VRAM MiB | Exact |
|---|---:|---:|---:|---:|
| Agent, 128 tokens | 22.013 | 15.763 | 1250 | 128/128 |
| Rust, 128 tokens | 19.408 | 15.724 | 1252 | 128/128 |
| Explanation, 128 tokens | 16.929 | 13.634 | 1252 | 128/128 |
| Held-out event-log design, 256 tokens | 16.619 | 14.581 | 1124 | 256/256 |
| Same event-log design, 256 MiB less hot | 15.912 | 13.974 | 1383 | 256/256 |

The held-out prompt used context capacity 1024. At the same 8064 MiB hot
placement, E102 took 68 verifier passes and reached 15.427 tok/s. E105 took
63 passes with the same 256 target tokens. E105 reduced target residency
to 7808 MiB in the lower-VRAM run. This placement used `--cache-mib 6200`.

The documented launch is unpinned. The `--prompt` path uses the decoder's
existing approximate tokenizer. For a model-specific chat template or exact
external tokenization, provide the rendered prompt IDs via `--prompt-ids`.
The full token gates compare the speculative and dense paths on the same IDs.

## E106–E119 — Draft quality and real chat input

E106 trained a rank-16 draft-only residual on 15,859 verifier labels from
24 synthetic prompts. Six separate prompts provided 4,011 validation labels.
Root top-1 increased from 72.66% to 75.09%. In the runtime, E107 improved
the Rust prompt from 19.408 to 20.115 tok/s, but fell from 16.619 to 16.140
on the event-log design and from 16.929 to 16.409 on the explanation. All
three runs matched the dense reference. The adapter was kept experimental.

E108 ran a separately retrained Q4 native MTP head in linear mode. E109 added
best-first MTP trees and compacted the accepted target hidden states and KV.
It preserved exact output. The original MTP checkpoint is still available and
also benefited from branching. E112's Q8 draft improved some passes but lost
throughput on the explanation. See `results/e108-*`, `results/e109-*`, and
`results/e112-*` for individual measurements.

E110 and E119 use the target GGUF's actual chat template and llama.cpp token IDs.
Both reference outputs come from the dense Q4 runtime. E110's async-queue design
measured 14.716 generated tok/s on E105 at 8064 MiB hot and 15.047 at 8192 MiB.
E119's water-tower explanation reached 17.622 on the same E105 launcher at
8064 MiB hot. All 256 and 128 IDs, respectively, matched their references.

E113–E115 reuse exact quantized activations and fuse RMS/SiLU with quantization.
The component probe found no bit differences for the tested widths. On E110,
mean cycle time and total throughput were within measurement noise of the
control. E116 combined these kernels with packed integer matrix multiply and
24 verification positions. It used 66 verifier passes on E110 against 72 on
E105 and reached 15.366 tok/s with all 256 tokens matching.

E117 trained a rank-32 draft correction with the exact fourth-layer feature.
Validation root top-1 increased from 72.66% to 76.13%. E118 ran those four
target layers before drafting while leaving the full verifier authoritative.
The two independent chat runs took the same number of verifier passes as
E105 and reached 13.942 and 16.948 tok/s. Both were exact. The extra computation
and forced early-layer residency made it slower.

## E120–E124 — Proposal combinations and native MTP integration

E120 combined EAGLE4 and native MTP trees within the existing verification
width. It reduced E110's pass count from 72 to 67 and E119's from 30 to 27.
It reached 14.399 and 17.774 generated tok/s respectively. Extra drafting and
the smaller target residency needed for both heads outweighed the first gain.
E121 recycled exact verifier predictions from rejected sibling rows as draft
proposals. It reached 14.666 and 17.056 with exact output. E122 transferred
the first eight cold chunks during drafting using the existing GPU buffers.
On the matched E119 run, about 126 MB moved earlier per cycle, while throughput
was 17.495 against 17.611 in the control. The other pair ran under different
desktop memory pressure. E123 fit a 45-coefficient proposal confidence model.
Its held-out log loss improved from 1.00090 to 0.98732. The two chat runs took
more full verifier passes and slowed to 14.421 and 16.508 tok/s.

E124 shares the best-first tree builder between EAGLE4 and native MTP, including
history continuation when the drafter agrees with the first token. Native MTP
now defaults to this tree with the original checkpoint. `--mtp-tree 0` selects
the legacy linear MTP path. The EAGLE4 launch remains the same. All values below
are generated for one sequence. Request rates include prompt evaluation after
weights load. `Free VRAM` is the minimum observed during each run and also
reflects desktop allocations.

| E124 native MTP | Generated tok/s | Including prefill | Free VRAM MiB | Exact |
|---|---:|---:|---:|---:|
| Agent | 22.235 | 14.913 | 1000 | 128/128 |
| Rust | 17.497 | 14.090 | 1222 | 128/128 |
| Explanation | 16.954 | 13.166 | 1222 | 128/128 |
| Async-queue chat | 15.570 | 13.607 | 904 | 256/256 |
| Water-tower chat | 17.082 | 13.481 | 906 | 128/128 |
| Rust, linear MTP control | 14.229 | 11.895 | 1222 | 128/128 |

The EAGLE4 regression run took 24 passes and reached 21.941 generated tok/s
with all 128 agent IDs matching. The 40–60 generated tok/s target remains open.

## E125–E127 — Wider prefill and the installed single-user launcher

E125 moved native MTP candidate selection to the GPU. It saved 0.7 ms of
drafting per verifier pass on the async-queue prompt. Measured generation
stayed at 15.55 tok/s with or without GPU selection. The CPU selector stayed
in production. Native MTP's 32-position prompt evaluation reduced prefill
from 2.437 s to about 1.95–2.00 s on the same prompt. E127 installs the wider
prompt path when the MTP and target output weights are verified identical.

E126 generated EAGLE4 proposals at each position of an exact dense stream.
The target first reproduced all reference IDs with the forced prefix. Each
proposal was then scored against the dense continuation. This is a diagnostic
of proposal coverage. Its pass counts are derived from the trees and exclude
wall-time costs. On async-queue chat, widths 16, 24, 32 and 64 would need
72, 66, 64 and 63 passes respectively. The target's next token was missing
from the first level at 39 of 255 positions even with width 64. The wider
runtime E116 lost throughput on the agent, Rust and event-log prompts.

E127 sets the local launcher to the original native MTP checkpoint by default
at 8064 MiB of resident target weights. It passed the full token check on six
prompts at context capacity 1024. EAGLE4 is available with
`GUM_DRAFTER=eagle4` and remains faster on the measured Rust and water-tower
prompts. The native default has the higher minimum measured generation rate
across this suite and requires the existing native checkpoint.

| E127 native MTP at 8064 MiB hot | Generated tok/s | Including prefill | Free VRAM MiB | Exact |
|---|---:|---:|---:|---:|
| Agent | 21.367 | 15.331 | 1023 | 128/128 |
| Rust | 17.122 | 14.277 | 1030 | 128/128 |
| Explanation | 16.596 | 13.427 | 1031 | 128/128 |
| Event-log design | 17.134 | 14.967 | 1034 | 256/256 |
| Async-queue chat, installed launcher | 15.236 | 13.700 | 1023 | 256/256 |
| Water-tower chat | 16.609 | 13.771 | 989 | 128/128 |

The installed EAGLE4 option reached 17.591 tok/s and 128/128 on water-tower
chat. Raising native target residency to 8192 MiB reached 15.492 tok/s on
async-queue chat, with 902 MiB free in that run. The 0.6B CPU reference test
checked 17 positions with zero argmax mismatches. All 202 resident chunks
passed the upload check.

## E128–E130 — Where the draft trees lose the target continuation

E128 repeated the E126 coverage probe with the original MTP head. On async-queue
chat, native MTP missed the next token at 26 of 255 prefixes, EAGLE4 at 39, and
both missed at 17. Choosing the longer exact prefix of two independently built
16-position trees would take 60 passes, compared with 68 for native MTP and 72
for EAGLE4 individually. That calculation uses both full candidate budgets and
the known reference. It measures complementary coverage. E120 measures the
actual cost of combining the drafters within one budget.

E129 rescored EAGLE4's shortlisted tokens using its original BF16 output
weights while leaving the target and the rest of the draft unchanged. Both
chat runs matched dense. They slowed to 13.862 and 16.153 generated tok/s, and
each took more verifier passes than E105.

E130 supplied the true target features at matching draft nodes in the forced
prefix diagnostic. At width 16, native MTP's async-queue pass count changed
from 68 to 61 and water-tower from 31 to 22. EAGLE4 changed from 72 to 60 and
29 to 23. These are diagnostic pass counts with future features supplied by
the verified reference. The feature capture in E132 is intended to test
whether a learned correction can recover any of this gain in real decoding.

## E131–E133 — Partial CPU verification and draft-state regression

E131 moved one chunk of each target FFN gate projection to eight AVX-VNNI
workers, with activation and result transfers on a separate CUDA stream.
The CPU component matched all 9,288 test outputs bit-for-bit against the
scalar reduction order. On async-queue chat, both placements removed about
1.07 GB of streamed weights per verifier pass and matched all 256 dense IDs.
They reached 14.895 and 14.849 generated tok/s versus 15.166 in the paired
control. Prefill improved from 1.975 s to 1.713–1.733 s. Generation stayed
slower, so the installed verifier remains on the GPU.

E132 captured 9,480 native MTP feature forecasts on 30 synthetic prompts
rendered with the actual chat template. The target reproduced all reference
tokens during collection. Six prompts were held out. The rank-32 adapter
raised held-out cosine similarity to the true target features from 0.655 to
0.725 at the first draft step and from 0.487 to 0.632 at the fourth step.
The diagonal adapter also improved feature similarity. Neither correction
improved generation in E133. A 30% blend reached 14.773 and 15.745 on the two
independent chat prompts, with all IDs matching. Full-strength and diagonal
corrections were slower. E135 uses the next-token training loss through a
frozen copy of the native draft block.

## E135–E140 — Grouped native draft branches

E135 trained the draft correction against next-token labels through a frozen
MTP block. Its training implementation reproduced native CUDA root features
at 0.999967 cosine similarity on 48 samples. On separate validation prompts,
the best loss was 1.79335 versus 1.96388 without the correction. The real
decoder slowed on both chat prompts and the event-log design, with exact
outputs. E136 trained two recursive steps. It did not improve the validation
loss over its warm start. E137's native probability cutoff sweep gave no broad
gain over 0.06.

E138 evaluates up to four ready native MTP branches together so they reuse
draft weights. The target still verifies one sequence's tree. On async-queue
chat, drafting fell from 21.1 ms to 9.7 ms per cycle. The 16-position tree
used the same 68 verifier passes and reached 15.927 generated tok/s. Moving
straight to a 32-position verifier cut the pass count to 58 but slowed to
12.239 tok/s with the original verifier matrix kernels.

E140 installs a shared tree builder and grouped native MTP with the same
16-position target verifier. `--mtp-group 1` evaluates one draft branch at a
time. All six prompt outputs matched the dense reference. The grouped path
improved five of the six measured generation rates. The explanation used two
more passes and went from 16.596 to 16.317 tok/s. The warm EAGLE4 regression
pair matched at 21.951 and 21.910 tok/s. Linear native MTP also passed.

The installed launcher measured 15.954 on async-queue chat and 23.382 on the
agent trace. Reducing the hot target budget by 256 MiB gave 15.164 on the same
chat with 1530 MiB of free GPU memory. The launch with 8064 MiB hot had 1274 MiB
free in that pair.

## E139–E143 — Wider integer verification and launch overhead

E139 combines grouped native MTP with 24-position integer Tensor Core
verification. It reached 16.630 tok/s on async-queue chat. A 32-position packed
verifier still cost too much on that prompt. E141 moves the hot-weight layout
conversion to the GPU using an existing transfer slot. Its paired hot load
took 1.194 s instead of 1.731 s. All 535 resident chunks passed the byte check.

E142 cleans up the packed path and exposes it as `--packed-verify 1`. The
24-position option spends more work per pass and usually needs fewer passes.
The table uses single-sequence generation and includes process wall time from
the measured launch. Every emitted ID matched dense.

| E142 packed native MTP | Generated tok/s | Including prefill | Process wall s | Free VRAM MiB | Exact |
|---|---:|---:|---:|---:|---:|
| Agent | 23.579 | 16.208 | 10.96 | 1228 | 128/128 |
| Rust | 18.254 | 14.566 | 11.87 | 1226 | 128/128 |
| Explanation | 17.189 | 13.485 | 12.56 | 1228 | 128/128 |
| Event-log design | 18.240 | 15.655 | 19.42 | 1226 | 256/256 |
| Async-queue chat | 16.611 | 14.659 | 20.56 | 1228 | 256/256 |
| Water-tower chat | 17.397 | 13.804 | 12.34 | 1228 | 128/128 |
| Async-queue chat, 256 MiB less hot | 16.084 | 14.227 | 21.15 | 1484 | 256/256 |

On async-queue chat, the paired 16-position path reached 15.949 generated
tok/s and took 20.96 s of process wall time. The packed mode ran slightly
longer on some of the shorter prompts because of startup and prefill costs.
E143 augmented the fixed native draft shortlist from the first proposal's
full vocabulary. It reduced one pass on water-tower chat and made no material
change on async-queue chat.

## E144–E149 — Packed verification in the single-user launcher

E144 replaced the native MTP cache of target hidden states with the last
committed state. Each prompt chunk or accepted path updates the MTP KV cache
before its target rows are reused. At context capacity 4096 this freed 80 MiB
in a paired exact run (1018 to 1098 MiB free). Both runs used 62 verifier
passes and matched all 256 tokens. E145 halved the transfer slot ring and
lost throughput. The launcher retains eight slots.

The E146 Nsight trace captured 20 generation passes. Verification averaged
232.3 ms per pass. Weight transfers occupied 214.6 ms at 24.42 GB/s, and GPU
kernels occupied 198.7 ms with overlap. The verifier transferred 5.242 GB
per pass. About 10.1 ms elapsed after the final weight transfer on an average
pass. Drafting occupied 10.8 ms in the corresponding CPU ranges.

The first E147 run of the recovered candidate matched the six references.
It measured 14.88 tok/s on async-queue chat, alongside 14.91 on the compact
control. The packed agent result also varied substantially under desktop
GPU load. E148 allowed 32 positions for a strong history match. One
such match was used on each of the agent and async-queue runs, and neither
run reduced verifier pass count. A separate CPU-only diagnostic indexed 4,800
tokens from 30 independent E132 completions. It found no four-token exact
retrieval at any prefix of the two chat references. That result applies to
this small corpus.

E149 distributes writes of partial sums across shared-memory banks in the
packed integer verifier while keeping the floating-point reduction order.
The paired kernel test compared 1,802,240 output values over the two FFN
matrix shapes and widths 8, 16, 24 and 32. All values matched bit-for-bit.
At width 24 the two shapes took 0.573 vs 0.782 ms and 0.458 vs 0.636 ms in
that test. In a full 256-token async-queue run, the same 62 verifier passes
averaged 226.7 vs 241.7 ms against E147. Generation was 16.994 vs 16.008 tok/s.
The installed-at-start E140 compact control reached 15.669 on the same
reference in that series.

| Single-user workload | E127 generated tok/s | E149 generated tok/s | E149 including prefill | E149 process wall s | E149 exact |
|---|---:|---:|---:|---:|---:|
| Agent trace | 21.367 | 23.868 | 16.712 | 10.83 | 128/128 |
| Rust merge | 17.122 | 18.578 | 14.821 | 11.79 | 128/128 |
| Explanation | 16.596 | 17.446 | 13.775 | 12.47 | 128/128 |
| Event-log design | 17.134 | 18.570 | 16.016 | 19.17 | 256/256 |
| Async-queue chat | 15.201 | 16.994 | 15.022 | 20.32 | 256/256 |
| Water-tower chat | 16.609 | 17.586 | 14.092 | 12.30 | 128/128 |

The suite used the launcher at context capacity 1024 and 8064 MiB hot.
Minimum free VRAM across the six runs was 1032 MiB. The low-hot async-queue
run reached 16.207 tok/s with 1290 MiB free and 256/256 matching. The EAGLE4
and linear MTP launcher checks matched all 128 tokens each. The 0.6B CPU
reference check passed 19 positions with zero counted argmax mismatches
and one near tie. All 202 hot weight chunks passed the upload check. The
installed packed launcher also passed all 535 hot chunk checks and matched
32/32 with a smaller RAM cache that streamed 0.20 GiB per pass from NVMe.

The normal native MTP launcher enables the 24-position path when the GPU
and compiled kernel support it. `--packed-verify 0` selects the 16-position
path. The installed launcher repeated async-queue at 16.971 generated and
15.005 including prefill, with 256/256 exact. The installed 16-position
override ran at 15.926 generated and also matched all 256 tokens.
The 40–60 tok/s target remains open. At the measured 242 ms per cycle
on async-queue, 40 tok/s would require about 9.7 generated tokens per pass,
versus 4.11 in E149. Raw evidence is in `results/e144-*` through
`results/e149-*`, including the E146 trace and the E149 kernel parity log.

## E150–E151 — Fixed and adaptive 32-position verification

E150 dispatches the 32-position packed multiply in four groups of eight
on both FFN shapes. At 32 positions, the async-queue verifier took 244.8
vs 249.3 ms per pass against E149's original 32-position dispatch. Both
needed 58 passes and matched all 256 IDs. E150 reached 16.712 tok/s versus
16.427 for the 32-position control and 16.971 in the 24-position launcher.
Water-tower chat reached 17.051 versus 17.586 at width 24, with exact IDs.

E151 starts at 24 positions and widens when the best queued proposal
score implies an upper bound on extra draft utility of at least 8.5%.
It widened 8 of 61 passes on async-queue, 0 of 30 on water-tower, and
4 of 22 on the agent trace. All outputs matched. Generation was
16.898 / 17.437 / 23.343 tok/s, below the measured 24-position default
on each prompt. The launcher stays at 24. See `results/e150-*` and
`results/e151-*`.

## E152 — Perfect drafts through the packed verifier

The diagnostic substitutes the next dense-reference IDs for the proposals,
while retaining the normal 24-position target verifier, full-vocabulary
greedy comparison, KV writes and recurrent state commit. The native MTP
allocation and shared output stay in place to preserve the production
weight placement. MTP proposal and cache update work is excluded.

The test used the async-queue and water-tower chat references at context
1024 and 8064 MiB hot. The table pools complete, full-width passes with a
state commit. The target emits one extra greedy token per successful
pass. The final pass can use fewer rows and skips a state commit.

| Draft tokens | Passes | Verify ms | Commit ms | Full cycle ms | Perfect-draft tok/s | Draft budget at 40 tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 54 | 220.46 | 2.65 | 223.21 | 31.36 | −48.21 ms |
| 8 | 42 | 222.69 | 3.13 | 225.94 | 39.83 | −0.94 ms |
| 10 | 34 | 222.02 | 3.24 | 225.37 | 48.81 | 49.63 ms |
| 12 | 28 | 225.35 | 3.93 | 229.43 | 56.66 | 95.57 ms |
| 15 | 22 | 224.98 | 4.33 | 229.46 | 69.73 | 170.54 ms |

All ten runs matched the dense references (1,920 token comparisons).
Every pass transferred 5,241,526,272 bytes. Full-cycle timing includes
embeddings, target token selection and commit. Draft budget is
`25 ms × (draft tokens + 1) − cycle time`, conditional on full acceptance.
A negative control changed one proposal ID. The verifier rejected it at
generated position 2 and exited 1. Raw measurements are in `results/e152-*`.

```bash
nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/real_decode.cu -o /tmp/vanta-decode
target="$HOME/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"
mtp="$HOME/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf"
fixture=e110-chat
reference="results/$fixture-dense-ids.txt"
/tmp/vanta-decode --model "$target" --mtp "$mtp" --packed-verify 1 --speculate 24 \
  --ctx 1024 --hot-mib 8064 --cache-mib 5600 --pld 0 --show-text 0 \
  --prompt-ids "$(cat "results/$fixture-prompt-ids.txt")" --gen-tokens "$(wc -l < "$reference")" \
  --draft-max 10 --oracle-ids "$reference" --expect-ids "$reference"
```

## E153 — Lightweight parallel Q4 drafter

Two compact learned heads propose ten tokens in one forward pass from the
previous target hidden state and the current token's exact Q4 embedding.
The independent variant predicts all positions in parallel. The adjacent
variant also feeds each position's intermediate features to its successor
in a residual layer. These are small prototypes inspired by
[DFlash](https://arxiv.org/abs/2602.06036) and
[DSpine](https://arxiv.org/abs/2609.36173).

The training labels and hidden states come from the Q4-verified E132
captures. Training used 3,600 anchors from 24 prompts. Model selection
used 900 anchors from six other prompts, maximizing the exact prefix
before the first wrong token. The 1,237-ID training vocabulary covered
58.39% of validation labels. The independent head averaged 0.448
matching tokens before its first error, with 21.67% correct at position
one. The adjacent head averaged 0.463 with 22.89% at position one.
Neither head matched all ten positions at any validation anchor.

Train or replay the experimental heads with the same target:

```bash
PYTHONPATH="$HOME/llama.cpp/gguf-py" python3 scripts/train_parallel_draft.py \
  --captures /tmp/vanta-e132 --model "$target" --output /tmp/e153-models
fixture=e153-altitude
/tmp/vanta-decode --model "$target" --parallel-draft models/e153-parallel-adjacent.vpd \
  --packed-verify 1 --speculate 24 --draft-max 10 --ctx 1024 --hot-mib 8272 --cache-mib 5600 \
  --prompt-ids "$(cat "results/$fixture-prompt-ids.txt")" --gen-tokens 64 --show-text 0 \
  --expect-ids "results/$fixture-dense-ids.txt"
```

The live controls use the installed native MTP policy with 8064 MiB hot.
The parallel heads use 8272 MiB hot to put the draft-memory savings into
exact target residency, keeping GPU headroom close to native MTP.

| Fresh prompt | Drafter | Mean accepted prefix | Draft + draft state ms | Generated tok/s | Free VRAM MiB |
|---|---|---:|---:|---:|---:|
| Altitude explanation | Native MTP | 3.200 | 10.6 | 17.728 | 996 |
| Altitude explanation | Independent | 0.086 | 0.3 | 5.060 | 956 |
| Altitude explanation | Adjacent | 0.068 | 0.4 | 4.967 | 966 |
| Go stream merge | Native MTP | 2.938 | 10.7 | 16.621 | 997 |
| Go stream merge | Independent | 0.086 | 0.2 | 5.086 | 1009 |
| Go stream merge | Adjacent | 0.068 | 0.2 | 4.999 | 976 |

All six live runs matched 64/64 dense IDs. The native control includes its
history proposals. Prefix means use each decoder's real cycle boundaries.
The CUDA heads also reproduced 120/120 PyTorch proposal IDs per variant on
12 saved Q4-state anchors. Both heads miss the acceptance and speed gate.
The two `.vpd` checkpoints are about 17 MiB each. Training, live measurements
and dense references are in `results/e153-*`.

## E154 — Extra exact residency and fused scale decoding

The target output matrix was resident directly, using its original Q4_0
bytes. MTP was absent and the context capacity was 384. This freed enough
space to place 8824 MiB of the target on the GPU under the existing
allocation guard. The 8872 MiB request at context 1024 and the 8832 MiB
request at context 384 were rejected by that guard.

`--compress-packed 1` keeps the four-bit weight payloads unchanged and
encodes each group of 32 packed tiles' FP16 scales with low bytes, sign
bits and four-bit deltas from a common high byte. Groups with a wider
range keep their raw representation. The MMA kernel reconstructs each
scale in registers as it uses the tile. The CPU checked every encoded
group against the original packed bytes before using it. The CUDA parity
test matched all 12,288 float outputs bit-for-bit at widths 8, 12, 16, 24
and 32 on both common FFN shapes, including mixed raw and encoded groups.

| Prompt | Hot MiB | Codec | H2D GB/pass | Verify ms | Commit ms | Cycle ms | Prefill s | Free VRAM MiB |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| Async queue | 8064 | Off | 5.242 | 223.55 | 3.92 | 227.61 | 2.139 | 1272 |
| Async queue | 8064 | On | 5.154 | 217.78 | 3.27 | 221.17 | 2.681 | 1313 |
| Async queue | 8824 | Off | 4.445 | 187.10 | 3.13 | 190.62 | 1.832 | 556 |
| Async queue | 8824 | On | 4.370 | 184.22 | 3.07 | 187.40 | 2.489 | 556 |
| Water tower | 8824 | Off | 4.445 | 188.65 | 3.74 | 192.52 | 1.511 | 443 |
| Water tower | 8824 | On | 4.370 | 185.23 | 3.42 | 189.11 | 2.175 | 552 |

This table uses complete ten-proposal passes with a state commit. Async
queue has 23 timed passes per arm and water tower has 11. Every run
matched its full dense reference. On async queue, a separate seven-proposal
run at 4.370 GB averaged 187.83 ms over 31 complete passes, leaving
12.17 ms per pass for a drafter if it can consistently supply seven
exact proposals and the verifier supplies the eighth token. This is a
perfect-draft measurement with reduced context capacity.

Reproduce the high-residency configuration from E152 by removing
`--mtp "$mtp"` and using `--resident-output 1 --ctx 384 --hot-mib 8824
--compress-packed 1`. Use `--compress-packed 0` for the matched raw
control. To rerun matrix parity, build `tests/packed_scale_codec.cu` with
the same NVCC flags used for the decoder. Raw results are in `results/e154-*`.

## E155 — Native MTP with higher residency and the fused codec

The three async-queue runs used the normal grouped native MTP policy,
packed 24-position verification, context capacity 384 and the same 256-token
dense reference. Each used 62 verifier passes, emitted 4.113 tokens per
decode cycle and matched every dense ID.

| Hot MiB | Codec | H2D GB/pass | Verify ms | Cycle ms | Generated tok/s | Including prefill | Prefill s | Free VRAM MiB |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 8064 | Off | 5.242 | 227.8 | 243.7 | 16.874 | 14.891 | 2.079 | 1044 |
| 8448 | Off | 4.839 | 213.7 | 229.6 | 17.915 | 15.818 | 1.950 | 671 |
| 8448 | On | 4.758 | 212.8 | 228.9 | 17.968 | 15.175 | 2.678 | 671 |

Extra exact residency improved both generation and request time on this
prompt and used 373 MiB more measured VRAM. At 8448 MiB the codec saved
81 MB per pass. Its 0.9 ms verifier gain was small relative to the 0.728 s
prefill increase on this 256-token workload. Keep it opt-in pending a
faster cache-preparation path and a larger matched workload suite.
See `results/e155-*`.

## E156 — Weight traffic audit and the bytes-per-token ledger

The E146 Nsight trace (`results/e146-packed-native-20261111.nsys-rep`) was
re-audited by reconstructing the Q4_0 chunk layout from the target GGUF and
matching every H2D weight-copy size, in schedule order, to the
target-verification NVTX ranges. Every copy size matched exactly, so the
per-tensor split below is measured, not estimated. Raw evidence:
`results/weight-traffic-audit-20261111.txt`.

Traffic composition per verifier pass (5.242 GB H2D): 99.98% is Q4 model
weights. The feed-forward layers alone carry 3.33 GB (ffn_up 22.7%,
ffn_gate 20.5%, ffn_down 20.4%), ssm_out 14.2%, attn_qkv 14.0%, attn_q 5.7%.
Each Q4 block is 32 four-bit weights plus a two-byte scale; payload is
4.66 GB nibbles + 0.58 GB scales per pass.

**Bytes per exact output token ledger** (the metric to track alongside
tok/s; H2D weight bytes ÷ new tokens):

| Run | Cycles | New tokens | Cold GB | GB/token |
|---|---:|---:|---:|---:|
| E149 installed chat | 62 | 255 | 325.0 | 1.274 |
| E149 agent trace | 22 | 127 | 115.3 | 0.908 |
| E150 32-position chat | 58 | 255 | 304.0 | 1.192 |
| E152 oracle d10 chat | 24 | 255 | 125.8 | 0.493 |
| E152 oracle d15 chat | 16 | 255 | 83.9 | 0.329 |

The E152 oracle rows substitute the known dense continuation, so they isolate
what a drafter that reliably supplied 10 (or 15) exact proposals would save:
61% (or 74%) of traffic, and measured 47.4 tok/s (or the d15 equivalent) with
all 256 IDs exact. They demonstrate headroom; they do not predict a live
drafter.

**Compression sample:** separating Q4 nibbles from scales and compressing
with zstd-3 recovered 9.0–9.8% planar / 5.4–5.7% wire on whole chunks from
both tiers (9.8% separate-planes on the slice sample), byte-exact on every
zstd round trip. Nibble entropy is 3.747 bits — little headroom beyond
planar reordering. No GPU decompression timing exists; the forecast of
roughly 17–19 tok/s from bytes alone is an upper bound until an integrated
unpack cost is measured.

**Sparsity probe (negative result):** an instrumented temporary E149 build
checked every 32-value Q4 activation group during generation for a zero
scale across all live verifier rows: 2.16M ffn_down, 1.27M ffn_up_gate and
2.93M attention/recurrent groups on chat (0.77M/0.45M/1.04M on agent) —
zero eligible payload bytes in both runs. Skipping zero blocks recovers
nothing on this workload. Both probe runs still matched dense exactly
(256/256, 128/128).

Decision: longer correct drafts are the first lever (the ledger makes the
drafter's value concrete: every extra exact proposal is worth ~19 ms of
PCIe at the current cycle); lossless compression of the FFN tensors is the
second. Track GB/token alongside tok/s from now on.

> **CORRECTED in E161 (2026-11-11):** the compression sample above is an
> artifact. Re-measured on real tensor boundaries (the GGUF reader's
> actual `blk.*` tensors, 2 MiB planes): zstd-3 ≈ zstd-19 ≈ xz ≈ **0.943–
> 0.952 on both scale and nibble planes**; scale-plane order-0 entropy is
> 7.53 bits/byte and the nibble plane sits at its 3.747-bit/nibble bound.
> The 0.902 planar / 0.56 scale ratios and the "16% ceiling" quoted below
> were wrong; the measured whole-model lossless ceiling is **~5.5%**, and
> the payload is incompressible beyond it. See E161.

### E156a — 8448 MiB residency promotion gate on the second chat

E155's 8448 MiB raw configuration was evaluated on the water-tower chat
(`e119-chat`, context capacity 384, packed 24-position verifier, native MTP)
against a matched 8064 MiB control. The desktop shares the GPU, so each arm
ran twice; single runs on this prompt differ by up to 1.7 tok/s.

| Hot MiB | Generated tok/s (two runs) | Verify ms | GB/pass | Exact IDs |
|---:|:---:|---:|---:|:---:|
| 8064 | 17.125, 16.625 | 231.7 / 237.1 | 5.242 | 128/128 both |
| 8448 | 16.519 (noise-depressed), 18.362 | 237.9 / 214.7 | 4.839 | 128/128 both |

The mean favors 8448 by ~0.5 tok/s and the cleanest run of each arm favors
it by +1.2, consistent in direction with E155's +1.04 on async-queue. The
same tree decisions were made in every run (mean 4.233 emitted/cycle, PLD
25/27 accepted), so the difference is pure transfer time. Promotion is
limited by allocation: 8448 MiB only fits at context capacity 384
(prompt + generation ≤ 384), while the launcher default keeps context 1024.
Keep `--hot-mib 8448 --ctx 384` as a measured short-sequence option; do not
replace the launcher default until the context trade-off is decided with the
owner. Raw: `results/e156-e119-chat-hot{8064,8448}-*`.

## E157 — Full-vocabulary tied-head parallel drafter (negative result)

E153's heads could only emit the 1,237 IDs seen in training labels, which
covered 58.39% of validation labels — a candidate hard cap on position-one
accuracy. E157 replaced the free head with `logit_t = ℓ·(E_t·P)`: the head
weights are derived from the target's exact dequantized Q4 embedding matrix
(248,320 × 5120, frozen) times a trainable 5120×96 projection P, so every
token in the vocabulary receives a structured logit even when absent from
training. Stem, forecast and adjacent layers are unchanged from E153; the
same 30 E132 captures and 24/6 split were used. The CUDA parallel-draft
loader now accepts vocabulary counts up to the target vocabulary
(`benchmarks/parallel_draft.cuh`), and the 95 MiB `.vpd` files carry the
full tied head plus zero biases.

Offline held-out exact-prefix (higher is better; native MTP live prefix is
about 3.1–3.2 on these prompts):

| Variant | Best prefix | Position-1 accuracy | E153 restricted-vocab |
|---|---:|---:|---:|
| Independent | 0.424 (epoch 8) | 21.4% | 0.448 / 21.67% |
| Adjacent | 0.426 (epoch 13) | 21.7% | 0.463 / 22.89% |

Both variants match E153's accuracy almost exactly — slightly worse, within
noise. The 58.39% vocabulary coverage was **not** the binding constraint;
the 384→96 MLP bottleneck is. A one-pass drafter of this family cannot
outperform its own function class, and more training data or a larger
vocabulary will not change that.

A live smoke run with the adjacent head (altitude prompt, 64 tokens,
`--parallel-draft /tmp/e157-models/adjacent.vpd`) loaded the full-vocabulary
head correctly (`parallel_draft_plan,vocab=248320`), matched all 64 dense
IDs, and confirmed the offline verdict live: mean 1.068 emitted tokens per
cycle, draft acceptance 0.007. Live integration was therefore not pursued;
native MTP remains the installed drafter.

Decision: the parallel one-pass MLP drafter family is exhausted as a path to
longer drafts. Reaching the 40 tok/s bar at 8448 MiB (≈ 6.5–7.5 exact
proposals per cycle at the measured 187–229 ms cycles) requires a
drafter with target-like per-step accuracy — a trained sequential draft
model (DFlash/EAGLE-class), which is a data- and training-infrastructure
investment — or the cycle-time levers (residency, compression). The two
E157 checkpoints remain in `/tmp/e157-models`; the loader change is
retained as it is backward-compatible. Raw: `results/e157-train-20261111.txt`,
`results/e157-smoke-altitude-20261111.txt`.

## E158 — Trained sequential drafter and the transition-structure probes (negative)

The owner directive was to pursue 40 tok/s by any means. Three independent
probes now bound what a drafter can do on novel chat with the data we can
produce:

1. **Sequential self-conditional transformer** (`scripts/train_sequential_draft.py`):
   d=2048, 3 blocks (~150M params), rollout attention over its own prefix,
   residual hidden-state transition, and — the key correction versus E157 —
   the head is the **target's exact `output.weight` + `output_norm`** (fp16,
   frozen; the GGUF confirms the output head is separate from the embedding,
   which retroactively explains E157's flatline). Teacher chains from the
   E132 captures cover *every* generated position (H[p-1], emb(ref[p])) → H[p]
   — 3,456 train chains × 16 depths = 55k supervised transitions. An
   untrained model with the exact head echoes its input with 100% accuracy
   (head decode validated). After 4 teacher epochs: train loss 0.10
   (memorized), **held-out depth-1 survival 5.9–8.9%**. The rollout/scheduled-
   sampling phase OOM'd at batch 32 and was not reached — the generalization
   failure at depth 1 is already decisive: 55k transitions cannot teach the
   64-layer hybrid's dynamics to a from-scratch drafter. The published
   drafters that reach ~4–6 accepted tokens (EAGLE-class) trained on ~10⁶×
   more target-distribution tokens.

2. **Native MTP chained survival, measured from captures** (val prompts):
   marginal per depth 0.725 / 0.629 / 0.540 / 0.519 — the head does not
   collapse; it decays slowly. The live tree converts this to 4.2 emitted
   via best-first fanout. A drafter that beats the tree needs per-depth
   marginals ≈ 0.85+ sustained to depth 8 (product ≈ 0.35 at depth 8,
   mean ≈ 6–7 exact).

3. **Linear transition probe**: ridge (8k samples) predicting
   H[p]−H[p−1] from [emb, H] and rolling out recursively on val: survival
   ≈ 1%. The delta is high-rank (rank-256 energy 0.68) with ‖Δ‖ ≈ 0.93·‖H‖ —
   the hidden state rotates almost fully every token. No shortcut basis.

Consequence: on novel chat, **the native MTP head is the best drafter
available at our data scale**, and the emitted wall (~4.2/cycle) is not a
tuning artifact. The 40 tok/s chat target requires either (a) target-scale
drafter training data — capture throughput is the binding constraint (dense
capture ≈ 2.5 transitions/s; batched-capture via the multi-sequence decode
path would be ~12–30× and is the prerequisite infrastructure for any
further drafter attempt), or (b) accepting that 40 tok/s arrives on
agent-shaped workloads (PLD-exploitable spans), where E149 already measures
23.9 tok/s and the ceiling with today's drafter is ~27. Raw:
`results/e158-train-20261111.txt`, `/tmp/e158-models/seq_draft.pt`.

## E159 — Batched capture infrastructure and the HalfDraft exit-head probe (negative)

The owner reframed the data wall ("data isn't really an issue — walk around
it, and if we use our own model that isn't an issue"). Two responses were
built and measured in one session.

**Batched capture (positive, durable infrastructure).** The E132-era capture
source was instrumented (`/tmp/vanta-e158/real_decode.cu`,
`VANTA_MID_CAPTURE`/`VANTA_MID_CAPTURE_FILE`) to dump per-position raw
residuals at layers 8/16/24/32/40/48/56 plus the final pre-norm residual,
for every lockstep sequence of the `--batch 8` dense path. 30 prompts × 160
positions × 8 hidden states now capture in **four ~100 s runs (≈ 12.8
positions/s aggregate, 5× the dense single-stream rate and 6× faster
wall-clock than sequential runs)**. Validation: decoding each captured final
residual through the target's exact `output.weight` reproduces the run's own
next token with 99.4–100% agreement on all 30 prompts, so the capture is
self-consistent and correctly aligned. (Side note: the E132-era *speculative*
reference for sample-00 diverges from dense at generated position 74 — the
current E149-era packed verifier remains exact-gated against dense, but the
old reference files were evidently produced by the spec path.)

**HalfDraft feasibility (negative).** The proposal: restructure residency so
the first K layers are fully resident and exact, draft with the target's own
early layers (compute hidden under the transfer window), and decode through
a small trained exit head + the shared exact output head. Cold bytes per
pass are unchanged by composition (cold = model − hot), so the only new cost
is exit-head quality. Feasible residency (hot 8448 MiB minus output head,
states, draft KV) reaches approximately layer 35. Measured val argmax
agreement through the exact head (954 held-out positions):

| Exit depth | Zero-shot | Ridge map | Trained MLP |
|---:|---:|---:|---:|
| 32 | 6.2% | 15.5% | 24.6% (33.2% with token embedding) |
| 40 | 10.1% | 17.7% | 30.9% (35.3% with embedding) |
| 48 | 12.4% | 18.6% | 37.2% (40.4%) |
| 56 (not resident-feasible) | 31.4% | — | 59.4–59.9% |

Gate was ≥ 0.8 at a feasible depth. Best feasible is 35.3% — far below the
native MTP head's 73% step-1 from the true final hidden. The mapping from a
mid-depth residual to the target's argmax *is* the remaining layers'
nonlinear compute (dominated by delta-net blocks with SSM state); no small
head learns it from ~4k curated positions. This is the same wall as E158,
now measured from the other side.

**Combined conclusion after E153–E159 (five independent probes):** a drafter
that beats the authored MTP head must contain a capacity-comparable model of
the target's late layers — distilling that into small heads from
small curated data fails repeatedly and predictably. The realistic 40 tok/s
routes are: agent/PLD workloads (23.9 measured, ~27 ceiling), transfer-side
gains to ~20–22 on chat, a change of hardware (owner's 22 GB card — out of
scope by directive), or drafter training at EAGLE-scale data volume (the
batched capture makes that 5× cheaper but ~100× more volume is still
required). Raw: `results/e159-exit-probe-20261111.txt`,
`/tmp/vanta-e158/`.

## E160 — Exposed-vs-hidden transfer analysis (owner-directed, from the E146 trace)

The owner proposed prioritizing the memory engine (branching speculation,
fewer transferred bytes, exposed-latency removal) over further drafter
training, and asked the diagnostic question: of the ~200 ms verify pass,
how much is hidden behind compute? Answer, from the existing E146 Nsight
trace (`results/e146-packed-native-20261111.nsys-rep`, 21.9 verify passes
captured) and E149 cycle measurements:

| Quantity | Value |
|---|---:|
| Transfer per pass (5.24 GB @ 24.6 GB/s) | 213.1 ms |
| GEMV compute per pass (537 chunks × ~200 µs `q4_mma`) | 107.2 ms |
| Per-chunk copy vs per-chunk GEMV | 397 µs vs 200 µs |
| Measured verify (E149) | 226.7 ms |
| **Exposed non-transfer stall** | **13.6 ms** |
| GPU idle inside the streaming pipeline | ~50% |

The chunk pipeline overlaps GEMV for chunk L with the copy of chunk L+1;
since per-chunk compute (200 µs) is roughly half per-chunk copy (397 µs),
compute hides almost fully and **~94% of the verify pass is exposed PCIe
time**. Synchronization stalls are ~13.6 ms/cycle — small.

Two consequences the owner's framing makes precise:

1. **Byte cuts pay until 50% reduction**, after which GEMV compute (107
   ms/pass) becomes the bound. The lossless ceiling measured in E156 is
   ~9–10% (zstd-3 planar) with an entropy bound of ~15–16% — so lossless
   compression alone lands at ~179–185 ms cycles ≈ 22–24 tok/s with the
   current drafter. Non-lossy 3-bit cold weights alter logits and remain
   rejected by the no-quality-loss directive.
2. **Tree speculation with branch-local GDN state is already installed** —
   the packed 24-position verifier verifies a best-first tree (~10–14
   leaves) with per-node parent-derived recurrent state (`tree_path`/
   `tree_depth` kernel arguments, E124/E140/E142/E149). Its measured yield
   is 4.2 emitted/cycle, with E126 bounding the remaining width headroom at
   ~13% (nine passes of 62 on chat). The GDN Tree-Scan mechanism the owner
   cites (arXiv 2609.23900) is therefore largely implemented here; the
   marginal idea it adds is commit-time state selection, which our
   snapshot/rollback machinery already covers.

Revised lever stack toward 40 with the current drafter: 16% lossless byte
cut (→ ~185 ms cycles) × tree headroom (+13%) ≈ 24–27 tok/s chat, ~30+ on
agent traces. Exceeding that requires either a >50% byte cut (losslessly
impossible on a fixed Q4 model), a faster GEMV (the E051–E075 microbench
hill, all failed so far), more emitted tokens (EAGLE-scale drafter training,
data now 5× cheaper via E159's batched capture), or hardware. Raw:
`results/e160-` analysis inline above; trace:
`results/e146-packed-native-20261111.nsys-rep`.

> **CORRECTED in E161:** the "~16% lossless ceiling" and the byte-cut
> arithmetic built on it do not survive — the measured ceiling is ~5.5%.
> The per-chunk compute/transfer crossover (50%) and the 94%-exposed
> finding are unaffected.

## E161 — Codec feasibility: the planar-compression lever is void (negative)

The owner approved a planar-codec GPU-unpack benchmark (five metrics:
compression ratio, decompression time, H2D transfer time, verification-cycle
time, sustained tok/s; acceptance = reproducible end-to-end speedup with
exact reconstruction and no significant VRAM growth). The experiment stops
at its first gate: **the codec does not exist.**

Measurements, all on real tensor boundaries from the GGUF reader (not file
offsets), 2 MiB samples per plane:

| Sample | zstd-3 | zstd-19 | xz -6 | order-0 entropy |
|---|---:|---:|---:|---:|
| blk.0 ffn_up/gate/down scales (2 MiB) | 0.947 | — | — | 7.54 bits/byte |
| blk.24.ffn_up scales | 0.9435 | 0.9433 | 0.9525 | 7.53 |
| blk.24.ffn_up nibbles | 0.9435 | 0.9434 | 0.9506 | 7.50 (≈ 3.75 bits/nibble, the audit's own bound) |
| blk.0 attn_qkv / ssm_out / attn_q scales | 0.942–0.947 | — | — | 7.49–7.53 |
| LZ4 (wire and both planes, 16 MiB chunks) | 1.000 | — | — | no matches at all |
| FP16 scale fields (5-bit exponent, 2-bit mantissa-top, 8-bit mantissa-low) | — | — | — | 4.80/5, 2.00/2, 7.49/8 |

Findings:

1. **Both planes sit at their order-0 entropy limits.** The nibble plane's
   3.747-bit/nibble bound (from the E156 audit, which was correct on this
   point) leaves at most ~6.2%; zstd-19 already achieves 0.9434 against a
   0.937 ideal. The scale plane carries ~7.53 bits/byte with no field-level
   slack (exponent 4.8/5, mantissa 7.5/8) and no LZ structure (LZ4 = 1.000).
2. **The E156 planar/scale results were a sampling artifact** (file-offset
   windows rather than tensor boundaries). The whole-model lossless ceiling
   is **~5.5%**, not 9–16%. Cross-tensor redundancy, repeated structures and
   cross-layer correlation are all ruled out by the same evidence: any such
   redundancy would appear in zstd-19/xz, and it does not.
3. A GPU-decodable codec would recover at most ~4–5.5% (≈ 250 MB of 5.24 GB
   ≈ 10 ms of the 213 ms pass) while adding per-chunk decode work to a kernel
   path that currently has ~200 µs of headroom per 397 µs chunk copy. The
   end-to-end gain after decode overhead is realistically under 1 tok/s —
   the benchmark's acceptance criterion cannot be met.

Consequences: (a) the compression lever is void for throughput on the fixed
Q4 model — cold bytes per pass are irreducible in practice; (b) the owner's
residency-reduction goal on this hardware therefore maps ~1:1 to throughput
loss, making the 22 GB card the genuine residency answer; (c) E160's
findings (94% exposed transfer, 107 ms hidden compute, 50% crossover) are
unaffected and remain the true map of where time goes; (d) the remaining
measured levers are the agent/PLD path (23.9 → ~27), tree-width headroom
(~13%), sync-stall removal (~13.6 ms/cycle), and EAGLE-scale drafter
training (capture now 5× cheaper via E159). Raw:
`results/e161-codec-feasibility-20261111.txt` (inline measurements above).

## E162 — Zero-copy mapped memory vs explicit staging (owner-directed, negative)

The owner proposed benchmarking a zero-copy GEMV path (GPU kernels reading
mapped pinned host memory directly, no staging buffers) against the staged
engine, targeting the memory-capacity objective rather than throughput.
`benchmarks/microbench/e162-zero-copy-q4.cu` measures the memory-path
question on real Q4_0 payload (512 MiB slice of a mid-file FFN tensor),
with the packed-verifier access pattern (each weight byte read once, shared
by 24 rows) and a pinned-DMA control (24.83 GB/s, matching E099):

| Path | Rate | Pass-equivalent (5.24 GB) |
|---|---:|---:|
| Staged DMA (pinned → VRAM) | 24.83 GB/s | 0.211 s |
| SM-issued mapped reads (read-sum kernel) | 18.18 GB/s | 0.288 s |
| Mapped reads, first run (same kernel) | 21.76 GB/s | 0.241 s |

Findings:

1. **SM-issued mapped reads sustain only ~73–87% of the DMA rate** (18.2–
   21.8 vs 24.8 GB/s; run-to-run desktop variance). Post-iommu mapping is
   far better than the E031-era 14.7 GB/s, but still loses to explicit DMA
   on this machine's sequential-stream workload.
2. Substituting mapped reads for staging would lengthen verify cycles from
   ~213–227 ms to ~244–275 ms — **~10–14% slower**, failing the owner's
   "comparable speed" acceptance bar. Zero-copy also saves only the staging
   slots (~8 × 16 MiB ≈ 128 MiB of VRAM), not the hot arena.
3. **The hot arena cannot move to mapped RAM**: hot weights are re-read by
   compute every pass; at ~18–22 GB/s instead of ~360 GB/s VRAM bandwidth,
   the 8.4 GiB hot residency would add roughly +390 ms per pass. Staging
   buffers are the only VRAM zero-copy could free, and they are the small
   part of the budget.
4. Instrument note: the GEMV-shaped kernel variant was instruction-bound
   (~1.2 s for 512 MiB in both paths — path-independent), so the read-sum
   kernel is the valid instrument for the memory-path question. The 1/24
   output mismatch between paths is atomicAdd ordering nondeterminism, not
   a memory-path defect.

Verdict: **zero-copy does not clear the owner's bar** — slower cycles for a
~128 MiB VRAM saving. The two-level design (DMA staging for cold weights,
VRAM residency for hot weights) is confirmed as the right architecture on
this hardware. The owner's framing stands: RAM is now *usable* as a large
GPU-accessible pool (18–22 GB/s via mapping, 24.8 via DMA), but making that
pool *fast* still requires the staging pipeline. Raw:
`results/e162-zero-copy-q4-20261111.txt`.

## E163 — Funnel session: A1 promotion, A2 trace correction, B1 closure

Executed per RESEARCH_FUNNEL.md (owner-approved next-session structure).

### A1 — 8448/ctx-384 promoted (SUPPORTED)

`scripts/run_single_user.sh` gains `GUM_RESIDENT=1` (8448 MiB hot,
context capacity 384). Revalidation on identical workloads, all
`--expect-ids` exact: e110-chat 256/256 (16.01 tok/s), e119-chat 128/128
(17.02), e153-altitude 64/64 (17.45), e153-go 64/64 (16.17); absolute rates
ran ~1 tok/s low under desktop load, so a same-session paired control was
run: e119 8448 = 16.974 vs 8064 = 15.980 — **the +1.0 tok/s paired delta
reproduces** (E156a: +0.5–1.2). The launcher default remains 8064/1024 for
long sequences; flipping the default is a one-line owner decision.

### A2 — synchronization trace analysis (ELIMINATED UNCERTAINTY + PLAN CHANGE)

The E146 sqlite was re-attributed per NVTX range ("Target verification",
20 ranges, 537 copies and 1,893 kernels each):

| Quantity | Per pass |
|---|---:|
| Range span | 227.2 ms (median) |
| **Kernel time** | **214.3 ms** |
| Copy time (97.63 GiB @ 24.7 GB/s) | 214.5 ms |
| GPU-timeline idle inside range | 2.4 ms (largest gap 0.65 ms) |

**E160 correction:** kernel compute per verify pass is ~214 ms, not ~107 ms
— E160 counted only the dominant `q4_mma<8,16,3>` family (~55% of kernel
time) and wrongly concluded the rest hides under transfer. The pipeline is
**balanced**: kernels ≈ copies ≈ 214 ms, running concurrently, span ≈ both.
There is no 13.6 ms removable sync stall — intra-range idle is 2.4 ms.
A2 closes with no implementation (per the owner's rule).

Consequences, each cross-checked against an existing measurement:

1. **Byte→span coupling is near 1:1 across the measured range.** From
   E154/E155 rows: (5.242 GB, 227.8 ms) → (4.839, 213.7) → (4.445, 187.1):
   slope ≈ 36–46 ms/GB ≈ 0.9–1.1× the DMA slope. The E160 "50% crossover"
   is wrong; the crossover sits near a ~17–20% byte cut (solo compute
   ≈ 185–190 ms). Copy-engine writes contend with GEMV for memory
   bandwidth, so fewer bytes speed up *both* streams — which is also why
   8448 residency beat 8064 by more than the pure transfer delta.
2. **Compression stays closed, with a corrected mechanism.** A 5.5% byte
   cut is worth ~10 ms of span, but a GPU decoder adds compute to the
   co-critical side; E155's measured codec delta (0.9 ms for 81 MB) shows
   the trade is already sub-linear. E161's verdict stands.
3. **Compute is co-primary for Track A.** The "GEMV hill is hidden so it
   doesn't matter" premise is retired. However, kernel time also scales
   with residency contention, so the strongest measured lever remains
   residency — and the residency ladder is capped: 8824 MiB leaves ~376 MiB
   after states, which cannot hold the 918 MiB MTP head, and no drafter
   ≤ 376 MiB with ≥ 4 exact mean exists (E153/E157/E158). 8448 + MTP stays
   the optimal live configuration.

### B1 — schedule-level byte-skipping (CLOSED, negative)

Dependency analysis, no code: every verifier row requires all 64 layers
(exact greedy semantics); row rejection is only observable at the final
logits, so no row can be retired mid-pass; every layer's weights are needed
for every live row in every pass. Schedule changes can reorder but not
remove bytes. Additionally, with the corrected balanced pipeline, even
skipped bytes would partly re-balance into compute time. B1 closed: no
valid mechanism exists under exact-output constraints.

### Session outcome

(b) eliminated uncertainty (E160 correction, B1 closure) + (c) plan change
(compute co-critical; residency-vs-drafter identified as the structural
Track A trade-off). Raw: `results/e163-*-20261111.txt`,
`results/e146-packed-native-20261111.sqlite`.

## E164 — C0 benefit ceiling: byte→span is ~1:1; the compression lever reopens (corrects E163)

Owner-approved C0 before building the C1 quality harness. Oracle-mode
verify sweeps (`--oracle-ids`, d10, ctx 384, no MTP, packed verifier) at
8448 and 8656 MiB were added to E154's two anchors (8064, 8824). Parsing
note: the first pass averaged `commit_seconds` fields by regex accident;
the corrected extraction uses `verify_seconds` from `oracle_cycle` lines
only.

| Cold GB | Verify ms | Source |
|---:|---:|---|
| 5.242 | 223.6 | E154 (8064) |
| 4.839 | 211.6 | E164 (8448) |
| 4.621 | 200.5 | E164 (8656) |
| 4.445 | 187.1 | E154 (8824) |

**Linear fit: span = 44.2 ms/GB × cold bytes − 5.7 ms, residuals ±3.6 ms**
(DMA slope would be 40.5 ms/GB). Byte→span is ~1:1 with DMA across the
entire measured range. **E163's "balanced pipeline / kernels ≈ 214 ms
floor" model is falsified**: the kernel durations in the E146 trace
stretch to match streaming copies at the 8064 operating point (memory-
latency-bound on cold data), they do not bound the pass. E160's ~107 ms
`q4_mma` count was the closer estimate of true compute; adding the other
kernels puts the compute floor around ~150 ms (estimated — no data point
below 4.445 GB yet), i.e. the crossover sits near a **24% byte cut at
8448**, not 17–20% and not 50%.

C0 projections at the 8448 operating point (live tree, 4.2 emitted):
10% byte cut → ~22.5 tok/s; 25% → ~27 tok/s; 40% → compute floor
(~28 tok/s) without compute-side improvements. **Verdict: mixed-precision
cold weights are worth ~+45% at the quality border — C1 is GO**, with the
note that the owner's 25% target sits near the crossover edge, so the
compute side (cheaper low-bit GEMV under TOLERANCE) becomes the second
lever exactly where byte cuts saturate.

Ledger honesty: E160 was corrected in E163 (kernel attribution), and E163's
crossover model is corrected here (streaming-stretched kernels). The
byte-ledger and funnel bounds table are updated to the fit above. Raw:
`results/e164-oracle-e110-hot{8448,8656}-ctx384-20261111.txt`,
`results/e164-oracle-e110-hot8824-e154config-20261111.txt` (config rejected
by the current binary — resident-output now requires a draft source; kept
for provenance).

**Owner reconciliation (adopted):** the first published projections mixed
baselines — the 27 tok/s at 25% figure used the 8448 floor against an
8064-style gain. Corrected: 25% byte cut from the 8448 baseline gives span
155 ms, cycle ~171 ms, **24.6 tok/s (+33% transfer-component)**; byte cuts
saturate at ~25.3 tok/s at the ~33% crossover (150 ms floor estimate, no
data below 4.445 GB — extrapolation flagged as hypothesis). The fit's
44.2 ms/GB ≈ 22.6 GB/s effective; four points establish a *local*
relationship only.

## E165 — C1 Stage S: screening harness validated, first sensitivity points

The Stage S instrument (`scripts/c1_requant.py`) requantizes selected Q4_0
tensors to symmetric k-bit grids with per-G-group fp16 scales and stores
the result **losslessly back as Q4_0** — a k-bit grid is a subset of Q4_0's
16-level grid when the same scale is kept, so engine runs measure the
candidate format's true quality with zero storage distortion; byte savings
are analytic (bpw = k + 16/G vs Q4_0's 4.5). Getting here required
replicating the original quantizer's exact convention, discovered
empirically: `d = signed-max-magnitude element / −8` (negative scale),
`n = trunc(w/d + 8.5)` clipped to [0,15]. **Harness validation: the
(4, 32) control rebuilds the original GGUF byte-for-byte** (cmp clean over
15.1 GB).

Screening metric: teacher-forced argmax agreement (`--multi 1 --feed-ids`)
against the EXACT dense continuation — per-position greedy agreement
without error cascade. Identity control on e119-chat: **0/128 mismatches**
✓. Ladder so far:

| Format | bpw (analytic) | Byte cut | e119 agreement | e110 agreement |
|---|---:|---:|---:|---:|
| (4, 32) control | 4.5 | 0% | 100% (0/128) | — |
| (3, 64) all tensors | 3.25 | 27.8% | 82.0% (23/128) | 80.1% (51/256) |
| (2, 64) all tensors | 2.25 | 50.0% | 53.9% (59/128) | — |

Reading: the 25% feasibility gate is arithmetically met by (3,64), but its
~81% per-position greedy agreement is a screening yellow flag — Stage V's
task-level gates (perplexity ≤2%, task ≤1 pp) decide, and the steep
82%→54% drop between 3.25 and 2.25 bpw suggests the quality knee lies
between. Next Stage S points: (3,32) (3.5 bpw, 22.2% cut), (3,128)
(3.125 bpw, 30.6%), and FFN-only mixes — mapping the knee before any
Stage V commitment. Per the owner's rule, two success definitions are
recorded (performance ≥25% vs memory-efficiency 15–20% safe), and no
drafter campaign launches from a C1 failure. Raw:
`results/e165-*-20261111.txt` (screen logs), `/tmp/gum-q*.gguf` variants.

## E166 — MMA width-64 validation probe (positive, prerequisite)

Owner-required independent validation of the packed-verify dispatch at verify widths
48/64 before any decoder change. Standalone probe (`benchmarks/microbench/e166-mma-width-probe.cu`)
instantiating `q4_mma<8,8,8>` (K<17408), `q4_mma<16,8,4>` (K≥17408) and `q4_mma<8,8,6>`
(width-48 reserve) against (a) per-group `q4_mma<NB,8,1>` decomposition — must be
BIT-IDENTICAL since GROUPS only offsets activation/y pointers — and (b) the dp4a
`k_mat_q4<NB>` fallback. All 8 cases (K=4096/5120/12288/17408 × rows=33/47/48/63/64)
bit_diff=0, dp4a worst_rel=0. Occupancy: 56 regs (NB=8, max_threads 1024), 106 regs
(NB=16, max_threads 512) at 256 threads — no pressure. Probe-harness lessons recorded:
activation buffers must be sized for the verify WIDTH (not rows), and the tiled
weight layout rounds rows up to 16. Raw: `results/e166-mma-width-probe-20261111.txt`.

## E167 — Deep-γ PLD: verify width 64 + adaptive direct-chain override (positive; exact-gated)

**Decoder changes** (all in `real_decode.cu` / `eagle4_draft.cuh`): packed default
γ 24→64 (`--speculate 24 --packed-verify 1` preserved byte-for-byte); new MMA dispatch
`q4_mma<8,8,8>`/`q4_mma<16,8,4>` for widths 33–64 (E166-validated); `speculate > 64`
cap; prefill pinned at 32 rows (a 64-row prefill flips near-tie argmaxes at token 0
via the E050 reduction-order class — found and fixed during gating); tree node budget
stays at `mtp_limit` (deeper trees only added rejected width); replay buffer (+~190 MiB
VRAM) now included in the VRAM plan; `draft_hist` per-cycle draft-length distribution
added to the summary (owner gate metric — expected accepted prefix is the conditional
sum, not the per-position rate).

**Deep-γ engagement mechanism**: the E149 guided-tree caps history chains at 23. New
override — when the longest exact history continuation exceeds the tree budget, verify
it as one direct linear chain (no drafter forwards, no tree), gated on the running
acceptance of DIRECT chains only (≥0.75; cold start probes +16 positions past the
budget to buy evidence cheaply). `lookup_drafts` now prefers the occurrence with the
longest continuation (the most-recent-occurrence scan capped chains at one repeat
period). Tree-mode acceptance was unchanged by deeper budgets (measured: identical
22 cycles / acceptance, only wasted width), which is why the tree stays at 23.

**Exact gate: all six prompts lossless** (`results/e167-deepgamma-*-20261111.txt`),
plus preserved γ=24 controls and boundary tests: oracle full-accept at 46 drafts in
one 534 ms verify pass (47 tokens, one cold-set transfer — the thesis in one line);
end-of-generation chains shrink with remaining budget, lossless.

**Measured (verbatim-repeat workload, 128 gen tokens, lossless 124/124):**
11 cycles vs 15 tree-mode, 11.18 emitted/cycle (tree 8.20), **36.86 tok/s** —
deep chains to 52 positions. This is the first measured >30 tok/s single-user
number on exact/repetitive structure, losslessly.

**Chat neutrality proven by same-state controls**: γ=24 control vs γ=64 back-to-back
on the e110 chat prompt: 16.500 vs 16.370 tok/s, verify 0.2323 vs 0.2346 s — within
noise. Agent prompt: 23.022 (γ=24 ctrl) vs 20.856–21.414 (γ=64, includes the one
bounded cold-probe cycle + machine-state drift). **Machine-state caveat: the desktop
held 300–700 MiB more VRAM than during E149 and every pass ran ~5–15 ms slower at
BOTH γ=24 and γ=64; the E149 table should be re-baselined on a calm desktop before
comparing absolute tok/s.** Six-prompt E167 runs: agent 20.86 / coding 16.80 /
explain 15.87 / eventlog 17.83 / chat 16.30 / chat2 15.78 — all match=yes.

## E167b — EAGLE-scale capture pipeline (infrastructure, pilot positive)

`scripts/capture_when_idle.sh` drives the E159 batched rig under owner-required
safeguards: exclusive GPU flock (`/tmp/gum-gpu.lock`, shared with all benchmark runs),
run_when_idle-style preflight per cycle, atomic *.part→validate→publish, per-sequence
dedup ledger, 40 GB storage budget, quarantine for failed validations, `capture/STOP`
switch. Pilot (3 groups): 1280 positions/group at 163,841 B/position (~200 MiB/group,
~85 s), 7/8 new sequences per group (dedup caught one truncated-prompt collision),
~14 unique positions/s. 100k unique positions ≈ 18.7 GB. Hourly idle-gated automation
enabled (first acts: lock + preflight + STOP check); schedule gated on pilot metrics
per owner. Capture products live in `capture/` (gitignored). The pilot used
`/tmp/vanta-e158`; E169 moved capture into `benchmarks/real_decode`.
Raw: `capture/LEDGER.md`, `results/e167-*`.

## E168 — Realistic-session measurement: deep-γ is span-conditional, drafter quality binds (honest negative + infrastructure)

Owner correction: 36.9 tok/s was a literally-repeating prompt — a synthetic ceiling, not a
session. New fixture `results/e168-session-prompt.txt`: multi-turn agent dialogue (design
discussion, corrections, "Proceed", "what if" turns, tool-call boilerplate, novel
reasoning), 732 rendered tokens, 77-token response, exact-gated (`results/e168-session-*.txt`).

**Measured, same machine state, all lossless:** γ=64 default 16.944 / γ=24 control 16.974 /
`--chain-min 14` 16.878 / `--chain-min 10` 16.886 / `--chain-min 6` 16.816 tok/s — all
identical cycle profiles, 4.222 emitted/cycle, longest exact continuation on this fixture:
**15 tokens**. Deep-γ never fires (correctly) and the threshold knob is a measured wash:
direct chains save the ~10 ms drafter forwards but forfeit the tree's model proposals.
**Session tok/s is drafter-bound, not span-bound.** The EWMA recovery gate (replaces the
cumulative direct-chain q, which could never re-fire after one bad stretch) and the
`--chain-min` knob landed with this experiment; six-prompt gate re-passed lossless on the
EWMA build.

**Two product findings for sessions:** (1) the approximate `--prompt` tokenizer inflated the
732-token fixture past ctx 1024 — sessions must use `--prompt-ids`, and the ctx=1024
launcher holds only ~250 generated tokens of session headroom; ctx 1536-2048 is feasible
VRAM-wise (~201-268 MiB KV) but is a policy change against the E149 dense references.
(2) Request-level session latency includes re-prefill of the whole context each turn
(prefill_s ≈ 2-6 s at 32-row chunks).

**Where session tok/s can still move, ranked:** (1) drafter quality on novel text
(4.2 emitted/cycle → 6-8 would put sessions at 21-28 tok/s) — the E167b capture pipeline
is the only measured-physical route and is accumulating hourly; (2) sync-stall trim
(~13.6 ms/cycle, E160 — out of scope by owner); (3) prefill width (policy: changes the
dense-reference ULP baseline). Deep-γ's value remains exactly what it was designed for:
verbatim-heavy spans (file dumps, logs, repeated templates), where it measured 36.9.

## E169 — MTP fine-tune Phase A (2026-11-11, no-go: need volume)

`scripts/finetune_mtp.py` reconstructed the native nextn forward from the
original MTP GGUF. It read each final pre-norm target residual, applied the
target output RMS norm, and paired the previous hidden state and current
token embedding with the next greedy token. The pilot used groups 000–002
for training (24 streams, 3,792 pairs) and held out group 003 (8 streams,
1,264 pairs). The training and validation groups had no exact prompt-ID
overlap.

| Run | Val top-1 | Val top-8 | Val loss |
| --- | ---: | ---: | ---: |
| Original head | 79.27% | 97.63% | 0.7377 |
| Shakedown, best top-1 epoch | 74.45% | 96.91% | 0.8745 |
| Gentle run, best top-1 epoch | 79.35% | 97.78% | 0.7224 |

The original head's 79.27% top-1 validated the capture-to-token alignment
on the pilot. In the shakedown, train loss fell from 0.9263 to 0.0608 while
val loss rose from 0.8745 to 1.1882. The gentle run's +0.08 percentage
points in top-1 amounted to one extra correct validation label. Phase A
called for +2 points in top-1 or +3 in top-8. With 97.63% baseline top-8,
the second threshold was above the split's 100% ceiling. Neither run
cleared the gate. **Need volume** before another training decision.
Raw runs: [shakedown](results/e169-finetune-shakedown-20261111.txt) and
[gentle](results/e169-finetune-gentle-20261111.txt).

### Capture restoration (2026-11-12)

The idle wrapper uses the in-tree decoder and a greedy Python tokenizer
over the model's GGUF vocabulary.
With `GUM_MID_CAPTURE=8,16,24,32,40,48,56` and `GUM_MID_CAPTURE_FILE` set,
dense batch generation writes one fp32 record per step. The header is
`[position, layer count, batch width]`, followed by the layer residuals
in ascending order and the final pre-norm residual. Prefill writes no
records. The wrapper checks every record header, file size, finite values,
and generated stream lengths before publishing the batch.

The shipped seed pool now includes 16 synthetic multi-turn sessions with
revisions, follow-up questions, and tool-call text. Equal-length batching
keeps the final user turn and assistant handoff. Once the authored pool is
exhausted, the wrapper samples continuation slices from all eligible
generated streams.

The RTX 3060 smoke held out a new eight-session group with 68 prompt
tokens and 32 generation steps. It published 256 positions in 40 MiB.
The 32 records covered positions 68–99, with seven mid-layer residuals
and one final pre-norm residual per sequence at each step. All values
were finite. Eval-only on the original MTP head scored **78.33% top-1**
and **98.33% top-8** on 240 pairs. A second decode with capture disabled
produced the same 256 token IDs. A second capture exercised seed exhaustion
and published 64 positions using slices of the existing 160-token streams.
The smoke group lives in
`capture/e169-smoke-20261112/` and was evaluated with legacy group 004
as the split companion. Raw: [capture and eval-only smoke](results/e169-capture-restored-20261112.txt).

## E170 — Ten verified tokens per pass on a single-user repository job

The benchmark used the current E169 publisher and a real crash-window
reproduction. An isolated directory contained a complete group 004 and
group 005's published text sidecar with its binary still absent. The
wrapper's group scan advanced to 006 and seed index 48. The evaluator's
binary glob saw only group 004. Turn one produced the triage record from
that tool output. Turn two turned it into a regression fixture. Both
structured answers were checked against the reproduction and reached
EOS. Exploratory prompts that hit the output limit or misidentified the
next group were excluded from the job timing.

The RTX 3060 ran one sequence at a time with the original 27B Q4_0 target.
The GGUF chat template and exact tokenizer produced 805 and 939 prompt
IDs. Every timed arm returned the same 113 output IDs as the installed
greedy verifier. External wall timing starts at submission of the rendered
IDs and includes model startup where applicable, prefill, and generation.
The two-turn total sums the requests without time between turns.

| Configuration | Two-turn wall s | Full-job tok/s | Follow-up wall s | Follow-up tok/s | Tokens/pass across both turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Installed native MTP, restart for each turn | 27.481 | 4.11 | 14.352 | 4.18 | 6.53 |
| Persistent session, default draft policy | 16.867 | 6.70 | 4.060 | 14.78 | 6.53 |
| Persistent session, history drafting off, 23 draft nodes | 14.536 | 7.77 | 2.845 | 21.09 | 9.25 |
| Persistent session, history drafting off, 31 draft nodes | 14.640 | 7.72 | 2.799 | 21.44 | **10.09** |

Tokens per pass exclude the first token of each turn, which prefill
produces. The 31-node run emitted 52 tokens in six verification passes
on the first turn and 59 in five on the follow-up. The 23-node run had
similar job latency. Widening the tree while leaving history drafting
enabled took 29.235 seconds. The history-enabled baseline accepted 17
out of 284 history proposals across the job.

Session mode checkpoints the recurrent state at a full prefill chunk
boundary and keeps the model and attention state loaded. The follow-up
reused 800 prompt IDs and evaluated 139 new ones. Separate dense-reference
checks passed 512/512 IDs across fresh, changed-prefix, reused-prefix,
and changed-back requests. The two-turn benchmark establishes performance
for this repository job. The draft flags remain opt-in for workloads
where repeating text can make history drafting valuable.

The normal launcher reproduced 111 generated tokens in 11 passes and
113/113 matching IDs. Its two-turn wall time was 27.511 seconds with
restarts and 14.592 seconds with `--session 1 --pld 0 --draft-max 31`.
The follow-up went from 14.285 to 2.750 seconds. Both used
`--ctx 2048 --gen-tokens 384`. Raw [launcher repeat](results/e170-launcher-repeat-20261112/timings.jsonl).

Reproduce via `scripts/bench_e170_job.py` with arms `baseline` and
`session-no-history-31`. Raw [timings](results/e170-verified-job-20261112/timings.jsonl),
[first answer](results/e170-verified-job-20261112/answer-1.txt),
[test fixture](results/e170-verified-job-20261112/answer-2.txt), and
[dense gate](results/e170-verified-job-20261112/dense-boundary.log).

## E171 — Reuse committed reply state and group exact prefill

The E170 repository job was replayed on one RTX 3060 sequence with the
same two rendered prompts and the same 113 reference output IDs. Both
turns completed. External wall time includes process startup, prompt
processing, and generation, and sums the two requests.

The decoder now pipelines initial hot-weight uploads through its existing
pinned staging ring. The grouped prompt path streams weights once for
up to four canonical 32-token chunks. It keeps the FP32 reduction order
and native MTP cache updates. In a session, a follow-up can reuse already
committed reply tokens when the complete token prefix matches. If it
changes, the decoder restores the matching prefill checkpoint or resets
state and evaluates the new request.

The E170 follow-up reused 858 of 939 input IDs. Its 81 new prompt IDs
took two target passes. The prior session processed 139 new IDs in five
passes. The five generation passes produced the same 59 tokens after
the first token from prefill. The uninstrumented release candidate
finished in 12.860 seconds across both turns, including a 2.329-second
follow-up (25.76 tok/s).

| Matched normal-launcher run | Full job | Follow-up | Follow-up tok/s |
| --- | ---: | ---: | ---: |
| E170 control | 15.062 s | 2.754 s | 21.79 |
| E171 | 13.035 s | 2.370 s | 25.32 |

The installed run matched 113/113 IDs. Follow-up prefill fell from
1.277 to 0.821 seconds, and the number of generation passes stayed at
five. The raw-upload CPU control matched all 202 hot chunks byte for
byte and had zero argmax mismatches over 16 positions.

Correctness checks compared separate fingerprints over the full recurrent
and convolution state, live target KV, live MTP KV, and final hidden state
at the end of prefill. They matched on both job turns and on six turns
across two independent prompts, with 192/192 matching output IDs on the
independent checks. These included a changed reply that fell back to the
96-token prefill checkpoint, a reset, and committed-state reuse after an
output limit. All 535 resident packed chunks matched the reference byte
layout. State instrumentation runs are excluded from the speed result.

The tested launcher flags are `--session 1 --session-tail 1 --prefill-chunks 4
--speculate 32 --packed-verify 1 --pld 0 --draft-max 31 --ctx 2048 --gen-tokens 384`. Draft
budgets of 35 and 39 took longer on this job. Repacking the shared output
head yielded a small follow-up change with no convincing job-time gain.

Raw [timings and exact checks](results/e171-session-20261112/timings.jsonl),
[follow-up state check](results/e171-session-20261112/release-state-2.log),
and [changed-prefix gate](results/e171-session-20261112/release-e110-chat-optimized.log).

## E172 — Draft retrieval and one-pass follow-up prefill

The interrupted run probed the E171 repository job with native-MTP tree
traces. Full-vocabulary drafting left both follow-up miss points outside the
top eight and retained five verification passes. History retrieval increased
the first accepted span from eight to thirteen positions but still required
five passes. A digit prior alone raised the count to seven.

Merging the two follow-up prefill passes preserved the output IDs and reduced
the pass count to one. In the matched run, prefill took 0.824 s against
0.829 s for the two-pass control. A subsequent probe measured 0.736 s.
The wall-time gain was unstable. The unfinished previous-reply proposal
prototype aborts on rejection and remains an experimental source.

[E172 timings](results/e172-20261112/timings.jsonl) and
[baseline follow-up trace](results/e172-20261112/baseline-2.log).

## E173 — Wider exact prefill

For the same 805-token first request, allowing eight canonical chunks per
weight pass reduced prompt evaluation from eight passes to five. Each
32-token chunk retains its per-row reduction order. Target and MTP weights,
resident memory budget, and the 32-position decode verifier stay fixed.

Two alternating normal-launcher runs per mode gave median job times of
11.848 s at eight chunks and 12.941 s at four chunks, with all 113 output
IDs matching in each run and both turns reaching EOS. Median first-turn
prefill fell from 6.610 to 5.551 s. The follow-up stayed at five verification
passes and about 25.9 output tok/s including prompt evaluation.
Eight chunks used about 55 MiB more device memory at the minimum-free watermark.

One 800-row group, under a smaller hot-weight budget, reduced first-turn
prefill to two target passes but was slower than 256 rows at that same budget.
The 32-row MMA tile and first-use direct pinned packing also failed to improve
wall time. A 64-position draft removed one generation pass on the first turn
but took 3.505 s on the follow-up against 2.345 s for its matched 32-position
control. It kept five passes and raised mean target verification from roughly
0.27 s to 0.49 s. Splitting the wide raw matvec into two 32-row launches
recovered little and still left the follow-up at 3.45 s with five passes.
One prefill arm was stopped by the existing swap guard and excluded.

The original follow-up needs to finish within 1.50 s to reach 40 output
tok/s. At roughly 0.83 s prefill and 0.30 s per verification round, that
means at most two rounds, or faster prompt evaluation paired with three
rounds. Five rounds transfer about 26.2 GB of target weights. Larger
trees at the current proposal quality failed to reduce that traffic.

Reproduce with the E171 flags and `--prefill-chunks 8`.
[Normal launcher timings](results/e173-launcher-20261112/timings.jsonl) and
[exploratory runs](results/e173-20261112/timings.jsonl).

## E174 — Reuse shared context across separate requests

The E171 session can reuse the previous reply when the next request extends
it. The new optional `--session-prefix-tokens` boundary also serves
independent requests that diverge after a shared tool transcript. It stores
the target's recurrent and convolution state plus the MTP handoff hidden
state in host RAM. Target and MTP attention KV stay in their normal device
buffers. Reuse checks the rendered token IDs up to the 32-aligned boundary.
An edit before the boundary or a request too short to safely retain it
invalidates the shared checkpoint.

The single-user benchmark asks the same model for two independently rendered,
checkable repository maintenance outputs from the real E169 capture failure.
It repeats the pair after editing the shared context. The two modes use the
same 256-row prefill and 32-position verifier. Each run starts one process and
submits the four requests serially. The comparison includes startup in the
first request, and prefill and decode in every request.

In two runs per mode through the installed launcher, the median request
that could reuse 704 context IDs fell from 6.648 s to 2.277 s
(9.03 to 26.35 output tok/s). Prefill for those requests fell from 4.895 s
to 0.496 s. Median time for all four requests, including the edited context
and cold startup, fell from 29.516 s to 21.016 s. Both modes generated the
same 226 IDs per run, and every answer reached EOS and passed the task
check. The option requires about 150 MiB of additional host memory.
[Normal launcher timings](results/e174-launcher-20261112/timings.jsonl).

Run `python3 scripts/bench_e174_shared_prefix.py results/e174-repeat` with
the normal launcher and an exact model tokenizer. The harness verifies that
each answer finishes and matches the independently computed fixture, and
checks every emitted ID against the uncached control.

An instrumented seven-request sequence matched all 193 output IDs and the
fingerprints for recurrent state, convolution state, live target and MTP
KV for every head, and the MTP handoff hidden state. It included an edited
prefix, a short prompt that invalidated the checkpoint, and a rebuilt
checkpoint that subsequently hit. The instrumented timings are excluded.
[State probe](results/e174-20261112/prefill-state-probe.patch) and
[cache rebuild evidence](results/e174-20261112/live-state-cache-7.log).

## E175 — Verify tool drafts during prefill

The original E170 follow-up uses source files and the output of the capture
failure reproduction to request a JSON fixture. A client can construct a
candidate from those observations while the session is warm. The E175
client extracts the listing, visible groups, cursor, and seed information
from the actual prompt and tokenizes its JSON draft. Baseline output IDs
are used only by the checker. The timed requests include draft construction,
tokenization, prompt evaluation, verification and any fallback generation.
Tokenizer initialization is outside the per-request measurements.

`--session-draft-hints 1` appends draft rows to the final prompt pass.
The verifier uses the full target vocabulary and follows the same greedy
argmax. The convolution and delta-net kernels retain prompt-boundary state
and put candidate end state in the existing replay arena. A rejected fused
draft discards its entire trial state and restarts generation from the
prompt boundary. Accepted rows advance target KV, recurrent state and MTP
KV for the next request. EOS is verified by the target. If the proposed
response extends beyond the fused width, remaining candidates can be
verified in the regular decode loop. Prompts without hints follow the
ordinary path.

The original follow-up can verify all 60 generated tokens alongside the
81 uncached prompt tokens in a single target weight pass. At the same
resident weight budget, this replaces two prompt passes and five decode
passes. The one-pass follow-up streams 5.24 GB instead of 36.69 GB of cold
weights across those seven passes.

The two alternating launcher runs per arm gave medians of 2.295 s for
the follow-up control and 1.050 s with the tool draft (26.1 and 57.1
output tok/s). The two-turn job including model startup fell from
11.780 s to 9.079 s. Both turns reached EOS and all 113 output IDs matched
in both repeats. A second set on the same build measured 55.4 output tok/s.

**Interpretation caveat:** 57.1 tok/s (and the 55.4 repeat) is the rate for a
task-specific candidate constructed from the client's tool/source context and
then target-verified. The final committed IDs are real target output, but the
candidate is not ordinary self-drafted open-ended generation. Treat this as a
verified tool-answer fast path; it does not establish 55–57 tok/s novel-prompt
throughput or a general 60 tok/s single-user agent rate. E176 measures fresh,
unhinted prompts separately.

The feature applies to requests where a client has a useful proposal.
The deliberately corrupted fixture matched 47 of 60 tokens before
rejection. Fallback took 2.549 s and reproduced the full control output.
A third turn reused the 999-token prefix that included the accepted
fixture and produced all ten expected IDs without a draft.
Instrumented runs matched fingerprints for the complete recurrent and
convolution states, all live target and MTP KV, and the MTP handoff hidden
state after both original turns. A separate 128-token oracle test covered
the fused-width boundary, a late rejection in the ordinary verifier, and
the output limit. Both streams matched all 128 IDs. Those oracle timings
are excluded from the performance result.
On a changed capture incident, the target's baseline output itself
disagreed with the on-disk facts. The exact decoder reproduced that
baseline output. The altered incident is a fallback check and is outside
the speed result.

Reproduce with `python3 scripts/bench_e175_tool_drafts.py results/e175-repeat`
using the normal launcher and `llama-cpp-python` for the vocabulary-only
tokenizer. See [launcher timings](results/e175-final-20261112/timings.jsonl),
[rejection and recovery](results/e175-final-20261112/rejected-draft-2.log),
[changed reproduction](results/e175-final-20261112/changed-oracle.json),
and [state instrumentation](results/e175-state-20261112/session-state-probe.patch).

## E177 — Unhinted agent baseline and merged final prefill

The fixed development and held-out suites cover fresh coding, reasoning,
technical explanation, structured JSON, document context and a live file
tool workflow. The 27B packed native MTP baseline at 8064 MiB resident
delivered 17.56 and 18.33 aggregate committed decode tok/s and moved
5.242 GB per verifier cycle. A separate 14-cycle Nsight trace at 7168 MiB
resident recorded 263 ms of DMA per cycle, of which 105 ms had no active
GPU kernel. Increasing the native tree's depth priority and re-enabling
history proposals both increased the number of verifier passes.

The merged prefill path retains the exact prompt checkpoint while avoiding
one weight pass where the final prompt segment and the preceding group fit.
It improves prefill latency. The output and the decode cycle count match the
control, including a changed-suffix checkpoint restore. See the
[E177 ledger, raw data, and reproduction commands](results/e177-general-20261112/README.md).

## E182 — Hot-tier sweep under full-pin (stream era)

2026-11-13, live production path. The pure-streaming profile (hot=0) re-measured at
7.52 decode tok/s on the server's fixed pendulum fixture (46 prompt tokens, 200 output,
44 cycles, emitted/cycle 4.52, cycle 0.602 s) — the frozen 5.5 reference predates the
root full-pin deployment. The verify pass is now link-saturated: 0.575 s ≈ cold bytes
÷ ~25 GB/s, non-fetch overhead (MTP draft 23.7 ms + repair 2 ms) only 4.3% of the
cycle; the E160-era sync-stall lever is spent.

`systemctl` env overrides (`GUM_PROFILE=stream GUM_HOT_MIB=H`, sudo) swept the hot
tier through the production server, one 200-token run per point:

| hot MiB | verify pass | cycle | decode tok/s | min free VRAM |
|---:|---:|---:|---:|---:|
| 0 | 0.575 s | 0.602 s | 7.52 | 8.5 GiB |
| 1600 | 0.506 s | 0.533 s | 8.49 | 6.5 GiB |
| 3200 | 0.438 s | 0.464 s | 9.75 | 5.1 GiB |
| 6400 | 0.304 s | 0.330 s | 13.70 | 1.8 GiB |

All four outputs **token-identical** (869-char completion diffed byte-for-byte). The
E164 local byte law holds linearly: Δverify = 42–43 ms per hot GiB across all three
deltas, i.e. **+0.72 tok/s per hot GiB** at this fixture's acceptance. hot=6400 leaves
1.8 GiB free — unusable as a static desktop default, which is exactly the gap E183 fills.

Raw: results/e182-adaptive-residency-20261113/ (per-point JSON/text/spec summaries).

## E183 — GUM Adaptive: runtime residency supervisor (installed, default-on)

The owner's 3 GiB engine directive exists so desktop apps can use VRAM; when they
don't, that VRAM sat idle. E183 makes residency a runtime decision: a supervisor
thread in `local_chat/gum_openai_server.py` polls free VRAM every 10 s and respawns
the idle decoder with a resized `--hot-mib`:

- `target = hot + free − reserve(4600) − margin(400)`, clamped [0, 6400]
- grow after 3 stable samples; shrink after 2, or immediately when free < reserve
- never mid-request (guarded by the session lock), 120 s cooldown, ±400 MiB deadband
- skips respawns after long prompts (>2000 tokens): a respawn drops session KV and
  the re-prefill costs more than the residency gain (hard pressure overrides)
- respawn reload ≈ 60–90 s, absorbed at request boundaries; engine spawn path
  shared with the request path (`_spawn`)

Live-verified all three paths in production: grow 0→3308 MiB at free 8308; forced
shrink 3308→2183 under a synthetic 5 GiB desktop hog (6.2 GiB returned to the desktop);
regrow to equilibrium 4378 with free settling at 5002 MiB (= reserve + margin exactly).
Throughput at equilibrium: **9.84–10.8 tok/s decode, token-identical to stream**;
ceiling 13.7 when the desktop frees more VRAM; floor 7.52 (hot=0) under pressure.

Enabled via `/etc/systemd/system/gum-openai.service.d/adaptive.conf`
(`GUM_ADAPTIVE=1`); disable by removing the drop-in or setting 0. Policy knobs:
`GUM_ADAPT_CEILING_MIB`, `GUM_ADAPT_RESERVE_MIB`, `GUM_ADAPT_MARGIN_MIB`,
`GUM_ADAPT_LONG_PROMPT`, `GUM_ADAPT_POLL_S`, `GUM_ADAPT_GROW_SAMPLES`,
`GUM_ADAPT_COOLDOWN_S`. /health now reports `hot_mib`, `adaptive`, `free_vram_mib`;
scripts/verify_stream.sh expects throughput from the E182 law given the live hot tier.

Raw: results/e183-adaptive-turn2.json; journalctl gum-openai.service adaptive, lines.

## E184 — Data-path efficiency audit (stream era, 2026-11-13)

Direct decoder trace at hot=0, production flags (results/e184-datapath-audit-20261113/):
cold payload is **13.70 GB/cycle, each byte fetched exactly once** (target 14.4 GB minus
resident MTP head); steady-state path is staged-cache → single pinned DMA, no bounce
copies under full-pin. Live effective rate **22.95 GB/s = 93% of the 24.7 GB/s DMA
ceiling** (the 7% residue is kernel-stretch — kernels execute while weight tiles land).
The link itself delivers 78% of gen4 spec; that gap is protocol overhead, config-closed
(iommu=pt, dual-stream and zero-copy falsified, compression at entropy bound). Remaining
slack, sized: ~16 ms/cycle serial draft+repair tail with the link idle (prefetch-during-
draft ≈ +4–5%) and the 7% stretch (deeper staging prefetch queue). Combined realistic
recovery ~6–8%; the link constant is otherwise closed.

## E185 — Stacking-small-efficiencies audit (2026-11-13)

Hunted the "1–2% items that stack". Results:

- **GPU clocks: falsified.** SW power cap never active during decode; SM held 1942 MHz
  through the pass, downshifting to 817 only after the run (the 2928 s lifetime
  power-cap counter came from other workloads). No clock-locking change warranted.
- **PCIe: clean.** gen4 x16 held through decode, persistence on.
- **CPU affinity: real win, installed.** Same-prompt A/B (fresh runs, hot=0):
  taskset 0–3 (production) 6.89 tok/s / verify 0.614 s; **0–7 → 7.04 tok/s / verify
  0.600 s (+2.3%)**; 0–15 (all SMT siblings) 6.98 — worse than 0–7, and unpinned 6.97.
  Topology (12700K, 20 threads): cpu 0–7 = one thread per physical P-core; 8–15 = SMT
  siblings; 16–19 = E-cores. Production was pinning to HALF its P-cores; 0–7 wins
  because the staging readers + draft CPU scoring get 8 real cores without SMT
  contention. Installed via systemd drop-in (CPUAffinity=0-7, overrides the unit's
  0 1 2 3) and server `_spawn` taskset set {0..7}. Token stream identical to the E182
  baseline; production warm-run delta +0.5–2.3% (run-to-run variance ±1%).
- Cycle composition is tight everywhere (verify+mtp+repair = cycle within ~1 ms) —
  no scheduling gaps left. Remaining stacking candidates, sized: kernel-stretch
  ~42 ms/cycle (deeper staging prefetch queue, E184) and the draft's 13.5 ms internal
  split (CPU tree top-k scoring — accuracy closed by E178, but the COST side is
  unaudited). Repair 2 ms and logits scratch are not worth touching.

## E186 — GUM-Fuse wavefront: deeper in-flight ring (installed, 2026-11-13)

The E186 audit showed ~19–42 ms/cycle of kernel-stretch; the pipeline's in-flight
buffer was 8 slots × 16 MiB ≈ 5 ms of link time with a 6-item CPU back-pressure sync.
Made the ring depth compile-time sweepable (`GUM_KSTAGE/GUM_KSLOTS/GUM_WINDOW`) and
swept under the production verify path, hot=0, same fixture, twice per point, every
point session_ids-identical to the dense reference:

| ring (stage/slots/window) | verify | cycle | tok/s |
|---|---|---|---|
| 8/8/6 (E047 baseline) | 0.6050 | 0.6208 | 6.98 |
| **16/16/12 (installed)** | **0.5880** | **0.6039** | **7.18 (+3.0%)** |
| 24/24/18 | 0.6347 | 0.6509 | 6.66 — reproducibly WORSE |

Deeper queues degrade past 16 (copy-engine scheduling / slot-cycling L2 effects —
mechanism unconfirmed, but the direction is consistent). Installed as compiled
defaults; production warm-run delta is smaller (+0.4%, within ±1% variance) because
the pinned-0-7 production path already ran nearer the floor than the unpinned A/B
shell. VRAM cost: +128 MiB slots, +128 MiB host staging ring.

## E187 — Prefetch-during-draft: negative (reverted)

Implemented the FlashAttention-style fusion: the weight schedule is static, so the
next pass's first copies can be issued during the draft's link-idle window
(position-keyed into the slot ring, skip-on-consume in mat). Token-exact everywhere,
but: **hot=0 net zero** (verify −6.3 ms, mtp +5.6 ms — the draft's logits readbacks
simply contend for the saturated link; hiding bytes in the draft window reshuffles
link demand, it does not create it) and **hot=4378 −4%** (verify +9 ms, mtp +7.5 ms,
despite 32 ms of measured link slack — ordering/L2-eviction interference, not
bandwidth). Reverted from the tree; the full patch, raw runs and IDs are preserved
in results/e187-prefetch-negative-20261113/. Do not re-attempt without a mechanism
that isolates the draft's D2H readbacks from prefetch H2D traffic (e.g., a dedicated
readback stream with priority, or batching the tree's logits readbacks into one
buffer — the latter doubles as the draft-cost audit item).

**Where this leaves the decode equation:** at hot=0 the cycle is link-saturated and
now 93→96% efficient; the remaining non-fetch tail (draft 14–19 ms + repair 2 ms)
is protected by contention, not idle bandwidth. The frontier is emitted/cycle
(drafter quality — capture-fed finetune) and prefill (A6 L2 window, untested).

## E188 — CPU second-stream feasibility (2026-11-13)

External claim tested: CPU Q4_0 matmul at verify width (32 rows) could absorb ~4 GB
of weights per cycle in <0.4 s (+40% ceiling). Measured with the user's llama.cpp
binary (GPU hidden, 8 P-cores): pp32 = 9.72 t/s = **3.29 s per 14.1 GiB weight pass =
4.3 GiB/s**, batch-independent (pp512 = 9.78). The batch-1 objection to CPU matmul is
indeed void — but the 0.4 s claim is 2.4× optimistic. Balanced GPU/CPU split ceiling:
**+15–20% with perfect per-layer overlap** (f≈16% of weight bytes), contingent on a
row-split pipeline, exactness gating of CPU-computed tensors, and modest RAM
contention. CPU compute is not the obstacle; per-layer overlap + exactness are.
Was excluded by owner directive; reopening is an owner call. Raw: results/e188-*.

## E189 — Fixed-VRAM audit: the reclaimable block is already spent (2026-11-13)

Follow-up to the CPU/codec evaluation, which proposed reclaiming ~0.73 GiB (254 MiB
logits + 489 MiB drafter FP32→Q8). Audited against the live decoder — both figures
were stale SYSTEM_MAP §6 numbers:

- **Logits: already compact.** `compact_target_logits` is default-on in speculate
  mode (logit_rows = speculate = 32 → ~32 MiB, not 254). Shipped long ago.
- **Drafter: already Q4_0-native.** Live load prints `mtp_plan,q4_MiB=910,f32_MiB=0`
  — no FP32 blob exists. Moreover **715 MiB of the 910 is the shared target output
  head** (`share_output`, verified byte-identical): it is device-resident and
  therefore already excluded from the 13.70 GB cold stream. The drafter-specific
  footprint is only ~260 MiB (Q4 weights ~195 + FP32 KV ~17 + scratch ~50), which
  five drafter probes (E153–E159) established as the smallest head that holds p₁.
  Raising drafter precision to Q8 would COST 195 MiB for no acceptance gain.

Corrected fixed overhead at 32K/stream: KV 723 (quality-bearing) + MTP-specific ~260
(already minimal) + activations ~500 (prefill-width structural) + attention scratch
151 + slots 256 (buys the E186 +3%) + logits 32 + CUDA ctx ~300 ≈ 2.0 GiB of true
overhead + the shared 715 MiB head. Reclaimable without quality loss: **tens of MiB
(drafter KV→FP16, misc) ≈ +1–2%, not +7%**. The +7% quick pair promised earlier does
not exist — withdrawn. The VRAM-neutral, quality-neutral levers that remain are the
capture-fed MTP finetune (emitted/cycle) and the A6 prefill L2 window.
