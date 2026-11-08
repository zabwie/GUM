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
