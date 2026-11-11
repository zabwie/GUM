# Controlled managed-memory oversubscription experiment

## Status

Prepared only. No allocation larger than the existing project caps has been created, and no over-VRAM run has been executed. The mission requires explicit user approval of a revised budget before execution.

## Question and success criteria

Can one CUDA managed allocation with a 13 GiB logical working set be fully initialized, accessed, and validated by GPU kernels on the RTX 3060 while the OS backs evicted pages in system RAM without entering swap?

A successful result must show all of the following:

- The single managed allocation is 13,312 MiB, larger than the GPU's 11,905 MiB physical VRAM.
- GPU initialization and all access/validation kernels cover every element. Allocation size alone is not evidence of residency.
- Sequential, randomized-tile, and locality-aware passes all produce zero validation errors.
- Timings separate GPU first-touch initialization from later full-working-set passes.
- Nsight Systems reports managed-memory migrations/page faults; NVIDIA telemetry records per-process and global VRAM use; system telemetry records available RAM and swap.
- Swap remains at zero for the entire run. The process aborts if available RAM falls below 8 GiB, global free VRAM falls below 2 GiB, the GPU reaches 80 C, or sustained heavy unrelated GPU use appears.
- Reports distinguish per-process observations from system-wide PCIe traffic and GPU memory readings. If physical backing or migration cannot be proven, the outcome is inconclusive rather than a pass.

Random access will permute 2 MiB tiles, so every tile is visited once while accesses inside each tile remain coalesced. The locality-aware pass will shuffle 32 MiB clusters while scanning tiles sequentially within each cluster. CPU work is limited to constructing small tile-order metadata and collecting telemetry; all dataset initialization, access, and correctness math runs on the GPU.

The first-touch metric will be labeled GPU initialization-plus-first-touch. A pure read-first latency would require preinitializing all 13 GiB in host RAM, which would exceed the present safe host-memory headroom. The experiment must not pretend those measurements are equivalent.

## Current resource estimate

Fresh samples from the small-model preflight on 2026-11-08 showed 11,082 MiB free VRAM out of 11,905 MiB total, 20,106 MiB system RAM available, and zero swap in use. The actual run must recompute these values and may proceed only if its resource gate passes.

For a 13,312 MiB working set, preserving 2,048 MiB of free VRAM leaves a current estimate of at most 9,034 MiB (about 8.82 GiB) available for incremental managed-page residency in VRAM. The remaining 4,278 MiB (about 4.18 GiB) would need host backing at that placement. Driver placement varies, so host use may be higher; live monitoring and abort limits are mandatory.

## Exact budget change requiring approval

Approve all of the following before the over-VRAM run:

- Raise the working-set limit from 2 GiB of additional device allocation to one 13 GiB CUDA managed allocation.
- Permit managed pages to occupy up to the live free-VRAM amount minus a 2 GiB guard. At the latest sample, that is about 8.82 GiB of additional VRAM residency, roughly 6.8 GiB above the existing 2 GiB cap.
- Raise the additional system-RAM backing cap from 4 GiB to 8 GiB.
- Keep at least 8 GiB of system RAM available, at least 2 GiB of VRAM free, and zero swap use throughout. Abort immediately if any threshold is crossed.
- Run only after the desktop and other GPU clients pass the usual idle gate. Do not disable swap or stop unrelated processes.

The requested 8 GiB host cap is above the expected 4.18 GiB minimum at the current VRAM estimate, but below the 12 GiB available-RAM envelope between the current 20.1 GiB availability and the 8 GiB hard reserve. If runtime telemetry indicates that the allocation would exceed these bounds, stop and report the run as deferred.

## Controlled procedure

1. Recheck GPU total/free memory, temperature, utilization and per-process SM activity; system MemAvailable, configured/used swap; process memory limits; and RAM headroom. Require no swap use, at least 16 GiB MemAvailable before launch, sufficient free VRAM to preserve the 2 GiB floor, and no sustained heavy unrelated GPU process.
2. Allocate one cudaMallocManaged buffer of exactly 13,312 MiB. Do not create a same-sized device copy or prefetch the whole allocation.
3. Launch a GPU grid-stride initializer that writes a deterministic index-derived value to every element. Record its CUDA-event time as initialization-plus-first-touch, not as a pure migration latency.
4. Run full-working-set validation passes in sequential, randomized 2 MiB tile order, and randomized 32 MiB cluster order. Each pass validates every element and reports elapsed time, logical bytes, throughput, and mismatch count.
5. Run repeated passes of each pattern. Keep initial-touch timing separate from repeated-pass medians and ranges.
6. Capture one unprofiled performance run with one-second PCIe dmon, 500 ms GPU-memory/utilization telemetry, and 500 ms MemAvailable/swap telemetry. Capture a separate Nsight Systems trace for a bounded pass to record managed migration and faults; exclude profiled timings from the performance table.
7. Record process RSS/smaps, per-process GPU memory if exposed by NVML, global VRAM use, MemAvailable, swap, utilization, temperature, and PCIe counters before, during, and after. Attribute system-wide counters only as aggregate observations.
8. Require zero mismatches across every region and zero swap use. If a safety limit is crossed or a monitor cannot run, terminate the experiment and record the failure without retrying at a higher size.

The allocation and run command will be added to the guarded harness after the revised budget is approved. Do not run by bypassing the ordinary 2 GiB/4 GiB resource gate.

## Interpretation boundaries

A passing run proves that this managed allocation was physically accessed across VRAM and system RAM on this machine. It does not prove that host pages are as fast as VRAM, that Linux HMM alone provides the mechanism, that a larger LLM will complete, or that throughput improved. The next model experiment must compare the fitting Qwen3 baseline with a larger model under identical decoding settings, and must report tokens/s and output correctness.

## Primary references

- [CUDA Runtime API memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html): managed memory can be oversubscribed when concurrent managed access is supported; runtime migration policy and prefetch behavior must be measured.
- [CUDA Unified Memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html): managed memory performance hints and prefetch behavior.
- [CUDA Driver API VMM](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html): explicit allocation, mapping, and minimum-granularity requirements.
- [Linux HMM documentation](https://docs.kernel.org/mm/hmm.html): driver-facing page-table mirroring and migration support.
