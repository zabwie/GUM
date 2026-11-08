# Next steps

1. **Completed:** Query CUDA capabilities and execute checksum-validated mapped-host, staged H2D, VRAM, and managed-memory scans.
2. **Completed:** Isolate H2D copy-only bandwidth; repeat VRAM scans with three warm-ups.
3. **Completed:** Measure 16, 24, 32, 64, and 128 MiB sequential scans. Staging was nearly tied at 16 MiB and faster at 24 MiB and above in these runs.
4. **Completed:** Compare managed CPU-first access, repeated GPU access, and explicit prefetch; repeat first access across five fresh allocations.
5. **Completed:** Use Nsight Systems to confirm host-to-device and device-to-host managed-memory migrations. Do not use profiler-inflated times as performance results.
6. **Completed:** Compare mapped reads, full H2D staging, and managed prefetch for 1%, 10%, 25%, and 100% strided access with 1, 2, and 4 passes. Evict L2 before every timed trial; all checksums passed.
7. **Next:** Add a bounded GPU-side cache path that gathers selected mapped-host values into compact VRAM, then reuses the compact set. Compare it with direct access and full prefetch for the same selectivity/reuse matrix.
8. **Then:** Inspect project-local llama.cpp and authorized model availability. Compare its unified-memory flag with ordinary offload using the same workload only when the model and run fit the resource limits.
9. Build an overlapped chunk-prefetch pipeline only if the cache and reuse measurements show a useful region.

## Guard behavior

Moderate desktop activity and an idle CUDA context do not automatically block a short bounded run. The guard defers on sustained heavy per-process GPU activity, repeated global utilization at or above 85%, temperature at or above 80 C, insufficient RAM/VRAM headroom, or invalid input limits. It never stops unrelated processes.
