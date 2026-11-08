# Next steps

1. **Completed:** Query CUDA capabilities and execute checksum-validated mapped-host, staged H2D, VRAM, and managed-memory scans.
2. **Completed:** Isolate H2D copy-only bandwidth; repeat VRAM scans with three warm-ups.
3. **Completed:** Measure 16, 24, 32, 64, and 128 MiB sequential scans. Staging was nearly tied at 16 MiB and faster at 24 MiB and above in these runs.
4. **Completed:** Compare managed CPU-first access, repeated GPU access, and explicit prefetch; repeat first access across five fresh allocations.
5. **Completed:** Use Nsight Systems to confirm host-to-device and device-to-host managed-memory migrations. Do not use profiler-inflated times as performance results.
6. **Next:** Implement a 64 MiB reuse/selectivity benchmark. Compare mapped-host reads with GPU prefetch plus local reads for 1, 2, and 4 passes over full, 25%, 10%, and 1% of the data. Record useful bytes, migration bytes, checksums, and total time to identify when staging pays for itself.
7. **Then:** Inspect the project-local availability of llama.cpp and an authorized model. Compare its existing unified-memory flag against the same ordinary-offload configuration only when the model and run fit the stated resource limits.
8. **After access-pattern results:** Prototype a bounded GPU-side cache or overlapped prefetch pipeline only if a measured reuse/selectivity region justifies it.

## Guard behavior

Moderate desktop activity and an idle CUDA context do not automatically block a short bounded run. The guard defers on sustained heavy per-process GPU activity, repeated global utilization at or above 85%, temperature at or above 80 C, insufficient RAM/VRAM headroom, or invalid input limits. It never stops unrelated processes.
