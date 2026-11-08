# Next steps

1. **Completed:** Query CUDA capabilities and execute checksum-validated mapped-host, staged H2D, VRAM, and managed-memory scans.
2. **Completed:** Isolate H2D copy-only bandwidth; repeat VRAM scans with three warm-ups.
3. **Completed:** Measure 16, 24, 32, 64, and 128 MiB sequential scans. Staging was nearly tied at 16 MiB and faster at 24 MiB and above in these runs.
4. **Completed:** Compare managed CPU-first access, repeated GPU access, and explicit prefetch; repeat first access across five fresh allocations.
5. **Completed:** Use Nsight Systems to confirm host-to-device and device-to-host managed-memory migrations. Do not use profiler-inflated times as performance results.
6. **Completed:** Compare mapped reads, full H2D staging, and managed prefetch for 1%, 10%, 25%, and 100% strided access with 1, 2, and 4 passes. Evict L2 before every timed trial; all checksums passed.
7. **Completed:** Gather selected mapped-host values into compact VRAM and compare against mapped reads, full H2D, and managed prefetch across selectivity and reuse. The cache helped at 1–10% with repeated passes in the synthetic gather.
8. **Completed:** Measure synthetic Q4_0 matvecs at 32, 64, and 128 MiB, then validate a bounded GEMV on a real 47.812 MiB model tensor slice. Managed prefetch won among host-backed paths when the matrix fit in VRAM.
9. **Completed:** Add serialized and double-buffered chunk copies, then sweep 4–64 MiB chunks on the same real tensor. Double buffering helped versus serialized chunks, but reloading weights each pass lost to one-time staging/prefetch; no stable chunk-size optimum was established.
10. **Completed:** Check whether CUDA can register read-only file-backed model pages. The device reports `cudaDevAttrHostRegisterReadOnlySupported=0`; the test skipped mapping before touching the model file.
11. **Completed for a fitting model:** Run Qwen3-0.6B Q4_0 in ordinary and managed-allocation modes. Both answers were correct and all 29 layers were offloaded; decode rates were effectively tied. This does not test oversubscription.
12. **Completed:** Add dependent random pointer chasing, with single-chain latency and disjoint multi-chain comparisons at 1–960 MiB; validate every output.
13. **Completed:** Measure the 960 MiB managed-memory chain-count crossover and profile 1, 256, and 4,096 chain migration traffic. At high chain counts, demand migration touches pages across nearly the full allocation.
14. **Completed:** Extend synthetic Q4_0 GEMV comparisons to 256 and 960 MiB. Managed prefetch remains faster than mapped or full H2D for these fitting-in-VRAM workloads; all CPU-reference checks passed.
15. **Next:** Compare prefetching only pages reached by the pointer chains against direct mapping, demand migration, and full-allocation prefetch at sparse and dense chain counts. Use the current results to test whether reduced migration bytes can also avoid the observed GPU-fault overhead.

16. **Completed:** Verify 1 GiB device plus 2 GiB mapped-host access in one GPU kernel and record PCIe activity with attribution limits.
17. **Completed:** Verify a contiguous mixed-backed CUDA VMM range on the RTX 3060 with zero GPU validation errors.
18. **Completed:** Run an actual small quantized LLM and compare ordinary CUDA allocation with the llama.cpp managed-memory switch.
19. **Completed (E026, user-approved):** Run the 13 GiB single-managed-allocation oversubscription experiment using the resource caps in OVERSUBSCRIPTION_PLAN.md. Do not execute before approval.
20. **Completed (synthetic, E025):** Page-aware residency manager with GPU access counters, hot-page prefetch, eviction under VRAM pressure and correctness checks; 3.4x vs direct mapped access on a 90/10 skewed trace, zero errors. Next: slab allocator instead of per-page cudaMalloc, then apply to a real over-VRAM workload.

## Guard behavior

Moderate desktop activity and an idle CUDA context do not automatically block a short bounded run. The guard defers on sustained heavy per-process GPU activity, repeated global utilization at or above 85%, temperature at or above 80 C, insufficient RAM/VRAM headroom, or invalid input limits. Q4_0 runs account for the full matrix, two configured chunk buffers, and scratch allowance; `CHUNK_MIB` is bounded to 4–64, and combinations above 2 GiB are rejected. The runner never stops unrelated processes.
For Q4_0 and host_latency runs, `MIB` is bounded at 960; the allocation estimate and Q4 executable enforce the 2 GiB VRAM cap, with host-RAM plans within the 4 GiB project limit.
