# Next steps

1. **Completed:** Compile `benchmarks/memory_paths.cu` with CUDA 13.2; no global CUDA state changed.
2. **Completed:** Query device attributes. The bandwidth/checksum run is still pending until GPU utilization falls below the guard threshold and no CUDA compute process is active.
3. Append raw samples and checksums to `BENCHMARKS.md`/`results/`, then update `AGENT_STATE.md`.
4. If pinned mapping works, compare sizes and access patterns; add a managed-memory case with explicit residency controls and separate cold-migration from warm access.
5. Use measured transfer bytes per token and access/reuse pattern to decide whether a cache/prefetch prototype is worth implementing.

Do not run the GPU experiment while the resource guard rejects it.
