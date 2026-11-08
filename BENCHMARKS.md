# Benchmark results

All rates below are five-sample medians of a synthetic sequential float4 reduction on the RTX 3060, CUDA 13.2. The benchmark uses three warm-ups per path after E003. “Effective input rate” is buffer bytes divided by elapsed time; it is not a hardware-link measurement or inference throughput. All measured checksums passed.

## Streaming memory paths

| Buffer | Mapped pinned host | H2D copy only | H2D plus scan | VRAM scan |
|---:|---:|---:|---:|---:|
| 16 MiB | 14.091 GB/s | 14.782 GB/s | 14.173 GB/s | 311.520 GB/s |
| 24 MiB | 13.293 GB/s | 14.744 GB/s | 14.036 GB/s | 320.078 GB/s |
| 32 MiB | 12.977 GB/s | 14.682 GB/s | 13.952 GB/s | 326.965 GB/s |
| 64 MiB | 11.388 GB/s | 14.719 GB/s | 13.656 GB/s | 332.617 GB/s |
| 128 MiB | 12.121 GB/s | 14.749 GB/s | 14.141 GB/s | 336.730 GB/s |

At 16 MiB, mapped and staged medians were within 1%. Staging was faster at each tested size from 24 through 128 MiB. The observed crossover is between 16 and 24 MiB for this scan only. The 64 MiB E003 baseline was noisier: mapped 10.009, staged 13.693, and VRAM 326.710 GB/s; E004 repeated 64 MiB with three warm-ups and isolated H2D timing. See [EXPERIMENTS.md](EXPERIMENTS.md) for ranges, conditions, and raw logs.

The pinned H2D-only path stayed near 14.7 GB/s over 16–128 MiB. E004 measured 4.559 ms median for a 64 MiB copy and 4.914 ms for copy plus scan. The difference between separate medians is only an approximate scan cost.

## Managed-memory behavior at 64 MiB

| Operation | Median | Range | Effective input rate |
|---|---:|---:|---:|
| CPU-initialized first GPU scan, no GPU prefetch (five fresh allocations) | 5.201 ms | 5.172–5.286 ms | 12.903 GB/s |
| Repeated scan after first GPU access | 0.210 ms | 0.202–0.220 ms | 319.688 GB/s |
| Explicit prefetch to GPU | 2.901 ms | 2.815–3.102 ms | 23.133 GB/s |
| GPU prefetch plus scan | 3.108 ms | 3.026–3.306 ms | 21.593 GB/s |

The managed-memory results come from [the fresh-allocation run](results/managed-memory-fresh-64mib-20261108.txt). Nsight Systems separately confirmed 640 MiB H2D and 320 MiB D2H migration across the designed trials; the profiled timings are excluded because fault tracing adds overhead. This is not an inference comparison.

## Reproducibility notes

- Hardware/runtime details are in [HARDWARE.md](HARDWARE.md).
- Raw outputs preserve per-sample timings, checksums, preflight resource samples, source revision, and command conditions.
- The 64 MiB VRAM scan had two outliers in the initial E003 run. Later 16–128 MiB samples were stable after three warm-ups.
- Do not infer LLM throughput, cache benefit, physical placement of every page, or a model-size threshold from these scans.
