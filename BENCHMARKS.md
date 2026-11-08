# Benchmark results

Streaming-memory rates are five-sample medians of synthetic sequential float4 reductions on the RTX 3060 using CUDA 13.2, with three warm-ups per path after E003. Later sections report separate workloads and list sample counts beside each result. “Effective input rate” is buffer bytes divided by elapsed time; it is not a hardware-link measurement or inference throughput. All measured checksums passed.

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

## Access selectivity and reuse at 64 MiB

The benchmark reads every 1st, 4th, 10th, or 100th float4 across the buffer, then repeats that read 1 or 4 times. Each H2D or managed-prefetch path moves the full 64 MiB once per trial. A 16 MiB device scan runs before each timed sample to evict prior L2 data; repeated passes within a sample can reuse data.

| Access fraction | Passes | Mapped host | Full H2D + scans | Managed prefetch + scans |
|---:|---:|---:|---:|---:|
| 100% | 1 | 5.561 ms | 4.742 ms | 3.130 ms |
| 100% | 4 | 21.389 ms | 5.334 ms | 3.661 ms |
| 25% | 1 | 4.460 ms | 4.638 ms | 2.974 ms |
| 25% | 4 | 17.891 ms | 4.931 ms | 3.267 ms |
| 10% | 1 | 2.059 ms | 4.599 ms | 2.915 ms |
| 10% | 4 | 8.730 ms | 5.373 ms | 2.998 ms |
| 1% | 1 | 0.380 ms | 4.687 ms | 2.894 ms |
| 1% | 4 | 1.260 ms | 4.639 ms | 2.949 ms |

For this synthetic strided gather, mapped access wins at 1% for all tested pass counts and at 10% for one pass. Managed prefetch wins at 10% with two or more passes and at 25% or 100% even for one pass. The benchmark tests full-buffer movement; it does not yet test packing only selected values into a compact cache. The useful-bandwidth column in the raw output counts requested float4 values, not physical PCIe transaction bytes.

Full samples and preflight data are in [the raw reuse run](results/reuse-selectivity-cold-64mib-20261108.txt); all checksums passed.

## Compact GPU-side cache prototype

The cache path gathers the selected float4 values from mapped host memory into a compact VRAM buffer once, then performs the requested local scans. The table shows four-pass medians from five samples per configuration; each run moved the selected payload into the cache, while managed prefetch and H2D staging moved the full 64 MiB.

| Access fraction | Mapped host | Full H2D staging | Full managed prefetch | Compact GPU cache |
|---:|---:|---:|---:|---:|
| 1% | 1.172 ms | 4.565 ms | 2.957 ms | 0.302 ms |
| 10% | 8.167 ms | 4.680 ms | 2.985 ms | 2.123 ms |
| 25% | 17.777 ms | 4.917 ms | 3.357 ms | 4.500 ms |
| 100% | 21.678 ms | 5.529 ms | 3.685 ms | 6.067 ms |

At one pass, cache gather was about tied with direct mapping at 1% (0.297 vs 0.295 ms) and 10% (2.048 vs 2.038 ms). At 1–10%, two or four passes amortized the gather and the compact cache beat both repeated mapped reads and full-buffer prefetch. For 25–100% access, full managed prefetch remained faster. `gathered_payload_MiB` is useful payload size; actual PCIe transaction bytes are not measured. Full results are in [the cache run](results/reuse-cache-64mib-20261108.txt).

## Q4_0 matrix-vector paths

The first Q4_0 benchmark uses synthetic packed weights and vectors, 4,096 rows, and the Q4_0 18-byte block layout. All weights fit in VRAM. Each staged or managed trial moves the matrix once, then reuses it for all passes; mapped reads cross the host link again on each pass. Five samples per path passed the CPU-reference check.

| Matrix | Passes | Mapped host | Pinned H2D + matvec | Managed prefetch + matvec | VRAM resident |
|---:|---:|---:|---:|---:|---:|
| 32 MiB | 1 | 3.022 ms | 2.535 ms | 1.749 ms | 0.240 ms |
| 32 MiB | 4 | 12.084 ms | 3.241 ms | 2.456 ms | 0.953 ms |
| 64 MiB | 1 | 6.328 ms | 5.139 ms | 3.430 ms | 0.504 ms |
| 64 MiB | 4 | 25.603 ms | 6.462 ms | 4.778 ms | 1.860 ms |
| 128 MiB | 1 | 14.390 ms | 10.420 ms | 7.010 ms | 1.203 ms |
| 128 MiB | 4 | 57.725 ms | 13.868 ms | 10.560 ms | 4.550 ms |
| 256 MiB | 1 | 28.442 ms | 20.596 ms | 13.506 ms | 2.271 ms |
| 256 MiB | 4 | 126.477 ms | 28.161 ms | 21.005 ms | 9.977 ms |
| 960 MiB | 1 | 108.887 ms | 77.317 ms | 51.965 ms | 8.604 ms |
| 960 MiB | 4 | 440.832 ms | 103.612 ms | 76.531 ms | 38.655 ms |

The isolated transfer-only medians were 14.849, 14.631, and 14.747 GB/s for pinned H2D at 32, 64, and 128 MiB; managed prefetch measured 22.105, 23.180, and 23.268 GB/s. Those are results for this procedure and CUDA stack, not general interface rates. Raw runs: [32 MiB](results/q4-transfer-isolation-32mib-20261108-030402.txt), [64 MiB](results/q4-transfer-isolation-64mib-20261108-030311.txt), and [128 MiB](results/q4-matvec-128mib-20261108-030701.txt).

The 256 MiB and 960 MiB synthetic cases also passed every CPU-reference validation. At 256 MiB, transfer-only medians were 18.293 ms (14.670 GB/s) for pinned H2D and 11.334 ms (23.678 GB/s) for managed prefetch. At 960 MiB they were 68.300 ms (14.738 GB/s) and 41.978 ms (23.980 GB/s). The recorded 960 MiB invocation planned 1,968 MiB device/1,920 MiB host memory and reported 1,974.728 MiB actual device buffers. A guard review found the old estimate omitted about 7 MiB of input/output scratch; the current planner reserves 8 MiB and estimates 1,976 MiB for the measured default 16 MiB chunks. The executable rejects exact device-buffer plans above 2,048 MiB. Double-buffering improved 960 MiB chunk movement versus serialized chunks (69.527 vs 81.818 ms at one pass; 278.928 vs 321.219 ms at four), but full managed prefetch plus matvec was faster at both pass counts. Raw runs: [256 MiB](results/q4-matvec-256mib-20261108.txt) and [960 MiB](results/q4-matvec-960mib-20261108.txt).

## Actual model tensor and chunk-copy pipeline


The benchmark read only the 47.812 MiB `blk.0.ffn_down.weight` Q4_0 tensor from the local 27B GGUF; the file itself was not changed. The matrix is 5,120 rows by 17,408 values. Input vectors were synthetic, so this is a real-weight GEMV microbenchmark, not inference.

| Passes | Mapped host | Full H2D staging | Managed prefetch | VRAM resident | 16 MiB chunks, serial | 16 MiB chunks, double-buffered |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.168 ms | 3.833 ms | 2.532 ms | 0.351 ms | 3.798 ms | 3.568 ms |
| 4 | 17.324 ms | 4.765 ms | 3.473 ms | 1.303 ms | 15.398 ms | 14.286 ms |

All paths passed validation after the CPU reference was corrected to decode each block's half-precision scale. The first attempt, [retained as a failed diagnostic](results/q4-model-tensor-20261108.txt), assumed the synthetic 1/16 scale and is excluded from performance analysis. The validated rerun is [here](results/q4-model-tensor-rerun-20261108.txt).

The same tensor was then measured with 4, 8, 16, 32, and 64 MiB double buffers. The table reports five-sample medians and ranges; each run had a separate resource preflight, with sampled GPU utilization maxima from 18% to 51%.

| Chunk size | Chunks | One-pass median (range) | Four-pass median (range) | Four-pass weight movement |
|---:|---:|---:|---:|---:|
| 4 MiB | 12 | 3.603 ms (3.508–3.853) | 14.063 ms (13.738–14.295) | 191.25 MiB |
| 8 MiB | 6 | 3.472 ms (3.465–3.532) | 13.788 ms (13.727–13.946) | 191.25 MiB |
| 16 MiB | 3 | 4.697 ms (4.058–5.946) | 19.387 ms (17.276–21.249) | 191.25 MiB |
| 32 MiB | 2 | 4.580 ms (3.670–5.182) | 16.887 ms (16.214–17.970) | 191.25 MiB |
| 64 MiB | 1 | 4.335 ms (4.284–5.674) | 19.722 ms (17.738–20.923) | 191.25 MiB |

8 MiB had the lowest observed medians, but run-to-run conditions varied and these separate runs do not establish an optimum. In the 8 MiB run, the four-pass chunk path took 13.788 ms versus 4.813 ms for full H2D staging and 3.524 ms for managed prefetch: chunk streaming transferred the tensor four times, while the full-buffer paths moved it once. The pipeline helped versus serialized chunks, but did not beat reuse of a fully staged or prefetched matrix. Raw chunk-size runs: [4 MiB](results/q4-model-tensor-chunk4-20261108.txt), [8 MiB](results/q4-model-tensor-chunk8-20261108.txt), [16 MiB](results/q4-model-tensor-chunk16-20261108.txt), [32 MiB](results/q4-model-tensor-chunk32-20261108.txt), and [64 MiB](results/q4-model-tensor-chunk64-20261108.txt).

## Dependent random pointer chasing

These results use a randomized pointer cycle and a dependent-load kernel. A 16 MiB scan evicted prior L2 data before each timed sample; correctness was checked against a CPU traversal. For multiple chains, “ns/load” is aggregate elapsed time divided by all chain loads, not the latency of one dependent load.

### One dependent chain

| Working set | Mapped host median (range) | VRAM median (range) |
|---:|---:|---:|
| 1 MiB | 586.47 ns (585.88–649.63) | 217.94 ns (217.78–332.10) |
| 4 MiB | 726.37 ns (724.72–831.26) | 257.25 ns (256.63–311.54) |
| 32 MiB | 749.37 ns (748.31–857.85) | 269.99 ns (269.72–324.28) |
| 64 MiB | 753.81 ns (753.19–871.38) | 273.78 ns (273.44–321.72) |
| 960 MiB | 1,108.27 ns (1,094.25–1,109.51) | 278.59 ns (278.25–279.00) |

Each row is the median of five samples. Mapped-host dependent access was about 2.7–2.8× the VRAM time at 1–64 MiB and about 4.0× at 960 MiB.

### Independent-chain scaling at 64 MiB

| Chains | Mapped host (ns/load) | Mapped throughput (million loads/s) | VRAM (ns/load) | VRAM throughput (million loads/s) |
|---:|---:|---:|---:|---:|
| 1 | 759.75 | 1.316 | 274.70 | 3.640 |
| 32 | 32.66 | 30.622 | 11.06 | 90.395 |
| 256 | 8.60 | 116.261 | 1.48 | 676.801 |
| 1,024 | 8.03 | 124.601 | 0.47 | 2,114.610 |
| 4,096 | 8.35 | 119.791 | 0.29 | 3,407.654 |

Mapped-host aggregate throughput rose with concurrency, then plateaued around 120–125 million loads/s. The VRAM path continued to scale across the measured range.

### Managed-memory size and chain-count comparisons

At 4,096 chains × 4,096 dependent steps, the 64, 256, and 960 MiB working sets produced these median elapsed times:

| Working set | Mapped host (ns/load; million loads/s) | Managed demand (ms) | Full prefetch + kernel (ms) | VRAM resident (ms) |
|---:|---:|---:|---:|---:|
| 64 MiB, 10 samples | 8.27; 120.977 | 9.069 | 9.852 | 4.931 |
| 256 MiB, 5 samples | 8.45; 118.330 | 21.753 | 16.693 | 5.121 |
| 960 MiB, 5 samples | 12.56; 79.612 | 66.444 | 47.305 | 5.171 |

At 960 MiB, the chain-count sweep measured these medians. Each step count was 4,096; the explicit-prefetch path moves the full 960 MiB allocation on every sample.

| Chains | Total loads | Mapped host (ms) | Managed demand (ms) | Full prefetch + kernel (ms) | VRAM resident (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 4.539 | 45.168 | 43.041 | 1.141 |
| 256 | 1,048,576 | 12.498 | 65.094 | 43.392 | 1.553 |
| 512 | 2,097,152 | 25.229 | 63.196 | 43.675 | 1.683 |
| 896 | 3,670,016 | 43.237 | 62.453 | 43.872 | 1.930 |
| 1,024 | 4,194,304 | 55.399 | 66.912 | 44.074 | 2.010 |
| 1,536 | 6,291,456 | 72.147 | 62.291 | 44.270 | 2.457 |
| 2,048 | 8,388,608 | 97.530 | 64.003 | 44.852 | 2.973 |
| 4,096 | 16,777,216 | 210.737 | 66.444 | 47.305 | 5.171 |

At 896 chains, mapped access and full prefetch were close. Full prefetch was faster by 1,024 chains; demand migration crossed later, between 1,024 and 1,536 chains. These observed brackets are specific to this kernel, allocation, and GPU; the 896-chain comparison is near a tie. The 64 MiB demand and prefetch ranges overlap, so that run does not establish a winner.

One 960 MiB Nsight trace covered two warmups and one measured sample of both managed paths. It recorded 5,932.446 decimal MB H2D and 5,033.165 MB D2H migration, with 2,880 CPU and 458 GPU page faults. The totals are close to six and five full 960 MiB movements respectively; the trace is aggregate and does not attribute bytes to individual arms. Profiling times are excluded.
Additional traces at one and 256 chains reported 3,823.829/1,817.969 MB H2D/D2H with 10,674 GPU faults, and 5,833.306/5,033.165 MB with 965 GPU faults, respectively. CPU faults totaled 2,880 in each trace. Together with the 4,096-chain trace, this indicates a chain-fanout effect in aggregate migration and fault counts, but the traces combine both managed arms and warmups. See E020/E022 in `EXPERIMENTS.md` for links and caveats.

All 960 MiB trials planned 1,936 MiB device and about 2,880 MiB host memory, within the project caps. The 960 MiB data set fits in the RTX 3060’s VRAM; these tests do not measure oversubscription or inference. Early multi-chain runs with overlapping chain starts are retained but excluded; the reported 64 MiB parallel results use disjoint chain segments.

Raw data: [serial 1 MiB](results/host-latency-1mib-20261108.txt), [4 MiB](results/host-latency-4mib-20261108.txt), [32 MiB](results/host-latency-32mib-20261108.txt), [64 MiB](results/host-latency-64mib-20261108.txt); [64 MiB, 1 chain](results/host-latency-spaced-64mib-c1-20261108.txt), [32 chains](results/host-latency-spaced-64mib-c32-20261108.txt), [256 chains](results/host-latency-spaced-64mib-c256-20261108.txt), [1,024 chains](results/host-latency-spaced-64mib-c1024-20261108.txt), and [4,096 chains](results/host-latency-spaced-64mib-c4096-20261108.txt); [64 MiB managed, 10 samples](results/host-managed-random-64mib-c4096-rep10-20261108.txt), [256 MiB managed](results/host-managed-random-256mib-c4096-20261108.txt), and [960 MiB managed](results/host-managed-random-960mib-c4096-20261108.txt).

960 MiB chain-count logs: [1](results/host-managed-random-960mib-c1-20261108.txt), [256](results/host-managed-random-960mib-c256-20261108.txt), [512](results/host-managed-random-960mib-c512-20261108.txt), [896](results/host-managed-random-960mib-c896-20261108.txt), [1,024](results/host-managed-random-960mib-c1024-20261108.txt), [1,536](results/host-managed-random-960mib-c1536-20261108.txt), [2,048](results/host-managed-random-960mib-c2048-20261108.txt), and [4,096](results/host-managed-random-960mib-c4096-20261108.txt). The migration trace is [here](results/unified-memory-faults-host_latency-960mib-20261108-105120.nsys-rep) with its [statistics](results/host-managed-random-profile-stats-960mib-20261108.txt).


## Reproducibility notes

- Hardware/runtime details are in [HARDWARE.md](HARDWARE.md).
- Raw outputs preserve per-sample timings, checksums, preflight resource samples, source revision, and command conditions.
- The 64 MiB VRAM scan had two outliers in the initial E003 run. Later 16–128 MiB samples were stable after three warm-ups.
- Do not infer LLM throughput, cache benefit, physical placement of every page, or a model-size threshold from these scans.
