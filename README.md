# GPU Unified Memory Research

An isolated, reproducible study of GPU access to system RAM for local inference. The initial microbenchmarks compare mapped pinned host memory, pinned H2D staging, CUDA managed memory, and VRAM scans on an RTX 3060.

## Current status

- Hardware discovery is recorded in [HARDWARE.md](HARDWARE.md).
- CUDA's memory model and constraints are in [FINDINGS.md](FINDINGS.md).
- The initial falsifiable experiment is specified in [HYPOTHESES.md](HYPOTHESES.md).
- The first benchmark is [benchmarks/memory_paths.cu](benchmarks/memory_paths.cu).
- Its resource gate is [scripts/run_when_idle.sh](scripts/run_when_idle.sh).
- Valid 16–128 MiB streaming measurements now exist for mapped host, H2D copy-only, H2D plus scan, and VRAM paths; five-sample results and caveats are in [BENCHMARKS.md](BENCHMARKS.md).
- A 64 MiB managed-memory test measured CPU-first GPU access, repeated access, explicit prefetch, and page migrations. Nsight recorded 640 MiB H2D and 320 MiB D2H across the controlled trials; details are in [EXPERIMENTS.md](EXPERIMENTS.md).
- These are synthetic float4 scans, not LLM inference. Inference throughput, cache benefit, and model-scale oversubscription remain unmeasured.

Build with the installed CUDA 13.2 toolkit:

```sh
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/memory_paths.cu -o benchmarks/memory_paths
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/managed_paths.cu -o benchmarks/managed_paths
```

Run only through the idle/resource gate:

```sh
scripts/run_when_idle.sh                         # mapped/staged/VRAM paths, 64 MiB
MIB=128 REPEATS=5 scripts/run_when_idle.sh      # select another bounded size
BENCHMARK=managed_paths scripts/run_when_idle.sh # managed first-touch/prefetch
```

The benchmarks default to a 64 MiB buffer and five measured repetitions, capped at 256 MiB. The resource guard samples GPU utilization, temperature, power, free VRAM, available RAM, compute-client memory, and per-process SM activity. It permits idle CUDA contexts and moderate desktop activity, and defers sustained heavy GPU use, temperature at or above 80 C, or insufficient headroom. It never stops unrelated processes.

To collect Unified Memory page-fault and migration data with Nsight Systems, run the managed benchmark through the same guard:

```sh
PROFILE=um_faults BENCHMARK=managed_paths MIB=64 REPEATS=5 scripts/run_when_idle.sh
nsys stats --report um_sum,um_total_sum \
  results/managed-memory-faults-64mib-20261108-022719.nsys-rep
```

Profiled timings are not used for performance comparisons because fault tracing adds overhead. The `.nsys-rep` file is the raw trace; the adjacent text records the generated summary.

The capability-only query allocates no benchmark buffers and launches no kernel:

```sh
benchmarks/memory_paths --capabilities-only
```

See [AGENT_STATE.md](AGENT_STATE.md) before resuming work.
