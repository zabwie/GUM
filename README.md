# GPU Unified Memory Research

An isolated, reproducible study of GPU access to system RAM for local inference. The work compares mapped pinned host memory, pinned H2D staging, CUDA managed memory, explicit VMM mappings, and VRAM on an RTX 3060, including synthetic Q4_0 matvecs, one real model-tensor slice, and a small quantized LLM.

## Current status

- Hardware discovery is recorded in [HARDWARE.md](HARDWARE.md).
- CUDA's memory model and constraints are in [FINDINGS.md](FINDINGS.md).
- The initial falsifiable experiment is specified in [HYPOTHESES.md](HYPOTHESES.md).
- The first benchmark is [benchmarks/memory_paths.cu](benchmarks/memory_paths.cu).
- Its resource gate is [scripts/run_when_idle.sh](scripts/run_when_idle.sh).
- Valid 16–128 MiB streaming measurements now exist for mapped host, H2D copy-only, H2D plus scan, and VRAM paths; five-sample results and caveats are in [BENCHMARKS.md](BENCHMARKS.md).
- A 64 MiB managed-memory test measured CPU-first GPU access, repeated access, explicit prefetch, and page migrations. Nsight recorded 640 MiB H2D and 320 MiB D2H across the controlled trials; details are in [EXPERIMENTS.md](EXPERIMENTS.md).
- A dependent random-load benchmark compares mapped pinned host memory, managed demand/prefetch, and VRAM. At 960 MiB it found mapped throughput plateauing near 80 million loads/s and measured workload-specific managed-memory crossover points; all data still fit in VRAM. See [BENCHMARKS.md](BENCHMARKS.md).
- A cache-evicted strided-access benchmark compared mapped reads, full staging, and managed prefetch across 1–100% selectivity and 1–4 passes; sparse one-pass reads favored direct mapping, while denser or reused reads favored prefetch in this synthetic test.
- A compact GPU-side gather/cache path was measured: for four passes it beat both repeated mapped reads and full prefetch at 1–10% selectivity; full prefetch remained faster for 25–100% access.
- Synthetic Q4_0 matvecs at 32–128 MiB favored managed prefetch over mapped reads and full staging. A 47.8 MiB Q4_0 tensor slice from the local model also passed CPU-reference validation; its input vectors were synthetic. Double-buffered chunk copies helped versus serialized chunks, but reloading the tensor on every pass was slower than reusing a fully prefetched copy.
- Newer random-load and Q4_0 measurements now reach 960 MiB while remaining within the project’s 2 GiB VRAM and 4 GiB host-RAM limits. The dependent-load and 960 MiB Q4_0 tables are in [BENCHMARKS.md](BENCHMARKS.md); all test data still fit in VRAM.
- A mixed-residency GPU kernel now processes 1 GiB of VRAM and 2 GiB of mapped pinned RAM in one launch, with GPU initialization and validation. All checks pass; the host region is about 30 times slower than VRAM by logical bandwidth, and the combined path did not improve this streaming workload.
- A small CUDA VMM library now maps device and host physical allocations into one contiguous GPU-visible range. A 64 MiB device plus 128 MiB host probe passed full-range GPU validation; the queried minimum granularity is 2 MiB.
- End-to-end inference has now been measured on Qwen3-0.6B Q4_0. All 29 model layers were offloaded in both llama.cpp modes, and both returned the correct arithmetic answer. Five-repetition decode means were 383.1 tokens/s with ordinary GPU allocation and 381.8 with the managed-memory switch; this fitting-in-VRAM model does not test oversubscription or show a performance gain.
- Actual working sets beyond VRAM remain untested. The 13 GiB managed-allocation experiment is prepared as a gated next step; its required resource-budget revision is listed in OVERSUBSCRIPTION_PLAN.md. Do not launch it before explicit approval.

Build with the installed CUDA 13.2 toolkit:

```sh
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/memory_paths.cu -o benchmarks/memory_paths
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/managed_paths.cu -o benchmarks/managed_paths
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/reuse_paths.cu -o benchmarks/reuse_paths
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/q4_matvec.cu -o benchmarks/q4_matvec
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/host_latency.cu -o benchmarks/host_latency
```

Run only through the idle/resource gate:

```sh
scripts/run_when_idle.sh                         # mapped/staged/VRAM paths, 64 MiB
MIB=128 REPEATS=5 scripts/run_when_idle.sh      # select another bounded size
BENCHMARK=managed_paths scripts/run_when_idle.sh # managed first-touch/prefetch
BENCHMARK=reuse_paths scripts/run_when_idle.sh   # selectivity/reuse comparison
BENCHMARK=q4_matvec MIB=128 scripts/run_when_idle.sh # synthetic Q4_0 matvec
BENCHMARK=q4_matvec MIB=960 REPEATS=5 scripts/run_when_idle.sh # largest bounded synthetic Q4_0 case
BENCHMARK=host_latency MIB=960 REPEATS=5 scripts/run_when_idle.sh --steps 4096 --chains 4096
```

Set `CHUNK_MIB=4..64` for the serialized versus double-buffered chunk comparison. A bounded real-tensor example, using the local GGUF without modifying it, is recorded in [the raw run](results/q4-model-tensor-rerun-20261108.txt).

The benchmarks default to a 64 MiB buffer and five measured repetitions, capped at 256 MiB. The resource guard samples GPU utilization, temperature, power, free VRAM, available RAM, compute-client memory, and per-process SM activity. It permits idle CUDA contexts and moderate desktop activity, and defers sustained heavy GPU use, temperature at or above 80 C, or insufficient headroom. It never stops unrelated processes.
The 256 MiB cap applies to memory_paths, managed_paths, and reuse_paths; q4_matvec and host_latency support inputs up to 960 MiB. The runner and Q4 benchmark reject plans above the 2 GiB device-allocation cap; Q4's full 960 MiB input fits with the default 16 MiB chunks.
The host_latency benchmark has a separate 960 MiB input cap; its maximum guarded plan is 1,937 MiB device memory and about 2,880 MiB host memory, within the project’s 2 GiB VRAM and 4 GiB system-RAM limits.

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

The over-VRAM run is still gated; see [OVERSUBSCRIPTION_PLAN.md](OVERSUBSCRIPTION_PLAN.md) for the exact requested budget. See [AGENT_STATE.md](AGENT_STATE.md) before resuming work.
