# GPU Unified Memory Research

An isolated, reproducible study of GPU access to system RAM for local inference. The first milestone is deliberately small: query CUDA's actual memory capabilities, then compare a streaming kernel over mapped pinned host RAM, host-to-device staging plus the same kernel, and a VRAM-resident scan.

## Current status

- Hardware discovery is recorded in [HARDWARE.md](HARDWARE.md).
- CUDA's memory model and constraints are in [FINDINGS.md](FINDINGS.md).
- The initial falsifiable experiment is specified in [HYPOTHESES.md](HYPOTHESES.md).
- The first benchmark is [benchmarks/memory_paths.cu](benchmarks/memory_paths.cu).
- Its resource gate is [scripts/run_when_idle.sh](scripts/run_when_idle.sh).
- A read-only CUDA capability query succeeded. The bandwidth benchmark is deferred while the GPU is busy; no performance result has been collected yet.

Build with the installed CUDA 13.2 toolkit:

```sh
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  benchmarks/memory_paths.cu -o benchmarks/memory_paths
```

Run only through the idle/resource gate:

```sh
scripts/run_when_idle.sh
```

The benchmark defaults to a 64 MiB buffer, five measured repetitions, and about 64 MiB of pinned RAM plus 64 MiB of VRAM. It caps buffer size at 256 MiB. The guard requires low GPU utilization, no CUDA compute process, at least 8 GiB available system RAM, and at least 2 GiB of free VRAM after the benchmark allocation.

The capability-only query allocates no benchmark buffers and launches no kernel:

```sh
benchmarks/memory_paths --capabilities-only
```

See [AGENT_STATE.md](AGENT_STATE.md) before resuming work.
