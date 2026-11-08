# Benchmark results

No CUDA benchmark has been run yet. The initial runtime experiment is built in `benchmarks/memory_paths.cu`; its guarded launch is `scripts/run_when_idle.sh`.

Two guarded attempts were deferred at 53% and 32% GPU utilization. Neither launched a kernel.

The capability-only observation is recorded in `results/capabilities-20261108.txt`. It contains no bandwidth or latency measurements.

Results must include repeated samples, warm-up, checksum correctness, input size, buffer type, GPU/driver/toolkit versions, runtime capability attributes, and contemporaneous memory/utilization. Do not substitute the PCIe theoretical rate for a measured number.
