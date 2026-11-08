# Experiment log

## E000 — read-only baseline discovery

- **Date:** 2026-11-08
- **Question:** What hardware/runtime is available, and is the GPU quiet enough for a controlled first run?
- **Procedure:** Read CPU, RAM, PCIe sysfs, `nvidia-smi`, installed compiler versions, and NVIDIA address mode. No hardware settings were changed and no CUDA kernel was run.
- **Observations:** See [HARDWARE.md](HARDWARE.md). RTX 3060 12 GiB, i7-12700K, PCIe 4.0 x16 at 16 GT/s, Linux kernel 7.0.0-38, NVIDIA driver 595.99.02, CUDA toolkits 12.0 and 13.2, HMM address mode. About 19 GiB RAM available; 3.9 GiB memlock allowance.
- **Resource decision:** GPU showed 54% utilization and active graphics workloads. Deferred GPU execution to avoid contaminating measurements or disrupting the desktop. No performance result is claimed.
- **Conclusion:** The HMM path is exposed by the installed driver, but actual runtime attributes and memory behavior remain to be queried/measured. The mapped pinned-host path is the first controlled test because it keeps the source physically host-resident.

## E001 — CUDA runtime capability query

- **Date:** 2026-11-08
- **Question:** Does the CUDA runtime expose mapped-host memory and full pageable-system-memory unified access on this GPU?
- **Procedure:** Built the project probe with `/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_86`; ran `benchmarks/memory_paths --capabilities-only`. This mode queries properties and attributes only: no benchmark buffers, no benchmark kernel, and no timing samples.
- **Observed output:** RTX 3060, CC 8.6, 28 SMs; driver/runtime API 13.2. `canMapHostMemory=1`, `unifiedAddressing=1`, `concurrentManagedAccess=1`, `pageableMemoryAccess=1`, `pageableMemoryAccessUsesHostPageTables=0`, `hostNativeAtomicSupported=0`.
- **Interpretation:** Mapped pinned host access is supported. The attribute combination is consistent with full Linux HMM unified-memory support using software coherence. It does not establish placement, bandwidth, or inference performance.
- **Conclusion:** H1's capability prerequisite is supported, but its kernel correctness part remains untested. The 64 MiB performance run remains deferred by the guard because GPU utilization was 53%.
