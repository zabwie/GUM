# Agent state

## Objective

Measure and improve GPU inference throughput when model state exceeds 12 GiB VRAM, using system RAM only through mechanisms that are supported and measured on this machine.

## Current verified baseline

- RTX 3060, 12 GiB, GA106; i7-12700K; 31 GiB RAM, about 19 GiB available at discovery.
- PCIe link 16 GT/s x16; Linux kernel 7.0.0-38; NVIDIA 595.99.02.
- CUDA 12.0 and 13.2 installed; HMM addressing mode reported.
- CUDA 13.2 reports CC 8.6, `canMapHostMemory=1`, `concurrentManagedAccess=1`, `pageableMemoryAccess=1`, and software-coherent HMM (`pageableMemoryAccessUsesHostPageTables=0`). The capability-only query ran no benchmark kernel.
- GPU utilization was 54% with desktop/browser/OBS graphics processes; no CUDA compute process. GPU experiment deferred.

## Current implementation

- Initial CUDA streaming benchmark source: `benchmarks/memory_paths.cu`.
- Guarded runner: `scripts/run_when_idle.sh`.
- Intended comparison: mapped pinned host read, pinned H2D staging plus scan, and VRAM-resident scan.

## Results

- No benchmark has run; there are no measured bandwidth or latency values.
- Capability query is verified. The mapped pinned-host kernel and its checksum are not yet verified.
- The benchmark is capped at 256 MiB and defaults to 64 MiB pinned host memory plus 64 MiB device memory.

## Active hypothesis

Mapped pinned host memory is kernel-accessible, but explicit staging will likely deliver higher effective bandwidth for a contiguous one-pass scan. Both are untested on this runtime.

## Immediate action

Run the mapped/staged/VRAM benchmark only via the resource-gated script when the GPU is sufficiently idle.
