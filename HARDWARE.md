# Hardware and software baseline

Observed 2026-11-08 before any GPU benchmark. Values are machine observations; the pasted brief's unverified RAM speed and previous availability estimate are not treated as current facts.

| Item | Observed value | Evidence |
|---|---|---|
| CPU | Intel Core i7-12700K, 12 cores / 20 logical CPUs | `lscpu` |
| Mainboard | ASUSTeK PRIME Z790-V AX | `/sys/class/dmi/id/board_name` |
| GPU | NVIDIA GeForce RTX 3060, GA106, 12,288 MiB VRAM | `nvidia-smi`, `lspci` |
| GPU PCI address | `0000:01:00.0`; IOMMU group 16 | sysfs, `lspci` |
| Negotiated PCIe link | 16.0 GT/s, x16 (PCIe 4.0 x16 signaling rate) | `/sys/bus/pci/devices/0000:01:00.0/current_link_{speed,width}` |
| NVIDIA driver | 595.99.02 | `nvidia-smi` |
| Driver's reported CUDA level | 13.2 | `nvidia-smi` header; this is not the selected compiler version |
| CUDA toolkits found | nvcc 12.0 at `/usr/bin/nvcc`; nvcc 13.2 at `/usr/local/cuda/bin/nvcc` | compiler version output |
| Kernel | Linux 7.0.0-38-generic, x86_64 | `uname -a` |
| NVIDIA addressing mode | HMM | `nvidia-smi -q` |
| CUDA compute capability | 8.6; 28 SMs | project capability-only runtime query |
| CUDA mapped host support | `canMapHostMemory=1` | project capability-only runtime query |
| Unified addressing | `unifiedAddressing=1` | project capability-only runtime query |
| Concurrent managed access | `concurrentManagedAccess=1` | project capability-only runtime query |
| Pageable system-memory access | `pageableMemoryAccess=1` | project capability-only runtime query |
| Host page tables used for pageable access | `pageableMemoryAccessUsesHostPageTables=0` | project capability-only runtime query |
| Host-native atomics | `hostNativeAtomicSupported=0` | project capability-only runtime query |
| System RAM | 31 GiB total, about 19 GiB available at discovery | `free -h` |
| Swap | 43 GiB configured, 0 used at discovery | `free -h` |
| Locked-memory limit | 3,977 MiB (`ulimit -l`) | shell limit |

The first resource-gate snapshots showed 53% and 32% global utilization with ample RAM/VRAM; those runs were deferred by an overly strict 15% utilization rule. Later bounded runs proceeded when sampled load, per-process activity, thermals, and headroom indicated low interference risk. An unrelated desktop app held a CUDA context using 98–126 MiB but had no measurable SM use in the sampled intervals; the guard records such clients and does not treat an idle context alone as active GPU load.

The CUDA 13.2 runtime capability-only query completed without allocating benchmark buffers or launching a kernel. It reported driver/runtime API level 13.2, compute capability 8.6, mapped-host support, unified addressing, concurrent managed access, pageable access, and software-coherent pageable access. It reported no host-native atomic support.

### Still to establish

- Effective host DRAM configuration/speed; no DIMM SPD or privileged firmware query was used.
- Current operating-system distribution string and precise driver/kernel module feature support beyond the reported HMM mode.
- Inference throughput, model-scale oversubscription behavior, and cache/prefetch benefit for real tensor access patterns.

Mapped-host and H2D streaming rates, managed-memory first-touch/prefetch timings, and Nsight migration totals are now measured; see [BENCHMARKS.md](BENCHMARKS.md) and [EXPERIMENTS.md](EXPERIMENTS.md).

### Capacity ceiling

PCIe 4.0 x16 has a nominal one-direction line-rate ceiling of about 31.5 GB/s before protocol and workload overhead. This is a theoretical link limit, not a measured result and not a prediction that host-memory kernels will achieve it.


### CUDA VMM capability check

A later CUDA Driver API probe reported virtual-memory-management support and host-backed VMM support on the RTX 3060. The minimum granularity for both device and host allocations was 2,097,152 bytes. The mixed-range experiment successfully mapped these distinct backings into adjacent segments of one GPU-visible address range; see E024 in EXPERIMENTS.md. The same successful probe does not imply that Linux HMM automatically implements the mapping policy.
