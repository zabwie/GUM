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

The initial GPU snapshot showed 826 MiB used, 54% utilization, 50 °C, and graphics processes for Xorg, Cinnamon, ChatGPT, a browser, and OBS. The first resource-gate snapshot showed 53% utilization, 11,213 MiB free VRAM, 20,038 MiB available RAM; a later attempt showed 32% utilization, 11,202 MiB free VRAM, and 20,178 MiB available RAM. Both reported no CUDA compute processes. The gate deferred the benchmark at both utilization levels.

The CUDA 13.2 runtime capability-only query completed without allocating benchmark buffers or launching a kernel. It reported driver/runtime API level 13.2, compute capability 8.6, mapped-host support, unified addressing, concurrent managed access, pageable access, and software-coherent pageable access. It reported no host-native atomic support.

### Still to establish

- Actual CUDA device attributes (`canMapHostMemory`, unified addressing, concurrent managed access, pageable-memory access, and host page-table use).
- Effective host DRAM configuration/speed; no DIMM SPD or privileged firmware query was used.
- Current operating-system distribution string and precise driver/kernel module feature support beyond the reported HMM mode.
- Measured PCIe, mapped-host, managed-memory, and staging bandwidth/latency.

### Capacity ceiling

PCIe 4.0 x16 has a nominal one-direction line-rate ceiling of about 31.5 GB/s before protocol and workload overhead. This is a theoretical link limit, not a measured result and not a prediction that host-memory kernels will achieve it.
