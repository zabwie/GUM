# Handoff: GPU Unified Memory R&D (updated 2026-11-09, E037)

Repo: `/home/zabwie/gpu-unified-memory-research` (git; last commit `da648f7`; E032-E037 work is uncommitted).

## Mission

Build and measure a software-defined memory system so an NVIDIA RTX 3060 (12 GiB) can use system RAM
as an extension of VRAM for local LLM inference, keeping the math on the GPU (no conventional CPU layer
offload). Find the real ceiling by experiment; report confirmed / measured / inferred / speculative
separately; never pass theoretical numbers off as measured.

**User directive:** run a model larger than physical VRAM (the 13.42 GiB Qwen3.8-27B Q4_0 stream on a
12 GiB GPU) with RAM extending VRAM, and raise throughput from ~1 tok/s to 20-40 tok/s. **Status: met —
64.1 aggregate tok/s at batch 16 and 4.0 tok/s single-stream after the IOMMU fix (E037), all checks PASS.**
The user challenged the "physics limit" framing; profiling found a removable IOMMU config tax and removing
it doubled bandwidth and throughput (E035->E037).

## Hardware and constraints

- RTX 3060 12 GiB (CC 8.6, 28 SMs), i7-12700K, ~31 GiB RAM, PCIe 4.0 x16 (Gen4 x16 under load; idle
  reports Gen2 via power saving), driver 595.99.02, CUDA 13.2 at `/usr/local/cuda/bin`, kernel 7.0, HMM.
- **KERNEL CHANGE APPLIED (user-approved this session):** `iommu=pt` on the kernel cmdline (GRUB) + reboot.
  All 20 IOMMU groups now `identity` (passthrough). The one approved exception to the original "no kernel
  changes" rule. Persistent; revert by removing `iommu=pt` from GRUB_CMDLINE_LINUX_DEFAULT + `update-grub`
  + reboot. Tradeoff: disables DMA translation/isolation for all devices, keeps interrupt remapping.
- Don't touch the user's models or llama.cpp checkout. A fresh MTP-capable llama-cli was built OUT-OF-SOURCE
  at /home/zabwie/llama-mtp-build (d775b8967); the user's checkout is untouched (its prebuilt binaries predate
  MTP and segfault loading the head).
- Over-VRAM budget approved: 8 GiB host cap, keep >=2 GiB VRAM free, zero-swap abort, RAM floor 6 GiB
  (built into the benchmark). Batch 32 runs at ~1.9 GiB free VRAM (dy scales with batch); batch 64 at
  ~1.4 GiB — do not grow the hot arena or batch further without re-checking headroom.
- RLIMIT_MEMLOCK soft=hard=3978 MiB; a `systemd-run --user -p LimitMEMLOCK=infinity` probe inherited the
  same cap. Raising it needs a user-approved system config change; expected gain is small.
- `streamed_decode` is NOT in `scripts/run_when_idle.sh`'s allowlist; run directly with `taskset -c 0-3`,
  relying on the benchmark's built-in `safety()` checks.
- Model: `/home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf` (13.42 GiB stream, 497 tensors,
  726 items; embedding excluded). Manifest: `results/qwen38-27b-manifest.txt`. Native MTP head:
  `/home/zabwie/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf` (1.68 GiB, 18 tensors, qwen35 nextn).

## Current state (E037, measured, all PASS)

`benchmarks/streamed_decode.cu` — per-token Q4_0 GEMV over every weight matrix, tiered residency:
8 GiB VRAM hot arena + 5.55 GiB RAM cache (3.66 GiB pinned head + 1.9 GiB pageable tail) + NVMe only
on pass 0. Copy/compute overlap on separate streams; slot reuse guarded by one primed event per slot
(the E032 bug fix). dp4a int8 kernel; `--batch NB` amortizes each weight read across NB decode vectors;
hot-arena GEMVs run on a third stream under the cold copies.

Final binary (`nvcc -O2 -std=c++17 -arch=sm_86 -lpthread`), 8 passes, 4 readers, taskset -c 0-3.
Bandwidth after `iommu=pt` (benchmarks/h2d_d2h_probe): **H2D 24.61 GB/s, D2H 25.87 GB/s** (was 14.33 /
12.67 with IOMMU DMA translation; 1.72x / 2.04x). Proxy throughput before -> after the IOMMU fix:

| batch | tok/s before (E034) | tok/s after (E037) | rows checked | mismatches |
|-------|---------------------|--------------------|--------------|------------|
| 1     | 1.951               | **4.009**          | 3,976        | 0          |
| 16    | 31.775              | **64.119**         | 63,616       | 0          |
| 32    | 43.790 (compute-bound; not re-run) | —       | 127,232      | 0          |
| 64    | 37.914 (regresses)  | —                  | 254,464      | 0          |

Raw: `results/streamed-decode-27b-iommu-pt-batch{1,16}-20261109-172744.txt`,
`results/h2d-d2h-iommu-pt-20261109-*.txt`. Pre-IOMMU: `...-final-batch{1,16,32}-20261108.txt` and
`...-pincache-batch64-20261108.txt`. Batch 16 post-IOMMU: ~2.6 GiB free VRAM, zero swap growth, zero NVMe
in steady state.

Reproduce best copy-bound config:
```
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/streamed_decode.cu -o benchmarks/streamed_decode
taskset -c 0-3 benchmarks/streamed_decode --model /home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf \
  --manifest results/qwen38-27b-manifest.txt --hot-mib 8192 --cache-mib 5553 --passes 8 --readers 4 --batch 16
```

## Real inference (E036, measured) — the gap that remains

Real Qwen3.8-27B through llama.cpp unified memory (GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, -ngl 99, model > VRAM):
baseline 0.63 t/s, native MTP draft head 1.04 t/s = **1.65x**, acceptance 23% (mean len 2.84).
**The >=3x bytes/accepted-token KPI FAILED:** PCIe H2D ~7 GiB/token in BOTH arms — managed-memory THRASHING
(weights + KV + draft > 12 GiB VRAM, pages evicted/re-faulted every token), not a speculation problem.
Temp-0 outputs diverge via batched-verification FP nondeterminism. Real inference (0.6-1.0 t/s) lags the
staged proxy (4 t/s single / 64 batch) because llama.cpp unified memory demand-migrates blindly with no
intelligent residency. Script: scripts/e036_mtp_spec.sh; raw: results/e036-{baseline,mtp}-20261109-171251*.txt.

## Key measured limits (post-IOMMU; don't re-litigate without new evidence)

- Single-stream ceiling ~4.2 tok/s: every token reads the 5.82 GB cold set over PCIe at ~23.3 GB/s effective
  (isolated pinned H2D 24.61). Measured 4.009 = 95% of ceiling. Faster single-stream needs FEWER bytes/token
  (approximate-layer drafting + exact verification done right), not more bandwidth — the Gen4 x16 link is now
  the limit and IOMMU is already removed.
- Batch 16 is copy-bound (4.007 passes/s, compute fully hidden) -> 64.1 aggregate. Batch 32 was compute-bound
  (0.73 s/pass, dp4a 72 regs no spills); closing the gap to its ~64 tok/s copy ceiling needs a tensor-core int8 MMQ kernel.
- Batch 64 regresses (96 regs -> 5 blocks/SM). Negative: packed float2 scales 27% slower at batch 32, reverted.
- Pinned-cache head removes CPU staging for 2/3 of the cold set; full pinning blocked by the memlock cap.
- E036 naive MTP speculation amortizes per-pass overhead (1.65x) but does NOT cut weight traffic; 23% acceptance.

## Next steps (in order)

1. **Highest leverage:** port the staged proxy's intelligent residency (hot VRAM arena + pinned RAM cache +
   overlapped H2D) into a REAL decode loop with attention/SSM math + KV state, to close the 4 tok/s proxy vs
   0.6-1.0 t/s llama.cpp unified-memory gap (E036). More direct win than speculation.
2. **Fewer bytes/token:** approximate-layer drafting + exact multi-token verification done right (SubSpec/
   SpecOffload-style), targeting the user's <=1 GB/accepted-token KPI. E036 showed naive MTP doesn't cut traffic.
3. **Commit the E032-E037 work** (user has not yet asked; ask first).
4. Optional: tensor-core int8 MMQ kernel for the batch-32 compute gap (ceiling ~64 tok/s).
5. Optional: ask the user about raising RLIMIT_MEMLOCK for full-cache pinning (small expected gain).

## Caveats

The streamed decode is a weight-streaming PROXY: no attention/SSM math, no KV state, no sampling,
synthetic input vectors. It bounds weight-bandwidth-limited decode from above. Batch tok/s is
serving-style aggregate (NB parallel vectors), not single-user latency. All measured numbers are
CPU-reference validated (zero mismatches); theoretical numbers are labeled as ceilings.
