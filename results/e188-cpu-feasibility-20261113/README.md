# E188 — CPU second-stream feasibility probe (2026-11-13)

External analysis claimed: verify is multi-row, so CPU matmul avoids the batch-1
limitation; ~4 GB/cycle of weights on 8 P-cores in <0.4 s → cycle ~0.4 s → +40%.

Measured with the user's llama.cpp build (read-only run of its binary, GPU hidden,
taskset 0-7 = 8 physical P-cores), 27B Q4_0:

- pp512: 9.78 ± 0.61 t/s → 512/9.78 = 52.4 s per weight pass → 0.269 GiB/s... per-pass
  reading is amortized across the batch; the per-pass wall time is the number:
  **one full 14.1 GiB weight pass at 512 rows = 52.4 s (0.52 TFLOP/s int8, compute-bound)
- pp32: 9.72 ± 0.07 t/s → **32/9.72 = 3.29 s per weight pass = 4.3 GiB/s of weights**
- Rate is batch-INDEPENDENT (32 vs 512 rows identical) — the reviewer is right that
  CPU matmul is not batch-1-limited, wrong that 4 GB fits in 0.4 s (it takes 0.93 s
  at llama.cpp's kernel rate; 2.4× over the claim).

Balanced split arithmetic (CPU rate 4.3 GiB/s vs GPU stream 23 GiB/s):
f = R_cpu/(R_cpu+R_gpu) = 4.3/27.3 ≈ 15.7% of weight bytes to CPU → balanced cycle
≈ 0.50 s vs 0.59 s → ceiling ≈ **+15–20% with perfect per-layer overlap**, not +40%.
Requires: row-split pipeline (per-layer CPU time 7.8 ms budget vs ffn_down's 11.6 ms
at llama.cpp rate — needs finer row-split or 2× faster CPU kernels), activation
exchange (640 KB/layer D2H + partial outputs H2D), exactness gate (accumulation-order
change on CPU-computed tensors — near-tie flips must be suite-gated), RAM contention
(DMA 24.7 + CPU ~6 GB/s inside DDR5-5600's ~65 GB/s real).

Note: the first bench run (CUDA visible, ngl 0) reported 208 t/s — silently GPU-assisted;
ignore it. CPU-only number is 9.72-9.78.
