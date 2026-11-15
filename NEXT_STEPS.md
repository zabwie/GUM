# Next steps

1. **Completed:** Query CUDA capabilities and execute checksum-validated mapped-host, staged H2D, VRAM, and managed-memory scans.
2. **Completed:** Isolate H2D copy-only bandwidth; repeat VRAM scans with three warm-ups.
3. **Completed:** Measure 16, 24, 32, 64, and 128 MiB sequential scans. Staging was nearly tied at 16 MiB and faster at 24 MiB and above in these runs.
4. **Completed:** Compare managed CPU-first access, repeated GPU access, and explicit prefetch; repeat first access across five fresh allocations.
5. **Completed:** Use Nsight Systems to confirm host-to-device and device-to-host managed-memory migrations. Do not use profiler-inflated times as performance results.
6. **Completed:** Compare mapped reads, full H2D staging, and managed prefetch for 1%, 10%, 25%, and 100% strided access with 1, 2, and 4 passes. Evict L2 before every timed trial; all checksums passed.
7. **Completed:** Gather selected mapped-host values into compact VRAM and compare against mapped reads, full H2D, and managed prefetch across selectivity and reuse. The cache helped at 1–10% with repeated passes in the synthetic gather.
8. **Completed:** Measure synthetic Q4_0 matvecs at 32, 64, and 128 MiB, then validate a bounded GEMV on a real 47.812 MiB model tensor slice. Managed prefetch won among host-backed paths when the matrix fit in VRAM.
9. **Completed:** Add serialized and double-buffered chunk copies, then sweep 4–64 MiB chunks on the same real tensor. Double buffering helped versus serialized chunks, but reloading weights each pass lost to one-time staging/prefetch; no stable chunk-size optimum was established.
10. **Completed:** Check whether CUDA can register read-only file-backed model pages. The device reports `cudaDevAttrHostRegisterReadOnlySupported=0`; the test skipped mapping before touching the model file.
11. **Completed for a fitting model:** Run Qwen3-0.6B Q4_0 in ordinary and managed-allocation modes. Both answers were correct and all 29 layers were offloaded; decode rates were effectively tied. This does not test oversubscription.
12. **Completed:** Add dependent random pointer chasing, with single-chain latency and disjoint multi-chain comparisons at 1–960 MiB; validate every output.
13. **Completed:** Measure the 960 MiB managed-memory chain-count crossover and profile 1, 256, and 4,096 chain migration traffic. At high chain counts, demand migration touches pages across nearly the full allocation.
14. **Completed:** Extend synthetic Q4_0 GEMV comparisons to 256 and 960 MiB. Managed prefetch remains faster than mapped or full H2D for these fitting-in-VRAM workloads; all CPU-reference checks passed.
15. **Next:** Compare prefetching only pages reached by the pointer chains against direct mapping, demand migration, and full-allocation prefetch at sparse and dense chain counts. Use the current results to test whether reduced migration bytes can also avoid the observed GPU-fault overhead.

16. **Completed:** Verify 1 GiB device plus 2 GiB mapped-host access in one GPU kernel and record PCIe activity with attribution limits.
17. **Completed:** Verify a contiguous mixed-backed CUDA VMM range on the RTX 3060 with zero GPU validation errors.
18. **Completed:** Run an actual small quantized LLM and compare ordinary CUDA allocation with the llama.cpp managed-memory switch.
19. **Completed (E026, user-approved):** Run the 13 GiB single-managed-allocation oversubscription experiment using the resource caps in OVERSUBSCRIPTION_PLAN.md. Do not execute before approval.
20. **Completed (synthetic, E025):** Page-aware residency manager with GPU access counters, hot-page prefetch, eviction under VRAM pressure and correctness checks; 3.4x vs direct mapped access on a 90/10 skewed trace, zero errors. Next: slab allocator instead of per-page cudaMalloc, then apply to a real over-VRAM workload.
21. **Completed (E031, measured):** Streamed decode proxy on the 13.42 GiB Qwen3.8-27B STRICT Q4_0 GGUF (497 tensors, per-token Q4_0 GEMV over every weight, CPU-reference validated, 3,976 rows, zero mismatches in every config). Steady state: 0.721 tok/s all-cold host-resident (3.57 GiB pinned mapped + 1.85 GiB managed), 0.744 tok/s with a 4.5 GiB RAM cache + 0.93 GiB NVMe per token, 1.021 tok/s with a 5.55 GiB RAM cache covering every cold weight (zero NVMe in steady state). The full-cache rate's 14.7 GB/s effective bandwidth matches the pinned H2D rate from E021 (14.738 GB/s), consistent with the H2D copy of the cold set being the dominant serialized cost; copy and GEMV run on one CUDA stream and cannot overlap in the current pipeline.
22. **Completed (E032, measured):** Copy/compute overlap in the streamed decode proxy (H2D on one stream, GEMV on another). First build had a slot-reuse event bug (event indexed by staging-buffer modulus kStage=8 while slots cycle by kSlots=4; kHot items recorded no event) and FAILED verification (13 mismatches, 1.240 tok/s invalid). Fixed with one event per slot, primed at startup: now PASS, zero mismatches, 3976 rows. Steady state 1.247 tok/s (up from 1.021, ~22% faster), effective 17.9 GB/s.
23. **Completed (E033/E034, measured):** Speed optimization. dp4a int8 GEMV (~275 GB/s vs 24.6 fp32), decode batching (NB vectors share one weight pass), hot-arena GEMVs on a third stream, and a pinned RAM-cache head (3658 MiB within the 3978 MiB memlock hard cap; no CPU memcpy for pinned items in steady state). Final binary, all PASS zero mismatches: batch 1 = 1.951 tok/s single-stream, batch 16 = 31.8 aggregate, **batch 32 = 43.8 aggregate tok/s** (127,232 rows checked), batch 64 = 37.9 (occupancy regression). 43x the E031 baseline; meets the 20-40 tok/s target as serving-style aggregate throughput. Single-stream ceiling is ~2.5 tok/s (5.82 GB cold set/token over PCIe at ~11.5-14.7 GB/s) — physics, not software. Negative result: packed float2 scales 27% slower at batch 32, reverted.
24. **Completed (E036, measured):** Real-inference speculative decoding on the over-VRAM 27B. Built MTP-capable llama-cli out-of-source at $HOME/llama-mtp-build (d775b8967; user's checkout untouched, its prebuilt binaries predate MTP). Target via GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 -ngl 99; arm B adds the native NextN/MTP head (-md mtp-Qwen3.8-27B-Q4_0.gguf -ngld 99 --spec-type draft-mtp --spec-draft-n-max 8). Result: baseline 0.63 t/s, MTP 1.04 t/s = **1.65x**; acceptance 23% (70/304), mean len 2.84. **KPI verdict: the >=3x bytes/accepted-token reduction is NOT met** — PCIe H2D is ~7 GiB/token in BOTH arms (managed-memory thrashing: weights+KV+draft > 12 GiB VRAM, pages evicted/re-faulted every token). The 1.65x is per-pass overhead amortization, not traffic reduction. Temp-0 outputs diverge ("the massive number of"/"massive amounts of") from batched-verification FP nondeterminism. Real inference (0.6-1.0 t/s) is far below the staged proxy (~2 t/s single / 43.8 batch) because llama.cpp unified memory demand-migrates blindly with no intelligent residency. Raw: results/e036-{baseline,mtp}-20261109-171251.txt + -pcie.txt; script scripts/e036_mtp_spec.sh.
25. **Completed (E037, measured; user-approved kernel change):** `iommu=pt` applied + reboot. All 20 IOMMU groups flipped DMA-FQ -> identity (GPU group 16); link Gen4 x16 under load. **H2D 14.33 -> 24.61 GB/s (1.72x), D2H 12.67 -> 25.87 (2.04x)** — the IOMMU was a ~42% bandwidth tax, a config limiter not silicon (validates the user's challenge to the "physics" framing). Proxy throughput doubled: single-stream 1.951 -> **4.009 tok/s**, batch-16 31.8 -> **64.1 aggregate** (both PASS, zero mismatches). Single-stream now near the Gen4-passthrough ceiling (4.23 theoretical). Change is persistent in GRUB; revert by removing iommu=pt + update-grub + reboot. Raw: results/h2d-d2h-iommu-pt-*.txt, results/streamed-decode-27b-iommu-pt-batch{1,16}-20261109-172744.txt.
26. **Next (highest leverage):** The IOMMU bandwidth lever is now exhausted (E037); single-stream proxy is at the Gen4-passthrough ceiling (~4 tok/s, 5.8 GB cold/token). Two remaining paths: (a) port the staged-proxy's intelligent residency (hot VRAM arena + pinned RAM cache + overlapped H2D, E032-E034/E037) into a REAL decode loop with attention/SSM math + KV state, to close the gap between the 4 tok/s proxy and the 0.6-1.0 t/s llama.cpp unified-memory thrashing (E036); (b) read fewer bytes per accepted token via approximate-layer drafting + exact multi-token verification done right (E036's naive MTP did NOT cut traffic). Path (a) is the more direct win.
27. **Optional:** Close the batch-32 compute gap (0.73 s/pass vs ~0.5 s copy-bound, ceiling ~64 tok/s) with a tensor-core int8 MMQ kernel; requires Q4_0 fragment layout work and full CPU-reference revalidation.
28. **Optional:** Full-cache pinning (5.55 GiB > 3978 MiB memlock hard cap) needs a user-approved limits.conf/systemd change; expected gain is small since the pageable tail already hides under H2D.

## Guard behavior

Moderate desktop activity and an idle CUDA context do not automatically block a short bounded run. The guard defers on sustained heavy per-process GPU activity, repeated global utilization at or above 85%, temperature at or above 80 C, insufficient RAM/VRAM headroom, or invalid input limits. Q4_0 runs account for the full matrix, two configured chunk buffers, and scratch allowance; `CHUNK_MIB` is bounded to 4–64, and combinations above 2 GiB are rejected. The runner never stops unrelated processes.
For Q4_0 and host_latency runs, `MIB` is bounded at 960; the allocation estimate and Q4 executable enforce the 2 GiB VRAM cap, with host-RAM plans within the 4 GiB project limit.

29. **Completed (E038, measured):** Real decode loop with tiered residency (`benchmarks/real_decode.cu`):
    full qwen3/qwen35 layer math (attention+KV, gated-delta-net+SSM state, greedy sampling) over real
    GGUF weights with the E031-E037 hot-arena/RAM-cache/overlap streaming. Validated token-for-token
    against llama.cpp greedy (0.6B: 24/24; 27B: matches after fixing the delta-net k-head mapping
    `h % n_group`). 27B steady **3.39 tok/s** vs 0.63 t/s llama.cpp unified memory = **5.4x**, 5.42 GiB
    cold H2D/token, zero NVMe steady-state, >=2.87 GiB VRAM free. Raw: results/e038-*.
30. **Next:** batch-16 real decode (amortize the 5.42 GiB/token cold set across sequences, E033-style)
    for aggregate serving throughput; then approximate-layer drafting + exact verification to cut
    bytes/accepted-token (the E036 KPI).

31. **Completed (E039, measured):** Batched real decode: --batch B (power of 2) with NB-template
    dp4a GEMV, per-sequence KV/SSM/conv states, batched kernels. 27B: batch 4 = 10.70 aggregate
    tok/s (1.36 GiB cold/token), batch 8 = 13.17 (0.772 GiB/token, 7x less than single-stream),
    both token-for-token equivalent to the validated sequence across all sequences; 2 GiB VRAM
    floor held (batch 8: hot 7424/cache 6144; batch 16 busts the floor on SSM states). Key fix:
    flat-index RMW race in six elementwise kernels (grid (x, B) blocks all writing the same
    indices — nondeterministic lost updates) + delta-net qkv stride. Raw: results/e039-*.
32. **Next:** tensor-core int8 MMQ kernel (batch-8 step time is compute+copy mixed); approximate-
    layer drafting for the bytes/accepted-token KPI; optional fp16 SSM states for batch 16.

33. **In progress (E041/E042, first working cut):** speculative decoding with PLD drafter +
    E040 verify. Working: lossless greedy speculation loop (draft -> one cold-set pass ->
    accept-prefix -> correction), SSM/conv state snapshot-rollback (~150 MB D2D/cycle),
    shared-cache full-history attention. Measured 27B: chat 7.26 tok/s (2.3x dense, acceptance
    33%), repeat workload 6.8 tok/s (28%). KNOWN ISSUES (next session): (a) known/hist
    alignment drift — the carried known_next occasionally disagrees with the history tail,
    capping acceptance below the periodic-stream expectation (~100%); (b) deep-g (16-32)
    untested; (c) MTP-head drafter (E041a) not integrated; (d) proposal wrap-around for
    periodic streams partially handled. Debug hooks: --dbg-dump prints drafter inputs
    (dbg,draft lines) and verify-row argmaxes.
34. **E040 CORRECTION (same-day):** the first E040 sweep had a silent regression — g_batch was
    set before the multi-mode adjustment, so GEMV<1> computed only row 0 and the tie-excused
    checks passed vacuously on all-zero rows. Fixed (g_batch ordering + all-zero-logits
    garbage detector). CORRECTED numbers: B=1 0.350s/2.86 tok/s, B=4 0.386s/10.37, B=8
    0.430s/18.61 (~7.4 ms/position marginal). The E039 serving numbers were never affected
    (--batch sets g_batch correctly). Lesson: tie-excused validation needs a row-energy
    garbage detector; new modes must be checked against g_batch ordering.

35. **Current E054-E075 result — user target still open:** preserve the transposed delta-net state,
    activation L2 window, packed scale/sum metadata, and the exact-order CB=64 shared GEMV tile
    for NB=16, K>=5120. E071 passed the exact 27B gate at 9.629 tok/s in one run; a repeat was
    8.799 under lower free VRAM. Matched E072/E073 profiling measured ~7.5% lower summed GEMV
    time, but GEMV still accounts for ~80% of kernel time. **Do not claim 20–25 tok/s or E050's
    projected 13–15 tok/s.** Continue from the E072/E073 timeline and pursue a larger GEMV
    reduction; do not repeat E051/E068 scalar shared tiling, E058/E059 vector-warps dispatches,
    E056 WMMA, or E074 vector-warps plus shared tile without a new hypothesis. Re-run the exact
    token gate after every integrated change. Evidence: EXPERIMENTS.md E065-E075.

36. **Completed (E076):** Attribute MTP and PLD acceptance separately, correct tok/s timing to
    include MTP drafting, and avoid computing an MTP chain that the next PLD-selected cycle will
    discard. Default MTP-only `p_min=0.97`, γ=8 accepted 61/63 (96.8%) and matched a 128-token
    dense reference exactly in two runs; end-to-end was 5.77–5.94 tok/s. The 128-token fast
    hybrid was 7.397 tok/s, 30.8% overall acceptance, and also matched dense exactly. Evidence:
    EXPERIMENTS.md E076.
37. **Next:** Reduce the measured `k_mat_q4` verifier share (~80% of profiled kernel time) while
    preserving the E076 full-output gate. Separately measure the confidence threshold on more
    prompts before treating 96.8% as representative beyond the checked agent trace. Do not use the
    old verifier-only tok/s values as end-to-end throughput.

38. **Current, E089:** Use the installed best-first EAGLE4 path for Qwen3.8-27B.
    It measured 20.340 / 18.784 / 15.467 generated tok/s on the agent, Rust, and
    explanation prompts, with 128/128 dense matches on each. Preserve the same
    checks and report prompt-inclusive rates and free VRAM. The 40–60 target
    is open.
39. Compare E095 DSpine and E096 overlapped byte-exact decompression with E089.
    Try a wider verifier only with an exact kernel that improves actual cycle
    cost enough to cover additional draft positions. E091's 24 positions and
    E085's 32 positions did not clear that bar.
40. **Current, E127:** Run `scripts/run_single_user.sh` for the original native
    MTP tree. The six-prompt suite matched all dense IDs at 15.20–21.37 generated
    tok/s. E116's packed 24-position verifier reduced passes on one chat prompt
    and slowed the agent, Rust and event-log prompts. E126/E128 measure where
    each drafter first misses the true continuation. Use those results to
    prioritize improving multi-step prediction over wider verification.
41. **Installed, E149:** The packed 24-position native MTP launcher measured
    16.99–23.87 generated tok/s across the same six prompts, with 1,024/1,024
    tokens matching. Keep this six-prompt baseline for changes. On the
    slower chat, verification took 226.7 ms and emitted 4.11 tokens per pass.
    E148's deeper history path saved zero passes on the checked runs. Use
    separate prompts to measure sustained draft depth before changing the
    default tree again. The 40–60 tok/s target remains open.
42. **Measured, E150–E151:** The tuned fixed and adaptive 32-position
    modes both stayed below the 24-position launcher on the two chat
    references. Use E149 as the control when testing a new proposal source.
43. **Measured, E156/E156a:** The weight traffic audit puts chat at
    1.274 GB per exact output token (0.493 with a ten-token oracle) and
    showed FFN tensors are 3.33 of 5.24 GB per pass; activation sparsity
    recovered zero bytes. The 8448 MiB raw config beat the matched 8064
    control on both chats but only fits at context capacity 384 — a
    measured short-sequence option, not yet the launcher default (owner
    call on the context trade-off).
44. **Negative, E157:** The full-vocabulary embedding-tied parallel head
    matched E153's restricted-vocab accuracy (prefix 0.42 vs 0.45,
    position-1 ~21% both). Vocabulary coverage was not the constraint;
    the MLP bottleneck is. The one-pass MLP drafter family is exhausted.
45. **Negative, E158:** The 150M sequential self-conditional drafter with
    the target's exact output head memorized 55k transitions but reached
    6–9% held-out depth-1 survival; linear transition probes ~1%. Native
    MTP's chained marginals (0.725/0.629/0.540/0.519) confirm the head is
    strong and slowly decaying — the live tree's 4.2 emitted is structural,
    not a tuning gap. Do not retrain a drafter on the current 30 captures.
46. **Built, E159 (durable infrastructure):** Batched mid-layer capture on
    the `--batch 8` lockstep path — 30 prompts × 160 positions × 8 depths
    in ~7 minutes (~12.8 positions/s, 5× dense single-stream), validated
    99.4–100% self-consistent through the exact output head. Source:
    `/tmp/vanta-e158/real_decode.cu` (copy of the E132-era decoder with
    `VANTA_MID_CAPTURE` hooks); parser: `scripts/e159_exit_head_probe.py`.
    Preserve this source — it is the base for any future capture campaign.
47. **Negative, E159:** HalfDraft exit heads (small trained maps from
    mid-depth residuals to the target argmax through the exact shared head)
    reach only 24.6–40.4% at resident-feasible depths (gate ≥ 0.8); 59.9%
    at layer 56, which does not fit the hot budget. The mid-depth→argmax
    mapping is the remaining layers' nonlinear compute. With E153–E158 this
    closes the small-drafter direction: five probes agree that beating the
    authored MTP head requires capacity-comparable drafter weights and
    EAGLE-scale training volume.
48. **Negative, E161:** The compression lever is void. On real tensor
    boundaries both Q4 planes sit at their order-0 entropy limits
    (zstd-19/xz ≈ 0.943–0.952; LZ4 = 1.000); E156's planar 9–10% and the
    "16% ceiling" were sampling artifacts. Measured whole-model lossless
    ceiling ≈ 5.5% — under 1 tok/s after GPU decode overhead. Do not build
    GPU-decompression codecs for this model.
49. **Negative, E162 (owner-directed):** Zero-copy mapped memory: SM reads
    at 18.2–21.8 GB/s vs 24.8 DMA → verify cycles ~10–14% slower, and only
    ~128 MiB of staging VRAM is freeable (the 8.4 GiB hot arena cannot move
    to RAM). The staged two-level design is confirmed. Post-iommu mapping
    is nonetheless far better than E031-era numbers (14.7 → 18–22 GB/s).
50. **Next (near-term gains):** 8448/ctx-384 residency (validated E156a),
    sync-stall trim (~13.6 ms/cycle, E160), and tree-width headroom
    (~13%, E126). Expected: ~20 tok/s chat; the agent/PLD path (23.9 →
    ~27) remains the fastest route toward the owner's 40 goal.

51. **Installed, E182/E183 (2026-11-13):** Stream profile re-baselined at 7.52 decode
    tok/s (full-pin, calm desktop; verify 0.575 s is link-saturated at ~25 GB/s,
    non-fetch overhead 4.3% — sync-stall lever spent). Hot tier is token-exact and
    linear: 42–43 ms/GiB, +0.72 tok/s per GiB (7.52/8.49/9.75/13.70 at
    hot 0/1600/3200/6400, all outputs identical). **GUM Adaptive installed and
    default-on:** server supervisor respawns the idle decoder with the hot tier the
    free VRAM affords (target = hot + free − 5000, clamp 0–6400, stream profile only);
    live-verified grow/shrink/regrow; equilibrium ~4378 MiB → 9.8–10.8 tok/s, ceiling
    13.7, floor 7.52 under desktop pressure. verify_stream.sh bands derive from live
    hot_mib. A8 shortlist coverage: E178 logs show widening gained ≤0.1 emitted/cycle
    — closed negative.
52. **Next:** (a) prefetch the next verify pass's weight window during the ~25 ms
    draft/repair tail (+4–5%, decoder surgery — respect the stream-discipline rules);
    (b) capture-fed MTP finetune when capture volume suffices (E169 gate: +2pp val
    top-1 + token-exact bench) — the emitted/cycle term is the only one left at fixed
    bytes; (c) batched parallel agent streams (E039: 13.17 aggregate tok/s at batch 8,
    0.77 GiB/token) for throughput-shaped workloads. Do not re-open: compression,
    zero-copy, small drafters, shortlist widening.
53. **Measured, E186/E187 (2026-11-13):** Wavefront ring 16/16/12 installed (+3% A/B,
    exact-gated; 24/24/18 reproducibly worse — do not go deeper). Pre-draft prefetch is
    a measured NEGATIVE at both hot=0 and hot=4378 (contention, not bandwidth) —
    reverted, patch preserved. The draft-time "idle window" is not free: the tree's
    logits readbacks already own it. If attacking it, batch the readbacks first
    (one pinned buffer per tree level) — that is also the draft-cost audit.
54. **E193 WIP (2026-11-13):** CPU row-split infrastructure works and is exact in the
    constrained-pinned regime; open bug = deterministic per-tensor shadow mismatch in
    full-pin (see EXPERIMENTS.md E193 for the full state + fixed bugs). Next session:
    isolate the snapshot-vs-GEMV data mismatch (suspects: a second xq writer between
    cpu_qev and the GEMVs — k_rms_quant at 2842 / k_quant_x at 2752 — or the packed-tile
    item boundary at non-4848-row items). The scalar-reference tool in --cpu-verify is
    itself buggy (disagrees even on CLEAN tensors) — fix it first; it is the fastest
    path to the answer.
55. **BF16 source downloaded (52 GB):** $HOME/Desktop/Chip/models/bf16-source
    = Qwen/Qwen3.8-27B @ 1d4bf0f2ff (Chip provenance pin). Codec 3-way KL test
    (Q4_0→BF16, (3,64)→BF16, teacher-forced) is NOW RUNNABLE — the codec lever (+28%)
    is live pending that measurement. Also enables true-BF16 quality comparisons for
    any future TOLERANCE-track work.
56. **Codec test RUNNING (2026-11-13 evening):** BF16 GGUF exported
    (~/Desktop/Chip/models/Qwen3.8-27B-BF16.gguf, 54.6 GB, from the provenance-pinned
    source). (3,64) requant building via scripts/c1_requant.py (losslessly Q4_0-stored).
    Teacher-forced perplexity suite (llama-perplexity, 200 chunks of wikitext-2-raw test,
    hybrid GPU offload) running in background: /tmp/codec_ppl.log —
    models: Q4_0, k3g64, BF16. DECISION RULE: compare each format's distance-from-BF16;
    if (3,64) ≤ Q4_0's distance → reopen the TOLERANCE track for the +28% codec lever
    (real 3-bit storage + GPU trellis/LUT kernel build). If (3,64) > Q4_0's distance →
    lever stays standby. Results land in /tmp/codec_ppl.log — copy them into
    results/e194-codec-kl/ when done. Runtime ~2-3 h total.
57. **E193 open bug (see #54):** strongest lead from the offline forensics — broken
    tensors' CPU output is ~10x smaller than the GPU's (reads like wrong cache region);
    clean tensors match to 1e-7. The dump tool + dump_analysis.py are in
    results/e193-cpu-rowsplit-wip-20261113/. Next: dump the GPU's actual slot bytes for
    a broken item and diff against pinned_cache[item1]; suspect the res_off/head
    interaction or a second writer of the cache region.
58. **Codec suite status (2026-11-13 night):** first run FAILED on three fronts, all
    diagnosed: (1) Q4_0 run — output lost by the `tail -5` capture, rerun with full log;
    (2) k3g64 — the suite's gate opened on a PARTIAL requant file → "data not within
    file bounds"; the requant has since COMPLETED (15.1 GB, "27.8% cut" summary printed)
    → k3g64 rerun is ready on the complete file; (3) BF16 — llama.cpp's qwen35 loader
    classifies blk.64 (the source MTP/nextn block, fully exported: eh_proj + a full
    layer set) as a hybrid repeating layer and demands blk.64.ssm_conv1d.weight which
    the MTP module doesn't have. FIX (next session): strip the 16 blk.64.* tensors +
    the qwen35.nextn_predict_layers KV from the BF16 GGUF (they sit at the file END →
    truncate-friendly), or check convert_hf_to_gguf.py for a nextn-skip option. The
    ad-hoc python GGUF parser derails on the tokenizer-array KVs — use llama-gguf r n
    for offsets (works) or pip gguf in a venv.
59. **Queue order (next session):** (a) rerun ppl: Q4_0 + k3g64 (both ready, full logs);
    (b) fix BF16 nextn strip → BF16 ppl run; (c) codec verdict → TOLERANCE go/no-go;
    (d) E193 bug: worker pointer escapes to RAW GGUF data (blk.1.attn_gate row 3491
    proven exact, nowhere else in file) — print res_off/pin_bytes at split time;
    (e) capture corpus accumulating (group-012 published 2026-11-13 21:04, 8 seqs).
60. **E195 16:1 tier port (COMPLETED; corrected and measured 2026-11-14):**
    port Vanta's tier geometry ((ctx−131k)/16 + 65k/4 + 65k ≈ 0.8 GiB @1M, from
    ~/Vanta/model/docs/TTKV_1M_DESIGN.md) into the CUDA TTKV, gated on needle recall
    at depth. OWNER WARNING: previous needle failures were likely MODEL YaRN-training
    limits, not cache limits (Qwen3.6 was extensible-but-needed-training) — the gate
    must separate model-context limits from cache-compression limits (test needles
    WITHIN the trained window first). Qwen3.8 rope_base=10,000,000 in our config =
    long-context-trained base — promising natively. Qwen4 reportedly ships 1M in
    configs (leaks); 512K/256K acceptable fallbacks. The measured 256K KV pool is
    2.23 GiB, with a 199.8 MiB attention scratch allocation; the 17-20 tok/s projection
    was not achieved. E196's separate old-KV attention reuse experiment raised full-run
    decode to 3.258 tok/s at the same 4,308 MiB decoder peak.
    Phase B prefill-overlay (from the Vanta doc) is the follow-on for true 1M.
60a. **E195 WIP history (superseded by 60b):** the compacted-row
    overlay is IMPLEMENTED and its core promise MEASURED: at ctx 262144 the
    baseline ABORTS (VRAM plan 10.79 GiB + 2 GiB floor > 11.63 GiB card) while
    the compact build BOOTS with the tier pool at 522.9 MiB (span=262144,
    rows=1060, kv_MiB=2982 in the plan) — 11x smaller than the baseline's
    ~5.9 GiB tier pool. Needle-recall gate: cache-INDEPENDENT (ttkv 1/5,
    turboquant no-pooling 1/5, fp32 oracle 0/5 at 13.7K with 5 needles) —
    the deep misses are model/task, not cache (owner's YaRN-vs-cache warning
    validated); recall parity holds, no loss from pooling or compaction.
    Bit-exactness: proven through 2560-token contexts (bisect), plain decode,
    and the dbg-constant repro; THREE bugs fixed (old-tier kh-stride span/16,
    overlay-copy pending cursor, W1 fresh_from recycle gate). ONE BUG OPEN:
    speculative decode diverges from baseline at >2560-token contexts
    (deterministic, any SP depth, isolated repro = long_canal A/B; layout,
    growth, capacity, recycling all excluded via GUM_TTKV_FULL_ROWS /
    GUM_TTKV_W1_FULL / GUM_TTKV_PAD env tests; device asserts clean). The initial
    divergence was closed by the pack-range correction in 60b. The final corrected
    build passed the 20-prompt suite and the 256K run. Retain this entry as debugging
    history.

60b. **E195 pack-range fix (2026-11-14):** The 4:1 and 16:1 launch counts
    divided the starting group twice. The extra writes wrapped the middle
    tier and changed the final prefill pass. The corrected, stripped binary
    matched all 256 `long_canal` IDs. The corrected cache also matched the
    pre-E195 decoder on a 7,840-token prefix with 48 generated IDs. The
    256K attention fixture passed under unfiltered initcheck. See E195 in
    [EXPERIMENTS.md](EXPERIMENTS.md#e195--corrected-ttkv-compact-row-commit-2026-11-14).
60c. **E196 (2026-11-15):** shared old-KV attention work across six query heads per
    KV head and enumerated retained rows directly. On the 261,888-token constructed
    repeat prompt, prefill rose from 8.99 to 46.41 tok/s and decode from 0.761 to
    3.258 tok/s; all 256 IDs
    matched E195 and decoder peak stayed at 4,308 MiB. Full details and raw results:
    [E196 record](results/e196/README.md).

61. **BF16 nextn-strip:** gguf-py script committed (scripts/strip_nextn_gguf.py,
    needs PYTHONPATH=/tmp/ggufpkg); FIRST RUN LAUNCHED but unverified — the earlier
    manual splice corrupted the header ("key '' has invalid GGUF type 808348524").
    If the script's output fails llama-gguf validation too, fall back: check
    convert_hf_to_gguf.py for a nextn-skip flag, or run the BF16 ppl on CPU-only
    (ngl 0, ~3 h) against the ORIGINAL BF16 gguf — the nextn block is unused during
    plain perplexity IF the loader tolerates it... it does not; the strip is required.
    Codec suite (Q4_0 + k3g64 ppl) running in background; BF16 stage auto-fires when
    the stripped file validates. Verdict rule unchanged (NEXT_STEPS #56).
62. **QTIP-class 3-bit test (QUEUED; E195 is complete; owner-approved 2026-11-13):** E194
    closed the NAIVE uniform 3-bit grid (PPL 8.75 vs Q4_0 7.04) but NOT the trellis/VQ
    class. Two-step falsification path, each cheap: (1) run an open-source QTIP/QuIP#
    pipeline on the BF16 source (~/Desktop/Chip/models/Qwen3.8-27B-BF16-nonextn.gguf,
    calibration on wikitext + capture corpus) → PPL decision rule: trellis-3bit ≤
    Q4_0's 7.04 = quality passes; (2) if quality passes, measure trellis decode
    throughput with the E191 methodology — our decode is LINK-BOUND (SMs 90% idle),
    so the kernel only needs to beat 24.7 GB/s, not dp4a — 10-15 GB/s SM-side suffices.
    Both pass → +28% lever reopens with a real kernel build. Fails either → lever
    closes with data. NOTE: requantize from the BF16 source directly, never from Q4_0.
