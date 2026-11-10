# Vanta Technical Design — real_decode.cu reference

The technical reference for `benchmarks/real_decode.cu` (~1,600 lines). Everything here was
learned the hard way — mostly by per-op bisection against llama.cpp. A cold-start reader
should NOT have to re-derive any of it. Read AGENT.md (mentality) and HANDOFF.md (state)
first; this file is the "how it works."

## 1. The model: Qwen3.8-27B ("qwen35" arch)

A hybrid: 64 blocks in a 3:1 pattern — **48 gated-delta-net (linear attention) layers +
16 full-attention layers** (blocks where il%4==3). Verified facts:

- **Dims**: n_embd 5120, 24 Q heads / 4 KV heads × 256 head dim, FFN 17408, vocab 248320.
- **Delta-net layers**: fused `attn_qkv` [5120, 10240] = q(16×128) | k(16×128) | v(48×128);
  `attn_gate` [5120, 6144] (output gate z); `ssm_beta`/`ssm_alpha` [5120, 48];
  `ssm_dt.bias`/`ssm_a` [48]; `ssm_conv1d` F32 [C=10240, 4 taps] (w[c*4+j], taps fastest!);
  `ssm_norm` [128]; `ssm_out` [6144, 5120]. Persistent state per layer: 48×128×128 fp32
  (3 MiB) + conv state 10240×3.
- **Attention layers**: `attn_q` [5120, 12288] = per head (24): q(256) | gate(256)
  INTERLEAVED (q at h*512, gate at h*512+256); `attn_k/v` [5120, 1024] (4 KV heads × 256);
  `attn_output` [6144, 5120]; q/k norms [256] per head.
- **Delta-net recurrence** (per v-head h, EXACT ggml semantics — ggml-cpu/ops.cpp
  `ggml_compute_forward_gated_delta_net_one_chunk`):
  k-head mapping is **hk = h % n_group (16)** — NOT h/(H/n_group). Head 0 is the only head
  where both agree; single-head tests mask this bug.
  State S transposed in memory: row j at st[j*S+i] represents S[i][j].
  Per token: `S *= exp(g)` (g = softplus(alpha+dt_bias)·ssm_a, scalar per head);
  `delta_j = (v[j] − dot(S_row_j, k))·beta`; `S_row_j += delta_j·k`;
  `out_j = dot(S_row_j, q)·(1/√S)`. q/k are L2-normalized per head BEFORE the recurrence:
  scale = 1/max(√(Σx²), eps) — **eps floors the norm, not added to the sum**.
- **RoPE**: qwen3 = NEOX half-split pairs (i, i+n_rot/2); qwen35 = IMROPE, but IMROPE also
  uses half-split in the ggml rope switch, and for text all mrope section positions are
  equal → **plain half-split RoPE**, theta = pos·base^(−2i/n_rot) with n_rot=64 (27B) or
  128 (0.6B). qwen35 rope base 1e7; partial rope (n_rot of 256 head dim).
- **Conv1d**: F32 [C, 4] ggml order (taps fastest), window = [state(3) | new token],
  state update = last 3 columns. gguf-py `tensor.shape` == GGML ne order (NOT numpy
  reversed) for this checkout.
- **GGUF**: Q4_0 = ggml type **2** (8 is Q8_0). Q4_0 block = fp16 scale + 16 bytes
  (byte j: low nibble = element j, high = element j+16). v3 strings have u64 length.
- **GEMV semantics** (the dp4a kernel): raw-nibble dot then ONE zero-point subtraction
  `s − 8·xsum`; folding (q−8) into the dot AND subtracting xsum doubles it. Activations
  quantized to int8 per 32-block (round-to-nearest-even) — matches llama.cpp MMQ; llama.cpp
  decode itself uses fp32 activations × dequant weights, yet greedy outputs match anyway.

## 2. The decode graph (per token, qwen35)

```
embed row (CPU pread + dequant, H2D)
per layer:
  x = rms_norm(x, attn_norm)
  if recurrent:
    qkv  = W_qkv·x          [10240]     gate = W_gate·x        [6144]
    beta = sigmoid(W_beta·x)[48]        g = softplus(W_alpha·x+dtb)·ssm_a [48]
    conv: out = silu(conv1d(state, qkv)); shift state   (q/k slices L2-normalized;
      v slice NOT)                                       (k slice at 2048, v at 4096!)
    delta_net(state, q,k,v, g, beta) → out[6144]        (sequential state update)
    att = rms_norm(out, ssm_norm)·silu(gate)            (gated norm, per head)
    y = W_ssm_out·att      [5120]
  else:
    q|g = W_q·x [12288]; q-norm per head (stride 512); k = W_k·x; k-norm; v = W_v·x
    rope(q, k); append k,v to cache at pos; attention (GQA h/6, scale 1/√256)
    att *= sigmoid(gate slice)                          (qwen35 output gate)
    y = W_o·att            [5120]
  x += y
  x = rms_norm(x, post_attention_norm) + FFN(silu(W_gate·x)·W_up·x) via W_down; x += ·
final: logits = W_output·rms_norm(x, output_norm); argmax (CPU)
```

Layer indices: recurrent layers are 0,1,2,4,5,6,… (il%4≠3); **recurrent-state index
ri = il − (il+1)/4** — indexing state arrays by raw il overflows (48 allocated, il≤62).

## 3. Modes

- **Batch serving** (`--batch B`): B sequences, per-seq KV/SSM/conv states, lockstep.
- **Multi-position verify** (`--multi B --feed-ids t,t,...`): ONE sequence, B positions per
  cold-set pass. Teacher-forced argmax checks vs the feed sequence.
- **Speculative** (`--speculate γ`): PLD n-gram drafter (CPU, ~free) → γ-position verify →
  accept confirmed prefix → emit correction → rewind `cur_pos` → SSM/conv snapshot-rollback.
- **Validation** (`--validate 1`, batch 1 only): mmap-backed fp32 CPU reference mirroring the
  GPU math exactly (same int8 activation quantization); per-step argmax comparison with a
  tie threshold (<0.25 logit gap) AND a row-energy garbage detector (all-zero logits are
  failures, not ties).

## 4. Buffer/state layout

- Activations: [B][...] stacked (x, t_norm, yA/yB pitch = max tensor N + slack, t_ffnh,
  t_att, t_dn, t_conv, logits [B][vocab]).
- KV cache: [B][n_attn][2][n_kv·ctx·hd] floats. **Batch mode: one slice per sequence
  (kv_seq_stride = n_attn·2·n_kv·ctx·hd). Multi/verify/spec mode: ALL rows share slice 0**
  (one sequence — full-history causal attention; per-row slices read sparse history and
  silently degrade).
- SSM state: [B][n_recr][48][128×128] + conv [B][n_recr][10240][3]; spec mode adds
  snapshot buffers for rollback.
- Weight tiers: hot arena (VRAM, loaded once) → staged cache (RAM, 3658 MiB pinned head) →
  NVMe (first pass only). Slots: 4×32 MiB pinned ring, 8-deep staging, reader threads.

## 5. Bug classes (EVERY one was a real bug — check new code against all)

1. **Flat-index RMW race**: grid (x, B) elementwise kernels with
   `i = blockIdx.x*blockDim.x + threadIdx.x` — every blockIdx.y block touches the same i.
   `y[i] += x[i]` becomes a nondeterministic read-modify-write race (lost updates).
   Sanitizer-invisible (racecheck = shared memory only; memcheck clean). Symptom: rare token
   divergence that worsens with B and is masked by debug syncs. Fix: global index
   `(blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x`.
2. **Per-sequence stride confusion**: dense [B][N] (tensor's own N) vs buffer pitch
   (max_rows) vs byte-vs-float units. Each GEMV writes y[v*N + row]; consumers index dense.
3. **g_batch ordering**: the NB-template dispatch must follow multi/speculate mode
   adjustments — otherwise GEMV<1> silently computes row 0 only and tie-excused checks pass
   vacuously on all-zero rows (E040's regression; caught by printing per-row argmaxes).
4. **Stream discipline**: KV appends and anything consuming kernel output go on the compute
   stream (stc). A copy-stream race corrupted every layer ≥ 1 (E038).
5. **State indexing**: recurrent arrays indexed by ri, not raw il (overflow: 195 MB into
   151 MB — illegal access).
6. **SSM rollback**: rejected speculative positions corrupt the cumulative delta-net state;
   KV overwrite is insufficient. Snapshot ~150 MB D2D per cycle (~1 ms).
7. **Garbage rows vs ties**: all-zero logits make every comparison a perfect tie — a
   tie-excused mismatch check passes vacuously. Always add a row-energy detector
   (max|logit| < 1e-2 → hard failure).
8. **Emission accounting in speculation**: feed[0] is the carried correction (already
   emitted); only feed[1..m] + the new correction are new tokens. The prefill's known_next
   must be emitted exactly once. Off-by-ones here double the text silently.

## 6. Validation protocol (run before believing ANY result)

1. `--validate 1` (0.6B): GPU vs CPU-reference argmax per step, ties excused only with the
   garbage detector active.
2. Replicated-prompt equivalence (`batch_equivalence` line): all sequences must produce the
   identical token stream.
3. Teacher-forced check (`--multi --feed-ids`): row argmaxes must reproduce the feed
   sequence — with the garbage detector ON.
4. Cross-check vs llama.cpp greedy (token IDs via /tmp/llama-dbg tooling, see
   RESEARCH_PLAN.md §7; the tooling recipe: patch cb() to rename nodes, build CPU-only,
   cb_eval dumps per-op values).
5. compute-sanitizer memcheck/initcheck/racecheck: catches OOB/uninit/shared races, but NOT
   global-memory RMW races (class 1) — only equivalence runs catch those.
