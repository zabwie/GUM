# Handoff: GPU Unified Memory R&D

Source: Codex thread `01a1b9b3-2ce5-7ed2-87e9-6f8c23419740` ("Research GPU unified memory").
Transcript: `~/.codex/sessions/2026/11/08/rollout-2026-11-08T01-48-09-01a1b9b3-2ce5-7ed2-87e9-6f8c23419740.jsonl`
Repo: `/home/zabwie/gpu-unified-memory-research` (git; last commit `853d6cb`; later work is uncommitted).

## What the user is trying to do
Build and measure a software-defined memory system so an NVIDIA RTX 3060 (12 GiB) can use system RAM as an
extension of VRAM for local LLM inference on models larger than VRAM, while keeping the math on the GPU (no
conventional CPU layer offload). Goal: get as close as possible to the speed of a GPU with enough VRAM. The user
says explicitly not to assume this is achievable; the point is to find the real ceiling by experiment.

The user wants an autonomous, hypothesis-driven researcher: observe -> hypothesize -> test -> measure -> revise.
Requirements:
- Keep unified addressing, physical sharing, remote access, migration and VRAM-equivalent bandwidth distinct.
- Quantitative, honest reporting (confirmed / measured / inferred / speculative). No theoretical numbers as measured.
- Protect the machine. No BIOS, kernel or driver changes. Don't touch the user's existing models or llama.cpp checkout.
- Keep project files (AGENT_STATE.md, EXPERIMENTS.md, FINDINGS.md, etc.) current.

## Hardware (see HARDWARE.md)
i7-12700K, ASUS PRIME Z790-V AX, RTX 3060 12 GiB, PCIe 4.0 x16 (~31.5 GB/s nominal), ~31 GiB RAM, 43 GiB swap,
driver 595.99.02, nvcc 13.2 at `/usr/local/cuda/bin`, kernel 7.0, HMM mode. Locked-memory limit ~3.9 GiB.
Capabilities: mapped host memory, UVA, concurrent managed access, pageable access, VMM + host-backed VMM
(2 MiB granularity). No host-native atomics. No read-only file-backed registration.

## Resource limits and rules
- Default caps: 2 GiB device, 4 GiB extra host memory. Do not exceed without explicit user approval.
- Run benchmarks through `scripts/run_when_idle.sh` (resource guard; never kills other processes).
- The user later closed other apps to free VRAM, but still wants care.

## Results so far (details in FINDINGS.md, BENCHMARKS.md, EXPERIMENTS.md)
- Microbenchmarks done: VRAM vs mapped host vs staged H2D vs managed (first touch, prefetch, repeat).
- Nsight confirmed demand migration. Managed prefetch beat mapped and full H2D when the working set fit in VRAM.
- A GPU-side gather cache helped at 1-10% selectivity with reuse (synthetic).
- Q4_0 GEMV validated against a CPU reference on real GGUF tensors, up to 960 MiB. Double-buffered chunks beat
  serialized chunks, but reloading weights every pass lost to one-time staging/prefetch.
- Dependent random pointer-chase latency was measured, with a chain-count crossover at 960 MiB: mapped wins at
  low fanout, managed wins at high fanout.
- Mixed residency: one kernel read 1 GiB device + 2 GiB mapped host, validated, with PCIe attribution caveats.
- CUDA VMM: one contiguous GPU VA range backed by both device and host memory worked (E024).
- Real LLM: Qwen3-0.6B Q4_0 ran with ordinary vs llama.cpp managed allocation. Correct, and decode rates
  tied. This is a fitting model, not oversubscription.
- Not yet measured: any model larger than VRAM, dynamic cache hit rates, expert-routing benefit, true oversubscription.
- No novelty claim made.

## Latest user mission ("NEXT RESEARCH MISSION: ACTUAL VRAM EXTENSION")
1. Mixed-residency kernel: done.
2. Real oversubscription (13-14 GiB managed allocation): **prepared only**, see OVERSUBSCRIPTION_PLAN.md
   (proposed 8 GiB host cap, 2 GiB VRAM reserve, 8 GiB RAM reserve, zero-swap abort). **Do not run until the user
   approves the budget.**
3. Unified address space prototype (VMM / managed / library): VMM probe done. A small abstraction library
   was started in `src/`.
4. Dynamic residency manager (track hot pages, prefetch, evict): in progress at the thread's end.
   `src/residency_manager.hpp` and `benchmarks/residency_manager.cu` were being written and debugged within
   existing caps. Its status is unverified, so check it and its tests.
Later priorities in the brief are truncated in my reading. Read the thread's user message at step 5271 for the full list.

## Suggested next steps
1. Read AGENT_STATE.md, NEXT_STEPS.md, git status and `git diff`. Uncommitted work exists.
2. Finish and verify the residency manager (build, correctness, benchmark within caps), then commit.
3. Ask the user to approve the revised budget in OVERSUBSCRIPTION_PLAN.md, then run the 13 GiB test.
4. Test page-selective prefetch (NEXT_STEPS item 15).
5. Run an over-VRAM model end to end (a 14.1 GiB GGUF existed historically; it was not loaded under the 4 GiB cap)
   and compare llama.cpp managed memory against a custom path. Check MoE expert-routing prediction.
