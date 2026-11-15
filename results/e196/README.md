# E196: 256K old-KV reuse

E196 changed the long-context attention kernel while keeping the E195 weight
paging and memory profile. It packs surviving old-key rows and shares each KV
load across the six query heads that use the same KV head.

## Full run

| Field | Value |
|---|---|
| GPU | NVIDIA RTX 3060, 12 GiB |
| Target | Qwen3.8-27B STRICT Q4_0 |
| Context | 262,144 tokens |
| Input / output | 261,888 prompt IDs / 256 generated IDs |
| Prompt | Repeated prose segments from the frozen prompt set |
| Weight profile | 0 MiB hot, 10 GiB host cache, NVMe spill |
| Draft and KV | Native MTP, up to 31 draft nodes; TurboQuant-TTKV |
| Prefill batching | Eight 32-position chunks per weight pass |
| Run cycles | 48 |

| Metric | E195 baseline | E196 |
|---|---:|---:|
| Prefill | 8.99 tok/s | 46.41 tok/s |
| Decode | 0.761 tok/s | 3.258 tok/s |
| Mean verifier pass | 6.863 s | 1.512 s |
| Decoder-process peak VRAM | 4,308 MiB | 4,308 MiB |
| Total-device peak VRAM | 6,211.31 MiB | 6,250.38 MiB |
| Generated IDs matching E195 | — | 256 / 256 |

NVML sampled memory every 200 ms. At the E196 total-device peak, 1,942.38
MiB was outside the decoder process. The 39.06 MiB device-wide peak increase
therefore did not come from a higher decoder allocation.

This prompt is a constructed repeat benchmark, not an independent 256K
quality suite. Exactness was checked against E195 for this same input; it does
not prove token agreement for other inputs or establish task quality.

## Weight paging

The run moved 12.76 GiB of target weights per verifier pass. Of that, 3,658
MiB was pinned host cache copied directly to the GPU, the pageable portion of
the 10 GiB host cache used pinned staging, and the remaining 2.76 GiB came
from NVMe through four direct-read workers. A 16-buffer pinned host ring and
16 reusable 16 MiB GPU slots were connected by CUDA events and a 12-item
overlap window. Measured H2D time was about 0.56 s per pass.

Grouped prefill processed 256 prompt positions per pass and completed 1,023
passes. E196 did not improve weight transfer bandwidth or reduce weight
bytes. It reduced repeated old-KV attention work while weights were already
moving.

## Decode bandwidth accounting

The 12.76 GiB is per target verifier pass, not per emitted token. The decode
summary reports 48 passes, 256 emitted IDs (255 counted decode tokens), 5.312
emitted tokens per pass, and 78.2656 seconds of decode time. That is about
612 GiB of target-weight traffic across the decode, or 2.40 GiB per counted
token and 7.82 GiB/s averaged over decode time. The NVMe-sourced 2.76 GiB per
pass works out to about 132.5 GiB total, or 1.69 GiB/s averaged over decode.

The logged 0.56 seconds of H2D time per pass is the sum of CUDA event intervals
around asynchronous copy calls. Dividing 12.76 GiB by that interval gives an
event-derived active-copy rate of about 22.8 GiB/s; copies overlap verifier
compute. This is not an independent PCIe bandwidth counter, and the run record
does not capture the negotiated PCIe link width or speed. The inferred rate is
below PCI-SIG's nominal PCIe 4.0 x16 limit of 31.51 GB/s per direction, but
the benchmark does not independently establish sustained bus throughput
([PCI-SIG bandwidth table](https://pcisig.com/sites/default/files/files/PCI-SIG-Seamless_Transition_to_PCIe_5.0_in_System_Implementations_FINAL.pdf)).

## Attention fixture

The 32-position, one-layer fixture at the 256K context boundary improved from
454.759 ms to 56.137 ms, with bit-identical output and unchanged cache and
scratch sizes. The head-tile sweep and protection-boundary checks are recorded
in the accompanying logs. See [the fixture source](attention.cu),
[timings and output hashes](attention.log),
[head-tile results](head-tiles.log), and
[protection checks](protection-boundaries.log).

## Evidence files

- [`ctx-262144.json`](ctx-262144.json): run summary.
- [`ctx-262144.log`](ctx-262144.log): sanitized decoder trace and allocation
  plan.
- [`ctx-262144.vram.csv`](ctx-262144.vram.csv): sampled VRAM usage.
- [`compare.log`](compare.log): E195/E196 rates, peaks, and output-ID match.
- [`build.sha256`](build.sha256): hashes for the measured binary and source.
- [`e195-reference.patch`](e195-reference.patch): baseline attention source
  patch used by the fixture.

The original unredacted command, full 256K prompt-ID file, and local binary
remain in the machine-local `results/e196-old-tier-20261115/` record; they are
not part of this public evidence set. The included
`scripts/build_repeated_prompt_ids.py` recreates the input if given the same
frozen prompt-ID corpus. The full-run binary and source hashes are recorded in
`build.sha256`. The current checkout's `benchmarks/real_decode.cu` hash differs
from the recorded measurement hash. The 48-ID integration gate passed on the
current source, but the 256K timing and 256-ID equivalence run were not
repeated from it. Treat the throughput figures above as results from the
hashed historical build; a full-scale rerun on the current checkout remains
outstanding.
