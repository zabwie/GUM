# GUM

GUM is an experimental CUDA inference runtime for running quantized language
models whose weights exceed available GPU memory. It treats VRAM, pinned and
pageable system RAM, and NVMe as explicit memory tiers, then reduces the cost
of moving weights by reusing each transfer across more verified work.

The project is built around one measurable objective: lower single-user
latency while preserving the target model's greedy token stream on gated
inputs. Every published throughput figure is tied to its hardware, model,
context, and memory profile.

## Best-supported result

E196 is the strongest verified long-context result in this repository. It ran
Qwen3.8-27B STRICT Q4_0 on an RTX 3060 12 GiB with a 261,888-token prompt and
256 generated tokens. Against E195 on the same input and settings, all 256
output token IDs matched.

| Profile and metric | E195 | E196 |
|---|---:|---:|
| Prefill, 256K context; zero hot weights, 10 GiB host cache | 8.99 tok/s | 46.41 tok/s |
| Decode, same profile and workload | 0.761 tok/s | 3.258 tok/s |
| Decoder-process peak VRAM | 4,308 MiB | 4,308 MiB |
| Total-device peak VRAM | 6,211 MiB | 6,250 MiB |

The E196 run used native MTP drafting, TurboQuant-TTKV, eight grouped
prefill chunks, and `--hot-mib 0 --cache-mib 10240`. Each verifier pass
streamed 12.76 GiB of target weights. The speedup came from reusing packed old
KV reads across query heads; the weight-paging profile stayed the same as the
baseline.

The 256K prompt was constructed by repeating prose segments from a frozen
prompt set. This is a controlled same-input comparison, not a diverse 256K
quality evaluation. Device-wide memory includes the desktop and other CUDA
processes; it is reported separately from decoder-process memory. See the
[benchmark record](results/e196/README.md) for run details, raw summaries,
and limits. The full-run build hashes are recorded there; the current checkout
passed the shorter 48-token gate but has not repeated the 256K benchmark. The
prompt-construction helper accepts the original frozen
token-ID corpus; that corpus is not distributed. The short release gate uses
separate included fixtures.

## Build and run the validation gate

The recommended release check is one end-to-end exact-token gate. It rebuilds
the decoder and compares 48 generated token IDs from a frozen 7,840-token
prompt against the E195 reference. It requires Linux, a compatible NVIDIA GPU
and CUDA toolkit, the Qwen3.8-27B STRICT Q4_0 model, and its matching native
MTP model. Model files are not included.

```sh
make decoder
GUM_TARGET_MODEL=/path/to/Qwen3.8-27B-STRICT-Q4_0.gguf \
GUM_MTP_MODEL=/path/to/mtp-Qwen3.8-27B-Q4_0.gguf \
make test
```

The build prefers `/usr/local/cuda/bin/nvcc` when present and otherwise uses
`nvcc` from `PATH`. It defaults to `CUDA_ARCH=sm_86` and the host flags used
on the RTX 3060 test machine. Set `NVCC` for a different toolkit, `CUDA_ARCH`
for a different GPU, and `HOST_FLAGS` for a different x86 CPU. The defaults
expect AVX2 and AVX-VNNI.

This is a hardware integration check, not a portable CPU-only test. The
recorded 256K benchmark is separate and takes substantially longer. See
[the validation and benchmark notes](docs/BENCHMARKS.md) before comparing
results across machines.

## Design and research record

- [GUM philosophy](docs/PHILOSOPHY.md): the design objective and evidence rules.
- [Architecture](docs/DESIGN.md): runtime, weight tiers, KV cache, and kernels.
- [Benchmark sources](benchmarks/README.md): active runtime versus archived probes.
- [Benchmark methodology](docs/BENCHMARKS.md): selected measurements and
  reproduction limits.
- [Local chat adapter](local_chat/README.md): connect the decoder to the local
  OpenAI-compatible harness.
- [Scripts](scripts/README.md): supported entry points and research helpers.
- [Contributing](CONTRIBUTING.md): correctness and benchmark requirements.
- [Archive guide](archive/README.md): where historical experiments and
  supporting probes are retained.

The research ledger includes 103 experiment entries numbered through E196.
They are historical evidence, not 200 separate tests. The primary public
validation path is `make test`; earlier probes and one-off tests are retained
for audit and reproduction.

## License

This repository does not yet include a license. No general permission to use,
modify, or redistribute the code has been granted. A license must be selected
before publishing it as an open-source project.
