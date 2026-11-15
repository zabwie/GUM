# Benchmarks and validation

## Recommended correctness gate

`make test` builds the current CUDA decoder and runs one end-to-end check on
the RTX 3060 configuration used for E196. It processes a frozen 7,840-token
prompt with Qwen3.8-27B STRICT Q4_0, native MTP, zero resident target weights,
and a 4 GiB host-cache budget. It passes only when all 48 generated token IDs
match the recorded E195 reference.

The gate exercises model loading, grouped prefill, the tiered weight path,
TurboQuant-TTKV attention, MTP speculation, and the session token protocol. It
is the project's one recommended hardware integration test. It is not a
quality evaluation across diverse prompts and is not expected to run in a
generic CPU-only CI worker.

Supply the model files through `GUM_TARGET_MODEL` and `GUM_MTP_MODEL` as shown
in the [build instructions](../README.md#build-and-run-the-validation-gate).
The test takes roughly two minutes on the recorded RTX 3060 host; runtime
depends on hardware, memory pressure, and model storage.

## E196: 256K controlled comparison

The full result used Qwen3.8-27B STRICT Q4_0, an RTX 3060 12 GiB, context
262,144, 261,888 prompt IDs, and 256 output IDs. The prompt was made by
repeating prose segments from the frozen prompt set. It provides a same-input
long-context comparison, not a novel prompt-quality suite.

| Measurement | E195 | E196 |
|---|---:|---:|
| Profile | zero hot weights, 10 GiB host cache | zero hot weights, 10 GiB host cache |
| Prefill | 8.99 tok/s | 46.41 tok/s |
| Prefill time | 8 h 5 min | 1 h 34 min |
| Decode | 0.761 tok/s | 3.258 tok/s |
| Mean verifier pass | 6.863 s | 1.512 s |
| Decoder-process peak | 4,308 MiB | 4,308 MiB |
| Total-device peak | 6,211 MiB | 6,250 MiB |
| Output agreement | — | 256/256 token IDs match E195 |

E196 ran 48 decode cycles after prefill. NVML sampled device memory at 200 ms
intervals. The decoder's peak allocation was unchanged; total-device peak
rose by 39 MiB outside the decoder. Prefill and decode rates are specific to
this model, input, profile, hardware, and run.

Each verifier pass streamed 12.76 GiB of target weights: 10 GiB from host
cache (3,658 MiB pinned and the rest pageable) and 2.76 GiB from NVMe. Measured
H2D time was about 0.56 s per pass. Eight 32-position prefill chunks shared
each pass across up to 256 prompt positions, for 1,023 grouped passes.

E196's attention fixture reduced the 32-position old-KV attention time from
454.759 ms to 56.137 ms per layer, with bit-identical output and unchanged
cache and scratch allocation sizes. The optimized kernel shares packed KV
loads across six query heads attached to one KV head. Weight paging and its
measured transfer volume did not change from E195; attention reuse accounts
for the full-run improvement.

## Evidence and reproduction

The [E196 result folder](../results/e196/README.md) includes the recorded JSON
summary, sanitized run log, device-memory samples, comparison, attention
fixture source, and exactness results. The source tree records the build
inputs. `scripts/build_repeated_prompt_ids.py` reconstructs the 256K prompt
from the frozen prompt-ID corpus when supplied with `--prompt-dir`. That
corpus and the original local model paths are kept outside the public result
set, so a fresh clone cannot regenerate the exact full-run input without that
data.

Run the short exactness gate with `make test`. A full 256K rerun requires the
same model and MTP files, the constructed prompt, approximately 21 GiB of
available system RAM for this profile, and more than 90 minutes for prefill
on the recorded host. Do not compare a result against the table unless the
model, input, context, and memory profile match.
