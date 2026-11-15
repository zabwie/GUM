# Benchmark sources

`real_decode.cu` and its supporting headers implement the current inference
runtime. Build the supported decoder with `make decoder`; the output is
`build/real_decode`.

`microbench/` and `probes/` retain source for historical measurements and
kernel investigations. They are research artifacts, not additional release
tests. The repository has one recommended hardware integration gate:
`make test`.

Previously tracked probe executables are preserved under
[`archive/build-artifacts/legacy-cuda-probes/`](../archive/build-artifacts/legacy-cuda-probes/).
They are machine-specific historical artifacts and do not match the supported
build contract. Build the current decoder from source before running it.
