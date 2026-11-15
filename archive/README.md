# Archive

This folder keeps supporting material that is no longer the primary user or
validation path. Files are retained for audit and reproduction; they have not
been deleted.

- [`tests/packed_scale_codec.cu`](tests/packed_scale_codec.cu) is the former
  focused codec parity test. The recommended release gate is now
  [`make test`](../README.md#build-and-run-the-validation-gate), which covers
  the integrated decoder and exact output IDs.
- `benchmarks/microbench/` and `benchmarks/probes/` preserve supporting
  research sources. They are documented in [`benchmarks/README.md`](../benchmarks/README.md)
  and are not part of the recommended user test path.
- [`build-artifacts/legacy-cuda-probes/`](build-artifacts/legacy-cuda-probes/)
  preserves the previously tracked compiled CUDA probes. These are
  machine-specific historical binaries; `make test` rebuilds the supported
  decoder from source and does not use them.
- Older experiment harnesses remain under `scripts/` and are summarized in
  [`scripts/README.md`](../scripts/README.md).
- [`docs/archive/`](../docs/archive/) holds superseded design notes and
  handoff history.
- [`EXPERIMENTS.md`](../EXPERIMENTS.md), [`RESULTS.md`](../RESULTS.md), and
  `results/` preserve the numbered experiments and their historical evidence.
  They are a research archive, not a list of tests that a user must run.

The research sources keep their original paths where experiment commands and
history link to them. New users should start with the README and run only the
recommended hardware gate.
