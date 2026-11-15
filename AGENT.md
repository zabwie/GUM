# GUM development rules

GUM is a measurement-led project about memory hierarchy and inference latency.
Use the constraints to guide experiments; do not report a performance claim
that the evidence does not support.

## Before measuring

- Verify the binary, active profile, model, context, and relevant flags.
- Compare runs only when hardware, model, inputs, output length, and memory
  profile match. Name the profile beside every throughput figure.
- Check system and GPU headroom before a long benchmark. Keep workloads
  bounded and avoid overlapping GPU runs.

## Before promoting a change

- State the bottleneck and a falsifiable hypothesis.
- Change one relevant factor at a time and retain failed measurements.
- Run `make test` with the supported target and MTP models. Token agreement
  is the gate for performance claims.
- Record model, exact input and output lengths, profile, hardware, command,
  elapsed time, process and device memory peaks, and result files.
- Preserve research history. Move superseded tests to `archive/`; do not
  erase measurements to make a result look cleaner.

## Reporting

Separate measurements from estimates and hypotheses. Identify a constructed
or repeated prompt as such. Distinguish decoder-process VRAM from total-device
use. Report negative results and environment limits plainly. See
[`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the design principles and
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the recommended validation gate.

The previous internal operating note is preserved at
[`archive/AGENT-legacy.md`](archive/AGENT-legacy.md).
