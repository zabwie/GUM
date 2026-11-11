# results/ — raw evidence

One file per measured run, named `eXXX-<what>-<date/time>.txt` (plus `.log`, `.nsys-rep`
profiler captures and a few pre-E000 capability files). This directory is intentionally flat:
[EXPERIMENTS.md](../EXPERIMENTS.md) cites these files by exact path, so renaming or grouping
them would break the evidence trail.

To find the runs behind a claim, look up the experiment number in EXPERIMENTS.md; the entry
names its raw result files. `.nsys-rep` files need Nsight Systems to open. Gitignored:
`*.csv`, `*.sqlite`.
