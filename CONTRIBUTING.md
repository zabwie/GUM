# Contributing

GUM is research software. Contributions should improve useful single-user
latency or make the runtime easier to reproduce without weakening its
correctness contract.

## Hardware scope

The current project targets my NVIDIA RTX 3060 12 GiB machine and its
Linux/CUDA/x86 CPU environment. Default host flags assume AVX2, AVX-VNNI,
F16C, and FMA. It is experimental and is not promised to build or run on other
systems or GPUs. After I achieve a stable version, I plan to work toward
support for a broader range of hardware.

## Before changing a kernel or runtime path

1. Identify the active model, hardware, and memory profile.
2. State the bottleneck and the measurement that supports it.
3. Make one focused change and preserve the target token stream.
4. Run `make test` with the supported Qwen3.8-27B and MTP files before
   reporting a speedup.
5. Record the command, model quantization, exact input and output lengths,
   profile, elapsed time, decoder and device memory peaks, and raw evidence.

When the end-to-end test is not available, report that limitation and use the
smallest relevant fixture. Do not present a microbenchmark as end-to-end
performance.

## Research records

Keep prior experiments and failed hypotheses. Add a new numbered entry to
`EXPERIMENTS.md` when a change is part of the research history, and store
large or machine-specific outputs outside the default user path. Preserve
older test sources under `archive/` when a better recommended gate replaces
them.

Do not add model weights, user captures, prompt databases, runtime state,
compiled executables, or machine-specific credentials to version control.
The repository ignore rules cover common local artifacts; check `git status`
before sharing changes.
