# Scripts

The supported entry points for a new checkout are:

- `make decoder` builds the current CUDA runtime.
- `make test` runs the one recommended exact-token integration gate.
- `run_single_user.sh` launches a local decoder session after the build and
  model paths are set.
- `build_repeated_prompt_ids.py` reconstructs the E196 long-context input
  when the original frozen prompt-ID corpus is available.

The remaining `bench_*`, training, capture, and codec scripts preserve
experiment-specific workflows. They are not required to build or validate
the current decoder. Read the matching E-numbered entry in `EXPERIMENTS.md`
before using one. Machine-local seed files, captures, models, and run outputs
are excluded from version control.
