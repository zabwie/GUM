# GUM philosophy

GUM treats memory capacity as a software design problem. When a quantized
model is larger than VRAM, host RAM and NVMe can hold the rest of its weights;
the runtime streams the current layer into a small reusable GPU working set.
This makes larger models runnable, while accepting that bytes crossing the
host-to-GPU link cost time.

The design goal is therefore to do more useful target-model work for every
byte transferred. Speculative verification checks several proposed tokens in
one target pass. Grouped prefill evaluates several prompt positions during the
same weight pass. Weight slots, host staging buffers, copy streams, and CUDA
events overlap movement with computation without allowing a slot to be reused
too early.

The measured cost can be viewed as:

```text
bytes per committed token
  = streamed weight bytes per verifier pass / accepted tokens per pass
    + KV and state traffic per committed token
```

This is a working model of this implementation, not a universal throughput
bound. Each term is a design choice to measure and improve. GUM uses explicit
staging for its weight path; it does not rely on demand-paged CUDA Unified
Memory to decide weight residency.

## Evidence rules

- Correctness comes before throughput. Compare generated IDs against a known
  good target stream and keep numerical or cache approximations visible.
- Compare the same model, input IDs, context, decode settings, and memory
  profile. State when any of them differ.
- Report decoder-process VRAM separately from total device use. Include
  hardware, model quantization, prompt and output lengths, and sampling method
  with each benchmark.
- Keep the raw evidence and the failing experiments. An optimization that
  does not survive a correctness gate is not a speedup.
- Treat a measured limit as a description of the tested design. Reframe it as
  an equation, identify its terms, and test which terms can change.

The 256K E196 run is an example of these rules: the exact input and settings
were reused from E195, the 256 generated IDs matched, and the speedup came
from reducing repeated old-KV work while weight paging remained unchanged.
See [the benchmark record](../results/e196/README.md).
