# Local Qwen agent in DeepSeek Harness

This connects the installed DeepSeek Harness Web UI to the existing GUM Qwen3.8-27B STRICT Q4_0 runner. It uses the local checkpoint and MTP draft weights; it does not download model files. The local GUM endpoint binds only to `127.0.0.1`.

## Start

Run `make decoder`, set `GUM_TARGET_MODEL` and `GUM_MTP_MODEL`, then run `./local_chat/start.sh` with the installed DeepSeek Harness package. Open the Web URL printed by the launcher to sign in. Start a new session with **GUM Agent** and **Qwen3.8-27B (GUM local)**. Select the workspace where you want it to work. The first prompt loads the 27B model. Later turns reuse the loaded model. Press Ctrl+C in the launch terminal to stop both services.

The model has a **196,608-token context** and up to **131,072 output tokens per request**. That leaves 65,536 tokens for the prompt, tool schemas, tool results, and history when reserving the full output budget. GUM Agent exposes Harness's filesystem, search, and shell tools with workspace permissions and local conversation compaction. The decoder uses 31 native MTP draft nodes, 32-position packed verification, eight prefill chunks, 3,456 MiB target weight residency, and a 10,240 MiB host-cache budget. Text streams as the decoder commits it. The MTP proposer keeps an 8,192-token rolling window. The target reads the full tiered context.

The default `turboquant-ttkv` cache keeps recent K/V in FP32, uses TurboQuant-MSE-style 4-bit keys and 4-bit grouped values for older rows, and pools eligible 4- and 16-token groups by age. Structural markers, paths, identifiers and numbers preserve their individual packed rows. Older K/V is approximate. An adapter launched with `GUM_KV_CACHE=fp32` defaults to a 24,576-token context and 8,192 output tokens. Match the Harness provider's context and output settings when using a different capacity.

The adapter logs `spec_summary` for each request. `tok_per_s` measures committed decode tokens, and `request_tok_per_s` includes prefill. Both omit process startup and browser overhead. The [benchmark notes](../docs/BENCHMARKS.md) distinguish the verified 256K research run from this local chat configuration.

The adapter defaults to `build/real_decode`. Set `GUM_DECODER=/path/to/real_decode` when the decoder is stored elsewhere.

The endpoint can be checked at `http://127.0.0.1:8765/health`. Its OpenAI-compatible model id is `qwen3.8-27b-gum`.
