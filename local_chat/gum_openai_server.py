#!/usr/bin/env python3
"""Small OpenAI Chat Completions adapter for the existing GUM Qwen runner."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import codecs
import ctypes
from functools import lru_cache
import fcntl
import hmac
import json
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from jinja2 import Environment, TemplateError
from llama_cpp import Llama, llama_token_to_piece
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from qwen_tools import TOOL_START, normalize_messages, parse_reply


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get(
    "GUM_TARGET_MODEL", Path.home() / "Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"))
MTP = Path(os.environ.get(
    "GUM_MTP_MODEL", Path.home() / "Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf"))
DECODER = Path(os.environ.get("GUM_DECODER", ROOT / "build/real_decode"))
PORT = int(os.environ.get("GUM_API_PORT", "8765"))
KV_CACHE = os.environ.get("GUM_KV_CACHE", "turboquant-ttkv")
# GUM_PROFILE picks the residency/context split. "chat" right-sizes the context to the
# working set (compaction keeps conversations bounded) and moves the freed VRAM into
# resident target weights: measured 2026-11-13, token-exact vs the 196K profile on the
# frozen 20-prompt suite, decode +39/+30/+21% at 128/1k/4k prompt lengths with MORE free
# VRAM (results/perf-auto/ctx32k). "196k" is the E181 long-output profile (131072-token
# outputs, 4.3 GiB KV tier) — same decoder, same code path, one env var away.
PROFILE = os.environ.get("GUM_PROFILE", "stream")
_PROFILE = {"stream": (32768, 32768, 0, 15600),    # pure streaming: ~3.6 GiB engine, ~8.7 GiB free, ~5.5 tok/s
            "nano": (32768, 32768, 1600, 14000),   # ~4.6 GiB engine, ~7.1 GiB free, 6.27 tok/s
            "lean": (32768, 32768, 3200, 12400),   # ~6.1 GiB engine, ~5.3 GiB free, 7.19 tok/s
            "chat": (32768, 32768, 6400, 8700),    # ~10.2 GiB engine, 10.79 tok/s on the frozen suite
            "196k": (196608, 131072, 3456, 10240)} # E181 long-output profile, 4.3 GiB KV tier
_pctx, _pgen, _phot, _pcache = _PROFILE[PROFILE]
CTX = int(os.environ.get("GUM_CTX", str(_pctx)))
GEN_TOKENS = int(os.environ.get("GUM_GEN_TOKENS", str(_pgen)))
HOT_MIB = os.environ.get("GUM_HOT_MIB", str(_phot))
CPU_ROWS = os.environ.get("GUM_CPU_ROWS", "0")   # E193: CPU row-split fraction (0 = off)
CACHE_MIB = os.environ.get("GUM_CACHE_MIB", str(_pcache))
API_KEY = os.environ.get("LOCAL_GUM_API_KEY", "local-gum-key")
MODEL_ID = "qwen3.8-27b-gum"

# GUM Adaptive (E183): runtime residency manager. The stream profile keeps the
# engine at ~3 GiB so desktop apps can use VRAM; when the desktop is NOT using
# it, that VRAM is wasted on nothing. This supervisor respawns the idle decoder
# with a larger --hot-mib (E182: token-exact, +0.72 tok/s per hot GiB) and
# shrinks back when free VRAM gets tight. Policy:
#   target_hot = hot_now + free_vram - RESERVE - MARGIN, clamped [0, CEILING]
#   grow if target >= hot+400 MiB stable for GROW_SAMPLES polls (hysteresis)
#   shrink if target <= hot-400 MiB for 2 polls, or 1 poll when free < reserve
#   never mid-request (guarded by the session lock); skipped after long prompts
#   (a respawn drops session KV and a long re-prefill costs more than it saves)
GUM_ADAPTIVE = os.environ.get("GUM_ADAPTIVE", "0") == "1"
ADAPT_CEILING = int(os.environ.get("GUM_ADAPT_CEILING_MIB", "6400"))
ADAPT_RESERVE = int(os.environ.get("GUM_ADAPT_RESERVE_MIB", "4600"))
ADAPT_MARGIN = int(os.environ.get("GUM_ADAPT_MARGIN_MIB", "400"))
ADAPT_LONG_PROMPT = int(os.environ.get("GUM_ADAPT_LONG_PROMPT", "2000"))
ADAPT_POLL_S = float(os.environ.get("GUM_ADAPT_POLL_S", "10"))
ADAPT_GROW_SAMPLES = int(os.environ.get("GUM_ADAPT_GROW_SAMPLES", "3"))
ADAPT_COOLDOWN_S = float(os.environ.get("GUM_ADAPT_COOLDOWN_S", "120"))


class GumSession:
    def __init__(self) -> None:
        if not TARGET.is_file() or not MTP.is_file() or not DECODER.is_file():
            raise SystemExit(f"Missing GUM decoder/model file: {TARGET}, {MTP}, or {DECODER}")

        # The vocab-only load maps tokenizer metadata without allocating target weights.
        self.vocab = Llama(str(TARGET), vocab_only=True, n_ctx=512, verbose=False)
        self.token_piece = lru_cache(maxsize=65536)(self._token_piece)
        self.template = Environment().from_string(self.vocab.metadata["tokenizer.chat_template"])
        self.template.globals["raise_exception"] = self.invalid_template
        self.eos = self.vocab.token_eos()
        self.proc: subprocess.Popen[str] | None = None
        self.gpu_lock = None
        self.lock = threading.Lock()
        self.hot_mib: int | None = None          # hot tier of the live decoder
        self.last_prompt_tokens = 0              # KV re-prefill cost of a respawn
        if GUM_ADAPTIVE:
            threading.Thread(target=self._adaptive_loop, daemon=True).start()

    @staticmethod
    def invalid_template(message: str) -> None:
        raise ValueError(message)

    def _token_piece(self, token: int) -> bytes:
        # The binding's detokenizer has a fixed 32-byte buffer, too small
        # for Qwen tokens such as long indentation and horizontal rules.
        buffer = ctypes.create_string_buffer(256)
        size = llama_token_to_piece(self.vocab._model.vocab, token, buffer, len(buffer), 0, False)
        if size < 0:
            buffer = ctypes.create_string_buffer(-size)
            size = llama_token_to_piece(self.vocab._model.vocab, token, buffer, len(buffer), 0, False)
        if size < 0:
            raise RuntimeError("GUM could not decode a Qwen token")
        return buffer.raw[:size]

    def detokenize(self, ids: list[int]) -> bytes:
        return b"".join(self.token_piece(token) for token in ids if token != self.eos)

    def _read_stdout(self, proc: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                lines.put(line.rstrip("\r\n"))
        finally:
            lines.put(None)

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
            self.proc = None
        if self.gpu_lock is not None:
            self.gpu_lock.close()
            self.gpu_lock = None

    def _spawn(self, hot_mib: int) -> None:
        command = [
            str(DECODER), "--model", str(TARGET), "--mtp", str(MTP),
            "--session", "1", "--session-tail", "1", "--session-stream", "1", "--prefill-chunks", "8",
            "--speculate", "32", "--packed-verify", "1", "--pld", "0",
            "--draft-max", "31", "--ctx", str(CTX), "--gen-tokens", str(GEN_TOKENS),
            "--hot-mib", str(hot_mib), "--cache-mib", CACHE_MIB, "--kv-cache", KV_CACHE, "--show-text", "0",
            "--cpu-rows", CPU_ROWS,
            "--prompt-ids", "-",
        ]
        benchmark_cpus = sorted(os.sched_getaffinity(0) & {0, 1, 2, 3, 4, 5, 6, 7})
        if benchmark_cpus and shutil.which("taskset"):
            command = ["taskset", "-c", ",".join(map(str, benchmark_cpus)), *command]
        self.lines = queue.Queue()
        self.gpu_lock = open("/tmp/gum-gpu.lock", "a")
        try:
            fcntl.flock(self.gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                pass_fds=(self.gpu_lock.fileno(),),
            )
        except BlockingIOError:
            self.close()
            raise RuntimeError("Another GUM job is using the GPU. Retry when it finishes.") from None
        except OSError:
            self.close()
            raise
        self.hot_mib = hot_mib
        threading.Thread(target=self._read_stdout, args=(self.proc, self.lines), daemon=True).start()

    @staticmethod
    def _free_vram_mib() -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            return int(out.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None

    def _adaptive_respawn(self, target: int, free: int, action: str) -> None:
        print(f"gum-api: adaptive,{action},hot_from={self.hot_mib},hot_to={target},free_vram_mib={free}", flush=True)
        self.close()
        try:
            self._spawn(target)
        except RuntimeError as exc:
            print(f"gum-api: adaptive_respawn_failed,error={exc}", flush=True)

    def _adaptive_loop(self) -> None:
        # E183 policy is calibrated at ctx 32768 (E182 law). Under 196k the engine
        # is ~10.4 GiB and free sits below reserve permanently — adaptive would
        # read that as pressure and destroy long-context sessions for nothing.
        if PROFILE != "stream":
            print(f"gum-api: adaptive_disabled,profile={PROFILE},reason=calibrated_for_stream_only", flush=True)
            return
        print(f"gum-api: adaptive_enabled,ceiling_mib={ADAPT_CEILING},reserve_mib={ADAPT_RESERVE},"
              f"poll_s={ADAPT_POLL_S}", flush=True)
        grow_streak = 0
        shrink_streak = 0
        last_action = 0.0
        while True:
            time.sleep(ADAPT_POLL_S)
            free = self._free_vram_mib()
            # Holding the session lock means a turn is NOT in flight — the only
            # safe moment to respawn the decoder.
            if free is None or not self.lock.acquire(blocking=False):
                continue
            try:
                hot = self.hot_mib
                if self.proc is None or hot is None or self.proc.poll() is not None:
                    continue
                target = max(0, min(ADAPT_CEILING, hot + free - ADAPT_RESERVE - ADAPT_MARGIN))
                long_ctx = self.last_prompt_tokens > ADAPT_LONG_PROMPT
                pressure = free < ADAPT_RESERVE
                if time.monotonic() - last_action < ADAPT_COOLDOWN_S:
                    continue
                if pressure:
                    # Desktop needs VRAM now; long-context guard yields only to
                    # hard pressure, where losing session KV beats starving it.
                    if target < hot and (free < 1500 or not long_ctx):
                        self._adaptive_respawn(target, free, "shrink")
                        last_action = time.monotonic()
                        grow_streak = shrink_streak = 0
                elif target <= hot - ADAPT_MARGIN:
                    shrink_streak += 1
                    grow_streak = 0
                    if shrink_streak >= 2 and not long_ctx:
                        self._adaptive_respawn(target, free, "shrink")
                        last_action = time.monotonic()
                        shrink_streak = 0
                elif target >= hot + ADAPT_MARGIN:
                    grow_streak += 1
                    shrink_streak = 0
                    if grow_streak >= ADAPT_GROW_SAMPLES:
                        self._adaptive_respawn(target, free, "grow")
                        last_action = time.monotonic()
                        grow_streak = 0
                else:
                    grow_streak = shrink_streak = 0
            finally:
                self.lock.release()

    def _wait_turn(self, on_tokens: Callable[[list[int]], None] | None,
                   on_prefill: Callable[[], None] | None) -> tuple[list[int], str]:
        output_ids: list[int] | None = None
        completion = "length"
        metrics = ""
        reuse = ""
        # Abort a stuck decoder, while allowing arbitrarily long active decode.
        idle_timeout = 600.0
        deadline = time.monotonic() + idle_timeout
        tail: deque[str] = deque(maxlen=20)
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(0.1, min(1.0, deadline - time.monotonic())))
            except queue.Empty:
                if self.proc and self.proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            deadline = time.monotonic() + idle_timeout
            if line.startswith(("session_delta,", "session_prefill,")):
                try:
                    if line.startswith("session_delta,") and on_tokens:
                        on_tokens([int(value) for value in line.split(",")[1:]])
                    elif line.startswith("session_prefill,") and on_prefill:
                        on_prefill()
                except ConnectionError:
                    self.close()
                    raise
                continue
            tail.append(line)
            if line.startswith("spec_summary,"):
                metrics = line
            elif line.startswith("session_turn,"):
                completion = "stop" if "completion=eos" in line else "length"
                reuse = line
            elif line.startswith("session_ids"):
                parts = line.split(",")
                output_ids = [int(value) for value in parts[1:] if value]
                break
        if output_ids is None:
            self.close()
            details = "\n".join(tail)
            raise RuntimeError("GUM did not finish the turn. " + details[-5000:])
        print(f"gum-api: {metrics} {reuse}", flush=True)
        return output_ids, completion

    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int, fit_context: bool,
                 on_tokens: Callable[[list[int]], None] | None = None,
                 on_prefill: Callable[[], None] | None = None) -> tuple[str, int, int, str]:
        rendered = self.template.render(
            messages=normalize_messages(messages),
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.vocab.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
        if fit_context:
            max_tokens = min(max_tokens, max(1, CTX - len(prompt_ids)))
        if len(prompt_ids) + max_tokens > CTX:
            raise ValueError(
                f"Prompt of {len(prompt_ids)} tokens exceeds the model context window of {CTX}; at most "
                f"{CTX - max_tokens} prompt tokens with the requested {max_tokens}-token output limit."
            )
        prompt = f"{max_tokens}:" + ",".join(map(str, prompt_ids))
        with self.lock:
            print(f"gum-api: admitted_prompt_tokens={len(prompt_ids)},output_budget={max_tokens}", flush=True)
            self.last_prompt_tokens = len(prompt_ids)
            if self.proc is not None and self.proc.poll() is not None:
                self.close()
            if self.proc is None:
                self._spawn(int(HOT_MIB))
            assert self.proc.stdin is not None
            try:
                self.proc.stdin.write(prompt + "\n")
                self.proc.stdin.flush()
            except BrokenPipeError:
                self.close()
                raise RuntimeError("GUM decoder exited before accepting the turn.") from None
            ids, finish_reason = self._wait_turn(on_tokens, on_prefill)

        visible_ids = [token for token in ids if token != self.eos]
        text = self.detokenize(visible_ids).decode("utf-8", errors="replace")
        return text, len(prompt_ids), len(visible_ids), finish_reason


SESSION = GumSession()


class Handler(BaseHTTPRequestHandler):
    server_version = "GUM-Local/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print("gum-api: " + fmt % args, flush=True)

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:3080")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:3080")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": MODEL_ID, "profile": PROFILE,
                             "loaded": SESSION.proc is not None and SESSION.proc.poll() is None,
                             "context": CTX, "max_tokens": GEN_TOKENS, "kv_cache": KV_CACHE,
                             "hot_mib": SESSION.hot_mib, "adaptive": GUM_ADAPTIVE, "cpu_rows": CPU_ROWS,
                             "free_vram_mib": SESSION._free_vram_mib()})
            return
        if self.path == "/v1/models":
            if not self._authorized():
                self._json(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})
                return
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}]})
            return
        self._json(404, {"error": {"message": "Not found", "type": "not_found"}})

    def _authorized(self) -> bool:
        value = self.headers.get("Authorization", "")
        candidate = value[7:] if value.startswith("Bearer ") else ""
        return hmac.compare_digest(candidate, API_KEY)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "Not found", "type": "not_found"}})
            return
        if not self._authorized():
            self._json(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})
            return
        created = int(time.time())
        response_id = "chatcmpl-" + uuid.uuid4().hex[:16]
        streaming = started = disconnected = False
        streamed_chars = 0
        pending_text = ""
        buffering_call = False
        functions = {}
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def event(payload: dict) -> None:
            nonlocal disconnected
            if disconnected:
                return
            chunk = {"id": response_id, "object": "chat.completion.chunk", "created": created,
                     "model": MODEL_ID, **payload}
            try:
                self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            except ConnectionError:
                disconnected = True

        def choice(delta: dict, finish: str | None = None) -> None:
            event({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

        def start_stream() -> None:
            nonlocal started, disconnected
            if started:
                return
            started = True
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:3080")
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                disconnected = True
            choice({"role": "assistant"})

        def send_tokens(tokens: list[int]) -> None:
            nonlocal streamed_chars, pending_text, buffering_call, disconnected
            if not buffering_call:
                pending_text += utf8.decode(SESSION.detokenize(tokens))
            start_stream()
            if buffering_call:
                size = 0
            elif functions and TOOL_START in pending_text:
                size = pending_text.index(TOOL_START)
                buffering_call = True
            elif functions:
                size = max(0, len(pending_text) - len(TOOL_START) + 1)
            else:
                size = len(pending_text)
            if size:
                choice({"content": pending_text[:size]})
                streamed_chars += size
                pending_text = pending_text[size:]
            if buffering_call and not size:
                try:
                    self.wfile.write(b": gum\n\n")
                    self.wfile.flush()
                except ConnectionError:
                    disconnected = True
            if disconnected:
                raise ConnectionResetError("Harness disconnected")

        def send_prefill_progress() -> None:
            start_stream()
            # Use a comment so prefill does not become visible assistant text.
            self.wfile.write(b": gum prefill\n\n")
            self.wfile.flush()
            if disconnected:
                raise ConnectionResetError("Harness disconnected")

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("Request body is empty or too large.")
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("A non-empty messages array is required.")
            tools = body.get("tools") or []
            if not isinstance(tools, list) or any(tool.get("type") != "function" for tool in tools):
                raise ValueError("Only function tools are supported.")
            if body.get("tool_choice") == "none":
                tools = []
            functions = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in tools}
            streaming = bool(body.get("stream"))
            # Respect per-call limits, including the lower cap on compaction.
            # The Harness route can also reduce the budget to the exact available space.
            requested = body.get("max_tokens", body.get("max_completion_tokens", GEN_TOKENS))
            max_tokens = min(max(1, int(requested)), GEN_TOKENS)
            fit_context = self.headers.get("X-GUM-Output-Budget") == "server"
            text, prompt_tokens, completion_tokens, finish_reason = SESSION.complete(
                messages, tools, max_tokens, fit_context, send_tokens if streaming else None,
                send_prefill_progress if streaming else None,
            )
        except (ValueError, TypeError, KeyError, AttributeError, TemplateError) as error:
            failure = {"error": {"message": str(error), "type": "invalid_request_error"}}
            if started:
                event(failure)
            else:
                self._json(400, failure)
            return
        except ConnectionError:
            return
        except Exception as error:
            print(f"gum-api request failed: {type(error).__name__}: {error}", flush=True)
            failure = {"error": {"message": str(error), "type": "server_error"}}
            if started:
                event(failure)
            else:
                self._json(503, failure)
            return

        text, tool_calls = parse_reply(text, functions)
        if tool_calls:
            finish_reason = "tool_calls"
        print(f"gum-api: prompt_tokens={prompt_tokens},completion_tokens={completion_tokens},"
              f"tool_calls={len(tool_calls)},finish_reason={finish_reason}", flush=True)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        if streaming:
            start_stream()
            if len(text) > streamed_chars:
                choice({"content": text[streamed_chars:]})
            for index, call in enumerate(tool_calls):
                choice({"tool_calls": [{"index": index, **call}]})
            choice({}, finish_reason)
            if (body.get("stream_options") or {}).get("include_usage"):
                event({"choices": [], "usage": usage})
            if not disconnected:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        message = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self._json(200, {
            "id": response_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        })


if __name__ == "__main__":
    print(f"GUM local endpoint: http://127.0.0.1:{PORT}/v1 (model {MODEL_ID})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
