#!/usr/bin/env python3
"""E195 needle-recall gate (NEXT_STEPS #60): does the compacted-row TTKV still
recall needles at depth?

One prompt carries five needle sentences at staggered depths (2K/6K/12K/24K/30K),
so a single prefill exercises the recent ring, the 4:1 middle tier, and the 16:1
old tier in one run. Needle codes contain digits, which the decoder classifies
as structural tokens -> protected groups -> individual compacted rows: the exact
path E195 changed. Owner warning baked in: all depths stay inside the 32K
trained window so a recall failure would point at the cache, not YaRN limits.

Usage: e195_needle.py <decoder-binary> <kv-mode> <out-prefix> [extra decoder flags...]
"""
import re
import subprocess
import sys

from llama_cpp import Llama

MODEL = "/home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"
TARGETS = [2048, 6144, 12288, 24576, 30720]
CODES = ["KITE-7341", "MARBLE-2918", "ORBIT-5603", "VELVET-8472", "CINDER-1096"]
ITEMS = ["harbor", "orchard", "lantern", "compass", "meadow"]

FILLER = ("The morning shift began with the usual checklist and a round of coffee. "
          "She sorted the invoices, flagged the overdue ones, and left the rest for "
          "the afternoon batch. By ten the loading dock was busy again, and the "
          "forklift operator waved as he passed the office window. Lunch came late "
          "and was eaten at the desk while the phone kept ringing. Nothing urgent "
          "was left by four, so the day closed with a quiet walk to the bus stop.")

QUERY = ("Before the checklist, five access codes were recorded for later lookup. "
         "Recall every access code together with its item name. Answer with one "
         "line per item, formatted exactly as: item = code.")


def build_prompt(tok) -> str:
    filler_ids = tok.tokenize(FILLER.encode(), add_bos=False, special=False)
    parts, pos = [], 0
    # Interleave needles and filler in document order; each needle is placed so
    # its START lands at the requested depth.
    schedule = sorted(zip(TARGETS, ITEMS, CODES))
    for depth, item, code in schedule:
        needle = f"The access code for the {item} is {code}. "
        nlen = len(tok.tokenize(needle.encode(), add_bos=False, special=False))
        while pos + nlen < depth:
            chunk = filler_ids[: min(len(filler_ids), depth - pos - nlen)]
            if not chunk:
                break
            parts.append(tok.detokenize(chunk).decode("utf-8", errors="replace"))
            pos += len(chunk)
        parts.append(needle)
        pos += nlen
    parts.append(QUERY)
    return "".join(parts)


def main() -> None:
    binary, kv_mode, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    extra = sys.argv[4:]
    tok = Llama(MODEL, vocab_only=True, verbose=False)
    text = build_prompt(tok)
    ntok = len(tok.tokenize(text.encode(), add_bos=False, special=False))
    print(f"prompt_tokens={ntok}", flush=True)

    cmd = ["taskset", "-c", "0-3", binary, "--model", MODEL, "--prompt", text,
           "--ctx", "33792", "--gen-tokens", "128", "--hot-mib", "3456",
           "--cache-mib", "10240", "--prefill-chunks", "8", "--show-text", "1",
           "--kv-cache", kv_mode, *extra]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    with open(f"results/e195-{prefix}.log", "w") as fh:
        fh.write(out)
    gen = ""
    for line in out.splitlines():
        if line.startswith("generated_text,"):
            gen = line[len("generated_text,"):]
    found = {item: re.search(rf"{item}\s*[=:]\s*([A-Z]+-\d+)", gen, re.I) for item in ITEMS}
    codes = {item: (m.group(1).upper() if m else "MISS") for item, m in found.items()}
    expect = dict(zip(ITEMS, CODES))
    hits = sum(codes[k] == expect[k] for k in ITEMS)
    print(f"{prefix},kv={kv_mode},recall={hits}/5," +
          ",".join(f"{k}={codes[k]}" for k in ITEMS), flush=True)


if __name__ == "__main__":
    main()
