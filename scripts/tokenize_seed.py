#!/usr/bin/env python3
"""Greedy byte-level tokenizer for Qwen capture seeds.

Uses the model's GGUF vocabulary and preserves chat markers. This is an
approximation to BPE. Emits one comma-separated ID sequence per seed, with
no automatic BOS. With no --text, reads one seed per stdin line (\n escapes
encode line breaks inside a seed).
"""
import argparse
import os
import re
import sys

from finetune_mtp import read_gguf


def byte_alphabet():
    # Qwen's byte-level vocabulary maps the remaining byte values above Latin-1.
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    alphabet = [chr(b) for b in range(256)]
    for i, b in enumerate(sorted(set(range(256)) - visible)):
        alphabet[b] = chr(256 + i)
    return alphabet


def encode(text, vocab, markers, alphabet):
    ids = []
    for part in markers.split(text):
        if markers.fullmatch(part):
            ids.append(vocab[part])
            continue
        data = "".join(alphabet[b] for b in part.encode("utf-8"))
        pos = 0
        while pos < len(data):
            for end in range(min(len(data), pos + 32), pos, -1):
                token_id = vocab.get(data[pos:end])
                if token_id is not None:
                    ids.append(token_id)
                    pos = end
                    break
            else:
                raise ValueError(f"no GGUF token for byte symbol {data[pos]!r}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"))
    ap.add_argument("--text", help="tokenize one seed instead of reading seed lines from stdin")
    args = ap.parse_args()

    kv, _, model_data, _ = read_gguf(args.model)
    model_data.close()
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv["tokenizer.ggml.token_type"]
    vocab = {token: i for i, token in enumerate(tokens)}
    special = [token for token, kind in zip(tokens, types) if kind in (3, 4) and token]
    pattern = "|".join(re.escape(token) for token in sorted(special, key=len, reverse=True))
    markers = re.compile(f"({pattern})" if pattern else r"(?!)")
    alphabet = byte_alphabet()
    seeds = [args.text] if args.text is not None else (
        line.rstrip("\n").replace(r"\n", "\n") for line in sys.stdin
    )
    for seed in seeds:
        try:
            ids = encode(seed, vocab, markers, alphabet)
        except ValueError as error:
            ap.error(str(error))
        if not ids:
            ap.error("empty seed")
        print(",".join(map(str, ids)))


if __name__ == "__main__":
    main()
