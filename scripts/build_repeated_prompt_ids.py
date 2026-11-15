#!/usr/bin/env python3
"""Construct the E196-style long-context input from frozen prompt IDs."""

import argparse
from pathlib import Path
import sys


def read_ids(path: Path) -> list[int]:
	return [int(value) for value in path.read_text().strip().split(",")]


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--prompt-dir",
		type=Path,
		required=True,
		help="directory containing the frozen prompt.*.ids input set",
	)
	parser.add_argument("--context", type=int, default=262144)
	parser.add_argument("--output-tokens", type=int, default=256)
	args = parser.parse_args()

	files = sorted(args.prompt_dir.glob("prompt.*.ids"))
	if not files:
		parser.error(f"No frozen prompt IDs found in {args.prompt_dir}")

	prose: list[int] = []
	header: list[int] | None = None
	tail: list[int] | None = None
	for path in files:
		ids = read_ids(path)
		if header is None:
			header, tail = ids[:6], ids[-128:]
		prose.extend(ids[6:-128])
	if not prose or header is None or tail is None:
		parser.error("Frozen prompt set contains no prose tokens")

	prompt_tokens = args.context - args.output_tokens
	needed = prompt_tokens - len(header) - len(tail)
	if needed <= 0:
		parser.error("Context must exceed the fixed chat framing tokens")
	prompt = header + (prose * (needed // len(prose) + 1))[:needed] + tail
	if len(prompt) != prompt_tokens:
		parser.error(f"Prompt construction produced {len(prompt)} IDs, expected {prompt_tokens}")

	print(f"{args.output_tokens}:" + ",".join(map(str, prompt)))
	print(
		f"Built {len(prompt)} prompt IDs from {len(files)} frozen files; "
		f"context={args.context}, output={args.output_tokens}.",
		file=sys.stderr,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
