#!/usr/bin/env python3
"""Time independent repository requests that share a tool transcript."""

import argparse
import json
import os
from pathlib import Path

from jinja2 import Environment

from bench_e170_job import (
    DEFAULT_TARGET, ROOT, decode, prepare_job, read_ids, render_ids, session,
)
from finetune_mtp import read_gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--launcher", type=Path, default=ROOT / "scripts/run_single_user.sh")
    parser.add_argument("--tokenizer", type=Path, default=Path.home() / "llama.cpp/build/bin/llama-tokenize")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    target = Path(os.environ.get("GUM_TARGET_MODEL", DEFAULT_TARGET))
    request, expected = prepare_job()
    expected[1]["touch"].sort()
    context, first_question = request.rsplit("\n\n", 1)
    context = context[context.index("`scripts/capture_when_idle.sh`"):]
    questions = [
        "Use the observed interruption to write a triage record for the maintainer. " + first_question,
        "Prepare a regression-test fixture for the interrupted capture. Return only JSON with touch "
        "(the filenames actually present), assert_before_fix (the current next group number), and "
        "assert_after_recovery (the group number to retry).",
    ]
    headers = [
        "Review this repository tool transcript.\n\n",
        "Review this edited repository tool transcript.\n\n",
    ]
    metadata, _, mapped, _ = read_gguf(target)
    mapped.close()
    template = Environment().from_string(metadata["tokenizer.chat_template"])
    vocabulary = metadata["tokenizer.ggml.tokens"]
    prompts = [
        render_ids(target, args.tokenizer, template,
                   [{"role": "user", "content": header + context + "\n\n" + question}],
                   args.output / f"job-{version * 2 + turn + 1}.ids")
        for version, header in enumerate(headers)
        for turn, question in enumerate(questions)
    ]
    shared = next(i for i, (left, right) in enumerate(zip(*prompts[:2])) if left != right)
    shared = shared // 32 * 32
    if not shared or prompts[0][:shared] == prompts[2][:shared]:
        raise RuntimeError("the edited prefix must invalidate the saved checkpoint")

    command = ["taskset", "-c", "0-3", str(args.launcher.resolve()), "--prefill-chunks", "8",
               "--session-tail", "1", "--speculate", "32", "--packed-verify", "1",
               "--pld", "0", "--draft-max", "31", "--ctx", "2048", "--gen-tokens", "384"]
    refs = [None] * len(prompts)
    for trial in range(args.repeats):
        control = f"control-{trial + 1}"
        cache = f"shared-prefix-{trial + 1}"
        runs = [(control, []), (cache, ["--session-prefix-tokens", str(shared)])]
        for arm, flags in runs if trial % 2 == 0 else reversed(runs):
            session(command + flags, args.output, arm, prompts, refs)
            for i in range(len(prompts)):
                result = args.output / f"{arm}-{i + 1}"
                log = result.with_suffix(".log").read_text()
                if ",completion=eos" not in log:
                    raise RuntimeError(f"{result.name} did not finish")
                if flags and f"reused_tokens={shared if i % 2 else 0},completion=eos" not in log:
                    raise RuntimeError(f"unexpected prefix reuse for {result.name}")
                ids = read_ids(result.with_suffix(".ids"))
                answer = "{" + decode(ids, vocabulary)
                record = json.loads(answer)
                if "touch" in record:
                    record["touch"].sort()
                if record != expected[i % 2]:
                    raise RuntimeError(f"incorrect task result for {result.name}: {record}")
                if refs[i] is None:
                    refs[i] = ids
                result.with_suffix(".answer.txt").write_text(answer + "\n")


if __name__ == "__main__":
    main()
