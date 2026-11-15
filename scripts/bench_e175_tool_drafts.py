#!/usr/bin/env python3
"""Propose maintenance answers from the repository tool output and verify them."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from jinja2 import Environment
from llama_cpp import Llama

from bench_e170_job import DEFAULT_TARGET, ROOT, decode, read_ids, render_ids, session
from finetune_mtp import read_gguf


def propose(prompt):
    """Extract facts from the tool transcript and format the requested record."""
    match = re.search(r"^ls: (.+)\nevaluator groups: (.*)\nwrapper next group: (\d+)\n"
                      r"wrapper next seed index: (\d+)\nInterrupted group to retry: (\d+)(?=$|<\|im_end\|>)",
                      prompt, flags=re.MULTILINE)
    if not match:
        return None
    files, groups, next_group, seed, retry = match.groups()
    request = prompt.rsplit("<|im_start|>user\n", 1)[-1]
    if all(key in request for key in ("touch", "assert_before_fix", "assert_after_recovery")):
        return {"touch": sorted(files.split()), "assert_before_fix": int(next_group),
                "assert_after_recovery": int(retry)}
    if all(key in request for key in ("visible_groups", "observed_next_group", "first_skipped_seed",
                                      "expected_retry_group")):
        count = re.search(r"\b(\d+|eight) fresh prompts\b", request)
        if count:
            batch = 8 if count[1] == "eight" else int(count[1])
            return {"visible_groups": groups.split(", ") if groups else [],
                    "observed_next_group": int(next_group), "first_skipped_seed": int(seed) - batch,
                    "expected_retry_group": int(retry)}
    return None


def changed_incident(directory, target, metadata, tokenizer_path):
    """Reproduce a second interrupted publish using the real cursor code."""
    crash_dir = directory / "capture"
    crash_dir.mkdir(exist_ok=True)
    for filename in ("group-012.bin", "group-012.txt", "group-013.txt"):
        (crash_dir / filename).touch()
    wrapper = (ROOT / "scripts/capture_when_idle.sh").read_text()
    cursor = wrapper[wrapper.index("GNUM=-1"):wrapper.index('SEEDS="$ROOT')]
    seed_expression = next(line for line in wrapper.splitlines() if line.startswith("first="))
    env = os.environ | {"CAPTURE_DIR": str(crash_dir.resolve()), "N_MEM": "4"}
    next_group, seed = map(int, subprocess.check_output(
        ["bash", "-c", cursor + seed_expression + '\nprintf "%d %d\\n" "$GNUM" "$first"'],
        env=env, text=True).split())
    files = sorted(path.name for path in crash_dir.iterdir())
    groups = sorted(path.stem for path in crash_dir.glob("group-*.bin"))
    prompt = (
        "Write a JSON triage record for this repository reproduction. The interrupted capture had "
        "4 fresh prompts. Output only the fields visible_groups (array of group names), "
        "observed_next_group (integer), first_skipped_seed (the first seed of the interrupted "
        "group), and expected_retry_group (integer).\n\n"
        f"ls: {' '.join(files)}\nevaluator groups: {', '.join(groups)}\n"
        f"wrapper next group: {next_group:03d}\nwrapper next seed index: {seed}\n"
        "Interrupted group to retry: 013"
    )
    expected = {"visible_groups": groups, "observed_next_group": next_group,
                "first_skipped_seed": seed - 4, "expected_retry_group": 13}
    template = Environment().from_string(metadata["tokenizer.chat_template"])
    ids = render_ids(target, tokenizer_path, template,
                     [{"role": "user", "content": prompt}], directory / "changed-job.ids")
    return ids, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--launcher", type=Path, default=ROOT / "scripts/run_single_user.sh")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/e170-launcher-repeat-20261112")
    parser.add_argument("--tokenizer", type=Path, default=Path.home() / "llama.cpp/build/bin/llama-tokenize")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    target = Path(os.environ.get("GUM_TARGET_MODEL", DEFAULT_TARGET))
    kv, _, mapped, _ = read_gguf(target)
    mapped.close()
    vocab = kv["tokenizer.ggml.tokens"]
    tokenizer = Llama(str(target), vocab_only=True, verbose=False)
    prompts = [read_ids(args.reference / f"job-{i}.ids", ",") for i in (1, 2)]
    refs = [read_ids(args.reference / f"baseline-{i}.ids") for i in (1, 2)]

    def draft_for(turn, prompt, corrupt=False):
        answer = propose(decode(prompt, vocab))
        if answer is None:
            return []
        if corrupt and turn == 1:
            answer["assert_before_fix"] += 1
        text = json.dumps(answer, indent=2)[1:]  # the rendered prompt ends in "{"
        return tokenizer.tokenize(text.encode(), add_bos=False)

    command = ["taskset", "-c", "0-3", str(args.launcher.resolve()), "--prefill-chunks", "8",
               "--session-tail", "1", "--speculate", "32", "--packed-verify", "1",
               "--pld", "0", "--draft-max", "31", "--ctx", "2048", "--gen-tokens", "384"]
    hints = command + ["--session-draft-hints", "1"]
    for i in range(args.repeats):
        arms = [(command, f"control-{i+1}", None), (hints, f"tool-draft-{i+1}", draft_for)]
        for cmd, name, drafter in arms if i % 2 == 0 else reversed(arms):
            session(cmd, args.output, name, prompts, refs, draft_for=drafter)
    session(hints, args.output, "rejected-draft", prompts, refs,
            draft_for=lambda turn, prompt: draft_for(turn, prompt, corrupt=True))
    if not re.search(r"^prefill_hint,proposed=\d+,matched=[1-9]\d*,committed=0,",
                     (args.output / "rejected-draft-2.log").read_text(), flags=re.MULTILINE):
        sys.exit("corrupted proposal did not exercise partial rejection")
    changed_prompt, changed_answer = changed_incident(args.output, target, kv, args.tokenizer)
    continuation = args.output / "continued-job.prompt.txt"
    continuation.write_text(
        decode(prompts[1], vocab) + decode(refs[1], vocab) +
        "<|im_end|>\n<|im_start|>user\n"
        "Return the retry group as a JSON object with the key retry_group.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n{")
    continued_ids = tokenizer.tokenize(continuation.read_bytes(), add_bos=False, special=True)
    if continued_ids[:len(prompts[1]) + len(refs[1])] != prompts[1] + refs[1]:
        sys.exit("continuation did not preserve the verified prefix")
    # Continue from accepted state, then reset on an independent reproduction.
    sequence = prompts + [continued_ids, changed_prompt]
    session(command, args.output, "changed-control", sequence, refs + [None, None])
    continued_ref = read_ids(args.output / "changed-control-3.ids")
    changed_ids = read_ids(args.output / "changed-control-4.ids")
    changed_record = json.loads("{" + decode(changed_ids, vocab))
    (args.output / "changed-oracle.json").write_text(json.dumps({
        "tool_result": changed_answer, "control_result": changed_record,
        "control_task_correct": changed_record == changed_answer,
    }, indent=2) + "\n")
    session(hints, args.output, "changed-draft", sequence, refs + [continued_ref, changed_ids],
            draft_for=lambda turn, prompt: draft_for(turn, prompt) if turn != 2 else [])
    if "prefill_hint," not in (args.output / "changed-draft-4.log").read_text():
        sys.exit("the changed incident did not exercise a draft")
    for log in args.output.glob("*.log"):
        if "completion=eos" not in log.read_text():
            sys.exit(f"{log} did not finish the job")


if __name__ == "__main__":
    main()
