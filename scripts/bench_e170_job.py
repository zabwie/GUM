#!/usr/bin/env python3
"""Time two turns of a real E169 maintenance job on one GPU sequence."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from jinja2 import Environment

from finetune_mtp import read_gguf
from tokenize_seed import byte_alphabet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path.home() / "Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"
GEN_LIMIT = 384
CTX = 2048


def prepare_job():
    wrapper = (ROOT / "scripts/capture_when_idle.sh").read_text()
    evaluator = (ROOT / "scripts/finetune_mtp.py").read_text()
    publish = wrapper[wrapper.index("    declare -A new_hashes"):].strip()
    cursor = wrapper[wrapper.index("GNUM=-1"):wrapper.index("SEEDS=\"$ROOT")]
    seed = next(line for line in wrapper.splitlines() if line.startswith("first="))
    discover = next(line.strip() for line in evaluator.splitlines() if "groups = sorted(" in line)
    files = ["group-004.txt", "group-004.bin", "group-005.txt"]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for name in files:
            (directory / name).touch()
        env = os.environ | {"CAPTURE_DIR": str(directory), "N_MEM": "8"}
        command = cursor + seed + "\nprintf '%s %s\\n' \"$GNUM\" \"$first\"\n"
        next_group, first_seed = map(int, subprocess.check_output(["bash", "-c", command], env=env).split())
        visible = sorted(path.stem for path in directory.glob("group-*.bin"))
    tool_result = (
        f"ls: {' '.join(sorted(files))}\n"
        f"evaluator groups: {', '.join(visible)}\n"
        f"wrapper next group: {next_group:03d}\n"
        f"wrapper next seed index: {first_seed}\n"
        "Interrupted group to retry: 005\n"
    )
    request = (
        "Turn this observed E169 capture crash into a triage record for the maintainer. "
        "Group 004 published successfully. Group 005 died after publishing its .txt "
        "but before moving the .bin. It contained eight fresh prompts.\n\n"
        f"`scripts/capture_when_idle.sh` cursor and publication code:\n```bash\n{cursor}{seed}\n{publish}\n```\n\n"
        f"`scripts/finetune_mtp.py` group discovery:\n```python\n{discover}\n```\n\n"
        f"Output from the isolated on-disk reproduction:\n```text\n{tool_result}```\n\n"
        "Output only JSON with the fields visible_groups (array of group names), "
        "observed_next_group (integer), first_skipped_seed (integer, 8 times the interrupted "
        "group number), and expected_retry_group (integer)."
    )
    return request, [
        {"visible_groups": visible, "observed_next_group": next_group,
         "first_skipped_seed": first_seed - 8, "expected_retry_group": 5},
        {"touch": files, "assert_before_fix": next_group, "assert_after_recovery": 5},
    ]


FOLLOWUP = (
    "Turn that review into a regression-test fixture. Return a single JSON object with keys "
    "touch (an array of filenames for the isolated directory), assert_before_fix (the next "
    "group number today), and assert_after_recovery (the group number to retry). Include the "
    "published and interrupted .txt and .bin files only where they would actually exist."
)


def render_ids(target, tokenizer, template, messages, destination):
    text = template.render(messages=messages, tools=[], add_generation_prompt=True, enable_thinking=False)
    text += "{"
    destination.with_suffix(".prompt.txt").write_text(text)
    output = subprocess.check_output(
        [tokenizer, "-m", target, "--ids", "--no-bos", "--no-escape", "--log-disable",
         "-f", destination.with_suffix(".prompt.txt")],
        text=True,
    )
    ids = json.loads(output)
    if len(ids) + GEN_LIMIT > CTX:
        raise RuntimeError(f"prompt has {len(ids)} tokens and exceeds the configured context budget")
    destination.write_text(",".join(map(str, ids)) + "\n")
    return ids


def decode(ids, vocabulary):
    inverse = {char: byte for byte, char in enumerate(byte_alphabet())}
    return bytes(inverse.get(char, ord(char)) for token in ids for char in vocabulary[token]).decode(
        "utf-8", errors="replace"
    )


def save_summary(directory, arm, turn, seconds, ids, prompt_ids, log, reference=None):
    match = re.search(r"^spec_summary,(.*)$", log, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing decoder summary for {arm} turn {turn}")
    metrics = dict(part.split("=", 1) for part in match.group(1).split(","))
    hit = re.search(r"^session_turn,reused_tokens=(\d+)", log, flags=re.MULTILINE)
    row = {
        "arm": arm,
        "turn": turn,
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(ids),
        "completed_before_limit": 0 < len(ids) < GEN_LIMIT,
        "seconds": round(seconds, 3),
        "single_user_tok_s": round(len(ids) / seconds, 3),
        "decode_tok_s": float(metrics["tok_per_s"]),
        "decode_tokens": int(metrics["decode_tokens"]),
        "passes": int(metrics["cycles"]),
        "emitted_per_pass": float(metrics["mean_emitted_per_cycle"]),
        "prefill_s": float(metrics["prefill_s"]),
        "min_free_vram_mib": int(metrics["min_free_vram_MiB"]),
        "reused_prompt_tokens": int(hit.group(1)) if hit else 0,
        "exact": ids == reference if reference is not None else None,
    }
    print(json.dumps(row), flush=True)
    with (directory / "timings.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    if reference is not None and ids != reference:
        raise RuntimeError(f"output drift for {arm} turn {turn}")


def read_ids(path, separator=None):
    return list(map(int, path.read_text().strip().split(separator)))


def request(cmd, directory, arm, turn, prompt_ids, reference=None):
    prefix = directory / f"{arm}-{turn}"
    ids_path = prefix.with_suffix(".ids")
    log_path = prefix.with_suffix(".log")
    arguments = [*cmd, "--prompt-ids", ",".join(map(str, prompt_ids)), "--out-ids", ids_path]
    with log_path.open("w") as stream:
        start = time.perf_counter()
        subprocess.run(arguments, stdout=stream, stderr=subprocess.STDOUT, check=True)
        seconds = time.perf_counter() - start
    ids = read_ids(ids_path)
    save_summary(directory, arm, turn, seconds, ids, prompt_ids, log_path.read_text(), reference)
    return ids


def session(cmd, directory, arm, prompts, refs, draft_for=None):
    def request_line(turn):
        prompt = prompts[turn]
        line = ",".join(map(str, prompt))
        draft = draft_for(turn, prompt) if draft_for else []
        return line + "|" + ",".join(map(str, draft)) if draft else line

    start = time.perf_counter()
    cmd = [*cmd, "--session", "1", "--prompt-ids", request_line(0)]
    with tempfile.TemporaryFile(mode="w+") as error, subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error,
        text=True, errors="replace",
    ) as proc:
        try:
            for turn, prompt in enumerate(prompts, 1):
                if turn > 1:
                    start = time.perf_counter()
                    proc.stdin.write(request_line(turn - 1) + "\n")
                    proc.stdin.flush()
                lines = []
                for line in proc.stdout:
                    lines.append(line)
                    if line.startswith("session_ids,"):
                        seconds = time.perf_counter() - start
                        ids = list(map(int, line.rstrip().split(",")[1:]))
                        break
                else:
                    error.seek(0)
                    raise RuntimeError(f"session exited before turn {turn}: {error.read()}")
                prefix = directory / f"{arm}-{turn}"
                prefix.with_suffix(".ids").write_text("\n".join(map(str, ids)) + "\n")
                prefix.with_suffix(".log").write_text("".join(lines))
                save_summary(directory, arm, turn, seconds, ids, prompt, "".join(lines), refs[turn - 1])
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            error.seek(0)
            raise RuntimeError(error.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("arm", choices=[
        "baseline", "wide", "narrow", "no-history", "eagle", "eagle-packed",
        "session", "session-eagle", "session-eagle-packed",
        "session-no-history", "session-no-history-31", "session-no-history-31-sequential",
    ])
    parser.add_argument("--binary", type=Path, default=ROOT / "benchmarks/real_decode")
    parser.add_argument("--target", type=Path, default=os.environ.get("GUM_TARGET_MODEL", DEFAULT_TARGET))
    parser.add_argument("--tokenizer", type=Path, default=Path.home() / "llama.cpp/build/bin/llama-tokenize")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cmd = ["taskset", "-c", "0-3", args.binary.resolve(), "--model", args.target,
           "--ctx", str(CTX), "--gen-tokens", str(GEN_LIMIT), "--hot-mib", "8064",
           "--cache-mib", "5600", "--show-text", "0"]
    if "eagle" in args.arm:
        cmd += ["--eagle4", ROOT / "models/Qwen3.8-27B-EAGLE4-Q4_0.gguf"]
        if args.arm.endswith("-packed"):
            cmd += ["--speculate", "24", "--packed-verify", "1"]
    else:
        cmd += ["--mtp", os.environ.get("GUM_MTP_MODEL", args.target.with_name("mtp-Qwen3.8-27B-Q4_0.gguf")),
                "--speculate", "64", "--packed-verify", "1"]
    if args.arm == "wide":
        cmd += ["--draft-max", "39", "--p-min", "0.025"]
    if args.arm == "narrow":
        cmd += ["--p-min", "0.25"]
    if "no-history" in args.arm:
        cmd += ["--pld", "0"]
    if "-31" in args.arm:
        cmd += ["--draft-max", "31"]
    if args.arm.endswith("-sequential"):
        cmd += ["--mtp-group", "1"]

    if args.arm == "baseline":
        kv, _, mapped, _ = read_gguf(args.target)
        mapped.close()
        template = Environment().from_string(kv["tokenizer.chat_template"])
        request_text, expected = prepare_job()
        messages = [{"role": "user", "content": request_text}]
        for turn in (1, 2):
            prompt_ids = render_ids(args.target, args.tokenizer, template, messages, output / f"job-{turn}.ids")
            ids = request(cmd, output, args.arm, turn, prompt_ids)
            if not 0 < len(ids) < GEN_LIMIT:
                raise RuntimeError(f"turn {turn} did not finish before the output limit")
            answer = "{" + decode(ids, kv["tokenizer.ggml.tokens"])
            (output / f"answer-{turn}.txt").write_text(answer + "\n")
            response = json.loads(answer)
            if turn == 2:
                response["touch"] = sorted(response["touch"])
                expected[1]["touch"] = sorted(expected[1]["touch"])
            if response != expected[turn - 1]:
                raise RuntimeError(f"turn {turn} output disagrees with the on-disk reproduction: {response}")
            messages.extend([{"role": "assistant", "content": answer}, {"role": "user", "content": FOLLOWUP}])
    else:
        prompts = [read_ids(output / f"job-{turn}.ids", ",") for turn in (1, 2)]
        refs = [read_ids(output / f"baseline-{turn}.ids") for turn in (1, 2)]
        if args.arm.startswith("session"):
            session(cmd, output, args.arm, prompts, refs)
        else:
            for turn, prompt in enumerate(prompts, 1):
                request(cmd, output, args.arm, turn, prompt, refs[turn - 1])


if __name__ == "__main__":
    main()
