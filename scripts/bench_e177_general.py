#!/usr/bin/env python3
"""Frozen no-hint GUM workloads, with paired exact-output and task checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import statistics
import subprocess
import sys
import time

from jinja2 import Environment
from llama_cpp import Llama

from bench_e170_job import DEFAULT_TARGET, ROOT
from finetune_mtp import read_gguf


# Freeze the suite before collecting E177 results. The second member of every
# category is held out from candidate selection.
SPLITS = {
    "development": {
        "coding": "A Python service drops events when two workers call a TTL cache with the same missing key. The fetch may raise, invalidation can arrive during a fetch, and timestamps come from a monotonic clock. Write a corrected single-flight cache with a small concurrency test and explain the invalidation rule.",
        "reasoning": "A relief depot has 18 crates in North and 12 in South. Sites A, B, C need 12, 10, 8. Costs per crate from North are 1, 4, 7 and from South are 3, 2, 1. North-to-A has capacity 7 and South-to-C has capacity 6. Find a minimum-cost allocation and prove optimality. Then explain how the allocation changes if North-to-A can take 9.",
        "technical": "Explain why an append-only data file can survive a power loss while its newly created filename disappears. Give an atomic replacement sequence on a local filesystem, the assumptions for each sync, and a small fault-injection test matrix.",
        "structured": "A manifest lists sealed jobs as ink 7, copper 11, glass 4, and an unsealed job chalk 9. Return only JSON with keys eligible (sealed names in alphabetical order), total (sum of their counts), and largest (name of the largest sealed job).",
        "long": "Compare the river gauge maintenance records. Determine which stations must be revisited under the bulletin and explain how conflicting entries should be resolved. Include a brief handover note.",
    },
    "heldout": {
        "coding": "A Python parser reads UTF-8 CSV chunks from a pipe. It incorrectly treats every newline as a record boundary. Design a streaming standard-library reader that handles split UTF-8, CRLF, and quoted embedded newlines. Include executable code and tests, and explain the malformed-input policy.",
        "reasoning": "A ferry can carry at most 12 pallets or 15 tonnes per trip. Loads contain six 2.7-tonne pallets, eleven 1.3-tonne pallets, and eight 0.6-tonne pallets. Heavy pallets can only travel on the first and third departures. Find a minimum trip schedule and explain why fewer trips fail.",
        "technical": "A compiler team sees a speedup from a change that removes stores, but hardware counters report more LLC misses and a larger instruction footprint. Explain mechanisms consistent with the observations and design a controlled experiment to distinguish them.",
        "structured": "Return only JSON with keys late (alphabetically sorted names) and over (total minutes past deadline). Ada arrived 08:43 for 08:45, Inez 08:56 for 08:45, Theo 08:30 for 08:30, and Noor 09:01 for 08:52.",
        "long": "Compare the river gauge maintenance records. Identify which stations can reopen under the bulletin and which need more evidence. Explain how to handle duplicate reports and give an auditable decision summary.",
    },
}
STRUCTURED = {
    "development": {"eligible": ["copper", "glass", "ink"], "total": 22, "largest": "copper"},
    "heldout": {"late": ["Inez", "Noor"], "over": 20},
}
RECORDS = """North district log. Ash Bridge recorded a stable 1.92 m at 06:30 and 1.91 m at 06:40. The operator listed the 06:40 reading as a verification, but the two reports share instrument certificate A-17. A field technician left a note about mud around the reference mark. The photograph from 07:15 shows the staff gauge clear up to the lowest visible graduation. No independent electronic reading has arrived.

East district log. Lake Cut stopped reporting after a crew unplugged the cell modem during a battery swap. The local logger still held five minute observations and exported them at 10:52. The report includes logger serial E-55, its replacement battery lot, and a photograph of the calibration plate. A separate contractor took a manual reading from a surveyed reference point at 11:04 and signed the entry.

South district log. Tern Weir reported 2.01 m at 07:05, 2.02 m at 07:18, and 1.97 m at 08:32. The 07:05 upload says gauge SW-8 while the second upload shows SW-8R. The procurement list says the replacement SW-8R was delivered Tuesday. A supervisor found a hand-written swap note that gives no time. The 08:32 observation came from a second instrument outside the intake and is physically independent.

West district log. Alder Bend was offline during road resurfacing. At noon a patrol sent a photograph of the meter, serial W-19, and reported that the safety fence had been moved. A contractor later measured the staff gauge from the bank. The two signed readings agree within tolerance, and a patrol manager confirmed that the fence was put back. Access was limited until the manager gave clearance in a separate message.

Incident review, earlier shift. Automated email marked Lake Cut and Ash Bridge both available. The email was based on a heartbeat, and a later incident review explained that heartbeats can arrive when a battery disconnect prevents measurement. A radio transcript lists Tern Weir as clear, but it refers to clearing access debris rather than checking the water-level sensor. The maintenance calendar describes W-19 as calibrated last month. It gives no corresponding dates for the backup meter.

Bulletin. Reopening requires two independently obtained measurements within tolerance, a traceable instrument identifier, and separate clearance of any known access issue. Multiple uploads from one instrument count once. If an instrument was swapped and the time is unknown, hold the station until a signed record establishes which reading came from which instrument. A heartbeat or status email may be used to route a visit but is insufficient for reopening."""
TOOL_FILES = {
    "development": ("plant.tsv", "line\tstate\tunits\nL8\tready\t11\nL2\thold\t7\nL6\tready\t13\n", 24),
    "heldout": ("outbound.tsv", "line\tstate\tunits\nG3\tready\t5\nG9\tready\t18\nG1\thold\t22\n", 23),
}
FLAGS = ["--session", "1", "--session-tail", "1", "--prefill-chunks", "8", "--speculate", "32",
         "--packed-verify", "1", "--pld", "0", "--draft-max", "31", "--ctx", "2048",
         "--gen-tokens", "256", "--show-text", "0", "--session-draft-hints", "0"]


def record(line):
    return dict(field.split("=", 1) for field in line.strip().split(",") if "=" in field)


def quantile(values, percent):
    if not values:
        return None
    return sorted(values)[max(0, (len(values) * percent + 99) // 100 - 1)]


def parse_json(text):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        return None


def run_suite(split, destination, template, tokenizer, command):
    destination.mkdir(parents=True)
    prompts = list(SPLITS[split].items())
    tool_file, contents, ready_total = TOOL_FILES[split]
    (destination / tool_file).write_text(contents)
    prompts += [("tool", f"Look up today's warehouse counts in {tool_file}. The only available "
                         'tool is read_file. First request it by returning only JSON in the form '
                         '{"tool":"read_file","file":"<filename>"}. Wait for the result.')]
    rows = []
    proc = None
    conversation = []
    tool_result = ""
    tool_seconds = 0.0
    try:
        for task, prompt in prompts + [("tool_followup", "")]:
            prefix = destination / task
            started = time.monotonic()
            if task == "tool_followup":
                conversation.append({"role": "user", "content": tool_result + "\nReturn only JSON "
                                     'with the key "ready_total" and the total units in ready rows.'})
            else:
                conversation = [{"role": "user", "content": (RECORDS + "\n\n" if task == "long" else "") + prompt}]
            rendered = template.render(messages=conversation, tools=[], add_generation_prompt=True,
                                       enable_thinking=False)
            ids = tokenizer.tokenize(rendered.encode(), add_bos=False, special=True)
            payload = ",".join(map(str, ids))
            prefix.with_suffix(".prompt.txt").write_text(rendered)
            prefix.with_suffix(".prompt.ids").write_text(payload + "\n")
            request_started = time.monotonic()
            if proc is None:
                invocation = command + ["--prompt-ids", payload]
                (destination / "command.txt").write_text(shlex.join(invocation) + "\n")
                proc = subprocess.Popen(invocation, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, errors="replace")
            else:
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
            cycles = []
            ids_out = None
            metrics = {}
            turn = {}
            with prefix.with_suffix(".log").open("w") as log:
                for line in proc.stdout:
                    observed = time.monotonic() - request_started
                    log.write(line)
                    if line.startswith("spec_cycle,"):
                        cycles.append(record(line) | {"observed_s": observed})
                    elif line.startswith("spec_summary,"):
                        metrics = record(line)
                    elif line.startswith("session_turn,"):
                        turn = record(line)
                    elif line.startswith("session_ids,"):
                        ids_out = list(map(int, line.rstrip().split(",")[1:]))
                        break
            if ids_out is None or not metrics:
                raise RuntimeError(f"decoder exited during {task}; see {prefix.with_suffix('.log')}")
            finished = time.monotonic()
            eos_id = tokenizer.token_eos()
            text = tokenizer.detokenize([i for i in ids_out if i != eos_id]).decode(errors="replace")
            prefix.with_suffix(".ids").write_text("\n".join(map(str, ids_out)) + "\n")
            prefix.with_suffix(".answer.txt").write_text(text)
            completion = turn["completion"]
            parsed = parse_json(text)
            task_pass = None
            if task == "structured":
                task_pass = parsed == STRUCTURED[split]
            elif task == "tool":
                task_pass = parsed == {"tool": "read_file", "file": tool_file}
                conversation.append({"role": "assistant", "content": text})
                before_tool = time.monotonic()
                if task_pass:
                    reply = subprocess.check_output(["cat", "--", str(destination / tool_file)], text=True)
                    tool_result = f"read_file({tool_file}) returned:\n{reply}"
                else:
                    tool_result = "No tool ran because the request was invalid."
                tool_seconds = time.monotonic() - before_tool
            elif task == "tool_followup":
                task_pass = rows[-1]["task_pass"] and parsed == {"ready_total": ready_total}
            ttft_estimate = max(0, cycles[0]["observed_s"] - float(cycles[0]["seconds"])) if cycles else observed
            # The decoder commits a burst per cycle. From the first token onward,
            # charge the wait to the first token in a burst and zero to the rest.
            arrivals = [ttft_estimate]
            for cycle in cycles:
                arrivals.extend([cycle["observed_s"]] * int(cycle["emitted"]))
            gaps = [max(0, right - left) for left, right in zip(arrivals, arrivals[1:])]
            byte_count = sum(int(c["cold_bytes"]) for c in cycles)
            dma_s = sum(float(c.get("cold_h2d_ms", 0)) for c in cycles) / 1000
            row = {"task": task, "prompt_tokens": len(ids), "committed": len(ids_out) - ids_out.count(eos_id),
                   "completion": completion, "task_pass": task_pass, "session": "cold" if not rows else "warm",
                   "tokenize_and_prepare_s": request_started - started, "wall_s": finished - request_started,
                   "ttft_estimated_s": ttft_estimate, "itl_median_s": statistics.median(gaps) if gaps else None,
                   "itl_p95_s": quantile(gaps, 95), "h2d_weight_bytes": byte_count,
                   "h2d_weight_s": dma_s, "effective_weight_h2d_gb_s": byte_count / dma_s / 1e9 if dma_s else None,
                   "tool_s": tool_seconds if task == "tool" else 0.0, "decode": metrics, "turn": turn}
            if task == "tool_followup":
                row["full_tool_task_s"] = (rows[-1]["wall_s"] + rows[-1]["tokenize_and_prepare_s"]
                                           + tool_seconds + finished - started)
            rows.append(row)
            with (destination / "timings.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        raise
    finally:
        if proc is not None:
            proc.stdin.close()
            try:
                status = proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise
            if status != 0 and not sys.exc_info()[0]:
                raise RuntimeError(f"decoder exited with {proc.returncode}; see {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("--binary", type=Path, help="candidate decoder, otherwise use the installed launcher")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cpu", default="0-3")
    parser.add_argument("--flags", default="", help="space-separated experimental flags")
    args = parser.parse_args()
    target = Path(os.environ.get("GUM_TARGET_MODEL", DEFAULT_TARGET))
    mtp = Path(os.environ.get("GUM_MTP_MODEL", DEFAULT_TARGET.with_name("mtp-Qwen3.8-27B-Q4_0.gguf")))
    launcher = ROOT / "scripts/run_single_user.sh"
    invocation = [str(launcher)] if args.binary is None else [str(args.binary.resolve()), "--model", str(target),
        "--mtp", str(mtp), "--hot-mib", "8064", "--cache-mib", "5600"]
    command = ["stdbuf", "-oL", "-eL", "taskset", "-c", args.cpu, *invocation,
               *FLAGS, "--measure-h2d", "1", *shlex.split(args.flags)]
    args.output.mkdir(parents=True)
    suite = {"splits": SPLITS, "structured": STRUCTURED, "records": RECORDS, "tool_files": TOOL_FILES,
             "flags": FLAGS}
    data = json.dumps(suite, indent=2, sort_keys=True) + "\n"
    (args.output / "suite.json").write_text(data)
    print(f"suite_sha256={hashlib.sha256(data.encode()).hexdigest()}", flush=True)
    if args.prepare_only:
        return
    if os.environ.get("GUM_DRAFTER", "mtp") != "mtp" or os.environ.get("GUM_RESIDENT", "0") != "0":
        parser.error("use default GUM_DRAFTER and GUM_RESIDENT")
    kv, _, mapped, _ = read_gguf(target)
    mapped.close()
    template = Environment().from_string(kv["tokenizer.chat_template"])
    tokenizer = Llama(str(target), vocab_only=True, verbose=False)
    run_suite(args.split, args.output / args.split, template, tokenizer, command)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    (args.output / "process.json").write_text(json.dumps({"child_user_s": usage.ru_utime,
        "child_system_s": usage.ru_stime, "child_peak_rss_kib": usage.ru_maxrss}, indent=2) + "\n")


if __name__ == "__main__":
    main()
