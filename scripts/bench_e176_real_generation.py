#!/usr/bin/env python3
"""Run fresh long greedy prompts against llama.cpp and GUM target-only/MTP paths."""

import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from jinja2 import Environment
from llama_cpp import Llama

from finetune_mtp import read_gguf


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get(
    "GUM_TARGET_MODEL", Path.home() / "Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"))
MTP = Path(os.environ.get(
    "GUM_MTP_MODEL", Path.home() / "Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf"))
BINARY = ROOT / "results/e176-real-generation-20261112/real_decode"
OUTPUT = ROOT / "results/e176-real-generation-20261112"
MAX_TOKENS = 512
CTX = 2048

PROMPTS = [
    {
        "id": "reasoning_microgrid",
        "system": "You are an operations-research analyst. Reason from the supplied data, show calculations, and distinguish assumptions from conclusions.",
        "user": "A coastal town has a 6 MW solar field, a 4 MW wind farm, and a battery rated 5 MWh with a 2 MW discharge limit. At 18:00, demand is 8 MW and forecast renewable supply is 3 MW for the next three hours. The battery is at 90% charge, but the emergency plan requires keeping at least 25% charge for possible overnight outages. A diesel unit can supply up to 4 MW at a variable fuel cost. The town is considering a voluntary demand response that can reduce load by 1.5 MW for two hours, but only if the price signal is announced by 16:00. Analyze the decision using only these facts. State the energy balance and the battery's usable energy under the reserve rule; compare dispatching diesel, using demand response, and combining them; identify what cannot be decided without efficiency, ramp, and forecast-error data; and propose a robust operating rule for the evening. Explain at least one case where a locally cheaper choice creates a larger outage risk. Write a detailed answer of at least 850 words with explicit arithmetic and a final recommendation.",
    },
    {
        "id": "coding_event_ledger",
        "system": "You are a senior Python engineer. Produce executable, maintainable standard-library code and explain its invariants and failure cases.",
        "user": "Design a Python 3 module for reconciling an append-only event stream from three warehouse scanners. Each event has event_id, device_id, device_sequence, item_id, event_type (RECEIVE, MOVE, SHIP, or RETRACT), location, and optional target_event_id. Delivery is at least once and may be out of order. A repeated event_id with identical fields is a duplicate; the same event_id with different fields is a conflict that must be quarantined. RETRACT can cancel only an earlier accepted event from the same device, and a retracted RETRACT does not restore its target. A shipment is valid only if the item is currently at the stated location. Do not invent timestamps or assume cross-device ordering. Give the data structures, a deterministic processing algorithm, Python code for the core ledger and validation path, and a set of table-driven tests covering duplicates, conflicting IDs, late arrivals, invalid shipments, valid retractions, and retraction cycles. State which results are provisional when device streams have gaps. Write at least 850 words in total; include enough code to make the core behavior directly implementable.",
    },
    {
        "id": "technical_dma_race",
        "system": "You are a systems engineer teaching a technical audience. Explain mechanisms precisely and use the stated numbers rather than claiming external measurements.",
        "user": "A data acquisition board has a DMA engine that writes 4 KiB records into a 64-slot ring. The CPU consumes records using a producer index and a consumer index. The DMA engine wraps at slot 63. Under load, a record occasionally appears to contain a new header and an old payload. The board and CPU are coherent for ordinary cached memory, but the descriptor ring is allocated as write-combined memory. The interrupt is raised after the DMA engine updates the producer index. Explain several plausible causes, including ordering, cacheability, ownership, wraparound, and interrupt timing. Give a step-by-step debugging plan that separates a device-side partial write from a CPU visibility bug, and propose a safe ownership protocol with sequence numbers and barriers. Include a concrete 64-slot wraparound example and explain why adding a delay is not a correctness fix. Use only the supplied setup, label assumptions, and write a detailed technical explanation of at least 850 words.",
    },
    {
        "id": "instruction_following_incident_runbook",
        "system": "You are an incident-response lead. Follow output constraints exactly and make each action operationally testable.",
        "user": "Write a runbook for a fictional service that accepts signed telemetry batches. The service has three regions, a queue, a signature verifier, and a durable object store. Operators report that a batch is visible in the queue but absent from the object store, and retries sometimes create duplicate notifications. Use exactly these four top-level headings: Objective, Steps, Escalation, Verification. Under Steps, provide exactly eight numbered steps. Each step must name an observable check and an action; do not tell operators to delete data or disable signature verification. In Escalation, distinguish an invalid signature, a stalled queue, and a storage timeout. In Verification, define how to show that one batch has one durable object and one notification after retry. Do not assume a region is healthy just because its dashboard is green. Write at least 850 words overall while preserving the exact headings and exactly eight numbered steps.",
    },
    {
        "id": "design_review_privacy_search",
        "system": "You are a skeptical architecture reviewer. Compare alternatives fairly, identify hidden costs, and do not claim that a design is private merely because it is encrypted.",
        "user": "A library consortium wants search across 40 independently administered catalogs. Each catalog has 2 to 8 million records and may update hourly. Users should search titles and author names without sending raw queries to every catalog. The consortium can afford one regional coordinator, but each library controls its own key and may disconnect for a day. Candidate designs are: (A) send encrypted queries to every library and decrypt results at the coordinator; (B) build a replicated index from normalized metadata; (C) use local query agents that return bounded ranked results; and (D) use a hybrid index for common fields plus local agents for exact or recent data. Analyze threat boundaries, metadata leakage, stale results, revocation, ranking comparability, bandwidth, and outage behavior. Propose an evaluation plan with measurable acceptance criteria and explain what evidence would make you reject your favored design. State assumptions explicitly. Write a detailed review of at least 850 words, including a decision matrix and a phased rollout recommendation.",
    },
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path):
    return [int(value) for value in path.read_text().split()]


def compare_ids(actual, reference):
    common = 0
    for left, right in zip(actual, reference):
        if left != right:
            break
        common += 1
    return {"exact": actual == reference, "common_prefix_tokens": common,
            "first_difference_at": None if actual == reference else common}


def render_prompts():
    kv, _, mapped, _ = read_gguf(TARGET)
    mapped.close()
    template = Environment().from_string(kv["tokenizer.chat_template"])
    tokenizer = Llama(str(TARGET), vocab_only=True, verbose=False)
    rows = []
    for item in PROMPTS:
        start = time.perf_counter()
        text = template.render(
            messages=[{"role": "system", "content": item["system"]},
                      {"role": "user", "content": item["user"]}],
            tools=[], add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.tokenize(text.encode(), add_bos=False, special=True)
        (OUTPUT / f"{item['id']}.prompt.txt").write_text(text)
        (OUTPUT / f"{item['id']}.prompt.ids").write_text("\n".join(map(str, ids)) + "\n")
        rows.append({"id": item["id"], "prompt_tokens": len(ids),
                     "prepare_seconds": time.perf_counter() - start,
                     "text": str(OUTPUT / f"{item['id']}.prompt.txt"),
                     "ids": str(OUTPUT / f"{item['id']}.prompt.ids")})
    del tokenizer
    gc.collect()
    return rows


def run_gum(rows, arm):
    result_rows = []
    for index, row in enumerate(rows):
        prefix = OUTPUT / f"{row['id']}.{arm}"
        ids_path = Path(str(prefix) + ".ids")
        log_path = Path(str(prefix) + ".log")
        command = ["taskset", "-c", "0-3", str(BINARY),
                   "--model", str(TARGET), "--ctx", str(CTX),
                   "--gen-tokens", str(MAX_TOKENS), "--hot-mib", "8064",
                   "--cache-mib", "5600", "--show-text", "0",
                   "--prompt-ids", ",".join(map(str, read_ids(Path(row["ids"])))),
                   "--out-ids", str(ids_path), "--measure-h2d", "1"]
        if arm == "gum-mtp":
            command.extend(["--mtp", str(MTP), "--speculate", "32",
                            "--packed-verify", "1", "--draft-max", "31",
                            "--p-min", "0.06", "--pld", "0",
                            "--prefill-chunks", "8"])
        else:
            command.extend(["--speculate", "0"])
        command_text = " ".join(command)
        start = time.perf_counter()
        with log_path.open("w") as log:
            log.write("command=" + command_text + "\n")
            log.flush()
            env = os.environ | {"CUDA_VISIBLE_DEVICES": "0"}
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        seconds = time.perf_counter() - start
        output_ids = read_ids(ids_path) if ids_path.exists() else []
        log_text = log_path.read_text(errors="replace")
        summary = re.search(r"^spec_summary,(.*)$", log_text, flags=re.MULTILINE)
        dense = re.search(r"^summary,(.*)$", log_text, flags=re.MULTILINE)
        summary_fields = {}
        if summary:
            summary_fields = dict(part.split("=", 1) for part in summary.group(1).split(","))
        cycle_rows = []
        for line in log_text.splitlines():
            if line.startswith("spec_cycle,"):
                fields = dict(part.split("=", 1) for part in line.split(",")[2:])
                cycle_rows.append({"emitted": int(fields["emitted"]),
                                   "seconds": float(fields["seconds"]),
                                   "cold_bytes": int(fields["cold_bytes"]),
                                   "cold_h2d_ms": float(fields["cold_h2d_ms"])})
        result = {
            "id": row["id"], "arm": arm, "exit_code": completed.returncode,
            "prompt_tokens": row["prompt_tokens"], "output_tokens": len(output_ids),
            "wall_seconds_process_start_to_exit": seconds,
            "wall_tokens_per_second": len(output_ids) / seconds if seconds else 0,
            "prepared_end_to_end_seconds": seconds + row["prepare_seconds"],
            "prepared_end_to_end_tokens_per_second": len(output_ids) / (seconds + row["prepare_seconds"]),
            "completed_at_limit": len(output_ids) == MAX_TOKENS,
            "completion": "token_limit" if len(output_ids) == MAX_TOKENS else "eos_or_error",
            "decode_tokens": int(summary_fields.get("decode_tokens", 0)),
            "decode_seconds": float(summary_fields.get("decode_s", 0)),
            "decode_tokens_per_second": float(summary_fields.get("tok_per_s", 0)),
            "prefill_seconds": float(summary_fields.get("prefill_s", 0)),
            "request_tokens_per_second": float(summary_fields.get("request_tok_per_s", 0)),
            "cycles": cycle_rows,
            "config_line": next((line for line in log_text.splitlines() if line.startswith("config,")), None),
            "spec_config_line": next((line for line in log_text.splitlines() if line.startswith("spec_config,")), None),
            "summary_line": summary.group(0) if summary else (dense.group(0) if dense else None),
        }
        result_rows.append(result)
        with (OUTPUT / "timings.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
    return result_rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompt_rows = render_prompts()
    kv, tensors, mapped, _ = read_gguf(TARGET)
    mapped.close()
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": str(TARGET), "target_sha256": sha256(TARGET),
        "mtp": str(MTP), "mtp_sha256": sha256(MTP),
        "gum_binary": str(BINARY), "gum_binary_sha256": sha256(BINARY),
        "target_manifest": {key: kv.get(key) for key in (
            "general.architecture", "general.name", "general.file_type",
            "general.quantization_version", "qwen35.block_count",
            "qwen35.embedding_length", "qwen35.feed_forward_length")},
        "target_tensor_count": len(tensors),
        "max_generated_tokens": MAX_TOKENS, "context_tokens": CTX,
        "sampling": {"method": "greedy argmax", "temperature": 0.0,
                     "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                     "repeat_penalty": 1.0, "frequency_penalty": 0.0,
                     "presence_penalty": 0.0},
        "prompts": prompt_rows,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Both paths execute every target layer on CUDA. Host memory stages weights;
    # it does not run target-model layers. Dense GUM is the target-only control,
    # not an independent llama.cpp implementation.
    dense_rows = run_gum(prompt_rows, "gum-dense")
    mtp_rows = run_gum(prompt_rows, "gum-mtp")
    dense_by_id = {row["id"]: row for row in dense_rows}
    for row in mtp_rows:
        dense_path = OUTPUT / f"{row['id']}.gum-dense.ids"
        mtp_path = OUTPUT / f"{row['id']}.gum-mtp.ids"
        row["dense_gum_comparison"] = (
            compare_ids(read_ids(mtp_path), read_ids(dense_path))
            if dense_path.exists() and mtp_path.exists() else None)
    successful = [row for row in mtp_rows if row["exit_code"] == 0 and row["output_tokens"]]
    total_committed = sum(row["output_tokens"] for row in successful)
    total_process_seconds = sum(row["wall_seconds_process_start_to_exit"] for row in successful)
    total_prepared_seconds = sum(row["prepared_end_to_end_seconds"] for row in successful)
    total_decode_tokens = sum(row["decode_tokens"] for row in successful)
    total_decode_seconds = sum(row["decode_seconds"] for row in successful)
    total_request_seconds = sum(row["prefill_seconds"] + row["decode_seconds"] for row in successful)
    process_rates = [row["wall_tokens_per_second"] for row in successful]
    request_rates = [row["request_tokens_per_second"] for row in successful]
    decode_rates = [row["decode_tokens_per_second"] for row in successful]
    cycle_rates = [cycle["emitted"] / cycle["seconds"]
                   for row in successful for cycle in row["cycles"] if cycle["seconds"]]
    aggregate = {
        "successful_runs": len(successful),
        "runs_with_at_least_512_committed_tokens": sum(row["output_tokens"] >= 512 for row in successful),
        "committed_output_tokens": total_committed,
        "process_wall_seconds": total_process_seconds,
        "process_wall_aggregate_tokens_per_second": total_committed / total_process_seconds if total_process_seconds else 0,
        "prepared_end_to_end_seconds": total_prepared_seconds,
        "prepared_end_to_end_aggregate_tokens_per_second": total_committed / total_prepared_seconds if total_prepared_seconds else 0,
        "request_seconds_including_prefill": total_request_seconds,
        "request_prefill_plus_decode_aggregate_tokens_per_second": total_committed / total_request_seconds if total_request_seconds else 0,
        "decode_committed_tokens": total_decode_tokens,
        "decode_seconds": total_decode_seconds,
        "novel_prompt_sustained_decode_tokens_per_second": total_decode_tokens / total_decode_seconds if total_decode_seconds else 0,
        "peak_cycle_committed_tokens_per_second": max(cycle_rates, default=0),
        "process_wall_per_run_tokens_per_second": {"median": statistics.median(process_rates) if process_rates else 0,
            "minimum": min(process_rates, default=0), "maximum": max(process_rates, default=0)},
        "request_prefill_plus_decode_per_run_tokens_per_second": {"median": statistics.median(request_rates) if request_rates else 0,
            "minimum": min(request_rates, default=0), "maximum": max(request_rates, default=0)},
        "decode_only_per_run_tokens_per_second": {"median": statistics.median(decode_rates) if decode_rates else 0,
            "minimum": min(decode_rates, default=0), "maximum": max(decode_rates, default=0)},
    }
    report = {"prompt_count": len(prompt_rows), "target_tokens_per_prompt": MAX_TOKENS,
              "comparison": "GUM GPU-only speculative path versus GUM GPU-only dense path; not an independent implementation",
              "gum_dense": dense_rows, "aggregate": aggregate, "gum_mtp": mtp_rows}
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
