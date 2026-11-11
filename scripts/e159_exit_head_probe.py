#!/usr/bin/env python3
"""E159: exit-head feasibility probe for HalfDraft.

Batched mid-depth captures (layers 8,16,24,32,40,48,56 + final pre-norm
residual) for the 30 E132 prompts, 8 lockstep sequences per run. Record
layout: [pos, n_layers, B, per-layer B rows, final B rows], float32.

Measures per exit depth, train/val split 24/6 prompts:
  1. zero-shot: argmax(shared_head(norm(h_mid))) vs the true next token;
  2. ridge linear map h_mid -> h_final, evaluated through the shared head.

Gate: val agreement >= 0.8 at some depth makes HalfDraft worth building.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch

WIDTH = 5120
LAYERS = [8, 16, 24, 32, 40, 48, 56]


def load_group(cap_dir, g):
    raw = np.fromfile(cap_dir / f"batch-group{g}.bin", dtype=np.float32)
    meta = json.loads((cap_dir / f"group{g}-lens.json").read_text())
    members, pad_to = meta["members"], meta["pad_to"]
    # the decoder replicates prompts[0] to fill the lockstep batch; the record
    # header carries the actual sequence count (may exceed len(members))
    header_stride = 3 + 0  # [pos, n_layers, B]
    first_B = int(raw[2])
    per = header_stride + first_B * (len(LAYERS) + 1) * WIDTH
    count = len(raw) // per
    rec = raw[: count * per].reshape(count, per)
    seqs = first_B
    mids = rec[:, 3: 3 + len(LAYERS) * seqs * WIDTH].reshape(count, len(LAYERS), seqs, WIDTH)
    final = rec[:, 3 + len(LAYERS) * seqs * WIDTH: 3 + (len(LAYERS) + 1) * seqs * WIDTH]
    final = final.reshape(count, seqs, WIDTH)
    lines = (cap_dir / f"batch-group{g}.ids").read_text().strip().splitlines()
    # only the first len(members) sequences are real
    gen = [[int(x) for x in line.strip(",").split(",") if x != ""] for line in lines][:len(members)]
    out = []
    for s, prompt_idx in enumerate(members):
        n = min(count, len(gen[s]) - 1)
        out.append({"prompt": prompt_idx, "mids": mids[:n, :, s], "final": final[:n, s],
                    "labels": gen[s][1:n + 1]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", default="/tmp/vanta-e158")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device
    cap_dir = Path(args.captures)

    target = gguf.GGUFReader(args.model)
    head_t = next(t for t in target.tensors if t.name == "output.weight")
    norm_t = next(t for t in target.tensors if t.name == "output_norm.weight")
    head = torch.from_numpy(np.asarray(
        gguf.quants.dequantize(head_t.data, head_t.tensor_type), dtype=np.float32)).half().to(device)
    nw = torch.from_numpy(np.array(norm_t.data, dtype=np.float32)).to(device)

    def argmax_of(hidden):
        t = hidden.to(device).float()
        normed = t * (t.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * nw
        return (normed.half() @ head.T).float().argmax(-1)

    data = []
    for g in range(4):
        data.extend(load_group(cap_dir, g))
    data.sort(key=lambda d: d["prompt"])
    train, val = data[:24], data[24:]
    print(f"train={sum(len(d['labels']) for d in train)} positions, "
          f"val={sum(len(d['labels']) for d in val)} positions", flush=True)

    for k, depth in enumerate(LAYERS):
        zhit = ztot = 0
        for d in val:
            zhit += int((argmax_of(torch.from_numpy(d["mids"][:, k])) ==
                         torch.tensor(d["labels"], device=device)).sum())
            ztot += len(d["labels"])
        X = torch.from_numpy(np.concatenate([d["mids"][:, k] for d in train])).float().to(device)
        Y = torch.from_numpy(np.concatenate([d["final"] for d in train])).float().to(device)
        lam = 3e2
        G = X.T @ X + lam * torch.eye(WIDTH, device=device) * (X.square().mean() / WIDTH)
        W = torch.linalg.solve(G, X.T @ Y)
        vhit = vtot = 0
        with torch.no_grad():
            for d in val:
                hv = torch.from_numpy(d["mids"][:, k]).float().to(device) @ W
                vhit += int((argmax_of(hv) == torch.tensor(d["labels"], device=device)).sum())
                vtot += len(d["labels"])
        print(f"layer {depth:2d}: zero_shot={zhit / ztot:.3f}  ridge_map={vhit / vtot:.3f}", flush=True)


if __name__ == "__main__":
    main()
