#!/usr/bin/env python3
"""E157: full-vocabulary parallel drafter with an embedding-tied head.

E153's heads could only emit the 1,237 IDs seen in training labels, which
covered 58.39% of validation labels -- a hard cap on position-one accuracy.
This variant replaces the free head with logit_t = l . (E_t @ P), where E is
the target's exact dequantized Q4 embedding matrix and P (5120x96) is the
only head parameter. Tokens absent from training still receive informative
logits through embedding geometry, so the coverage cap disappears.
"""
import argparse
from pathlib import Path
import struct

import gguf
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 5120
BLOCK = 10
STEM = 384
LATENT = 96
RECORD = np.dtype([("depth", "<i4"), ("features", "<f4", (3, WIDTH))])


def load_prompt(path):
    reference = np.loadtxt(path.with_suffix(".ids"), dtype=np.int32, ndmin=1)
    if f"reference_matches={len(reference)}" not in path.with_suffix(".capture.log").read_text():
        raise ValueError(f"Missing Q4 verification for {path}")
    records = np.memmap(path, mode="r", dtype=RECORD)
    roots = records[records["depth"] == 0]["features"]
    positions = np.arange(len(reference) - BLOCK)
    if len(roots) != len(reference) // 2:
        raise ValueError(f"Incomplete root capture for {path}")
    state = roots[positions // 2, (positions % 2) * 2]
    labels = reference[positions[:, None] + np.arange(1, BLOCK + 1)]
    if not np.isfinite(state).all():
        raise ValueError(f"Nonfinite target state in {path}")
    return state, reference[positions], labels


class TiedParallelDraft(nn.Module):
    """Same stem/forecast/adjacent as E153; head tied to target embeddings."""

    def __init__(self, embeddings, adjacent):
        super().__init__()
        self.stem = nn.Linear(2 * WIDTH, STEM)
        self.forecast = nn.Linear(STEM, BLOCK * LATENT)
        self.adjacent = nn.Linear(LATENT, LATENT, bias=False) if adjacent else None
        self.register_buffer("embeddings", embeddings)
        self.proj = nn.Parameter(torch.randn(WIDTH, LATENT) / np.sqrt(WIDTH))
        self.scale = nn.Parameter(torch.tensor(4.0))

    def head_logits(self, forecast_relu):
        # forecast_relu: [B, BLOCK, LATENT] -> logits over the full vocabulary.
        z = forecast_relu @ self.proj.T           # [B, BLOCK, WIDTH]
        return self.scale * (z @ self.embeddings.T)  # [B, BLOCK, vocab]

    def forward(self, inputs):
        rows = inputs.reshape(-1, 2, WIDTH)
        rows = rows * (rows.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        forecast = self.forecast(self.stem(rows.flatten(1)).relu()).reshape(-1, BLOCK, LATENT)
        if self.adjacent is not None:
            injected = self.adjacent(forecast[:, :-1].relu())
            forecast = forecast + F.pad(injected, (0, 0, 1, 0))
        return self.head_logits(forecast.relu())


@torch.no_grad()
def evaluate(model, inputs, labels):
    match = torch.cat([model(batch).argmax(dim=-1) for batch in inputs.split(32)]) == labels
    survival = match.int().cumprod(dim=1).float().mean(dim=0)
    return survival.sum().item(), survival.tolist(), match.float().mean(dim=0).tolist()


def export(path, model, n_vocab):
    layers = [model.stem, model.forecast]
    if model.adjacent is not None:
        layers.append(model.adjacent)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4s5I", b"VPD1", WIDTH, BLOCK, STEM, LATENT, n_vocab))
        stream.write(struct.pack("<I", model.adjacent is not None))
        stream.write(np.arange(n_vocab, dtype="<i4").tobytes())
        for layer in layers:
            stream.write(layer.weight.detach().cpu().numpy().astype("<f4").tobytes())
            if layer.bias is not None:
                stream.write(layer.bias.detach().cpu().numpy().astype("<f4").tobytes())
        # The tied head has no bias: weights E@P ([vocab, LATENT] row-major,
        # same layout as an nn.Linear(LATENT, vocab).weight) then zero biases.
        with torch.no_grad():
            head = model.embeddings @ model.proj
        stream.write(head.cpu().numpy().astype("<f4").tobytes())
        stream.write(np.zeros(n_vocab, dtype="<f4").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    files = sorted(args.captures.glob("sample-*.bin"))
    if len(files) != 30:
        raise ValueError("Expected the 30 independent E132 prompt captures")
    training = [load_prompt(path) for path in files[:24]]
    validation = [load_prompt(path) for path in files[24:]]
    target = gguf.GGUFReader(args.model)
    embeddings = next(t for t in target.tensors if t.name == "token_embd.weight")
    if embeddings.tensor_type.name != "Q4_0" or int(embeddings.shape[0]) != WIDTH:
        raise ValueError("Expected the exact 5120-wide Q4_0 target embeddings")
    n_vocab = int(embeddings.shape[1])
    print(f"full vocabulary={n_vocab}", flush=True)
    rows = gguf.quants.dequantize(embeddings.data, embeddings.tensor_type)
    table = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(args.device)
    del rows

    def dataset(samples):
        states, tokens, labels = map(np.concatenate, zip(*samples))
        emb_rows = gguf.quants.dequantize(embeddings.data[tokens], embeddings.tensor_type)
        x = np.concatenate((states, emb_rows), axis=1).astype(np.float32)
        return torch.from_numpy(x).to(args.device), torch.from_numpy(labels).long().to(args.device)

    x, y = dataset(training)
    val_x, val_y = dataset(validation)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = torch.exp(-0.20 * torch.arange(BLOCK, device=args.device))
    for adjacent in (False, True):
        torch.manual_seed(5393)
        name = "adjacent" if adjacent else "independent"
        model = TiedParallelDraft(table, adjacent).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        best = -1.0
        for epoch in range(args.epochs):
            model.train()
            for indices in torch.randperm(len(x), device=x.device).split(args.batch):
                prediction = model(x[indices])
                errors = F.cross_entropy(prediction.transpose(1, 2), y[indices], reduction="none")
                error = (errors * weights).mean()
                optimizer.zero_grad(set_to_none=True)
                error.backward()
                optimizer.step()
            model.eval()
            prefix, survival, accuracy = evaluate(model, val_x, val_y)
            print(f"{name},epoch={epoch+1},prefix={prefix:.4f},"
                  f"survival={np.round(survival, 4).tolist()},"
                  f"accuracy={np.round(accuracy, 4).tolist()}", flush=True)
            if prefix > best:
                best = prefix
                export(args.output / f"{name}.vpd", model, n_vocab)


if __name__ == "__main__":
    main()
