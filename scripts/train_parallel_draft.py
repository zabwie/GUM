#!/usr/bin/env python3
"""Train a small block drafter from E132's Q4-verified hidden-state captures."""
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
    # At even i use h[i-1], the native MTP root input. The same record also
    # contains h[i], which is the previous target state at the next (odd) i.
    state = roots[positions // 2, (positions % 2) * 2]
    labels = reference[positions[:, None] + np.arange(1, BLOCK + 1)]
    if not np.isfinite(state).all():
        raise ValueError(f"Nonfinite target state in {path}")
    return state, reference[positions], labels


def dataset(samples, embeddings, vocab, device):
    states, tokens, labels = map(np.concatenate, zip(*samples))
    rows = gguf.quants.dequantize(embeddings.data[tokens], embeddings.tensor_type)
    x = np.concatenate((states, rows), axis=1).astype(np.float32)
    y = np.searchsorted(vocab, labels).clip(0, len(vocab) - 1)
    y[vocab[y] != labels] = -100
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


class ParallelDraft(nn.Module):
    def __init__(self, vocab_size, adjacent):
        super().__init__()
        self.stem = nn.Linear(2 * WIDTH, STEM)
        self.forecast = nn.Linear(STEM, BLOCK * LATENT)
        self.adjacent = nn.Linear(LATENT, LATENT, bias=False) if adjacent else None
        self.head = nn.Linear(LATENT, vocab_size)

    def forward(self, inputs):
        rows = inputs.reshape(-1, 2, WIDTH)
        rows = rows * (rows.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        forecast = self.forecast(self.stem(rows.flatten(1)).relu()).reshape(-1, BLOCK, LATENT)
        if self.adjacent is not None:
            injected = self.adjacent(forecast[:, :-1].relu())
            forecast = forecast + F.pad(injected, (0, 0, 1, 0))
        return self.head(forecast.relu())


@torch.no_grad()
def evaluate(model, inputs, labels):
    match = torch.cat([model(batch).argmax(dim=-1) for batch in inputs.split(96)]) == labels
    survival = match.int().cumprod(dim=1).float().mean(dim=0)
    return survival.sum().item(), survival.tolist(), match.float().mean(dim=0).tolist()


def export(path, model, vocab):
    layers = [model.stem, model.forecast]
    if model.adjacent is not None:
        layers.append(model.adjacent)
    layers.append(model.head)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4s5I", b"VPD1", WIDTH, BLOCK, STEM, LATENT, len(vocab)))
        stream.write(struct.pack("<I", model.adjacent is not None))
        stream.write(vocab.astype("<i4").tobytes())
        for layer in layers:
            stream.write(layer.weight.detach().cpu().numpy().astype("<f4").tobytes())
            if layer.bias is not None:
                stream.write(layer.bias.detach().cpu().numpy().astype("<f4").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    files = sorted(args.captures.glob("sample-*.bin"))
    if len(files) != 30:
        raise ValueError("Expected the 30 independent E132 prompt captures")
    training = [load_prompt(path) for path in files[:24]]
    validation = [load_prompt(path) for path in files[24:]]
    vocab = np.unique(np.concatenate([labels.ravel() for _, _, labels in training]))
    target = gguf.GGUFReader(args.model)
    embeddings = next(t for t in target.tensors if t.name == "token_embd.weight")
    if embeddings.tensor_type.name != "Q4_0" or int(embeddings.shape[0]) != WIDTH:
        raise ValueError("Expected the exact 5120-wide Q4_0 target embeddings")
    x, y = dataset(training, embeddings, vocab, args.device)
    val_x, val_y = dataset(validation, embeddings, vocab, args.device)
    print(f"training={len(x)},validation={len(val_x)},vocab={len(vocab)},"
          f"validation_label_coverage={(val_y >= 0).float().mean().item():.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = torch.exp(-0.20 * torch.arange(BLOCK, device=args.device))
    for adjacent in (False, True):
        torch.manual_seed(5393)
        name = "adjacent" if adjacent else "independent"
        model = ParallelDraft(len(vocab), adjacent).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        best = -1.0
        for epoch in range(args.epochs):
            model.train()
            for indices in torch.randperm(len(x), device=x.device).split(64):
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
                export(args.output / f"{name}.vpd", model, vocab)


if __name__ == "__main__":
    main()
