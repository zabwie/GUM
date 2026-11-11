#!/usr/bin/env python3
"""E158: trained self-conditional sequential drafter distilled from target states.

Motivation: native MTP chains one dense block (mean exact prefix ~3.1) and
loses depth to feature drift; one-pass MLP heads (E153/E157) cap at ~0.45.
This drafter models the target's true transition (H[p-1], emb(ref[p])) -> H[p]
with a small transformer (d=2048, 3 blocks) that attends over its own rollout
prefix, recovers through a residual, and decodes through a head tied to the
target's exact Q4 embedding matrix. Training mixes teacher forcing with
scheduled sampling so the live rollout distribution (own hiddens, own tokens)
is what the model sees.

Teacher chains come from the E132 captures: every record carries the exact
target hidden H[root+depth], so consecutive transitions exist for every
generated position, not just the even roots E153 sampled.
"""
import argparse
from pathlib import Path

import gguf
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 5120
CHAIN = 16          # rollout depth trained/evaluated (E152's oracle measured d15)
DMODEL = 2048
LAYERS = 3
HEADS = 8
FFN = 5504
MAX_DEPTH = 24
RECORD = np.dtype([("depth", "<i4"), ("features", "<f4", (3, WIDTH))])


def load_prompt(path):
    """Return (full_hiddens, tokens, chain_starts).

    full[q] = target hidden after consuming q-1 tokens (full[0] = hidden after
    the prompt, i.e. H[-1]); full[q] for q>=1 equals H[q-1]. A transition is
    (full[q], emb(ref[q])) -> full[q+1] with label ref[q+1].
    """
    reference = np.loadtxt(path.with_suffix(".ids"), dtype=np.int32, ndmin=1)
    if f"reference_matches={len(reference)}" not in path.with_suffix(".capture.log").read_text():
        raise ValueError(f"Missing Q4 verification for {path}")
    records = np.memmap(path, mode="r", dtype=RECORD)
    n = len(reference)
    hidden = np.full((n, WIDTH), np.nan, dtype=np.float32)   # H[p]: after consuming ref[p]
    prior = None                                             # H[-1]
    root = -2
    for record in records:
        if record["depth"] == 0:
            root += 2                                        # roots are written for even generated indices
            if root == 0:
                prior = record["features"][0].copy()
        generated = root + int(record["depth"])
        if generated < n:
            hidden[generated] = record["features"][2]
    if prior is None or not np.isfinite(prior).all():
        raise ValueError(f"Missing prior hidden for {path}")
    valid = np.isfinite(hidden).all(axis=1)
    full = np.concatenate((prior[None], hidden[:n - 1]), axis=0)  # [n, WIDTH]
    full_valid = np.concatenate(([True], valid[:n - 1]))
    starts = [p for p in range(0, n - CHAIN) if full_valid[p:p + CHAIN + 1].all()]
    if len(starts) < CHAIN:
        raise ValueError(f"Too few teacher chains for {path}: {len(starts)}")
    return torch.from_numpy(full), torch.from_numpy(reference.astype(np.int64)), starts


def rms_half(x):
    rows = x.reshape(-1, 2, WIDTH)
    return (rows * (rows.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()).reshape(x.shape)


class Block(nn.Module):
    def __init__(self, d, heads, ffn, dropout):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n2 = nn.RMSNorm(d)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.heads = heads

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = k.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        v = v.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(att))
        x = x + self.drop(self.down(F.gelu(self.up(self.n2(x)))))
        return x


class SequentialDraft(nn.Module):
    """Head = the target's exact output.weight + output_norm (fp16, frozen).
    A tied embedding head cannot decode target hidden states -- E157 proved
    that empirically and the GGUF confirms output.weight is separate."""

    def __init__(self, embeddings, out_head, out_norm_w, device, dropout=0.1):
        super().__init__()
        self.embeddings = embeddings                                  # [vocab, WIDTH] fp16, frozen
        self.out_head = out_head                                      # [vocab, WIDTH] fp16, frozen
        self.register_buffer("out_norm_w", out_norm_w)                # [WIDTH] fp32, frozen
        self.in_proj = nn.Linear(2 * WIDTH, DMODEL)
        self.depth_emb = nn.Embedding(MAX_DEPTH, DMODEL)
        self.blocks = nn.ModuleList(Block(DMODEL, HEADS, FFN, dropout) for _ in range(LAYERS))
        self.out_norm = nn.RMSNorm(DMODEL)
        self.delta = nn.Linear(DMODEL, WIDTH, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.to(device)

    def _logits(self, hidden):
        normed = hidden * (hidden.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        normed = normed * self.out_norm_w
        return self.logit_scale * (normed.half() @ self.out_head.T).float()

    def step_input(self, bases, tokens, depth_ids):
        emb = F.embedding(tokens, self.embeddings)
        x = self.in_proj(rms_half(torch.cat((bases, emb), dim=-1)))
        return x + self.depth_emb(depth_ids)

    def teacher_forward(self, bases, tokens, depth_ids):
        """bases/tokens: [B*CHAIN, ...] depth-major. Returns hidden, logits (depth-major)."""
        count = len(depth_ids)
        batch = bases.shape[0] // count
        x = self.step_input(bases, tokens, depth_ids.repeat_interleave(batch))  # [B*CHAIN, D]
        x = x.reshape(count, batch, DMODEL).transpose(0, 1)                     # [B, CHAIN, D]
        for block in self.blocks:
            x = block(x)
        x = x.transpose(0, 1).reshape(-1, DMODEL)                               # depth-major
        hidden = bases + self.delta(self.out_norm(x))
        return hidden, self._logits(hidden)

    def rollout(self, flat_h, flat_t, offsets, starts, limit, own_prob):
        """Sequential rollout with scheduled sampling. Returns (tokens [B,limit],
        logits list, hidden list). The block *input* sequence is kept separate
        from block outputs; each step re-runs the blocks causally over the
        whole prefix, which reproduces the teacher path's per-position values.
        """
        device = self.logit_scale.device
        batch = len(starts)
        flat = [o + s for o, s in zip(offsets, starts)]
        base = flat_h[flat]
        token_in = flat_t[flat]
        x_inputs = self.step_input(base, token_in,
                                   torch.zeros(batch, dtype=torch.long, device=device)).unsqueeze(1)
        picks, step_logits, step_hiddens = [], [], []
        for depth in range(limit):
            y = x_inputs
            for block in self.blocks:
                y = block(y)
            hidden = base + self.delta(self.out_norm(y[:, -1]))
            logits = self._logits(hidden)
            picks.append(logits.argmax(dim=-1))
            step_logits.append(logits)
            step_hiddens.append(hidden)
            if depth + 1 < limit:
                own = torch.rand(batch, device=device) < own_prob
                teacher_hidden = flat_h[[o + s + depth + 1 for o, s in zip(offsets, starts)]]
                teacher_token = flat_t[[o + s + depth + 1 for o, s in zip(offsets, starts)]]
                base = torch.where(own.unsqueeze(1), hidden, teacher_hidden)
                token_in = torch.where(own, picks[-1], teacher_token)
                x_new = self.step_input(base, token_in,
                                        torch.full((batch,), depth + 1, dtype=torch.long, device=device))
                x_inputs = torch.cat((x_inputs, x_new.unsqueeze(1)), dim=1)
        return torch.stack(picks, dim=1), step_logits, step_hiddens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-epochs", type=int, default=4)
    parser.add_argument("--rollout-epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--feature-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device
    files = sorted(args.captures.glob("sample-*.bin"))
    if len(files) != 30:
        raise ValueError("Expected the 30 independent E132 prompt captures")
    train_data = [load_prompt(path) for path in files[:24]]
    val_data = [load_prompt(path) for path in files[24:]]

    target = gguf.GGUFReader(args.model)
    emb_tensor = next(t for t in target.tensors if t.name == "token_embd.weight")
    head_tensor = next(t for t in target.tensors if t.name == "output.weight")
    norm_tensor = next(t for t in target.tensors if t.name == "output_norm.weight")
    if head_tensor.tensor_type.name != "Q4_0" or int(head_tensor.shape[0]) != WIDTH:
        raise ValueError("Expected the exact 5120-wide Q4_0 output head")
    embeddings = torch.from_numpy(np.asarray(
        gguf.quants.dequantize(emb_tensor.data, emb_tensor.tensor_type), dtype=np.float32))
    head = torch.from_numpy(np.asarray(
        gguf.quants.dequantize(head_tensor.data, head_tensor.tensor_type), dtype=np.float32))
    norm_w = torch.from_numpy(np.asarray(np.array(norm_tensor.data), dtype=np.float32))
    embeddings, head = embeddings.half().to(device), head.half().to(device)
    norm_w = norm_w.to(device)

    def pack(data):
        hiddens, tokens, starts, owners, offsets = [], [], [], [], []
        cursor = 0
        for index, (full, reference, chain_starts) in enumerate(data):
            hiddens.append(full)
            tokens.append(reference)
            offsets.append(torch.full((len(chain_starts),), cursor, dtype=torch.long))
            starts.append(torch.tensor(chain_starts, dtype=torch.long))
            owners.append(torch.full((len(chain_starts),), index, dtype=torch.long))
            cursor += len(reference)
        return (torch.cat(hiddens).to(device), torch.cat(tokens).to(device),
                torch.cat(starts).to(device), torch.cat(owners).to(device),
                torch.cat(offsets).to(device))

    tr_h, tr_t, tr_s, tr_o, tr_off = pack(train_data)
    va_h, va_t, va_s, va_o, va_off = pack(val_data)
    print(f"train_chains={len(tr_s)},val_chains={len(va_s)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model = SequentialDraft(embeddings, head, norm_w, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    depths = torch.arange(CHAIN, device=device)

    def evaluate():
        model.eval()
        picks = []
        with torch.no_grad():
            for chunk in torch.arange(len(va_s)).split(96):
                got, _, _ = model.rollout(va_h, va_t, va_off[chunk].tolist(),
                                          va_s[chunk].tolist(), CHAIN, 1.0)
                picks.append(got)
        picks = torch.cat(picks)
        labels = torch.stack([va_t[va_off[va_o] + va_s + 1 + d] for d in range(CHAIN)], dim=1)
        match = (picks == labels).float()
        running = torch.ones(len(picks), device=device)
        survival = []
        for depth in range(CHAIN):
            running = running * match[:, depth]
            survival.append(running.mean().item())
        return survival

    best = -1.0
    for phase, epochs in (("teacher", args.teacher_epochs), ("rollout", args.rollout_epochs)):
        for epoch in range(epochs):
            own_prob = 0.0 if phase == "teacher" else min(0.9, 0.3 + 0.15 * epoch)
            model.train()
            order = torch.randperm(len(tr_s), device=device)
            losses = []
            for chunk in order.split(args.batch):
                owners, starts = tr_o[chunk], tr_s[chunk]
                if phase == "teacher":
                    bases = torch.cat([tr_h[tr_off[owners] + starts + d] for d in range(CHAIN)])
                    tokens = torch.cat([tr_t[tr_off[owners] + starts + d] for d in range(CHAIN)])
                    labels = torch.cat([tr_t[tr_off[owners] + starts + d + 1] for d in range(CHAIN)])
                    targets = torch.cat([tr_h[tr_off[owners] + starts + d + 1] for d in range(CHAIN)])
                    hidden, logits = model.teacher_forward(bases, tokens, depths)
                else:
                    _, step_logits, step_hiddens = model.rollout(
                        tr_h, tr_t, tr_off[chunk].tolist(), starts.tolist(), CHAIN, own_prob)
                    logits = torch.cat(step_logits)
                    hidden = torch.cat(step_hiddens)
                    labels = torch.cat([tr_t[tr_off[owners] + starts + d + 1] for d in range(CHAIN)])
                    targets = torch.cat([tr_h[tr_off[owners] + starts + d + 1] for d in range(CHAIN)])
                ce = F.cross_entropy(logits, labels, reduction="none")
                feat = 1 - F.cosine_similarity(hidden, targets, dim=-1)
                error = ce.mean() + args.feature_weight * feat.mean()
                optimizer.zero_grad(set_to_none=True)
                error.backward()
                optimizer.step()
                losses.append(error.item())
            survival = evaluate()
            prefix = sum(survival)
            emitted = prefix + 1.0
            est = [emitted / (v + 0.018) for v in (0.214, 0.184)]
            print(f"{phase},epoch={epoch+1},loss={np.mean(losses):.4f},prefix={prefix:.3f},"
                  f"survival={[round(v, 3) for v in survival]},"
                  f"est_tok_per_s={est[0]:.1f}/{est[1]:.1f}", flush=True)
            if prefix > best:
                best = prefix
                torch.save({"model": model.state_dict(), "config": {
                    "chain": CHAIN, "dmodel": DMODEL, "layers": LAYERS,
                    "heads": HEADS, "ffn": FFN}, "prefix": prefix},
                    args.output / "seq_draft.pt")
    print(f"best_prefix={best:.3f}")


if __name__ == "__main__":
    main()
