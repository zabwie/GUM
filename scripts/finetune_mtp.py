#!/usr/bin/env python3
"""E169: fine-tune the MTP (nextn) drafter head on E167b batch captures.

Lossless by construction: the drafter only proposes; the target's greedy verify
emits the tokens. A better drafter changes acceptance, never a token ID.

Pipeline:
  1. parse capture/group-*.bin (Mode-A records: [3 hdr][7 layers x B x 5120][B x 5120])
     + group-*.ids + group-*.txt (fed prompts) -> (h_prev_normed, emb[tok]) -> tok_next pairs
  2. rebuild the nextn block forward in torch (mirrors real_decode.cu mtp_forward)
  3. evaluate the ORIGINAL head on held-out groups (alignment self-test: ~random
     accuracy means the stream mapping is wrong, not the model)
  4. fine-tune FFN + eh_proj (attention/norms/head/emb frozen; head must stay
     byte-identical for the decoder's share_output verification)
  5. report top-1 / top-8 on the same val groups; --export re-quantizes changed
     tensors to Q4_0 and writes a swapped-in MTP GGUF

Phase A go/no-go: val top-1 >= +2 pts or top-8 >= +3 pts over the original head.
"""
import argparse
import glob
import json
import math
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

Q4_0, F32 = 2, 0


# ---------------------------------------------------------------- GGUF mini-reader
def read_gguf(path):
    f = open(path, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", path
    ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def rd_val(vtype):
        if vtype in (0, 1):
            return struct.unpack("<B", f.read(1))[0]
        if vtype in (2, 3):
            return struct.unpack("<H", f.read(2))[0]
        if vtype in (4, 5):
            return struct.unpack("<I", f.read(4))[0]
        if vtype == 6:
            return struct.unpack("<f", f.read(4))[0]
        if vtype == 7:
            return struct.unpack("<B", f.read(1))[0]
        if vtype == 8:
            return rstr()
        if vtype == 9:
            atype = struct.unpack("<I", f.read(4))[0]
            (cnt,) = struct.unpack("<Q", f.read(8))
            return [rd_val(atype) for _ in range(cnt)]
        if vtype in (10, 11):
            return struct.unpack("<q", f.read(8))[0]
        if vtype == 12:
            return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"gguf kv type {vtype}")

    kv = {}
    for _ in range(n_kv):
        k = rstr()
        kv[k] = rd_val(struct.unpack("<I", f.read(4))[0])
    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        (nd,) = struct.unpack("<I", f.read(4))
        ne = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (ty,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        n = 1
        for d in ne:
            n *= d
        nbytes = {F32: 4 * n, Q4_0: ((n + 31) // 32) * 18}.get(ty)
        tensors.append({"name": name, "ne": ne, "type": ty, "off": off, "n": n, "bytes": nbytes})
    align = kv.get("general.alignment", 32)
    data_start = (f.tell() + align - 1) // align * align
    import mmap

    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    f.close()
    return kv, tensors, mm, data_start


def tensor_bytes(mm, data_start, t):
    return mm[data_start + t["off"] : data_start + t["off"] + t["bytes"]]


def dequant_q4(raw, n):
    blocks = (n + 31) // 32
    b = np.frombuffer(raw[: blocks * 18], dtype=np.uint8).reshape(blocks, 18)
    d = b[:, 0:2].copy().view("<f2").reshape(-1).astype(np.float32)
    q = np.empty((blocks, 32), np.float32)
    q[:, :16] = (b[:, 2:18] & 15).astype(np.float32) - 8.0
    q[:, 16:] = (b[:, 2:18] >> 4).astype(np.float32) - 8.0
    return (q * d[:, None]).reshape(-1)[:n]


def get_f32(mm, data_start, t):
    return np.frombuffer(tensor_bytes(mm, data_start, t), dtype="<f4").astype(np.float32)


def get_q4(mm, data_start, t):
    return dequant_q4(tensor_bytes(mm, data_start, t), t["n"])


def quant_q4(x):
    """Q4_0 encode matching the decoder reader: d=amax/8 fp16, nibble=q+8, low nibbles first."""
    n = x.size
    blocks = (n + 31) // 32
    pad = np.zeros(blocks * 32, np.float32)
    pad[:n] = x
    b = pad.reshape(blocks, 32)
    amax = np.abs(b).max(axis=1)
    d = (amax / 8.0).astype(np.float16).astype(np.float32)
    d[d == 0] = np.float32(1e-8)
    q = np.rint(b / d[:, None]).clip(-8, 7).astype(np.int8) + 8
    out = np.empty((blocks, 18), np.uint8)
    out[:, 0:2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8)
    lo = q[:, :16].astype(np.uint8)
    hi = q[:, 16:].astype(np.uint8)
    out[:, 2:18] = lo | (hi << 4)
    return out.reshape(-1)


# ---------------------------------------------------------------- capture parsing
def parse_group(capdir, gname, emb_dequant, tok_rows):
    """Returns list per member of dicts {h: [R,5120] fp32 pre-norm, stream: int array}."""
    binp = os.path.join(capdir, f"{gname}.bin")
    idsp = os.path.join(capdir, f"{gname}.ids")
    txtp = os.path.join(capdir, f"{gname}.txt")
    size = os.path.getsize(binp)
    with open(binp, "rb") as f:
        hdr = np.fromfile(f, dtype="<f4", count=3)
        mid_count = int(hdr[1])
        B = int(hdr[2])
        if not (1 <= mid_count <= 16 and 1 <= B <= 64):
            mid_count, B = int(hdr[2]), int(hdr[1])
        rec_floats = 3 + mid_count * B * 5120 + B * 5120
        rec = rec_floats * 4
        assert size % rec == 0, f"{gname}: size {size} not divisible by record {rec}"
        n_rec = size // rec
        f.seek(0)
        raw = np.fromfile(f, dtype="<f4", count=n_rec * rec_floats).reshape(n_rec, rec_floats)
    finals = raw[:, 3 + mid_count * B * 5120 :].reshape(n_rec, B, 5120)  # [R, B, 5120]
    prompts = [np.array([int(x) for x in line.split(",") if x.strip()], np.int64)
               for line in open(txtp).read().strip().split(";")]
    ids_lines = [line.strip() for line in open(idsp) if line.strip()]
    members = []
    for b in range(B):
        # rig out-ids: one comma-TERMINATED line per member holding ONLY the generated
        # ids (160 = n_rec). Record r (0-based) is the gen step for position plen+r.
        out_ids = np.array([int(x) for x in ids_lines[b].split(",") if x.strip()], np.int64)
        prompt = prompts[b]
        assert len(out_ids) == n_rec, f"{gname}: ids {len(out_ids)} != records {n_rec}"
        stream = np.concatenate([prompt, out_ids])
        base = len(prompt)          # record r holds h AFTER processing stream[base + r]
        r0 = 1                      # first usable record (pair needs h[r-1])
        h = finals[:, b, :]           # [R, 5120] pre-norm hidden, record r -> position base+r
        members.append({"h": h, "stream": stream, "base": base, "r0": r0,
                        "prompt_len": len(prompt), "n_rec": n_rec})
    return members


def build_pairs(members, w_out_norm, eps):
    """Yield (h_prev_normed [P,5120] fp32, tokens [P], labels [P]) per member."""
    out = []
    for m in members:
        h = m["h"]
        stream = m["stream"]
        base, r0, R = m["base"], m["r0"], m["n_rec"]
        rows, toks, labels = [], [], []
        for r in range(max(r0, 1), R - 1):
            pos = base + r                    # record r processes stream[pos]
            hp = h[r - 1]                     # hidden at position pos-1
            ss = hp.astype(np.float64)
            inv = 1.0 / math.sqrt(ss.dot(ss) / 5120.0 + eps)
            rows.append((hp * inv * w_out_norm).astype(np.float32))
            toks.append(stream[pos])
            labels.append(stream[pos + 1])
        if rows:
            out.append({"h_prev": np.stack(rows), "tok": np.array(toks, np.int64),
                        "label": np.array(labels, np.int64),
                        "pos": np.arange(base + max(r0, 1), base + R - 1, dtype=np.int64)})
    return out


# ---------------------------------------------------------------- nextn forward
def rms_norm(x, w, eps):
    ss = x.float().pow(2).mean(-1, keepdim=True)
    return x.float() * torch.rsqrt(ss + eps) * w.float()


def rms_norm_h(x, w, eps):
    """x: [T, H, hd], w: [hd] — per-head RMS norm (q/k norm semantics)."""
    ss = x.float().pow(2).mean(-1, keepdim=True)
    return x.float() * torch.rsqrt(ss + eps) * w.float()


class NextN(torch.nn.Module):
    def __init__(self, P, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.f32 = torch.nn.ParameterDict()
        self.bf16 = torch.nn.ParameterDict()
        for name, t in P.items():
            key = name.replace(".", "_")   # torch ParameterDict keys cannot contain "."
            if name in ("ffn_gate", "ffn_up", "ffn_down", "eh_proj"):
                self.f32[name] = torch.nn.Parameter(t.to(device))
            else:
                self.bf16[key] = torch.nn.Parameter(t.to(torch.bfloat16).to(device), requires_grad=False)
        self.head = torch.nn.Parameter(cfg["head"].to(torch.bfloat16).to(device), requires_grad=False)

    def rope(self, x, pos):
        # half-split, n_rot=rot; x: [T, H, hd]; pairs (i, i+rot/2)
        rot = self.cfg["rot"]
        half = rot // 2
        i = torch.arange(half, device=x.device, dtype=torch.float32)
        theta = pos.float()[:, None] * (self.cfg["rope_base"] ** (-2.0 * i / rot))[None, :]
        c, s = theta.cos()[:, None, :], theta.sin()[:, None, :]
        a = x[..., :half]
        b = x[..., half:rot]
        x = x.clone()
        x[..., :half] = a * c - b * s
        x[..., half:rot] = a * s + b * c
        return x

    def forward(self, h_prev, emb, pos, chunk=48):
        """h_prev [T,5120], emb [T,5120], pos [T] -> logits [T, vocab] (bf16 chunks)."""
        cfg = self.cfg
        T = h_prev.shape[0]
        cat = torch.empty(T, 2 * cfg["n_embd"], device=h_prev.device, dtype=torch.float32)
        cat[:, : cfg["n_embd"]] = rms_norm(emb, self.bf16["nextn_enorm"], cfg["eps"])
        cat[:, cfg["n_embd"]:] = rms_norm(h_prev, self.bf16["nextn_hnorm"], cfg["eps"])
        resid = cat @ self.f32["eh_proj"].t()
        normed = rms_norm(resid, self.bf16["attn_norm"], cfg["eps"])
        qg = normed @ self.bf16["attn_q"].float().t()  # [T, 12288]
        k = normed @ self.bf16["attn_k"].float().t()   # [T, 1024]
        v = normed @ self.bf16["attn_v"].float().t()   # [T, 1024]
        H, KV, hd, gs = cfg["n_head"], cfg["n_kv"], cfg["head_v"], cfg["n_head"] // cfg["n_kv"]
        q = qg.view(T, H, 2 * hd)[..., :hd]
        gate = qg.view(T, H, 2 * hd)[..., hd:]
        k = k.view(T, KV, hd)
        v = v.view(T, KV, hd)
        qn = rms_norm_h(q, self.bf16["attn_q_norm"], cfg["eps"])
        kn = rms_norm_h(k, self.bf16["attn_k_norm"], cfg["eps"])
        q = self.rope(qn, pos)
        k = self.rope(kn, pos)
        kh = torch.arange(H, device=q.device) // gs
        kk = k[:, kh]                                   # [T, H, hd]
        vv = v[:, kh]                                   # [T, H, hd]
        att = torch.einsum("thd,shd->hts", q.float(), kk.float()) / math.sqrt(hd)
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask[None], float("-inf"))
        att = torch.softmax(att, -1)
        ctx = torch.einsum("hts,shd->thd", att, vv.float())   # [T, H, hd]
        ctx = ctx * torch.sigmoid(gate.float())               # per-head output gate
        ctx = ctx.reshape(T, H * hd)
        out = ctx @ self.bf16["attn_output"].float().t()
        resid = resid + out
        normed2 = rms_norm(resid, self.bf16["post_attention_norm"], cfg["eps"])
        g = normed2 @ self.f32["ffn_gate"].t()
        u = normed2 @ self.f32["ffn_up"].t()
        resid = resid + (F.silu(g.float()) * u.float()) @ self.f32["ffn_down"].t()
        hout = rms_norm(resid, self.bf16["nextn_shared_head_norm"], cfg["eps"])
        logits = torch.empty(T, cfg["vocab"], device=hout.device, dtype=torch.bfloat16)
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            logits[s:e] = (hout[s:e].to(torch.bfloat16) @ self.head.t()).float()
        return logits, hout


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default="capture")
    ap.add_argument("--target", default=os.path.expanduser("~/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf"))
    ap.add_argument("--mtp", default=os.path.expanduser("~/Desktop/Chip/models/mtp-Qwen3.8-27B-Q4_0.gguf"))
    ap.add_argument("--val-groups", type=int, default=1, help="last N groups held out")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-members", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default="models/mtp-ft.pt")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}")

    tkv, ttens, tmm, tstart = read_gguf(args.target)
    A = tkv["general.architecture"]
    eps = tkv.get(A + ".attention.layer_norm_rms_epsilon", 1e-6)
    cfg = {
        "n_embd": tkv[A + ".embedding_length"], "n_head": tkv[A + ".attention.head_count"],
        "n_kv": tkv[A + ".attention.head_count_kv"], "head_v": tkv[A + ".attention.value_length"],
        "rot": tkv[A + ".rope.dimension_count"], "vocab": len(tkv["tokenizer.ggml.tokens"]),
        "rope_base": tkv.get(A + ".rope.freq_base", 1e7), "eps": eps,
    }
    print("cfg:", cfg)
    out_norm_t = next(t for t in ttens if t["name"] == "output_norm.weight")
    w_out_norm = get_f32(tmm, tstart, out_norm_t)
    emb_t = next(t for t in ttens if t["name"] == "token_embd.weight")
    # token_embd is row-major [vocab, 5120], 18 bytes per 32-element block
    rowbytes = 5120 // 32 * 18
    emb_raw = tensor_bytes(tmm, tstart, emb_t)

    def emb_rows(ids):
        return np.stack([dequant_q4(emb_raw[int(i) * rowbytes : (int(i) + 1) * rowbytes], 5120)
                         for i in ids])

    mkv, mtens, mmm, mstart = read_gguf(args.mtp)
    P = {}
    shapes = {"attn_q": (12288, cfg["n_embd"]), "attn_k": (1024, cfg["n_embd"]),
              "attn_v": (1024, cfg["n_embd"]), "attn_output": (cfg["n_embd"], 6144),
              "ffn_gate": (17408, cfg["n_embd"]), "ffn_up": (17408, cfg["n_embd"]),
              "ffn_down": (cfg["n_embd"], 17408), "eh_proj": (cfg["n_embd"], 10240)}
    for name, (rows, cols) in shapes.items():
        prefix = "blk.64.nextn." if name == "eh_proj" else "blk.64."
        t = next(t for t in mtens if t["name"] == f"{prefix}{name}.weight")
        P[name] = torch.from_numpy(get_q4(mmm, mstart, t).reshape(rows, cols))
    for name in ("attn_norm", "post_attention_norm", "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"):
        t = next(t for t in mtens if t["name"] == f"blk.64.{name}.weight")
        P[name] = torch.from_numpy(get_f32(mmm, mstart, t))
    for name in ("attn_q_norm", "attn_k_norm"):
        t = next(t for t in mtens if t["name"] == f"blk.64.{name}.weight")
        P[name] = torch.from_numpy(get_f32(mmm, mstart, t))
    head_t = next(t for t in mtens if t["name"] == "output.weight")
    P_head = torch.from_numpy(get_q4(mmm, mstart, head_t).reshape(cfg["vocab"], cfg["n_embd"]))
    cfg["head"] = P_head

    groups = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.capture_dir, "group-*.bin")))
    if args.max_groups:
        groups = groups[: args.max_groups]
    assert len(groups) >= 2, "need at least 2 groups (train + val)"
    val_names = groups[-args.val_groups:]
    train_names = groups[:-args.val_groups]
    print(f"groups: {len(train_names)} train / {len(val_names)} val {val_names}")

    def load(names):
        data = []
        for g in names:
            members = parse_group(args.capture_dir, g, None, None)
            data.extend(build_pairs(members, w_out_norm, eps))
        return data

    train_data = load(train_names)
    val_data = load(val_names)
    n_train = sum(len(d["label"]) for d in train_data)
    n_val = sum(len(d["label"]) for d in val_data)
    print(f"pairs: train={n_train} val={n_val}")

    model = NextN(P, cfg, dev)
    model.eval()

    def evaluate(data, tag):
        model.eval()
        top1 = top8 = n = 0
        loss_sum = 0.0
        with torch.no_grad():
            for d in data:
                h = torch.from_numpy(d["h_prev"]).to(dev)
                e = torch.from_numpy(emb_rows(d["tok"])).to(dev)
                pos = torch.from_numpy(d["pos"]).to(dev)
                y = torch.from_numpy(d["label"]).to(dev)
                logits, _ = model(h, e, pos)
                lf = logits.float()
                loss_sum += F.cross_entropy(lf, y, reduction="sum").item()
                topk = lf.topk(8, dim=-1).indices
                top1 += (topk[:, 0] == y).sum().item()
                top8 += (topk == y[:, None]).any(-1).sum().item()
                n += len(y)
        print(f"{tag}: n={n} loss={loss_sum / n:.4f} top1={top1 / n:.4f} top8={top8 / n:.4f}")
        return top1 / n, top8 / n, loss_sum / n

    print("== original head (alignment self-test: ~random top1 means stream mapping is wrong)")
    orig_val = evaluate(val_data, "orig/val")
    if args.eval_only:
        return
    orig_train = evaluate(train_data, "orig/train")

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.f32.values(), lr=args.lr, weight_decay=0.0)
        print("optimizer: bnb AdamW8bit")
    except Exception as e:
        print(f"bnb unavailable ({e}); torch AdamW")
        opt = torch.optim.AdamW(model.f32.values(), lr=args.lr, weight_decay=0.0)
    import random

    rng = random.Random(7)
    best = (0.0, None)
    for epoch in range(args.epochs):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        model.train()
        tot_loss = tot_n = 0
        WIN = 64   # slice members into windows; backward per slice frees the graph (VRAM)
        for s in range(0, len(order), args.batch_members):
            batch = [train_data[i] for i in order[s : s + args.batch_members]]
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for d in batch:
                for w0 in range(0, len(d["label"]), WIN):
                    h = torch.from_numpy(d["h_prev"][w0 : w0 + WIN]).to(dev)
                    e = torch.from_numpy(emb_rows(d["tok"][w0 : w0 + WIN])).to(dev)
                    pos = torch.from_numpy(d["pos"][w0 : w0 + WIN]).to(dev)
                    y = torch.from_numpy(d["label"][w0 : w0 + WIN]).to(dev)
                    logits, _ = model(h, e, pos)
                    loss = F.cross_entropy(logits.float(), y, reduction="sum")
                    loss.backward()          # accumulate grads slice by slice
                    batch_loss += loss.item()
                    tot_n += len(y)
            torch.nn.utils.clip_grad_norm_(model.f32.values(), 1.0)
            opt.step()
            tot_loss += batch_loss
        val_m = evaluate(val_data, f"ft/val epoch {epoch}")
        if val_m[0] > best[0]:
            best = (val_m[0], {k: v.detach().clone() for k, v in model.f32.items()})
        print(f"epoch {epoch}: train_loss={tot_loss / max(1, tot_n):.4f}")

    print("== summary")
    print(f"original val top1/top8: {orig_val[0]:.4f}/{orig_val[1]:.4f}")
    print(f"best finetuned val top1: {best[0]:.4f} (delta {(best[0] - orig_val[0]) * 100:+.2f} pts)")
    torch.save({"cfg": {k: v for k, v in cfg.items() if k != "head"},
                "state": {k: v.cpu() for k, v in best[1].items()}}, args.out)
    print(f"saved best state -> {args.out}")


if __name__ == "__main__":
    main()
