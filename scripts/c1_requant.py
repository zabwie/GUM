#!/usr/bin/env python3
"""C1 Stage S: build a requantized variant of the target GGUF.

Selected Q4_0 tensors are requantized to a symmetric k-bit grid with
per-G-group fp16 scales. The resulting values are stored LOSSLESSLY as
Q4_0 (a k-bit grid d*{-2^(k-1)..2^(k-1)-1} is a subset of Q4_0's
d*{-8..7} levels when the same scale is kept), so engine runs measure the
candidate format's quality with no storage distortion. The hypothetical
format's byte savings are analytic: bpw = k + 16/G vs Q4_0's 4.5.

A --k 4 --group 32 run over all roles is the identity control.
"""
import argparse
import collections
import struct
import sys

import numpy as np

Q4_0 = 2  # GGML quant type id (8 is Q8_0!)


def dequant_blocks(raw, nblocks):
    """raw: (nblocks*18,) uint8 -> (nblocks*32,) float32."""
    d = raw.reshape(nblocks, 18)[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nblocks, 1)
    q = raw.reshape(nblocks, 18)[:, 2:18].astype(np.int32)
    lo = (q & 15) - 8
    hi = (q >> 4) - 8
    vals = np.concatenate([lo, hi], axis=1) * d
    return vals.reshape(-1)


def requant_tensor(raw, nblocks, k, g):
    """Return (new_raw_bytes, rmse) for requantization to a k-bit, G-group grid.

    Replicates the original Q4_0 convention (d = signed-max / -half,
    n = trunc(w/d + 8.5) clipped to [0,15]); k < 4 restricts the nibble
    range to a centered 2^k subset, so k = 4 is the exact identity control.
    """
    w = dequant_blocks(raw, nblocks).reshape(-1, g)
    half = 2 ** (k - 1)
    imax = np.abs(w).argmax(axis=1, keepdims=True)
    smax = np.take_along_axis(w, imax, axis=1)              # signed max-|w| element
    d = (smax / -half).astype(np.float16).astype(np.float32)
    d = np.where(d == 0, np.float16(1e-8).astype(np.float32), d)
    n = np.trunc(w / d + np.float32(8.5)).astype(np.int32).clip(8 - half, 8 + half - 1)
    err = n * d - w
    rmse = float(np.sqrt((err ** 2).mean()))
    n32 = n.reshape(-1, 32)
    d32 = np.repeat(d, g // 32, axis=0)                     # scale per 32-block
    out = np.zeros((len(n32), 18), dtype=np.uint8)
    out[:, 0:2] = d32.astype(np.float16).view(np.uint8)     # fp16 scale, LE
    lo = n32[:, :16].astype(np.uint8)
    hi = n32[:, 16:].astype(np.uint8)
    out[:, 2:18] = lo | (hi << 4)
    return out.reshape(-1), rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--roles", default="ffn,attn,ssm,other")
    args = ap.parse_args()
    roles = set(args.roles.split(","))

    def role_of(name):
        if any(x in name for x in ("ffn_up", "ffn_gate", "ffn_down")):
            return "ffn"
        if "attn" in name:
            return "attn"
        if "ssm" in name:
            return "ssm"
        return "other"

    with open(args.model, "rb") as f:
        magic, ver, ntens, nkvs = struct.unpack("<4sIQQ", f.read(4 + 4 + 8 + 8))
        assert magic == b"GGUF" and ver == 3
        kv_start = f.tell()
        # parse KVs enough to skip them; capture raw bytes
        def rd_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n)
        for _ in range(nkvs):
            rd_str()
            vt = struct.unpack("<I", f.read(4))[0]
            if vt == 8:
                rd_str()
            elif vt == 9:
                at = struct.unpack("<I", f.read(4))[0]
                cnt = struct.unpack("<Q", f.read(8))[0]
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 10: 8, 11: 8, 12: 8}
                for _j in range(cnt):
                    if at == 8:
                        rd_str()
                    else:
                        f.read(sizes[at])
            else:
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 10: 8, 11: 8, 12: 8}
                f.read(sizes[vt])
        kv_end = f.tell()
        f.seek(kv_start)
        kv_raw = f.read(kv_end - kv_start)
        # tensor directory
        tensors = []
        for _ in range(ntens):
            name = rd_str().decode()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, dims, ttype, off))
        data_start = f.tell()
        f.seek(0, 2)
        fsize = f.tell()

    # norms and small aux tensors are F32; they are always copied verbatim
    import mmap
    with open(args.model, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # plan: per-tensor payload size, new offsets, requantized payloads
    plan = []
    for name, dims, ttype, off in tensors:
        ncols = dims[0]
        nrows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        if ttype == Q4_0:
            nblocks = (ncols // 32) * nrows
            nbytes = nblocks * 18
        else:
            nblocks = 0
            nbytes = int(np.prod(dims)) * (4 if ttype == 0 else 2)
        plan.append({"name": name, "dims": dims, "off": off, "nblocks": nblocks,
                     "nbytes": nbytes, "ttype": ttype,
                     "requant": ttype == Q4_0 and role_of(name) in roles})
    total_bytes = sum(p["nbytes"] for p in plan)
    infos = []
    offset = 0
    for p in plan:
        offset = (offset + 31) & ~31
        p["new_off"] = offset
        infos.append((p["name"], p["dims"], p["nbytes"], offset))
        offset += p["nbytes"]
    data_start_new = (len(struct.pack("<4sIQQ", b"GGUF", 3, ntens, nkvs)) + len(kv_raw)
                      + sum(8 + len(n) + 4 + 8 * len(d) + 4 + 8 for n, d, _, _ in infos) + 31) & ~31

    per_role_orig, per_role_new = collections.Counter(), collections.Counter()
    with open(args.out, "wb") as out:
        out.write(struct.pack("<4sIQQ", b"GGUF", 3, ntens, nkvs))
        out.write(kv_raw)
        for (name, dims, nbytes, off), p in zip(infos, plan):
            out.write(struct.pack("<Q", len(name)))
            out.write(name.encode())
            out.write(struct.pack("<I", len(dims)))
            out.write(struct.pack("<" + "Q" * len(dims), *dims))
            out.write(struct.pack("<I", p["ttype"]))
            out.write(struct.pack("<Q", off))
        out.write(b"\0" * (data_start_new - out.tell()))
        for p in plan:
            raw = np.frombuffer(mm, dtype=np.uint8, count=p["nbytes"], offset=data_start + p["off"])
            role = role_of(p["name"])
            per_role_orig[role] += p["nbytes"]
            if p["requant"]:
                new_raw, rmse = requant_tensor(raw.copy(), p["nblocks"], args.k, args.group)
                out.write(new_raw.tobytes())
                per_role_new[role] += p["nbytes"] * (args.k / 8 + 2 / args.group) / 0.5625
                print(f"{p['name']}: rmse={rmse:.5f}", flush=True)
            else:
                out.write(raw.tobytes())
                per_role_new[role] += p["nbytes"]

    print(f"k={args.k},group={args.group},roles={args.roles}")
    tot_o = tot_n = 0.0
    for role in sorted(per_role_orig):
        o, n = per_role_orig[role] / 2**20, per_role_new[role] / 2**20
        tot_o += o; tot_n += n
        print(f"role {role}: {o:.1f} MiB -> {n:.1f} MiB packed-equivalent ({(1 - n/o)*100:.1f}% cut)")
    print(f"total: {tot_o/1024:.2f} GiB -> packed-equivalent {tot_n/1024:.2f} GiB ({(1 - tot_n/tot_o)*100:.1f}% cut)")


if __name__ == "__main__":
    main()
