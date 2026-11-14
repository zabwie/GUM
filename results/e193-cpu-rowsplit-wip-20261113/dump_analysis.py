#!/usr/bin/env python3
"""E193 forensic: decode a --cpu-verify dump and compare GPU vs CPU-SIMD vs python-ref.
Dump layout: i32 K, i32 B, i64 row_start, i64 ystride, then blocks*288 tile bytes
(tile(t=0,b) for b<blocks = suffix rows 0..15), then B*K xq, then B*blocks int2 meta,
then B*w f32 GPU rows, then B*w f32 CPU rows (w = ystride - row_start).
GGUF Q4_0 order: weights 0-15 = low nibbles of bytes 0-15, weights 16-31 = high nibbles.
Packed tile: tile[r*16..+16) = nibble bytes of tile-row r; scale fp16 at tile[256+r*2].
"""
import struct, sys
import numpy as np

for path in sys.argv[1:]:
    f = open(path, 'rb').read()
    K, B = struct.unpack('<ii', f[0:8])
    row_start, ystride = struct.unpack('<qq', f[8:24])
    off = 24
    blocks = K // 32
    W = np.frombuffer(f[off:off+blocks*288], dtype=np.uint8).reshape(blocks, 288); off += blocks*288
    XQ = np.frombuffer(f[off:off+B*K], dtype=np.int8).reshape(B, K); off += B*K
    META = np.frombuffer(f[off:off+B*blocks*8], dtype=np.int32).reshape(B, blocks, 2); off += B*blocks*8
    w = ystride - row_start
    G = np.frombuffer(f[off:off+B*w*4], dtype=np.float32).reshape(B, w); off += B*w*4
    C = np.frombuffer(f[off:off+B*w*4], dtype=np.float32).reshape(B, w)
    hs = W[:, 256:288].copy().reshape(-1).view(np.uint16).astype(np.uint32)
    bits = ((hs & 0x8000) << 16) | (((hs & 0x7C00) + 0x1C000) << 13) | ((hs & 0x3FF) << 13)
    dw = bits.view(np.float32).reshape(blocks, 16)
    out_ref = np.zeros((B, 16))
    for v in range(B):
        for b in range(blocks):
            dx = np.float32(struct.unpack('<f', struct.pack('<i', int(META[v,b,0])))[0])
            sx = int(META[v,b,1])
            for r in range(16):
                nib = W[b][r*16:(r+1)*16]
                s_int = int(np.dot(np.concatenate([(nib.astype(np.int32)&15)-8, (nib.astype(np.int32)>>4)-8]),
                                   XQ[v, b*32:(b+1)*32].astype(np.int32)))
                out_ref[v, r] += dw[b, r] * dx * (s_int - 8*sx)
    print(f'{path}: ref-vs-GPU {np.abs(out_ref-G[:,:16]).max():.4f} | ref-vs-SIMD {np.abs(out_ref-C[:,:16]).max():.4f}')
    print(f'  GPU[0,:4]={G[0,:4]}  SIMD[0,:4]={C[0,:4]}  ref[0,:4]={out_ref[0,:4]}')
