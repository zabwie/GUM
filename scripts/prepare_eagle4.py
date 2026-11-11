#!/usr/bin/env python3
"""Build the optional Qwen3.8-27B EAGLE4 draft GGUF without downloading its embeddings."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from urllib.request import Request, urlopen

REVISION = "f58a5985223040552ad404fba98f0dd27f5f676a"
URL = f"https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4/resolve/{REVISION}/model.safetensors"
FILE_SIZE = 6_178_152_384
HEADER_SHA256 = "30e005e3beea767d15ea3fe6ea6359c6d0f15df0c24141c0110d20bd3e6fe846"
RANGES = (
    (1984, 64_882_623, "bb1905277e088ea9b76aa6d280c936b1db81c0ca9c1b9ee94919e1d77ad59de8"),
    (2_607_679_424, FILE_SIZE - 1, "ce910be1ba7b91ed6c701a2e91c39dff878f3f501284e262edb7aa6c4dc47667"),
)
CHUNK = 4 << 20


def download_range(destination, start, end):
    request = Request(URL, headers={"Range": f"bytes={start}-{end}"})
    digest = hashlib.sha256()
    remaining = end - start + 1
    with urlopen(request, timeout=90) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{FILE_SIZE}":
            raise ValueError(f"unexpected server response for bytes {start}–{end}")
        destination.seek(start)
        while remaining:
            data = response.read(min(CHUNK, remaining))
            if not data:
                raise ValueError("incomplete checkpoint download")
            destination.write(data)
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def download(source):
    with source.open("wb") as destination:
        destination.truncate(FILE_SIZE)
        if download_range(destination, 0, RANGES[0][0] - 1) != HEADER_SHA256:
            raise ValueError("checkpoint header failed SHA-256 verification")
        for start, end, expected in RANGES:
            print(f"Downloading {end - start + 1:,} checkpoint bytes", flush=True)
            if download_range(destination, start, end) != expected:
                raise ValueError("checkpoint download failed SHA-256 verification")


def read_header(source):
    if source.stat().st_size != FILE_SIZE:
        raise ValueError("checkpoint size does not match the pinned revision")
    with source.open("rb") as data:
        header = data.read(RANGES[0][0])
    if hashlib.sha256(header).hexdigest() != HEADER_SHA256:
        raise ValueError("checkpoint header failed SHA-256 verification")
    header_size, = struct.unpack("<Q", header[:8])
    return header_size + 8, json.loads(header[8:])


def verify_local(source):
    with source.open("rb") as checkpoint:
        for start, stop, expected in RANGES:
            checkpoint.seek(start)
            remaining = stop - start + 1
            digest = hashlib.sha256()
            while remaining:
                data = checkpoint.read(min(CHUNK, remaining))
                if not data:
                    raise ValueError("incomplete local checkpoint")
                digest.update(data)
                remaining -= len(data)
            if digest.hexdigest() != expected:
                raise ValueError("local checkpoint differs from the pinned revision")


def convert(source, output):
    import gguf
    import numpy as np

    data_start, tensors = read_header(source)
    writer = gguf.GGUFWriter(output, "eagle4", use_temp_file=True)
    writer.add_name("Qwen3.8-27B-EAGLE4 draft")
    writer.add_array("eagle4.target_layers", [3, 31, 59])
    for name, tensor in sorted(tensors.items()):
        if name in ("__metadata__", "embed_tokens.weight"):
            continue
        shape = tuple(tensor["shape"])
        begin, end = (offset + data_start for offset in tensor["data_offsets"])
        if tensor["dtype"] != "BF16" or not any(start <= begin < end <= stop + 1 for start, stop, _ in RANGES):
            raise ValueError(f"unexpected checkpoint tensor {name}")
        values = np.memmap(source, mode="r", dtype="<u2", offset=begin, shape=shape)

        def fp32(bits):
            return (bits.astype(np.uint32) << 16).view(np.float32)

        if len(shape) == 1:
            writer.add_tensor(name, fp32(values))
        else:
            rows, columns = shape
            quant = gguf.GGMLQuantizationType.Q8_0 if name.startswith("efficient_lm_head") else gguf.GGMLQuantizationType.Q4_0
            block, size = gguf.GGML_QUANT_SIZES[quant]
            packed = np.empty((rows, columns // block * size), dtype=np.uint8)
            for start in range(0, rows, 128):
                packed[start:start + 128] = gguf.quants.quantize(fp32(values[start:start + 128]), quant)
            writer.add_tensor(name, packed, raw_dtype=quant)
        print(f"Converted {name}", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local checkpoint from the pinned revision")
    parser.add_argument("--gguf-py", type=Path, help="optional path to llama.cpp's gguf-py package")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gguf_py:
        sys.path.insert(0, str(args.gguf_py))
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    with tempfile.TemporaryDirectory(prefix=".eagle4-", dir=args.output.parent) as directory:
        folder = Path(directory)
        source = args.source or folder / "checkpoint.safetensors"
        if args.source:
            verify_local(source)
        else:
            download(source)
        draft = folder / "draft.gguf"
        convert(source, draft)
        draft.replace(args.output)
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
