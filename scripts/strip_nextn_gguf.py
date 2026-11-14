#!/usr/bin/env python3
"""Strip the MTP/nextn block (blk.64.*) + nextn KV from a BF16 GGUF so llama.cpp's
qwen35 loader accepts it as a plain 64-layer model. Uses gguf-py reader/writer.
Usage: PYTHONPATH=/tmp/ggufpkg python3 scripts/strip_nextn_gguf.py <in.gguf> <out.gguf>
"""
import sys
import gguf
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(dst, arch=r.fields['general.architecture'].contents[-1].decode() if False else 'qwen35')

# KVs: copy everything except the nextn-predict KV
for name, field in r.fields.items():
    if 'nextn_predict_layers' in name:
        print('dropping KV:', name); continue
    w.add_key_value(name, field.contents, field.type) if False else None
# gguf-py wants typed adds; use the reader's parsed contents with the field type
from gguf import GGUFValueType
for name, field in r.fields.items():
    if 'nextn_predict_layers' in name: continue
    t = GGUFValueType(field.types[0])
    if t == GGUFValueType.STRING:
        w.add_string(name, bytes(field.parts[-1]).decode())
    elif t == GGUFValueType.ARRAY:
        vt = GGUFValueType(field.types[1]) if len(field.types)>1 else t
        if vt == GGUFValueType.STRING:
            w.add_array(name, [bytes(p).decode() for p in field.parts[1:]], GGUFValueType.STRING)
        else:
            w.add_array(name, list(field.contents[1]) if hasattr(field.contents[1],'__len__') else field.contents[1], vt)
    else:
        w.add_key_value(name, field.contents, field.type)

kept = dropped = 0
for t in r.tensors:
    if t.name.startswith('blk.64.'):
        dropped += 1; continue
    shape = list(t.field.shape)[::-1]   # gguf-py stores reversed
    w.add_tensor(t.name, np.asarray(t.data).reshape(shape), raw_dtype=t.tensor_type)
    kept += 1
print(f'kept {kept} tensors, dropped {dropped}')
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(True)
print('done:', dst)
