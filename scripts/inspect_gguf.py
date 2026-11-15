import os, sys, re
sys.path.insert(0, os.path.expanduser(os.environ.get('GGUF_PYTHON_PATH', '~/llama.cpp/gguf-py')))
from collections import Counter
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for k, f in r.fields.items():
    if k.startswith('tokenizer.ggml.') and k not in ('tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id', 'tokenizer.ggml.model', 'tokenizer.ggml.pre'):
        continue
    try:
        v = f.contents()
        print(k, v if not isinstance(v, (list, bytes)) or len(v) < 16 else str(v[:8]) + '...')
    except Exception as e:
        print(k, '?', e)
print('tensors', len(r.tensors))
print(Counter(t.tensor_type.name for t in r.tensors))
for t in r.tensors[:12]:
    print(t.name, t.tensor_type.name, list(t.shape), t.data_offset, t.n_bytes)
names = sorted(set(re.sub(r'\.\d+\.', '.N.', t.name) for t in r.tensors))
print(names)
