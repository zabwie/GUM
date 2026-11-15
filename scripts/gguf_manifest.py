import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get('GGUF_PYTHON_PATH', '~/llama.cpp/gguf-py')))
from gguf import GGUFReader
# Read-only: emits "name K N absolute_offset nbytes" for 2-D Q4_0 tensors, in file order.
r = GGUFReader(sys.argv[1])
for t in sorted(r.tensors, key=lambda t: int(t.data_offset)):
    if t.tensor_type.name == 'Q4_0' and len(t.shape) == 2:
        print(t.name, int(t.shape[0]), int(t.shape[1]), int(t.data_offset), int(t.n_bytes))
