#!/usr/bin/env bash
# Strip E195 debug assets for perf runs: device asserts, unconditional prints.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python3 - <<'PYEOF'
import re
p = 'benchmarks/tiered_kv.cuh'
s = open(p).read()
# remove device asserts (unconditional perf cost)
s = s.replace('#include <cassert>\n', '')
s = re.sub(r'    assert\([^\n]*\n', '', s)
s = re.sub(r'\n *assert\([^;]*;', '', s)
# remove BAD_ATT/BAD_overlay prints (keep gated ones)
s = re.sub(r'        if \(os >= cache\.capacity\)\n            printf\("BAD_overlay[^;]*;', '', s)
s = re.sub(r'\n *if \(width == 1 && !straddle && cache\.overlay_route && cache\.overlay && cache\.overlay\[t\] < 0\)\n *printf\("BAD_ATT[^;]*;', '', s)
s = re.sub(r'\n *if \(width == 1 && !straddle && cache\.overlay && cache\.overlay\[t\] < 0\)\n *printf\("BAD_ATT[^;]*;', '', s)
open(p, 'w').write(s)

p = 'benchmarks/real_decode.cu'
s = open(p).read()
# remove the unconditional HOST_GAP scan
s = re.sub(r'            for \(long pos = flag_start; pos < end; \+\+pos\)\n                if \(tier_flags\[pos / kTierMidGroup \* kTierMidGroup \+ 1\] && tier_overlay\[pos\] < 0\)\n                    printf\("HOST_GAP[^;]*;\n', '', s)
open(p, 'w').write(s)
print("stripped")
PYEOF
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler "-mf16c -mfma -mavx2 -mavxvnni" -lpthread benchmarks/real_decode.cu -o benchmarks/real_decode && echo STRIP_BUILD_OK
