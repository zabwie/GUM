#!/usr/bin/env bash
# E033 batch sweep: one streamed weight pass serves B decode vectors.
set -u
cd "$(dirname "$0")/.."
/usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_86 -lpthread benchmarks/streamed_decode.cu -o benchmarks/streamed_decode || exit 1
for B in "$@"; do
  out=results/streamed-decode-27b-hot8-cache5.6-dp4a-hotstream-batch${B}-20261108.txt
  taskset -c 0-3 benchmarks/streamed_decode --model /home/zabwie/Desktop/Chip/models/Qwen3.8-27B-STRICT-Q4_0.gguf \
    --manifest results/qwen38-27b-manifest.txt --hot-mib 8192 --cache-mib 5600 --passes 8 --readers 4 --batch "$B" > "$out" 2>&1
  echo "== batch $B"; tr -d '\r' < "$out" | grep -E '^(alloc|verify|summary)|MISMATCH|ABORT'; tr -d '\r' < "$out" | grep '^token,7' | cut -c1-90
done
