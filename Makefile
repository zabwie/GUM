NVCC ?= $(if $(wildcard /usr/local/cuda/bin/nvcc),/usr/local/cuda/bin/nvcc,nvcc)
CUDA_ARCH ?= sm_86
HOST_FLAGS ?= -mf16c -mfma -mavx2 -mavxvnni
BUILD_DIR ?= build
DECODER := $(BUILD_DIR)/real_decode

.PHONY: all decoder test

all: decoder

decoder: $(DECODER)

$(DECODER): benchmarks/real_decode.cu benchmarks/tiered_kv.cuh \
	benchmarks/packed_scale_codec.cuh benchmarks/eagle4_draft.cuh \
	benchmarks/parallel_draft.cuh
	mkdir -p "$(BUILD_DIR)"
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) \
		-Xcompiler "$(HOST_FLAGS)" -lpthread $< -o $@

test: decoder
	GUM_DECODER="$(CURDIR)/$(DECODER)" ./tests/verify_e196.sh
