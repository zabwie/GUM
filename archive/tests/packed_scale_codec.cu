#define main real_decode_main
#include "../../benchmarks/real_decode.cu"
#undef main

int main() {
    constexpr int rows = 64;
    for (int columns : {5120, 17408}) {
        const int blocks = columns / 32;
        std::vector<uint8_t> packed((size_t)rows * blocks * 18);
        for (size_t p = 0; p < packed.size() / kRawPanel; ++p) {
            for (int tile = 0; tile < kPanelTiles; ++tile) {
                uint8_t* dest = packed.data() + p * kRawPanel + tile * kRawTile;
                for (int i = 0; i < kTileQuants; ++i) dest[i] = (p * 29 + tile * 17 + i * 41) & 255;
                for (int r = 0; r < 16; ++r) {
                    dest[kTileQuants + 2 * r] = (tile * 93 + r * 187) & 255;
                    const int high = 20 + (r + tile) % 16 + (p % 4 == 0 && r == 15 ? 16 : 0);
                    dest[kTileQuants + 2 * r + 1] = high | ((r & 1) << 7);
                }
            }
        }
        std::vector<uint8_t> compressed;
        const size_t encoded = encode_scale_panels(compressed, packed.data(), packed.size());
        if (encoded >= packed.size() || !verify_scale_panels(compressed.data(), packed.data(), packed.size())) return 1;
        uint8_t *raw_gpu, *encoded_gpu;
        float *activations, *output;
        int8_t* quantized;
        int2* metadata;
        check(cudaMalloc(&raw_gpu, packed.size()), "raw weights");
        check(cudaMalloc(&encoded_gpu, encoded), "encoded weights");
        check(cudaMemcpy(raw_gpu, packed.data(), packed.size(), cudaMemcpyHostToDevice), "upload raw weights");
        check(cudaMemcpy(encoded_gpu, compressed.data(), encoded, cudaMemcpyHostToDevice), "upload encoded weights");
        std::vector<float> input(32 * columns);
        for (size_t i = 0; i < input.size(); ++i) input[i] = sinf(i * .23f) * cosf(i * .37f);
        check(cudaMalloc(&activations, input.size() * sizeof(float)), "activations");
        check(cudaMemcpy(activations, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "upload activations");
        check(cudaMalloc(&quantized, input.size()), "quantized activations");
        check(cudaMalloc(&metadata, 32 * blocks * sizeof(int2)), "activation metadata");
        check(cudaMalloc(&output, 32 * rows * sizeof(float)), "output");
        k_quant_x<<<(32 * blocks + 127) / 128, 128>>>(activations, 32 * blocks, quantized, metadata);
        for (int width : {8, 12, 16, 24, 32}) {
            g_batch = width;
            const int count = (width == 12 ? 16 : width) * rows;
            std::vector<float> raw(count), fused(count);
            launch_packed_q4(raw_gpu, columns, rows, quantized, metadata, output, rows, nullptr);
            check(cudaMemcpy(raw.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost), "download raw outputs");
            launch_packed_q4<true>(encoded_gpu, columns, rows, quantized, metadata, output, rows, nullptr);
            check(cudaMemcpy(fused.data(), output, count * sizeof(float), cudaMemcpyDeviceToHost), "download fused outputs");
            const bool same = !memcmp(raw.data(), fused.data(), count * sizeof(float));
            printf("codec_parity,columns=%d,width=%d,values=%d,raw=%zu,encoded=%zu,bit_exact=%d\n",
                   columns, width, count, packed.size(), encoded, same);
            if (!same || !std::all_of(raw.begin(), raw.end(), [](float v) { return std::isfinite(v); })) return 1;
        }
        for (void* allocation : {static_cast<void*>(raw_gpu), static_cast<void*>(encoded_gpu),
                                 static_cast<void*>(activations), static_cast<void*>(output),
                                 static_cast<void*>(quantized), static_cast<void*>(metadata)})
            check(cudaFree(allocation), "release test allocation");
    }
}
