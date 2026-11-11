// Experimental block drafter trained by scripts/train_parallel_draft.py.
__global__ void k_draft_normalize(float* output, const float* state, const float* embedding, int width) {
    __shared__ float squares[256];
    const float* source = blockIdx.x ? embedding : state;
    float sum = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) sum += source[i] * source[i];
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) squares[threadIdx.x] += squares[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(squares[0] / width + 1e-6f);
    for (int i = threadIdx.x; i < width; i += blockDim.x) output[blockIdx.x * width + i] = source[i] * scale;
}

__global__ void k_draft_linear(float* output, const float* input, const float* weights,
                               int cols, int rows, bool bias, bool relu_input, bool relu_output) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    input += blockIdx.y * cols;
    const float* w = weights + (size_t)row * cols;
    float sum = 0;
    for (int i = lane; i < cols; i += 32) {
        const float x = input[i];
        sum = fmaf(w[i], relu_input ? fmaxf(x, 0.f) : x, sum);
    }
    for (int shift = 16; shift; shift /= 2) sum += __shfl_down_sync(0xffffffff, sum, shift);
    if (lane == 0) {
        if (bias) sum += weights[(size_t)rows * cols + row];
        output[blockIdx.y * rows + row] = relu_output ? fmaxf(sum, 0.f) : sum;
    }
}

__global__ void k_draft_adjacent(float* forecast, const float* injection, int count, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) forecast[i + width] += injection[i];
}

struct ParallelDraft {
    int width = 0, block = 0, stem = 0, latent = 0;
    std::vector<int32_t> vocab;
    std::vector<float> logits;
    float *first = nullptr, *forecast = nullptr, *adjacent = nullptr, *head = nullptr;
    float *scratch = nullptr, *state = nullptr;

    bool load(const std::string& path, long target_width, long target_vocab) {
        FILE* file = fopen(path.c_str(), "rb");
        if (!file) return false;
        uint32_t header[7];
        bool ok = fread(header, sizeof(uint32_t), 7, file) == 7 &&
            !memcmp(header, "VPD1", 4) && header[1] == target_width &&
            header[2] > 0 && header[2] <= 24 && header[3] > 0 && header[3] <= 1024 &&
            header[4] > 0 && header[4] <= 512 && header[5] > 0 &&
            header[5] <= (uint32_t)target_vocab && header[6] <= 1;
        if (ok) {
            width = header[1]; block = header[2]; stem = header[3]; latent = header[4];
            vocab.resize(header[5]);
            ok = fread(vocab.data(), sizeof(int32_t), vocab.size(), file) == vocab.size() &&
                vocab.front() >= 0 && vocab.back() < target_vocab &&
                std::adjacent_find(vocab.begin(), vocab.end(), std::greater_equal<int32_t>()) == vocab.end();
        }
        auto read_layer = [&](float** device, size_t count) {
            if (!ok) return;
            std::vector<float> weights(count);
            ok = fread(weights.data(), sizeof(float), count, file) == count &&
                std::all_of(weights.begin(), weights.end(), [](float value) { return std::isfinite(value); });
            if (!ok) return;
            check(cudaMalloc(device, count * sizeof(float)), "parallel draft weights");
            check(cudaMemcpy(*device, weights.data(), count * sizeof(float), cudaMemcpyHostToDevice), "parallel draft upload");
        };
        if (ok) {
            read_layer(&first, (size_t)stem * (2 * width + 1));
            read_layer(&forecast, (size_t)block * latent * (stem + 1));
            if (header[6]) read_layer(&adjacent, (size_t)latent * latent);
            read_layer(&head, vocab.size() * (latent + 1));
            ok = ok && fgetc(file) == EOF;
        }
        fclose(file);
        if (!ok) return false;
        logits.resize((size_t)block * vocab.size());
        const size_t scratch_count = 4 * width + stem + 2 * block * latent + logits.size();
        check(cudaMalloc(&scratch, scratch_count * sizeof(float)), "parallel draft scratch");
        state = scratch;
        printf("parallel_draft_plan,block=%d,vocab=%zu,adjacent=%d,scratch_bytes=%zu\n",
               block, vocab.size(), adjacent != nullptr, scratch_count * sizeof(float));
        return true;
    }

    void commit(const float* hidden, cudaStream_t stream) {
        check(cudaMemcpyAsync(state, hidden, width * sizeof(float), cudaMemcpyDeviceToDevice, stream),
              "parallel draft state");
    }

    std::vector<long> propose(const float* embedding, int limit, long eos, cudaStream_t stream) {
        if (!limit) return {};
        limit = std::min(limit, block);
        float* device_embedding = state + width;
        float* normalized = device_embedding + width;
        float* bottleneck = normalized + 2 * width;
        float* prediction = bottleneck + stem;
        float* injection = prediction + block * latent;
        float* device_logits = injection + block * latent;
        const int choices = (int)vocab.size();
        check(cudaMemcpyAsync(device_embedding, embedding, width * sizeof(float), cudaMemcpyHostToDevice, stream),
              "parallel draft embedding");
        k_draft_normalize<<<2, 256, 0, stream>>>(normalized, state, device_embedding, width);
        k_draft_linear<<<(stem + 7) / 8, 256, 0, stream>>>(bottleneck, normalized, first,
            2 * width, stem, true, false, true);
        k_draft_linear<<<(block * latent + 7) / 8, 256, 0, stream>>>(prediction, bottleneck, forecast,
            stem, block * latent, true, false, false);
        if (adjacent && limit > 1) {
            k_draft_linear<<<dim3((latent + 7) / 8, limit - 1), 256, 0, stream>>>(
                injection, prediction, adjacent, latent, latent, false, true, false);
            const int count = (limit - 1) * latent;
            k_draft_adjacent<<<(count + 255) / 256, 256, 0, stream>>>(prediction, injection, count, latent);
        }
        k_draft_linear<<<dim3((choices + 7) / 8, limit), 256, 0, stream>>>(
            device_logits, prediction, head, latent, choices, true, true, false);
        check(cudaGetLastError(), "parallel draft kernels");
        check(cudaMemcpyAsync(logits.data(), device_logits, (size_t)limit * choices * sizeof(float),
              cudaMemcpyDeviceToHost, stream), "parallel draft logits");
        check(cudaStreamSynchronize(stream), "parallel draft sync");
        std::vector<long> tokens;
        for (int pos = 0; pos < limit; ++pos) {
            const float* row = logits.data() + pos * choices;
            const long id = vocab[std::max_element(row, row + choices) - row];
            if (id == eos) break;
            tokens.push_back(id);
        }
        return tokens;
    }
};
