#pragma once

#include <queue>

#if __has_include(<cub/device/device_topk.cuh>) && __has_include(<cuda/std/execution>)
#define GUM_EAGLE_GPU_TOPK 1
#include <cub/device/device_topk.cuh>
#include <cuda/stream>
#include <cuda/std/execution>
#include <thrust/iterator/counting_iterator.h>

__global__ void k_eagle_rank(const uint8_t* w, const int8_t* xq, const int2* meta,
                             uint64_t* keys, int rows) {
    const int sub = threadIdx.x & 3;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
    if (row >= rows) return;
    const auto* weight = reinterpret_cast<const uint16_t*>(w + (size_t)row * 136 + sub * 34);
    const auto* input = reinterpret_cast<const int4*>(xq + sub * 32);
    const int4 first = input[0], second = input[1];
    const int values[8] = {first.x, first.y, first.z, first.w, second.x, second.y, second.z, second.w};
    int dot = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q8 = weight[1 + 2*i] | (uint32_t)weight[2 + 2*i] << 16;
        dot = __dp4a(q8, values[i], dot);
    }
    const float scale = __half2float(__ushort_as_half(weight[0]));
    float value = __int_as_float(meta[sub].x) * (scale * dot);
    value += __shfl_down_sync(0xffffffffu, value, 2, 4);
    value += __shfl_down_sync(0xffffffffu, value, 1, 4);
    if (!sub) {
        const uint32_t bits = __float_as_uint(value);
        const uint32_t ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
        keys[row] = (uint64_t)ordered << 32 | (uint32_t)~row;
    }
}

#else
#define GUM_EAGLE_GPU_TOPK 0
#endif

// The Qwen3.8 EAGLE4 draft uses three target layer outputs, one decoder layer,
// and a low-rank vocabulary pass before rescoring selected full-head rows.
__global__ void k_eagle_q8_mat(const uint8_t* weights, long K, long rows,
                               const int8_t* xq, const int2* meta, float* out) {
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const long blocks = K / 32;
    float sum = 0.f;
    for (long b = lane; b < blocks; b += 32) {
        const uint16_t* weight = (const uint16_t*)(weights + (row * blocks + b) * 34);
        const int4* activation = (const int4*)(xq + b * 32);
        const int4 lo = activation[0], hi = activation[1];
        const int values[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        int dot = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t word = weight[1 + 2 * i] | (uint32_t)weight[2 + 2 * i] << 16;
            dot = __dp4a((int)word, values[i], dot);
        }
        sum += __int_as_float(meta[b].x) * (__half2float(__ushort_as_half(weight[0])) * dot);
    }
    for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (!lane) out[row] = sum;
}

struct DraftChoice {
    long token;
    double probability;
};

struct DraftTree {
    std::vector<long> tokens;
    std::vector<int> parents{-1};
    std::vector<bool> from_history;
};

static constexpr int kDraftFanout = 8;
static constexpr int kMaxDraftGroup = 4;

struct DraftOptions {
    DraftChoice choices[kDraftFanout];
    int count = 0;
};

template<class Propose, class Lookup>
DraftTree build_draft_tree(long eos, int limit, int max_depth, double cutoff, int group_width,
                           Propose propose, Lookup lookup, int model_budget = -1) {
    struct Proposal {
        double score;
        long token;
        int parent;
        std::vector<long> continuation;
        bool use_history;
        bool from_history;
        bool operator<(const Proposal& other) const {
            if (score != other.score) return score < other.score;
            if (parent != other.parent) return parent > other.parent;
            return token > other.token;
        }
    };
    DraftTree result;
    if (!limit || !max_depth) return result;
    // model_budget caps MODEL-EXPANDED nodes (each costs a drafter forward);
    // history-guided nodes are free and may use the full `limit` (E167 deep-gamma).
    if (model_budget < 0) model_budget = limit;
    int model_nodes = 0;
    std::priority_queue<Proposal> pending;
    std::vector<std::vector<int>> paths{{0}};
    std::vector<double> scores{0.0};
    auto expand = [&](int first, int count) {
        DraftOptions options[kMaxDraftGroup]{};
        propose(first, count, paths, result, options);
        for (int row = 0; row < count; ++row) {
            const int node = first + row;
            const auto& path = paths[node];
            const int depth = (int)path.size() - 1;
            std::vector<long> prefix;
            for (size_t j = 1; j < path.size(); ++j) prefix.push_back(result.tokens[path[j] - 1]);
            for (int i = 0; i < options[row].count; ++i) {
                const auto& choice = options[row].choices[i];
                if (!(choice.probability > 0.0 && choice.probability >= cutoff) || choice.token == eos) continue;
                const double score = scores[node] + log(choice.probability * 0.9);
                const auto match = i == 0 ? lookup(prefix, choice.token, limit - depth) : std::vector<long>{};
                if (match.size() >= 4) {
                    pending.push({score, choice.token, node, {match.begin() + 1, match.end()}, true, false});
                } else {
                    pending.push({score, choice.token, node, {}, false, false});
                }
            }
        }
    };
    expand(0, 1);
    while (!pending.empty() && (int)result.tokens.size() < limit) {
        const int first = (int)result.tokens.size() + 1;
        int count = 0;
        while (!pending.empty() && (int)result.tokens.size() < limit && count < group_width) {
            const Proposal next = pending.top();
            const int depth = (int)paths[next.parent].size();
            const bool expandable = !next.use_history && depth < max_depth && model_nodes < model_budget;
            // A group contains independent candidates whose parents have
            // already been evaluated. History continuations need no model call.
            if (count && !expandable) break;
            pending.pop();
            const int node = (int)result.tokens.size() + 1;
            result.parents.push_back(next.parent);
            paths.push_back(paths[next.parent]);
            paths.back().push_back(node);
            result.tokens.push_back(next.token);
            result.from_history.push_back(next.from_history);
            scores.push_back(next.score);
            if (!next.use_history) ++model_nodes;
            if (next.use_history && depth < limit && !next.continuation.empty() && next.continuation.front() != eos)
                pending.push({next.score + log(0.86), next.continuation.front(), node,
                    {next.continuation.begin() + 1, next.continuation.end()}, true, true});
            if (!expandable) break;
            ++count;
        }
        if (count && (int)result.tokens.size() < limit) expand(first, count);
    }
    return result;
}

struct Eagle4 {
    static constexpr int hidden = 5120, inner = 16384, heads = 40, hd = 128;
    static constexpr int rank = 128, vocab = 248320, candidates = 128, fanout = kDraftFanout;
    static constexpr double proposal_temperature = 1.2;
    static constexpr size_t rowbytes = hidden / 32 * 18;
    int ctx = 0;
    cudaStream_t stream;
    uint8_t *weights = nullptr, *short_head = nullptr, *gathered = nullptr;
    const uint8_t* host_head = nullptr;
    std::map<std::string, uint64_t> offsets;

    float *features = nullptr, *hidden_context = nullptr, *embeddings = nullptr, *combined = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *keys = nullptr, *values = nullptr;
    float *attn = nullptr, *state = nullptr, *normed = nullptr, *out = nullptr;
    float* draft_states = nullptr;
    int *draft_depth = nullptr, *draft_path = nullptr;
    float *gate = nullptr, *up = nullptr, *lowrank = nullptr, *logits = nullptr;
    int8_t* quant = nullptr;
    int2* meta = nullptr;
    std::vector<float> host_logits;
    std::vector<int> ranking;

#if GUM_EAGLE_GPU_TOPK
    uint64_t *rank_keys = nullptr, *top_keys = nullptr;
    int* top_ids = nullptr;
    void* top_scratch = nullptr;
    size_t top_bytes = 0;

    cudaError_t top_k() {
        auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
            cuda::execution::output_ordering::unsorted);
        auto stream_env = cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>{
            cuda::get_stream_t{}, cuda::stream_ref{stream}};
        const cuda::std::execution::env<decltype(requirements), decltype(stream_env)> env{requirements, stream_env};
        return cub::DeviceTopK::MaxPairs(top_scratch, top_bytes, rank_keys, top_keys,
            thrust::counting_iterator<int>(0), top_ids, vocab, candidates, env);
    }

#endif

    struct Shape { const char* name; int columns, rows, type; };
    static constexpr Shape shapes[] = {
        {"fc.weight", 3 * hidden, hidden, kGGML_Q4_0},
        {"fc_norm.0.weight", hidden, 1, kGGML_F32},
        {"fc_norm.1.weight", hidden, 1, kGGML_F32},
        {"fc_norm.2.weight", hidden, 1, kGGML_F32},
        {"layers.0.input_layernorm.weight", hidden, 1, kGGML_F32},
        {"layers.0.hidden_norm.weight", hidden, 1, kGGML_F32},
        {"layers.0.self_attn.q_proj.weight", 2 * hidden, hidden, kGGML_Q4_0},
        {"layers.0.self_attn.k_proj.weight", 2 * hidden, hidden, kGGML_Q4_0},
        {"layers.0.self_attn.v_proj.weight", 2 * hidden, hidden, kGGML_Q4_0},
        {"layers.0.self_attn.o_proj.weight", hidden, hidden, kGGML_Q4_0},
        {"layers.0.post_attention_layernorm.weight", hidden, 1, kGGML_F32},
        {"layers.0.mlp.gate_proj.weight", hidden, inner, kGGML_Q4_0},
        {"layers.0.mlp.up_proj.weight", hidden, inner, kGGML_Q4_0},
        {"layers.0.mlp.down_proj.weight", inner, hidden, kGGML_Q4_0},
        {"norm.weight", hidden, 1, kGGML_F32},
        {"efficient_lm_head_down_proj.weight", hidden, rank, kGGML_Q8_0},
        {"efficient_lm_head.weight", rank, vocab, kGGML_Q8_0},
        {"lm_head.weight", hidden, vocab, kGGML_Q4_0},
    };

    static size_t device_bytes(long context, long rows) {
        size_t bytes = vocab * rowbytes;  // Exact target head.
        for (const auto& shape : shapes) {
            if (strcmp(shape.name, "lm_head.weight") == 0) continue;
            const size_t count = (size_t)shape.columns * shape.rows;
            if (shape.type == kGGML_F32) bytes += align_up(count * 4, 256);
            else bytes += align_up(count / 32 * (shape.type == kGGML_Q8_0 ? 34 : 18), 256);
        }
        const size_t buffers = (size_t)(3 * context + 1 + 10 * rows + 5) * hidden * sizeof(float)
            + (2 * inner + vocab + rank) * sizeof(float) + candidates * rowbytes
            + (rows * 3 * hidden + inner) * 5 / 4;
        return bytes + buffers + (GUM_EAGLE_GPU_TOPK ? (4u << 20) : 0);
    }

    template<class T> void alloc(T** ptr, size_t count) {
        check(cudaMalloc(ptr, count * sizeof(T)), "EAGLE allocation");
    }

    void load(const std::string& path, int context, int rows, cudaStream_t st) {
        ctx = context;
        stream = st;
        GgufKV kv;
        std::vector<GTensor> tensors;
        uint64_t start = 0;
        if (!parse_gguf(path.c_str(), kv, tensors, start)) {
            fprintf(stderr, "EAGLE GGUF parse failed\n");
            exit(1);
        }
        if (kv.strs["general.architecture"] != "eagle4" ||
            kv.arr_ints["eagle4.target_layers"] != std::vector<long>{3, 31, 59} ||
            tensors.size() != sizeof(shapes) / sizeof(shapes[0])) {
            fprintf(stderr, "Unexpected EAGLE4 GGUF architecture or feature taps\n");
            exit(1);
        }
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { perror("EAGLE4 open"); exit(1); }
        struct stat stbuf {};
        if (fstat(fd, &stbuf) != 0 || start > (uint64_t)stbuf.st_size) {
            fprintf(stderr, "EAGLE4 GGUF file size unavailable\n");
            exit(1);
        }
        const uint64_t available = (uint64_t)stbuf.st_size - start;
        uint64_t bytes = 0;
        for (const auto& shape : shapes) {
            const auto tensor = std::find_if(tensors.begin(), tensors.end(),
                [&](const GTensor& t) { return t.name == shape.name; });
            const int dims = shape.type == kGGML_F32 ? 1 : 2;
            if (tensor == tensors.end() || tensor->type != shape.type || tensor->nd != dims ||
                tensor->ne[0] != shape.columns || tensor->ne[1] != shape.rows ||
                tensor->off > available || tensor->bytes > available - tensor->off) {
                fprintf(stderr, "Invalid EAGLE4 tensor: %s\n", shape.name);
                exit(1);
            }
            if (tensor->name == "lm_head.weight") continue;
            offsets[tensor->name] = bytes;
            bytes += align_up(tensor->bytes, 256);
        }
        alloc(&weights, bytes);
        std::vector<uint8_t> buffer;
        for (const auto& tensor : tensors) {
            if (tensor.name == "lm_head.weight") {
                const uint64_t offset = (start + tensor.off) & ~4095ull;
                const uint64_t head = start + tensor.off - offset;
                const void* map = mmap(nullptr, tensor.bytes + head, PROT_READ, MAP_PRIVATE, fd, offset);
                if (map == MAP_FAILED) { perror("EAGLE head mmap"); exit(1); }
                host_head = (const uint8_t*)map + head;
                continue;
            }
            buffer.resize(tensor.bytes);
            if (pread(fd, buffer.data(), tensor.bytes, start + tensor.off) != (ssize_t)tensor.bytes) {
                fprintf(stderr, "EAGLE weight read failed\n");
                exit(1);
            }
            check(cudaMemcpy(weights + offsets.at(tensor.name), buffer.data(), tensor.bytes,
                             cudaMemcpyHostToDevice), "EAGLE weights upload");
        }
        close(fd);
        if (!host_head) { fprintf(stderr, "EAGLE full head missing\n"); exit(1); }
        const size_t n = mat_width(rows);
        alloc(&features, n * hidden * 3);
        alloc(&hidden_context, (size_t)(ctx + 1) * hidden);
        check(cudaMemset(hidden_context, 0, hidden * sizeof(float)), "EAGLE zero first context row");
        alloc(&embeddings, n * hidden);
        alloc(&combined, n * hidden * 2);
        alloc(&q, n * hidden);
        alloc(&k, n * hidden);
        alloc(&v, n * hidden);
        alloc(&keys, (size_t)ctx * hidden);
        alloc(&values, (size_t)ctx * hidden);
        alloc(&attn, hidden);
        alloc(&state, hidden);
        alloc(&draft_states, n * hidden);
        alloc(&draft_depth, 1);
        alloc(&draft_path, n);
        alloc(&normed, hidden);
        alloc(&out, hidden);
        alloc(&gate, inner);
        alloc(&up, inner);
        alloc(&lowrank, rank);
        alloc(&logits, vocab);
        alloc(&quant, n * hidden * 3 + inner);
        alloc(&meta, (n * hidden * 3 + inner) / 32);
        check(cudaMemset(quant, 0, n * hidden * 3 + inner), "EAGLE padded activation");
        check(cudaMemset(meta, 0, (n * hidden * 3 + inner) / 32 * sizeof(int2)), "EAGLE padded metadata");
        alloc(&short_head, candidates * rowbytes);
        check(cudaHostAlloc(&gathered, candidates * rowbytes, cudaHostAllocDefault), "EAGLE head staging");
        host_logits.resize(vocab);
#if GUM_EAGLE_GPU_TOPK
        ranking.resize(candidates);
        alloc(&rank_keys, vocab);
        alloc(&top_keys, candidates);
        alloc(&top_ids, candidates);
        check(top_k(), "EAGLE top-k workspace query");
        check(cudaMalloc(&top_scratch, top_bytes), "EAGLE top-k workspace");
#else
        ranking.resize(vocab);
#endif
        printf("eagle_plan,weights_MiB=%.1f,host_head_MiB=%.1f\n",
               bytes / 1048576.0, vocab * rowbytes / 1048576.0);
    }

    const float* norm(const std::string& name) const {
        return (const float*)(weights + offsets.at(name));
    }

    void quantize(const float* input, long K, int rows) {
        const long blocks = K / 32 * rows;
        g_batch = mat_width(rows);
        k_quant_x<<<(blocks + 127) / 128, 128, 0, stream>>>(input, blocks, quant, meta);
    }

    void matrix(const std::string& name, long K, long N, float* output, bool q8 = false) {
        const uint8_t* w = weights + offsets.at(name);
        if (q8) k_eagle_q8_mat<<<(N + 3) / 4, 128, 0, stream>>>(w, K, N, quant, meta, output);
        else launch_mat_q4(w, K, N, quant, meta, output, N, stream);
    }

    void tap(int layer, const float* source, int rows) {
        constexpr int taps[3] = {3, 31, 59};
        static const std::string names[3] = {"fc_norm.0.weight", "fc_norm.1.weight", "fc_norm.2.weight"};
        for (int i = 0; i < 3; ++i) {
            if (layer == taps[i])
                k_rms_norm<<<dim3(1, rows), 256, 0, stream>>>(features + i * hidden, source,
                    norm(names[i]), hidden, 1e-6f, 3 * hidden);
        }
    }

    void project(int position, int rows) {
        quantize(features, 3 * hidden, rows);
        matrix("fc.weight", 3 * hidden, hidden, hidden_context + (size_t)(position + 1) * hidden);
    }

    void attention(const float* incoming, int pos, int rows, bool draft, int storage = -1,
                   int base = -1, int depth = 0) {
        const dim3 rowgrid(1, rows);
        k_rms_norm<<<rowgrid, 256, 0, stream>>>(combined, embeddings,
            norm("layers.0.input_layernorm.weight"), hidden, 1e-6f, 2 * hidden);
        k_rms_norm<<<rowgrid, 256, 0, stream>>>(combined + hidden, incoming,
            norm("layers.0.hidden_norm.weight"), hidden, 1e-6f, 2 * hidden);
        quantize(combined, 2 * hidden, rows);
        if (draft) matrix("layers.0.self_attn.q_proj.weight", 2 * hidden, hidden, q);
        matrix("layers.0.self_attn.k_proj.weight", 2 * hidden, hidden, k);
        matrix("layers.0.self_attn.v_proj.weight", 2 * hidden, hidden, v);
        const int query_heads = draft ? heads : 0;
        k_rope<<<dim3(query_heads + heads, rows), 128, 0, stream>>>(q, k, pos,
            query_heads, heads, hd, hd, 1e7f, hd, 0, 1);
        k_append_kv<<<dim3(hidden / 256, rows), 256, 0, stream>>>(keys, values,
            k, v, storage < 0 ? pos : storage, ctx, heads, hd, 0, true);
        if (!draft) return;

        k_attn_decode<<<heads, 256, (pos + 1) * sizeof(float), stream>>>(attn, keys, values,
            q, base < 0 ? pos : base, ctx, heads, heads, hd, 1.f / sqrtf((float)hd), hd, 0, 1,
            base < 0 ? nullptr : draft_depth, base < 0 ? nullptr : draft_path, depth + 1);
        quantize(attn, hidden, 1);
        matrix("layers.0.self_attn.o_proj.weight", hidden, hidden, state);
        k_add<<<hidden / 256, 256, 0, stream>>>(state, incoming, hidden);
        k_rms_norm<<<1, 256, 0, stream>>>(normed, state, norm("layers.0.post_attention_layernorm.weight"), hidden, 1e-6f);
        quantize(normed, hidden, 1);
        matrix("layers.0.mlp.gate_proj.weight", hidden, inner, gate);
        matrix("layers.0.mlp.up_proj.weight", hidden, inner, up);
        k_silu_mul<<<inner / 256, 256, 0, stream>>>(gate, gate, up, inner);
        quantize(gate, inner, 1);
        matrix("layers.0.mlp.down_proj.weight", inner, hidden, out);
        k_add<<<hidden / 256, 256, 0, stream>>>(state, out, hidden);
        k_rms_norm<<<1, 256, 0, stream>>>(normed, state, norm("norm.weight"), hidden, 1e-6f);
    }

    void propose(const float* incoming, int base, const std::vector<int>& path, DraftChoice (&choices)[fanout]) {
        const int depth = (int)path.size() - 1;
        check(cudaMemcpyAsync(draft_depth, &depth, sizeof(depth), cudaMemcpyHostToDevice, stream), "EAGLE draft depth");
        check(cudaMemcpyAsync(draft_path, path.data(), path.size() * sizeof(int), cudaMemcpyHostToDevice, stream), "EAGLE draft path");
        attention(incoming, base + depth, 1, true, base + path.back(), base, depth);
        quantize(normed, hidden, 1);
        matrix("efficient_lm_head_down_proj.weight", hidden, rank, lowrank, true);
        quantize(lowrank, rank, 1);
#if GUM_EAGLE_GPU_TOPK
        const uint8_t* weights_head = weights + offsets.at("efficient_lm_head.weight");
        k_eagle_rank<<<vocab / 32, 128, 0, stream>>>(weights_head, quant, meta, rank_keys, vocab);
        check(top_k(), "EAGLE top-k selection");
        check(cudaMemcpyAsync(ranking.data(), top_ids, candidates * sizeof(int),
            cudaMemcpyDeviceToHost, stream), "EAGLE candidate indices");
        check(cudaStreamSynchronize(stream), "EAGLE candidate selection");
#else
        matrix("efficient_lm_head.weight", rank, vocab, logits, true);
        check(cudaMemcpyAsync(host_logits.data(), logits, vocab * sizeof(float),
            cudaMemcpyDeviceToHost, stream), "EAGLE candidate logits");
        check(cudaStreamSynchronize(stream), "EAGLE candidate selection");
        for (int id = 0; id < vocab; ++id) ranking[id] = id;
        std::partial_sort(ranking.begin(), ranking.begin() + candidates, ranking.end(), [&](int a, int b) {
            return host_logits[a] > host_logits[b] || (host_logits[a] == host_logits[b] && a < b);
        });
#endif
        for (int i = 0; i < candidates; ++i)
            memcpy(gathered + i * rowbytes, host_head + ranking[i] * rowbytes, rowbytes);
        check(cudaMemcpyAsync(short_head, gathered, candidates * rowbytes, cudaMemcpyHostToDevice, stream), "EAGLE selected head");
        quantize(normed, hidden, 1);
        launch_mat_q4(short_head, hidden, candidates, quant, meta, logits, candidates, stream);
        check(cudaMemcpyAsync(host_logits.data(), logits, candidates * sizeof(float),
            cudaMemcpyDeviceToHost, stream), "EAGLE selected logits");
        check(cudaGetLastError(), "EAGLE draft kernels");
        check(cudaStreamSynchronize(stream), "EAGLE draft");
        int order[candidates];
        for (int i = 0; i < candidates; ++i) order[i] = i;
        std::partial_sort(order, order + fanout, order + candidates, [&](int a, int b) {
            return host_logits[a] > host_logits[b] || (host_logits[a] == host_logits[b] && ranking[a] < ranking[b]);
        });
        const double peak = host_logits[order[0]];
        double mass = 0.0;
        for (int i = 0; i < candidates; ++i)
            mass += exp(((double)host_logits[i] - peak) / proposal_temperature);
        for (int i = 0; i < fanout; ++i)
            choices[i] = {ranking[order[i]], exp(((double)host_logits[order[i]] - peak) / proposal_temperature) / mass};
    }

    template<class Embed, class Lookup>
    DraftTree draft(long known, long eos, int base, int limit, int max_depth, double cutoff,
                     Embed embed, Lookup lookup) {
        return build_draft_tree(eos, limit, max_depth, cutoff, 1,
            [&](int node, int, const std::vector<std::vector<int>>& paths, const DraftTree& tree, DraftOptions* options) {
                const float* input = node ? draft_states + (size_t)tree.parents[node] * hidden :
                    hidden_context + (size_t)base * hidden;
                embed(node ? tree.tokens[node - 1] : known);
                propose(input, base, paths[node], options[0].choices);
                check(cudaMemcpyAsync(draft_states + (size_t)node * hidden, normed,
                    hidden * sizeof(float), cudaMemcpyDeviceToDevice, stream), "EAGLE graph hidden");
                options[0].count = fanout;
            }, lookup);
    }
};
