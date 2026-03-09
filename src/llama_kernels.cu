#include "llama_kernels.h"
#include "utils.h"
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Embedding lookup: out[i] = embeddings[token_ids[i]]
// No positional embedding (Llama uses RoPE instead)
// ---------------------------------------------------------------------------
__global__ void llama_embedding_lookup_kernel(float *out, const float *embeddings,
                                               const int *token_ids,
                                               int seq_len, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_embd) return;
    int row = idx / n_embd;
    int col = idx % n_embd;
    out[idx] = embeddings[token_ids[row] * n_embd + col];
}

void llama_embedding_lookup(float *out, const float *embeddings,
                            const int *token_ids, int seq_len, int n_embd) {
    int n = seq_len * n_embd;
    int block = 256;
    int grid = (n + block - 1) / block;
    llama_embedding_lookup_kernel<<<grid, block>>>(out, embeddings, token_ids,
                                                    seq_len, n_embd);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
// ---------------------------------------------------------------------------
__global__ void llama_rmsnorm_kernel(float *out, const float *inp,
                                      const float *weight,
                                      int rows, int cols, float eps) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *x = inp + row * cols;
    float *y = out + row * cols;

    // Compute sum of squares
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_ss += x[i] * x[i];
    }
    shared[threadIdx.x] = local_ss;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float rms = rsqrtf(shared[0] / cols + eps);

    // Scale
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = x[i] * rms * weight[i];
    }
}

void llama_rmsnorm(float *out, const float *inp, const float *weight,
                   int rows, int cols, float eps) {
    int block = 256;
    if (cols < block) block = cols;
    int b = 1;
    while (b < block) b <<= 1;
    block = b;
    if (block > 1024) block = 1024;

    size_t shared_mem = block * sizeof(float);
    llama_rmsnorm_kernel<<<rows, block, shared_mem>>>(out, inp, weight,
                                                       rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// RoPE + reshape + KV cache update
//
// Applies rotary position embedding to Q and K, reshapes to multi-head
// layout, and writes K/V into the cache.
//
// Q input:  [seq_len, n_head * head_dim]
// K input:  [seq_len, n_kv_head * head_dim]
// V input:  [seq_len, n_kv_head * head_dim]
//
// Q output: [n_head, seq_len, head_dim]  (with RoPE applied)
// K cache:  [n_kv_head, n_ctx, head_dim] (RoPE applied, written at past_len)
// V cache:  [n_kv_head, n_ctx, head_dim] (no RoPE, written at past_len)
// ---------------------------------------------------------------------------
// HuggingFace "rotate_half" RoPE convention:
// Pairs (x[j], x[j + D/2]) for j = 0..D/2-1
// y[j]       = x[j] * cos - x[j+D/2] * sin
// y[j + D/2] = x[j+D/2] * cos + x[j] * sin
__global__ void llama_rope_q_kernel(float *q_out, const float *q_in,
                                     int seq_len, int past_len,
                                     int n_head, int head_dim,
                                     float rope_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = seq_len * n_head * half_dim;
    if (idx >= total) return;

    // Decompose into (seq_pos, head, pair_idx)
    int pair_idx = idx % half_dim;
    int tmp = idx / half_dim;
    int head = tmp % n_head;
    int seq_pos = tmp / n_head;

    int pos = past_len + seq_pos;
    float freq = 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)head_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Read "half" pair from input [seq_len, n_head * head_dim]
    int in_offset = seq_pos * (n_head * head_dim) + head * head_dim;
    float x_first  = q_in[in_offset + pair_idx];
    float x_second = q_in[in_offset + pair_idx + half_dim];

    // Rotate (HF rotate_half convention)
    float y_first  = x_first * cos_a - x_second * sin_a;
    float y_second = x_second * cos_a + x_first * sin_a;

    // Write to output [n_head, seq_len, head_dim]
    int out_offset = head * seq_len * head_dim + seq_pos * head_dim;
    q_out[out_offset + pair_idx]            = y_first;
    q_out[out_offset + pair_idx + half_dim] = y_second;
}

__global__ void llama_rope_kv_cache_kernel(float *k_cache, float *v_cache,
                                            const float *k_in, const float *v_in,
                                            int seq_len, int past_len,
                                            int n_kv_head, int head_dim,
                                            int n_ctx, float rope_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = seq_len * n_kv_head * half_dim;
    if (idx >= total) return;

    int pair_idx = idx % half_dim;
    int tmp = idx / half_dim;
    int head = tmp % n_kv_head;
    int seq_pos = tmp / n_kv_head;

    int pos = past_len + seq_pos;
    float freq = 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)head_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // K: read "half" pair from [seq_len, n_kv_head * head_dim], apply RoPE
    int in_offset = seq_pos * (n_kv_head * head_dim) + head * head_dim;
    float k_first  = k_in[in_offset + pair_idx];
    float k_second = k_in[in_offset + pair_idx + half_dim];
    float ky_first  = k_first * cos_a - k_second * sin_a;
    float ky_second = k_second * cos_a + k_first * sin_a;

    int cache_pos = past_len + seq_pos;
    int cache_offset = head * n_ctx * head_dim + cache_pos * head_dim;
    k_cache[cache_offset + pair_idx]            = ky_first;
    k_cache[cache_offset + pair_idx + half_dim] = ky_second;

    // V: no RoPE, just copy to cache (both halves)
    v_cache[cache_offset + pair_idx]            = v_in[in_offset + pair_idx];
    v_cache[cache_offset + pair_idx + half_dim] = v_in[in_offset + pair_idx + half_dim];
}

void llama_rope_and_cache(float *q_out,
                          float *k_cache, float *v_cache,
                          const float *q_in, const float *k_in,
                          const float *v_in,
                          int seq_len, int past_len,
                          int n_head, int n_kv_head, int head_dim,
                          int n_ctx, float rope_theta) {
    int half_dim = head_dim / 2;
    int block = 256;

    // RoPE for Q + reshape
    int total_q = seq_len * n_head * half_dim;
    int grid_q = (total_q + block - 1) / block;
    llama_rope_q_kernel<<<grid_q, block>>>(q_out, q_in, seq_len, past_len,
                                            n_head, head_dim, rope_theta);
    CUDA_CHECK(cudaGetLastError());

    // RoPE for K + V cache write
    int total_kv = seq_len * n_kv_head * half_dim;
    int grid_kv = (total_kv + block - 1) / block;
    llama_rope_kv_cache_kernel<<<grid_kv, block>>>(k_cache, v_cache,
                                                    k_in, v_in,
                                                    seq_len, past_len,
                                                    n_kv_head, head_dim,
                                                    n_ctx, rope_theta);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Causal softmax (same algorithm as GPT-2 version)
// ---------------------------------------------------------------------------
__global__ void llama_causal_softmax_kernel(float *scores, int n_head,
                                             int seq_len, int total_len,
                                             int past_len, float scale) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = n_head * seq_len;
    if (row_idx >= total_rows) return;

    int head = row_idx / seq_len;
    int q_local = row_idx % seq_len;
    int q_abs = past_len + q_local;

    float *row = scores + head * seq_len * total_len + q_local * total_len;

    float max_val = -FLT_MAX;
    for (int k = 0; k < total_len; k++) {
        if (k <= q_abs) {
            row[k] *= scale;
            if (row[k] > max_val) max_val = row[k];
        } else {
            row[k] = -FLT_MAX;
        }
    }

    float sum = 0.0f;
    for (int k = 0; k < total_len; k++) {
        if (k <= q_abs) {
            row[k] = expf(row[k] - max_val);
            sum += row[k];
        } else {
            row[k] = 0.0f;
        }
    }

    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int k = 0; k < total_len; k++) {
        row[k] *= inv_sum;
    }
}

void llama_causal_softmax(float *scores, int n_head, int seq_len,
                          int total_len, int past_len, float scale) {
    int total_rows = n_head * seq_len;
    int block = 256;
    int grid = (total_rows + block - 1) / block;
    llama_causal_softmax_kernel<<<grid, block>>>(scores, n_head, seq_len,
                                                  total_len, past_len, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Attention reshape: [n_head, seq_len, head_dim] -> [seq_len, n_embd]
// ---------------------------------------------------------------------------
__global__ void llama_attention_reshape_kernel(float *out, const float *in,
                                                int seq_len, int n_head,
                                                int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_embd = n_head * head_dim;
    int total = seq_len * n_embd;
    if (idx >= total) return;

    int col = idx % n_embd;
    int row = idx / n_embd;
    int head = col / head_dim;
    int d = col % head_dim;

    out[idx] = in[head * seq_len * head_dim + row * head_dim + d];
}

void llama_attention_reshape(float *out, const float *in, int seq_len,
                             int n_head, int head_dim) {
    int n = seq_len * n_head * head_dim;
    int block = 256;
    int grid = (n + block - 1) / block;
    llama_attention_reshape_kernel<<<grid, block>>>(out, in, seq_len, n_head,
                                                     head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// SiLU(gate) * up — fused SwiGLU activation
// out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// ---------------------------------------------------------------------------
__global__ void llama_silu_multiply_kernel(float *out, const float *gate,
                                            const float *up, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate[idx];
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = silu_g * up[idx];
}

void llama_silu_multiply(float *out, const float *gate, const float *up, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    llama_silu_multiply_kernel<<<grid, block>>>(out, gate, up, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Residual add
// ---------------------------------------------------------------------------
__global__ void llama_residual_add_kernel(float *out, const float *a,
                                           const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

void llama_residual_add(float *out, const float *a, const float *b, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    llama_residual_add_kernel<<<grid, block>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Scramble weights (kill switch placeholder)
// ---------------------------------------------------------------------------
__global__ void llama_scramble_kernel(float *weights, const float *mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    weights[idx] *= mask[idx];
}

void llama_scramble_weights(float *weights, const float *mask, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    llama_scramble_kernel<<<grid, block>>>(weights, mask, n);
    CUDA_CHECK(cudaGetLastError());
}
