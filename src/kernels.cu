#include "kernels.h"
#include "utils.h"
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Embedding lookup: out[i] = wte[token_ids[i]] + wpe[pos_offset + i]
// ---------------------------------------------------------------------------
__global__ void embedding_lookup_kernel(float *out, const float *wte,
                                        const float *wpe, const int *token_ids,
                                        int seq_len, int n_embd, int pos_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_embd) return;

    int row = idx / n_embd;
    int col = idx % n_embd;
    int token_id = token_ids[row];
    int pos = pos_offset + row;

    out[idx] = wte[token_id * n_embd + col] + wpe[pos * n_embd + col];
}

void embedding_lookup(float *out, const float *wte, const float *wpe,
                      const int *token_ids, int seq_len, int n_embd,
                      int pos_offset) {
    int n = seq_len * n_embd;
    int block = 256;
    int grid = (n + block - 1) / block;
    embedding_lookup_kernel<<<grid, block>>>(out, wte, wpe, token_ids,
                                             seq_len, n_embd, pos_offset);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Layer normalization with shared-memory reduction
// ---------------------------------------------------------------------------
__global__ void layer_norm_kernel(float *out, const float *inp,
                                  const float *gamma, const float *beta,
                                  int rows, int cols, float eps) {
    // One block per row, threads cooperate on reduction
    extern __shared__ float shared[];
    float *s_sum  = shared;            // [blockDim.x]
    float *s_sum2 = shared + blockDim.x; // [blockDim.x]

    int row = blockIdx.x;
    if (row >= rows) return;

    const float *x = inp + row * cols;
    float *y = out + row * cols;

    // Compute partial sums
    float local_sum = 0.0f, local_sum2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        local_sum  += val;
        local_sum2 += val * val;
    }
    s_sum[threadIdx.x]  = local_sum;
    s_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]  += s_sum[threadIdx.x + stride];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / cols;
    float var  = s_sum2[0] / cols - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Normalize + scale + shift
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = gamma[i] * ((x[i] - mean) * inv_std) + beta[i];
    }
}

void layer_norm(float *out, const float *inp, const float *gamma,
                const float *beta, int rows, int cols, float eps) {
    int block = 256;
    if (cols < block) block = cols;
    // Round up to next power of 2 for reduction
    int b = 1;
    while (b < block) b <<= 1;
    block = b;
    if (block > 1024) block = 1024;

    size_t shared_mem = 2 * block * sizeof(float);
    layer_norm_kernel<<<rows, block, shared_mem>>>(out, inp, gamma, beta,
                                                    rows, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Add bias: out[r*cols + c] += bias[c]
// ---------------------------------------------------------------------------
__global__ void add_bias_kernel(float *out, const float *bias, int rows,
                                int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    out[idx] += bias[col];
}

void add_bias(float *out, const float *bias, int rows, int cols) {
    int n = rows * cols;
    int block = 256;
    int grid = (n + block - 1) / block;
    add_bias_kernel<<<grid, block>>>(out, bias, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Split fused QKV and populate KV cache
// qkv layout: [seq_len, 3*n_embd] where n_embd = n_head * head_dim
// Output Q: [n_head, seq_len, head_dim]
// K/V cache: [n_head, n_ctx, head_dim], we write at positions past_len..past_len+seq_len-1
// ---------------------------------------------------------------------------
__global__ void split_qkv_and_cache_kernel(float *q_out, float *k_cache,
                                            float *v_cache, const float *qkv,
                                            int seq_len, int past_len,
                                            int n_head, int head_dim,
                                            int n_ctx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_head * head_dim;
    if (idx >= total) return;

    int n_embd = n_head * head_dim;
    // Decompose idx into (seq_pos, head, d)
    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int head = tmp % n_head;
    int seq_pos = tmp / n_head;

    // Source index in fused QKV: row=seq_pos, col offsets for Q,K,V
    int qkv_row = seq_pos * (3 * n_embd);
    int q_offset = head * head_dim + d;
    int k_offset = n_embd + head * head_dim + d;
    int v_offset = 2 * n_embd + head * head_dim + d;

    // Write Q to [n_head, seq_len, head_dim]
    q_out[head * seq_len * head_dim + seq_pos * head_dim + d] =
        qkv[qkv_row + q_offset];

    // Write K,V to cache at position past_len + seq_pos
    int cache_pos = past_len + seq_pos;
    k_cache[head * n_ctx * head_dim + cache_pos * head_dim + d] =
        qkv[qkv_row + k_offset];
    v_cache[head * n_ctx * head_dim + cache_pos * head_dim + d] =
        qkv[qkv_row + v_offset];
}

void split_qkv_and_cache(float *q_out, float *k_cache, float *v_cache,
                          const float *qkv, int seq_len, int past_len,
                          int n_head, int head_dim, int n_ctx) {
    int total = seq_len * n_head * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    split_qkv_and_cache_kernel<<<grid, block>>>(q_out, k_cache, v_cache, qkv,
                                                 seq_len, past_len, n_head,
                                                 head_dim, n_ctx);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Causal softmax with scaling
// scores: [n_head, seq_len, total_len] where total_len = past_len + seq_len
// Each row is a query position attending to all past + current positions.
// Mask: position q (absolute: past_len + q_local) can attend to k <= q_abs.
// ---------------------------------------------------------------------------
__global__ void causal_softmax_kernel(float *scores, int n_head, int seq_len,
                                       int total_len, int past_len,
                                       float scale) {
    // One thread per (head, query_pos) row
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = n_head * seq_len;
    if (row_idx >= total_rows) return;

    int head = row_idx / seq_len;
    int q_local = row_idx % seq_len;
    int q_abs = past_len + q_local;

    float *row = scores + head * seq_len * total_len + q_local * total_len;

    // Scale + causal mask + find max
    float max_val = -FLT_MAX;
    for (int k = 0; k < total_len; k++) {
        if (k <= q_abs) {
            row[k] *= scale;
            if (row[k] > max_val) max_val = row[k];
        } else {
            row[k] = -FLT_MAX;
        }
    }

    // Softmax: exp and sum
    float sum = 0.0f;
    for (int k = 0; k < total_len; k++) {
        if (k <= q_abs) {
            row[k] = expf(row[k] - max_val);
            sum += row[k];
        } else {
            row[k] = 0.0f;
        }
    }

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int k = 0; k < total_len; k++) {
        row[k] *= inv_sum;
    }
}

void causal_softmax(float *scores, int n_head, int seq_len, int total_len,
                    int past_len, float scale) {
    int total_rows = n_head * seq_len;
    int block = 256;
    int grid = (total_rows + block - 1) / block;
    causal_softmax_kernel<<<grid, block>>>(scores, n_head, seq_len, total_len,
                                            past_len, scale);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Reshape multi-head output: [n_head, seq_len, head_dim] -> [seq_len, n_embd]
// ---------------------------------------------------------------------------
__global__ void attention_reshape_kernel(float *out, const float *in,
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

void attention_reshape(float *out, const float *in, int seq_len,
                       int n_head, int head_dim) {
    int n = seq_len * n_head * head_dim;
    int block = 256;
    int grid = (n + block - 1) / block;
    attention_reshape_kernel<<<grid, block>>>(out, in, seq_len, n_head,
                                              head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Approximate GELU
// ---------------------------------------------------------------------------
__global__ void gelu_kernel_impl(float *out, const float *inp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = inp[idx];
    // GPT-2 uses the approximate tanh version
    float c = 0.7978845608f; // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    out[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

void gelu(float *out, const float *inp, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_kernel_impl<<<grid, block>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Residual add: out = a + b
// ---------------------------------------------------------------------------
__global__ void residual_add_kernel(float *out, const float *a,
                                     const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

void residual_add(float *out, const float *a, const float *b, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    residual_add_kernel<<<grid, block>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Temperature scaling
// ---------------------------------------------------------------------------
__global__ void apply_temperature_kernel(float *logits, int n, float temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    logits[idx] /= temp;
}

void apply_temperature(float *logits, int n, float temperature) {
    if (temperature == 1.0f) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    apply_temperature_kernel<<<grid, block>>>(logits, n, temperature);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Scramble weights (kill switch placeholder)
// For now: element-wise multiply weights by mask
// Future: XOR with random, zero out, add noise, etc.
// ---------------------------------------------------------------------------
__global__ void scramble_weights_kernel(float *weights, const float *mask,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    weights[idx] *= mask[idx];
}

void scramble_weights(float *weights, const float *mask, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    scramble_weights_kernel<<<grid, block>>>(weights, mask, n);
    CUDA_CHECK(cudaGetLastError());
}
