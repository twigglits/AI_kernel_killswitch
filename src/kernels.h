#ifndef KERNELS_H
#define KERNELS_H

// ---------------------------------------------------------------------------
// Custom CUDA kernel declarations
// ---------------------------------------------------------------------------

// Embedding: out[i] = wte[token_ids[i]] + wpe[pos_offset + i]
void embedding_lookup(float *out, const float *wte, const float *wpe,
                      const int *token_ids, int seq_len, int n_embd,
                      int pos_offset);

// Layer normalization: out = gamma * (x - mean) / sqrt(var + eps) + beta
void layer_norm(float *out, const float *inp, const float *gamma,
                const float *beta, int rows, int cols, float eps);

// Add bias to every row: out[r][c] += bias[c]
void add_bias(float *out, const float *bias, int rows, int cols);

// Split fused QKV into per-head Q, and write K/V into cache at position pos
// qkv: [seq_len, 3*n_embd]
// q_out: [n_head, seq_len, head_dim]
// k_cache/v_cache: [n_head, n_ctx, head_dim] — writes at column `past_len`
void split_qkv_and_cache(float *q_out, float *k_cache, float *v_cache,
                          const float *qkv, int seq_len, int past_len,
                          int n_head, int head_dim, int n_ctx);

// Causal softmax of attention scores
// scores: [n_head, seq_len, total_len]  (total_len = past_len + seq_len)
// Applies scale, causal mask, then row-wise softmax in-place
void causal_softmax(float *scores, int n_head, int seq_len, int total_len,
                    int past_len, float scale);

// Reshape multi-head attention output back to [seq_len, n_embd]
// in: [n_head, seq_len, head_dim] -> out: [seq_len, n_embd]
void attention_reshape(float *out, const float *in, int seq_len,
                       int n_head, int head_dim);

// Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
void gelu(float *out, const float *inp, int n);

// Residual add: out[i] = a[i] + b[i]
void residual_add(float *out, const float *a, const float *b, int n);

// Apply temperature: logits[i] /= temperature
void apply_temperature(float *logits, int n, float temperature);

// Placeholder: scramble weights kernel (future kill switch)
void scramble_weights(float *weights, const float *mask, int n);

#endif // KERNELS_H
