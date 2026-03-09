#ifndef LLAMA_KERNELS_H
#define LLAMA_KERNELS_H

// ---------------------------------------------------------------------------
// Llama-specific CUDA kernel declarations
// ---------------------------------------------------------------------------

// Embedding lookup: out[i] = embeddings[token_ids[i]]
void llama_embedding_lookup(float *out, const float *embeddings,
                            const int *token_ids, int seq_len, int n_embd);

// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
void llama_rmsnorm(float *out, const float *inp, const float *weight,
                   int rows, int cols, float eps);

// Apply RoPE to Q and K, then reshape to multi-head layout
// q_in:  [seq_len, n_head * head_dim] -> q_out: [n_head, seq_len, head_dim]
// k_in:  [seq_len, n_kv_head * head_dim] -> writes to k_cache at past_len
// v_in:  [seq_len, n_kv_head * head_dim] -> writes to v_cache at past_len
void llama_rope_and_cache(float *q_out,
                          float *k_cache, float *v_cache,
                          const float *q_in, const float *k_in,
                          const float *v_in,
                          int seq_len, int past_len,
                          int n_head, int n_kv_head, int head_dim,
                          int n_ctx, float rope_theta);

// Causal softmax (same as GPT-2 version but separate for clarity)
void llama_causal_softmax(float *scores, int n_head, int seq_len,
                          int total_len, int past_len, float scale);

// Reshape multi-head: [n_head, seq_len, head_dim] -> [seq_len, n_embd]
void llama_attention_reshape(float *out, const float *in, int seq_len,
                             int n_head, int head_dim);

// SiLU (Swish) activation: out = x * sigmoid(x)
// Applied in-place to gate, then multiplied element-wise with up
// Combined: out = silu(gate) * up
void llama_silu_multiply(float *out, const float *gate, const float *up, int n);

// Residual add: out = a + b
void llama_residual_add(float *out, const float *a, const float *b, int n);

// Placeholder: scramble weights (kill switch)
void llama_scramble_weights(float *weights, const float *mask, int n);

#endif // LLAMA_KERNELS_H
