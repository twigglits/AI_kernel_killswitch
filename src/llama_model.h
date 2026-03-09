#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// Llama configuration (supports TinyLlama 1.1B, Llama 2/3, etc.)
// ---------------------------------------------------------------------------
typedef struct {
    int n_vocab;           // vocabulary size
    int n_ctx;             // max sequence length
    int n_embd;            // hidden dimension (dim)
    int n_head;            // number of query attention heads
    int n_kv_head;         // number of key/value heads (GQA)
    int n_layer;           // number of transformer layers
    int head_dim;          // n_embd / n_head
    int n_group;           // n_head / n_kv_head (GQA group size)
    int intermediate_size; // MLP intermediate dimension
    float rms_norm_eps;    // RMSNorm epsilon
    float rope_theta;      // RoPE base frequency
} LlamaConfig;

// TinyLlama 1.1B
static inline LlamaConfig llama_config_tinyllama(void) {
    LlamaConfig c;
    c.n_vocab           = 32003;  // 32000 base + 3 special tokens
    c.n_ctx             = 2048;
    c.n_embd            = 2048;
    c.n_head            = 32;
    c.n_kv_head         = 4;
    c.n_layer           = 22;
    c.head_dim          = 64;     // 2048 / 32
    c.n_group           = 8;      // 32 / 4
    c.intermediate_size = 5632;
    c.rms_norm_eps      = 1e-5f;
    c.rope_theta        = 10000.0f;
    return c;
}

// ---------------------------------------------------------------------------
// Weight pointers (all GPU memory, bare float*)
// HuggingFace nn.Linear stores [out_features, in_features]
// We use cublas_matmul with transpose_B=true: C = X * W^T
// ---------------------------------------------------------------------------
typedef struct {
    float *tok_embeddings;       // [n_vocab, n_embd]

    // Per-layer weights (arrays of n_layer pointers)
    float **attn_norm;           // [n_embd]  (RMSNorm weight)
    float **wq;                  // [n_head * head_dim, n_embd]
    float **wk;                  // [n_kv_head * head_dim, n_embd]
    float **wv;                  // [n_kv_head * head_dim, n_embd]
    float **wo;                  // [n_embd, n_head * head_dim]
    float **ffn_norm;            // [n_embd]  (RMSNorm weight)
    float **w_gate;              // [intermediate_size, n_embd]
    float **w_up;                // [intermediate_size, n_embd]
    float **w_down;              // [n_embd, intermediate_size]

    // Final
    float *final_norm;           // [n_embd]
    float *lm_head;              // [n_vocab, n_embd]
} LlamaWeights;

// ---------------------------------------------------------------------------
// Activation buffers (reused each forward pass)
// ---------------------------------------------------------------------------
typedef struct {
    float *residual;      // [n_ctx, n_embd]
    float *norm_out;      // [n_ctx, n_embd]
    float *q;             // [n_ctx, n_head * head_dim]
    float *k;             // [n_ctx, n_kv_head * head_dim]
    float *v;             // [n_ctx, n_kv_head * head_dim]
    float *q_rope;        // [n_head, n_ctx, head_dim] (after RoPE + reshape)
    float *attn_out;      // [n_ctx, n_embd]
    float *attn_proj;     // [n_ctx, n_embd]
    float *gate;          // [n_ctx, intermediate_size]
    float *up;            // [n_ctx, intermediate_size]
    float *mlp_out;       // [n_ctx, n_embd]
    float *logits;        // [n_vocab]

    // Attention intermediates
    float *attn_scores;   // [n_head, n_ctx, n_ctx]
    float *attn_v_out;    // [n_head, n_ctx, head_dim]
} LlamaActivations;

// ---------------------------------------------------------------------------
// KV cache (pre-allocated for max context)
// ---------------------------------------------------------------------------
typedef struct {
    float **k_cache;   // per-layer: [n_kv_head, n_ctx, head_dim]
    float **v_cache;   // per-layer: [n_kv_head, n_ctx, head_dim]
} LlamaKVCache;

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------
typedef struct {
    LlamaConfig config;
    LlamaWeights weights;
    LlamaActivations acts;
    LlamaKVCache kv_cache;
    cublasHandle_t cublas_handle;

    int seq_len;

    // --- Kill switch hook points (future) ---
    int kill_switch_armed;
    float *scramble_mask;
} LlamaModel;

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

void llama_build(LlamaModel *model, const char *weight_path);
void llama_forward(LlamaModel *model, const int *tokens, int n_tokens, int past_len);
void llama_free(LlamaModel *model);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_MODEL_H
