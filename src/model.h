#ifndef MODEL_H
#define MODEL_H

#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// GPT-2 124M configuration
// ---------------------------------------------------------------------------
typedef struct {
    int n_vocab;      // 50257
    int n_ctx;        // 1024  (max sequence length)
    int n_embd;       // 768
    int n_head;       // 12
    int n_layer;      // 12
    int head_dim;     // 64  (n_embd / n_head)
} GPT2Config;

static inline GPT2Config gpt2_config_124M(void) {
    GPT2Config c;
    c.n_vocab  = 50257;
    c.n_ctx    = 1024;
    c.n_embd   = 768;
    c.n_head   = 12;
    c.n_layer  = 12;
    c.head_dim = 64;
    return c;
}

// ---------------------------------------------------------------------------
// Weight pointers (all GPU memory, bare float*)
// Per-layer weights stored in arrays indexed by layer.
// ---------------------------------------------------------------------------
typedef struct {
    // Token + position embeddings
    float *wte;           // [n_vocab, n_embd]
    float *wpe;           // [n_ctx,   n_embd]

    // Per-layer weights (arrays of n_layer pointers)
    float **ln1_weight;   // [n_embd]
    float **ln1_bias;     // [n_embd]
    float **attn_qkv_weight; // [n_embd, 3*n_embd]  (fused Q,K,V)
    float **attn_qkv_bias;   // [3*n_embd]
    float **attn_proj_weight; // [n_embd, n_embd]
    float **attn_proj_bias;   // [n_embd]
    float **ln2_weight;   // [n_embd]
    float **ln2_bias;     // [n_embd]
    float **mlp_fc_weight;   // [n_embd, 4*n_embd]
    float **mlp_fc_bias;     // [4*n_embd]
    float **mlp_proj_weight; // [4*n_embd, n_embd]
    float **mlp_proj_bias;   // [n_embd]

    // Final layer norm
    float *ln_f_weight;   // [n_embd]
    float *ln_f_bias;     // [n_embd]

    // Note: lm_head shares weights with wte (weight tying)
} GPT2Weights;

// ---------------------------------------------------------------------------
// Activation buffers (all GPU memory, reused each forward pass)
// ---------------------------------------------------------------------------
typedef struct {
    float *residual;      // [n_ctx, n_embd]  — current residual stream
    float *ln_out;        // [n_ctx, n_embd]  — layernorm output
    float *qkv;           // [n_ctx, 3*n_embd] — fused QKV projection
    float *attn_out;      // [n_ctx, n_embd]  — attention output
    float *attn_proj;     // [n_ctx, n_embd]  — after attention projection
    float *mlp_fc;        // [n_ctx, 4*n_embd] — MLP first layer
    float *mlp_gelu;      // [n_ctx, 4*n_embd] — after GELU
    float *mlp_proj;      // [n_ctx, n_embd]  — MLP projection
    float *logits;        // [n_vocab]         — final logits (last token only for generation)

    // Attention intermediates
    float *attn_scores;   // [n_head, n_ctx, n_ctx] — QK^T scores
    float *attn_weights;  // [n_head, n_ctx, n_ctx] — softmax output (reuses attn_scores)
    float *attn_v_out;    // [n_head, n_ctx, head_dim] — after V multiply
} GPT2Activations;

// ---------------------------------------------------------------------------
// KV cache (pre-allocated for max 1024 tokens)
// ---------------------------------------------------------------------------
typedef struct {
    float **k_cache;   // per-layer: [n_head, n_ctx, head_dim]
    float **v_cache;   // per-layer: [n_head, n_ctx, head_dim]
} GPT2KVCache;

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------
typedef struct {
    GPT2Config config;
    GPT2Weights weights;
    GPT2Activations acts;
    GPT2KVCache kv_cache;
    cublasHandle_t cublas_handle;

    int seq_len;          // current sequence length (tokens generated so far)

    // --- Kill switch hook points (future) ---
    int kill_switch_armed;
    float *scramble_mask; // GPU buffer, same size as largest weight tensor
} GPT2Model;

// ---------------------------------------------------------------------------
// Functions declared in model.cu
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

void gpt2_build(GPT2Model *model, const char *weight_path);
void gpt2_forward(GPT2Model *model, const int *tokens, int n_tokens, int past_len);
void gpt2_free(GPT2Model *model);

#ifdef __cplusplus
}
#endif

#endif // MODEL_H
