#include "model.h"
#include "utils.h"
#include "kernels.h"
#include "cublas_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Weight file format:
//   Header: 8 ints (magic, version, n_vocab, n_ctx, n_embd, n_head, n_layer, 0)
//   Then all weights in order:
//     wte, wpe,
//     for each layer:
//       ln1_weight, ln1_bias,
//       attn_qkv_weight, attn_qkv_bias,
//       attn_proj_weight, attn_proj_bias,
//       ln2_weight, ln2_bias,
//       mlp_fc_weight, mlp_fc_bias,
//       mlp_proj_weight, mlp_proj_bias,
//     ln_f_weight, ln_f_bias
// ---------------------------------------------------------------------------

#define WEIGHT_MAGIC 0x47505432  // "GPT2"
#define WEIGHT_VERSION 1

static void load_gpu_buffer(float **d_ptr, FILE *f, size_t n_floats) {
    float *host = (float *)malloc(n_floats * sizeof(float));
    size_t read = fread(host, sizeof(float), n_floats, f);
    if (read != n_floats) {
        fprintf(stderr, "Weight read error: expected %zu floats, got %zu\n",
                n_floats, read);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMalloc(d_ptr, n_floats * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, host, n_floats * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(host);
}

void gpt2_build(GPT2Model *model, const char *weight_path) {
    memset(model, 0, sizeof(GPT2Model));
    GPT2Config cfg = gpt2_config_124M();
    model->config = cfg;
    model->seq_len = 0;
    model->kill_switch_armed = 0;
    model->scramble_mask = NULL;

    // cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&model->cublas_handle));
    // Use TF32 for better performance on Ampere+
    CUBLAS_CHECK(cublasSetMathMode(model->cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    fprintf(stderr, "Loading weights from %s...\n", weight_path);
    FILE *f = fopen(weight_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open weight file: %s\n", weight_path);
        exit(EXIT_FAILURE);
    }

    // Read header
    int header[8];
    if (fread(header, sizeof(int), 8, f) != 8) {
        fprintf(stderr, "Failed to read weight file header\n");
        exit(EXIT_FAILURE);
    }
    if (header[0] != WEIGHT_MAGIC || header[1] != WEIGHT_VERSION) {
        fprintf(stderr, "Invalid weight file magic/version\n");
        exit(EXIT_FAILURE);
    }
    // Verify config matches
    if (header[2] != cfg.n_vocab || header[3] != cfg.n_ctx ||
        header[4] != cfg.n_embd || header[5] != cfg.n_head ||
        header[6] != cfg.n_layer) {
        fprintf(stderr, "Weight file config mismatch\n");
        exit(EXIT_FAILURE);
    }

    // Allocate per-layer pointer arrays on host
    GPT2Weights *w = &model->weights;
    int L = cfg.n_layer;
    w->ln1_weight       = (float **)malloc(L * sizeof(float *));
    w->ln1_bias         = (float **)malloc(L * sizeof(float *));
    w->attn_qkv_weight  = (float **)malloc(L * sizeof(float *));
    w->attn_qkv_bias    = (float **)malloc(L * sizeof(float *));
    w->attn_proj_weight = (float **)malloc(L * sizeof(float *));
    w->attn_proj_bias   = (float **)malloc(L * sizeof(float *));
    w->ln2_weight       = (float **)malloc(L * sizeof(float *));
    w->ln2_bias         = (float **)malloc(L * sizeof(float *));
    w->mlp_fc_weight    = (float **)malloc(L * sizeof(float *));
    w->mlp_fc_bias      = (float **)malloc(L * sizeof(float *));
    w->mlp_proj_weight  = (float **)malloc(L * sizeof(float *));
    w->mlp_proj_bias    = (float **)malloc(L * sizeof(float *));

    // Load embeddings
    int E = cfg.n_embd;
    int V = cfg.n_vocab;
    int C = cfg.n_ctx;

    load_gpu_buffer(&w->wte, f, (size_t)V * E);
    load_gpu_buffer(&w->wpe, f, (size_t)C * E);

    // Load per-layer weights
    for (int l = 0; l < L; l++) {
        load_gpu_buffer(&w->ln1_weight[l],       f, E);
        load_gpu_buffer(&w->ln1_bias[l],         f, E);
        load_gpu_buffer(&w->attn_qkv_weight[l],  f, (size_t)E * 3 * E);
        load_gpu_buffer(&w->attn_qkv_bias[l],    f, 3 * E);
        load_gpu_buffer(&w->attn_proj_weight[l],  f, (size_t)E * E);
        load_gpu_buffer(&w->attn_proj_bias[l],    f, E);
        load_gpu_buffer(&w->ln2_weight[l],       f, E);
        load_gpu_buffer(&w->ln2_bias[l],         f, E);
        load_gpu_buffer(&w->mlp_fc_weight[l],    f, (size_t)E * 4 * E);
        load_gpu_buffer(&w->mlp_fc_bias[l],      f, 4 * E);
        load_gpu_buffer(&w->mlp_proj_weight[l],  f, (size_t)4 * E * E);
        load_gpu_buffer(&w->mlp_proj_bias[l],    f, E);
    }

    // Final layer norm
    load_gpu_buffer(&w->ln_f_weight, f, E);
    load_gpu_buffer(&w->ln_f_bias,   f, E);

    fclose(f);

    // ---------------------------------------------------------------------------
    // Allocate activation buffers (max sequence length)
    // ---------------------------------------------------------------------------
    GPT2Activations *a = &model->acts;
    CUDA_CHECK(cudaMalloc(&a->residual,    (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->ln_out,      (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->qkv,         (size_t)C * 3 * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->attn_out,    (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->attn_proj,   (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->mlp_fc,      (size_t)C * 4 * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->mlp_gelu,    (size_t)C * 4 * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->mlp_proj,    (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->logits,      (size_t)V * sizeof(float)));

    // Attention intermediates
    int H = cfg.n_head;
    CUDA_CHECK(cudaMalloc(&a->attn_scores, (size_t)H * C * C * sizeof(float)));
    // attn_weights reuses attn_scores (softmax is in-place)
    a->attn_weights = a->attn_scores;
    CUDA_CHECK(cudaMalloc(&a->attn_v_out,  (size_t)H * C * cfg.head_dim * sizeof(float)));

    // ---------------------------------------------------------------------------
    // Allocate KV cache
    // ---------------------------------------------------------------------------
    GPT2KVCache *kv = &model->kv_cache;
    kv->k_cache = (float **)malloc(L * sizeof(float *));
    kv->v_cache = (float **)malloc(L * sizeof(float *));
    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMalloc(&kv->k_cache[l],
                              (size_t)H * C * cfg.head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&kv->v_cache[l],
                              (size_t)H * C * cfg.head_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(kv->k_cache[l], 0,
                              (size_t)H * C * cfg.head_dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(kv->v_cache[l], 0,
                              (size_t)H * C * cfg.head_dim * sizeof(float)));
    }

    // Print VRAM usage
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    fprintf(stderr, "Model loaded. VRAM: %.0f MB used, %.0f MB free\n",
            (total_mem - free_mem) / 1e6, free_mem / 1e6);
}

// ---------------------------------------------------------------------------
// Forward pass: processes tokens[0..n_tokens-1] at position past_len
// After return, model->acts.logits contains logits for the last token.
// ---------------------------------------------------------------------------
void gpt2_forward(GPT2Model *model, const int *h_tokens, int n_tokens,
                  int past_len) {
    GPT2Config *cfg = &model->config;
    GPT2Weights *w = &model->weights;
    GPT2Activations *a = &model->acts;
    GPT2KVCache *kv = &model->kv_cache;
    cublasHandle_t cublas = model->cublas_handle;

    int E = cfg->n_embd;
    int H = cfg->n_head;
    int D = cfg->head_dim;
    int C = cfg->n_ctx;
    int S = n_tokens;   // current sequence length being processed
    int T = past_len + S; // total length including past

    // Copy token IDs to GPU
    int *d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, S * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens, S * sizeof(int),
                          cudaMemcpyHostToDevice));

    // 1. Token + position embedding
    embedding_lookup(a->residual, w->wte, w->wpe, d_tokens, S, E, past_len);

    CUDA_CHECK(cudaFree(d_tokens));

    // 2. Transformer layers
    for (int l = 0; l < cfg->n_layer; l++) {
        // --- Kill switch pre-layer hook (future) ---
        // if (model->kill_switch_armed) { scramble_weights(...); }

        // LayerNorm 1
        layer_norm(a->ln_out, a->residual, w->ln1_weight[l], w->ln1_bias[l],
                   S, E, 1e-5f);

        // QKV projection: [S, E] x [E, 3E] = [S, 3E]
        cublas_matmul(cublas, a->qkv, a->ln_out, w->attn_qkv_weight[l],
                      S, 3 * E, E, 1.0f, 0.0f, false);
        add_bias(a->qkv, w->attn_qkv_bias[l], S, 3 * E);

        // Split QKV and update KV cache
        // q: [H, S, D], k_cache/v_cache: [H, C, D]
        split_qkv_and_cache(a->attn_out, kv->k_cache[l], kv->v_cache[l],
                            a->qkv, S, past_len, H, D, C);
        // Now a->attn_out holds Q in [H, S, D] layout

        // Attention scores: Q * K^T = [H, S, T]
        // Q: [H, S, D], K_cache: [H, C, D] (we use first T columns)
        // We need Q * K^T for the first T key positions
        cublas_batched_matmul(cublas,
                              a->attn_scores,  // [H, S, T]
                              a->attn_out,     // Q: [H, S, D]
                              kv->k_cache[l],  // K: [H, C, D] (stride over full C)
                              H,               // batch
                              S, T, D,         // M, N, K
                              S * D,           // strideA = S * D
                              C * D,           // strideB = C * D (full cache stride)
                              S * T,           // strideC = S * T
                              1.0f, 0.0f,
                              true);           // transpose B -> K^T

        // Causal softmax with scaling
        float scale = 1.0f / sqrtf((float)D);
        causal_softmax(a->attn_scores, H, S, T, past_len, scale);

        // Attention output: softmax_weights * V = [H, S, D]
        // weights: [H, S, T], V_cache: [H, C, D] (first T rows)
        cublas_batched_matmul(cublas,
                              a->attn_v_out,   // [H, S, D]
                              a->attn_scores,  // [H, S, T]
                              kv->v_cache[l],  // V: [H, C, D]
                              H,               // batch
                              S, D, T,         // M, N, K
                              S * T,           // strideA
                              C * D,           // strideB (full cache stride)
                              S * D,           // strideC
                              1.0f, 0.0f,
                              false);          // no transpose

        // Reshape multi-head: [H, S, D] -> [S, E]
        attention_reshape(a->attn_out, a->attn_v_out, S, H, D);

        // Attention projection: [S, E] x [E, E] = [S, E]
        cublas_matmul(cublas, a->attn_proj, a->attn_out, w->attn_proj_weight[l],
                      S, E, E, 1.0f, 0.0f, false);
        add_bias(a->attn_proj, w->attn_proj_bias[l], S, E);

        // Residual connection
        residual_add(a->residual, a->residual, a->attn_proj, S * E);

        // LayerNorm 2
        layer_norm(a->ln_out, a->residual, w->ln2_weight[l], w->ln2_bias[l],
                   S, E, 1e-5f);

        // MLP: FC -> GELU -> Proj
        cublas_matmul(cublas, a->mlp_fc, a->ln_out, w->mlp_fc_weight[l],
                      S, 4 * E, E, 1.0f, 0.0f, false);
        add_bias(a->mlp_fc, w->mlp_fc_bias[l], S, 4 * E);

        gelu(a->mlp_gelu, a->mlp_fc, S * 4 * E);

        cublas_matmul(cublas, a->mlp_proj, a->mlp_gelu, w->mlp_proj_weight[l],
                      S, E, 4 * E, 1.0f, 0.0f, false);
        add_bias(a->mlp_proj, w->mlp_proj_bias[l], S, E);

        // Residual connection
        residual_add(a->residual, a->residual, a->mlp_proj, S * E);

        // --- Kill switch post-layer hook (future) ---
    }

    // 3. Final layer norm (only for the last token position for generation)
    // For efficiency during generation (S=1), we only process the last token
    float *last_residual = a->residual + (S - 1) * E;
    layer_norm(a->ln_out, last_residual, w->ln_f_weight, w->ln_f_bias,
               1, E, 1e-5f);

    // 4. LM head: logits = ln_out * wte^T  (weight tying)
    // ln_out: [1, E], wte: [V, E] -> logits: [1, V]
    cublas_matmul(cublas, a->logits, a->ln_out, w->wte,
                  1, cfg->n_vocab, E, 1.0f, 0.0f, true);
}

// ---------------------------------------------------------------------------
// Free all GPU memory
// ---------------------------------------------------------------------------
void gpt2_free(GPT2Model *model) {
    GPT2Config *cfg = &model->config;
    GPT2Weights *w = &model->weights;
    GPT2Activations *a = &model->acts;
    GPT2KVCache *kv = &model->kv_cache;

    // Weights
    cudaFree(w->wte);
    cudaFree(w->wpe);
    for (int l = 0; l < cfg->n_layer; l++) {
        cudaFree(w->ln1_weight[l]);
        cudaFree(w->ln1_bias[l]);
        cudaFree(w->attn_qkv_weight[l]);
        cudaFree(w->attn_qkv_bias[l]);
        cudaFree(w->attn_proj_weight[l]);
        cudaFree(w->attn_proj_bias[l]);
        cudaFree(w->ln2_weight[l]);
        cudaFree(w->ln2_bias[l]);
        cudaFree(w->mlp_fc_weight[l]);
        cudaFree(w->mlp_fc_bias[l]);
        cudaFree(w->mlp_proj_weight[l]);
        cudaFree(w->mlp_proj_bias[l]);
    }
    cudaFree(w->ln_f_weight);
    cudaFree(w->ln_f_bias);
    free(w->ln1_weight);
    free(w->ln1_bias);
    free(w->attn_qkv_weight);
    free(w->attn_qkv_bias);
    free(w->attn_proj_weight);
    free(w->attn_proj_bias);
    free(w->ln2_weight);
    free(w->ln2_bias);
    free(w->mlp_fc_weight);
    free(w->mlp_fc_bias);
    free(w->mlp_proj_weight);
    free(w->mlp_proj_bias);

    // Activations
    cudaFree(a->residual);
    cudaFree(a->ln_out);
    cudaFree(a->qkv);
    cudaFree(a->attn_out);
    cudaFree(a->attn_proj);
    cudaFree(a->mlp_fc);
    cudaFree(a->mlp_gelu);
    cudaFree(a->mlp_proj);
    cudaFree(a->logits);
    cudaFree(a->attn_scores);
    // attn_weights is aliased to attn_scores, don't double-free
    cudaFree(a->attn_v_out);

    // KV cache
    for (int l = 0; l < cfg->n_layer; l++) {
        cudaFree(kv->k_cache[l]);
        cudaFree(kv->v_cache[l]);
    }
    free(kv->k_cache);
    free(kv->v_cache);

    // Scramble mask
    if (model->scramble_mask) cudaFree(model->scramble_mask);

    // cuBLAS
    cublasDestroy(model->cublas_handle);
}
