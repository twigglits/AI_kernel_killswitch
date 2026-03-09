#include "llama_model.h"
#include "utils.h"
#include "llama_kernels.h"
#include "cublas_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Weight file format:
//   Header: 12 ints (magic, version, n_vocab, n_ctx, n_embd, n_head,
//                     n_kv_head, n_layer, intermediate_size, 0, 0, 0)
//   Then all weights in order as float32:
//     tok_embeddings [n_vocab, n_embd]
//     for each layer:
//       attn_norm [n_embd]
//       wq [n_head * head_dim, n_embd]
//       wk [n_kv_head * head_dim, n_embd]
//       wv [n_kv_head * head_dim, n_embd]
//       wo [n_embd, n_head * head_dim]
//       ffn_norm [n_embd]
//       w_gate [intermediate_size, n_embd]
//       w_up [intermediate_size, n_embd]
//       w_down [n_embd, intermediate_size]
//     final_norm [n_embd]
//     lm_head [n_vocab, n_embd]
// ---------------------------------------------------------------------------

#define LLAMA_WEIGHT_MAGIC 0x4C4C414D  // "LLAM"
#define LLAMA_WEIGHT_VERSION 1

static void load_gpu_buffer(float **d_ptr, FILE *f, size_t n_floats) {
    float *host = (float *)malloc(n_floats * sizeof(float));
    size_t nread = fread(host, sizeof(float), n_floats, f);
    if (nread != n_floats) {
        fprintf(stderr, "Weight read error: expected %zu floats, got %zu\n",
                n_floats, nread);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMalloc(d_ptr, n_floats * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_ptr, host, n_floats * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(host);
}

void llama_build(LlamaModel *model, const char *weight_path) {
    memset(model, 0, sizeof(LlamaModel));
    model->kill_switch_armed = 0;
    model->scramble_mask = NULL;
    model->seq_len = 0;

    // cuBLAS
    CUBLAS_CHECK(cublasCreate(&model->cublas_handle));
    // Use full FP32 precision (TF32 causes too much error accumulation in deep models)
    CUBLAS_CHECK(cublasSetMathMode(model->cublas_handle, CUBLAS_DEFAULT_MATH));

    fprintf(stderr, "Loading Llama weights from %s...\n", weight_path);
    FILE *f = fopen(weight_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open weight file: %s\n", weight_path);
        exit(EXIT_FAILURE);
    }

    // Read header
    int header[12];
    if (fread(header, sizeof(int), 12, f) != 12) {
        fprintf(stderr, "Failed to read weight file header\n");
        exit(EXIT_FAILURE);
    }
    if (header[0] != LLAMA_WEIGHT_MAGIC || header[1] != LLAMA_WEIGHT_VERSION) {
        fprintf(stderr, "Invalid weight file magic/version: %x/%d\n",
                header[0], header[1]);
        exit(EXIT_FAILURE);
    }

    // Set config from header
    LlamaConfig *cfg = &model->config;
    cfg->n_vocab           = header[2];
    cfg->n_ctx             = header[3];
    cfg->n_embd            = header[4];
    cfg->n_head            = header[5];
    cfg->n_kv_head         = header[6];
    cfg->n_layer           = header[7];
    cfg->intermediate_size = header[8];
    cfg->head_dim          = cfg->n_embd / cfg->n_head;
    cfg->n_group           = cfg->n_head / cfg->n_kv_head;
    cfg->rms_norm_eps      = 1e-5f;
    cfg->rope_theta        = 10000.0f;

    fprintf(stderr, "Config: vocab=%d ctx=%d embd=%d heads=%d kv_heads=%d "
            "layers=%d intermediate=%d\n",
            cfg->n_vocab, cfg->n_ctx, cfg->n_embd, cfg->n_head,
            cfg->n_kv_head, cfg->n_layer, cfg->intermediate_size);

    int E = cfg->n_embd;
    int V = cfg->n_vocab;
    int C = cfg->n_ctx;
    int H = cfg->n_head;
    int KVH = cfg->n_kv_head;
    int D = cfg->head_dim;
    int L = cfg->n_layer;
    int I = cfg->intermediate_size;

    // Allocate per-layer pointer arrays
    LlamaWeights *w = &model->weights;
    w->attn_norm = (float **)malloc(L * sizeof(float *));
    w->wq        = (float **)malloc(L * sizeof(float *));
    w->wk        = (float **)malloc(L * sizeof(float *));
    w->wv        = (float **)malloc(L * sizeof(float *));
    w->wo        = (float **)malloc(L * sizeof(float *));
    w->ffn_norm  = (float **)malloc(L * sizeof(float *));
    w->w_gate    = (float **)malloc(L * sizeof(float *));
    w->w_up      = (float **)malloc(L * sizeof(float *));
    w->w_down    = (float **)malloc(L * sizeof(float *));

    // Load embeddings
    load_gpu_buffer(&w->tok_embeddings, f, (size_t)V * E);

    // Load per-layer weights
    for (int l = 0; l < L; l++) {
        load_gpu_buffer(&w->attn_norm[l], f, E);
        load_gpu_buffer(&w->wq[l],        f, (size_t)(H * D) * E);
        load_gpu_buffer(&w->wk[l],        f, (size_t)(KVH * D) * E);
        load_gpu_buffer(&w->wv[l],        f, (size_t)(KVH * D) * E);
        load_gpu_buffer(&w->wo[l],        f, (size_t)E * (H * D));
        load_gpu_buffer(&w->ffn_norm[l],  f, E);
        load_gpu_buffer(&w->w_gate[l],    f, (size_t)I * E);
        load_gpu_buffer(&w->w_up[l],      f, (size_t)I * E);
        load_gpu_buffer(&w->w_down[l],    f, (size_t)E * I);
    }

    // Final norm + lm_head
    load_gpu_buffer(&w->final_norm, f, E);
    load_gpu_buffer(&w->lm_head,   f, (size_t)V * E);

    fclose(f);

    // ---------------------------------------------------------------------------
    // Allocate activation buffers
    // ---------------------------------------------------------------------------
    LlamaActivations *a = &model->acts;
    CUDA_CHECK(cudaMalloc(&a->residual,    (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->norm_out,    (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->q,           (size_t)C * (H * D) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->k,           (size_t)C * (KVH * D) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->v,           (size_t)C * (KVH * D) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->q_rope,      (size_t)H * C * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->attn_out,    (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->attn_proj,   (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->gate,        (size_t)C * I * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->up,          (size_t)C * I * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->mlp_out,     (size_t)C * E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->logits,      (size_t)V * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&a->attn_scores, (size_t)H * C * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->attn_v_out,  (size_t)H * C * D * sizeof(float)));

    // ---------------------------------------------------------------------------
    // Allocate KV cache
    // ---------------------------------------------------------------------------
    LlamaKVCache *kv = &model->kv_cache;
    kv->k_cache = (float **)malloc(L * sizeof(float *));
    kv->v_cache = (float **)malloc(L * sizeof(float *));
    for (int l = 0; l < L; l++) {
        CUDA_CHECK(cudaMalloc(&kv->k_cache[l],
                              (size_t)KVH * C * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&kv->v_cache[l],
                              (size_t)KVH * C * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(kv->k_cache[l], 0,
                              (size_t)KVH * C * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(kv->v_cache[l], 0,
                              (size_t)KVH * C * D * sizeof(float)));
    }

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    fprintf(stderr, "Model loaded. VRAM: %.0f MB used, %.0f MB free\n",
            (total_mem - free_mem) / 1e6, free_mem / 1e6);
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
void llama_forward(LlamaModel *model, const int *h_tokens, int n_tokens,
                   int past_len) {
    LlamaConfig *cfg = &model->config;
    LlamaWeights *w = &model->weights;
    LlamaActivations *a = &model->acts;
    LlamaKVCache *kv = &model->kv_cache;
    cublasHandle_t cublas = model->cublas_handle;

    int E   = cfg->n_embd;
    int H   = cfg->n_head;
    int KVH = cfg->n_kv_head;
    int D   = cfg->head_dim;
    int C   = cfg->n_ctx;
    int I   = cfg->intermediate_size;
    int G   = cfg->n_group;
    int S   = n_tokens;
    int T   = past_len + S;

    // Copy tokens to GPU
    int *d_tokens;
    CUDA_CHECK(cudaMalloc(&d_tokens, S * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens, S * sizeof(int),
                          cudaMemcpyHostToDevice));

    // 1. Token embedding (no positional — Llama uses RoPE)
    llama_embedding_lookup(a->residual, w->tok_embeddings, d_tokens, S, E);
    CUDA_CHECK(cudaFree(d_tokens));

    // 2. Transformer layers
    for (int l = 0; l < cfg->n_layer; l++) {
        // --- Kill switch pre-layer hook (future) ---

        // Pre-attention RMSNorm
        llama_rmsnorm(a->norm_out, a->residual, w->attn_norm[l],
                      S, E, cfg->rms_norm_eps);

        // Q, K, V projections (separate, not fused)
        // Q: [S, E] x [H*D, E]^T = [S, H*D]
        cublas_matmul(cublas, a->q, a->norm_out, w->wq[l],
                      S, H * D, E, 1.0f, 0.0f, true);
        // K: [S, E] x [KVH*D, E]^T = [S, KVH*D]
        cublas_matmul(cublas, a->k, a->norm_out, w->wk[l],
                      S, KVH * D, E, 1.0f, 0.0f, true);
        // V: [S, E] x [KVH*D, E]^T = [S, KVH*D]
        cublas_matmul(cublas, a->v, a->norm_out, w->wv[l],
                      S, KVH * D, E, 1.0f, 0.0f, true);

        // Apply RoPE to Q/K, reshape Q to multi-head, write K/V to cache
        llama_rope_and_cache(a->q_rope, kv->k_cache[l], kv->v_cache[l],
                             a->q, a->k, a->v,
                             S, past_len, H, KVH, D, C, cfg->rope_theta);
        // q_rope: [H, S, D]
        // k_cache: [KVH, C, D], v_cache: [KVH, C, D]

        // Grouped-query attention
        // For each KV head group: Q_group * K^T, softmax, * V
        float scale = 1.0f / sqrtf((float)D);

        for (int g = 0; g < KVH; g++) {
            // Q for this group: G consecutive heads starting at g*G
            // Q_group: [G, S, D] at offset g*G*S*D in q_rope
            float *q_group = a->q_rope + g * G * S * D;
            // K for this KV head: [1, C, D] at offset g*C*D in k_cache
            float *k_head = kv->k_cache[l] + g * C * D;
            // V for this KV head: [1, C, D]
            float *v_head = kv->v_cache[l] + g * C * D;
            // Scores for this group: [G, S, T]
            float *scores_group = a->attn_scores + g * G * S * T;
            // V output: [G, S, D]
            float *v_out_group = a->attn_v_out + g * G * S * D;

            // Q * K^T: [G, S, D] x [1, T, D]^T = [G, S, T]
            // strideB=0 means all G batches use the same K
            cublas_batched_matmul(cublas,
                                  scores_group, q_group, k_head,
                                  G, S, T, D,
                                  S * D,    // strideA: between Q heads
                                  0,        // strideB: same K for all heads in group
                                  S * T,    // strideC
                                  1.0f, 0.0f, true);

            // Causal softmax
            llama_causal_softmax(scores_group, G, S, T, past_len, scale);

            // Scores * V: [G, S, T] x [1, T, D] = [G, S, D]
            // V is [C, D] but we only read first T rows
            cublas_batched_matmul(cublas,
                                  v_out_group, scores_group, v_head,
                                  G, S, D, T,
                                  S * T,    // strideA
                                  0,        // strideB: same V for all heads in group
                                  S * D,    // strideC
                                  1.0f, 0.0f, false);
        }

        // Reshape: [H, S, D] -> [S, E]
        llama_attention_reshape(a->attn_out, a->attn_v_out, S, H, D);

        // Output projection: [S, E] x [E, E]^T = [S, E]
        cublas_matmul(cublas, a->attn_proj, a->attn_out, w->wo[l],
                      S, E, H * D, 1.0f, 0.0f, true);

        // Residual
        llama_residual_add(a->residual, a->residual, a->attn_proj, S * E);

        // Post-attention RMSNorm
        llama_rmsnorm(a->norm_out, a->residual, w->ffn_norm[l],
                      S, E, cfg->rms_norm_eps);

        // SwiGLU MLP
        // Gate: [S, E] x [I, E]^T = [S, I]
        cublas_matmul(cublas, a->gate, a->norm_out, w->w_gate[l],
                      S, I, E, 1.0f, 0.0f, true);
        // Up: [S, E] x [I, E]^T = [S, I]
        cublas_matmul(cublas, a->up, a->norm_out, w->w_up[l],
                      S, I, E, 1.0f, 0.0f, true);

        // SiLU(gate) * up
        llama_silu_multiply(a->gate, a->gate, a->up, S * I);

        // Down: [S, I] x [E, I]^T = [S, E]
        cublas_matmul(cublas, a->mlp_out, a->gate, w->w_down[l],
                      S, E, I, 1.0f, 0.0f, true);

        // Residual
        llama_residual_add(a->residual, a->residual, a->mlp_out, S * E);

        // --- Kill switch post-layer hook (future) ---
    }

    // 3. Final RMSNorm (last token only for generation)
    float *last_residual = a->residual + (S - 1) * E;
    llama_rmsnorm(a->norm_out, last_residual, w->final_norm,
                  1, E, cfg->rms_norm_eps);

    // 4. LM head: [1, E] x [V, E]^T = [1, V]
    cublas_matmul(cublas, a->logits, a->norm_out, w->lm_head,
                  1, cfg->n_vocab, E, 1.0f, 0.0f, true);
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------
void llama_free(LlamaModel *model) {
    LlamaConfig *cfg = &model->config;
    LlamaWeights *w = &model->weights;
    LlamaActivations *a = &model->acts;
    LlamaKVCache *kv = &model->kv_cache;

    cudaFree(w->tok_embeddings);
    for (int l = 0; l < cfg->n_layer; l++) {
        cudaFree(w->attn_norm[l]);
        cudaFree(w->wq[l]);
        cudaFree(w->wk[l]);
        cudaFree(w->wv[l]);
        cudaFree(w->wo[l]);
        cudaFree(w->ffn_norm[l]);
        cudaFree(w->w_gate[l]);
        cudaFree(w->w_up[l]);
        cudaFree(w->w_down[l]);
    }
    free(w->attn_norm); free(w->wq); free(w->wk); free(w->wv); free(w->wo);
    free(w->ffn_norm); free(w->w_gate); free(w->w_up); free(w->w_down);
    cudaFree(w->final_norm);
    cudaFree(w->lm_head);

    cudaFree(a->residual);  cudaFree(a->norm_out);
    cudaFree(a->q);         cudaFree(a->k);         cudaFree(a->v);
    cudaFree(a->q_rope);    cudaFree(a->attn_out);   cudaFree(a->attn_proj);
    cudaFree(a->gate);      cudaFree(a->up);         cudaFree(a->mlp_out);
    cudaFree(a->logits);    cudaFree(a->attn_scores); cudaFree(a->attn_v_out);

    for (int l = 0; l < cfg->n_layer; l++) {
        cudaFree(kv->k_cache[l]);
        cudaFree(kv->v_cache[l]);
    }
    free(kv->k_cache); free(kv->v_cache);

    if (model->scramble_mask) cudaFree(model->scramble_mask);
    cublasDestroy(model->cublas_handle);
}
