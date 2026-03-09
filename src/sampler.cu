#include "sampler.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>

// ---------------------------------------------------------------------------
// Simple xorshift64 RNG
// ---------------------------------------------------------------------------
static unsigned long long xorshift64(unsigned long long *state) {
    unsigned long long x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float random_f32(unsigned long long *state) {
    return (float)(xorshift64(state) >> 11) / (float)(1ULL << 53);
}

#define HISTORY_CAP 2048

void sampler_init(Sampler *s, float temperature, int top_k, float top_p,
                  float repetition_penalty, unsigned long long seed) {
    s->temperature = temperature;
    s->top_k = top_k;
    s->top_p = top_p;
    s->repetition_penalty = repetition_penalty;
    s->rng_state = seed;
    s->history_cap = HISTORY_CAP;
    s->history_len = 0;
    s->history = (int *)calloc(HISTORY_CAP, sizeof(int));
}

void sampler_free(Sampler *s) {
    free(s->history);
    s->history = NULL;
}

// Comparison function for sorting logits by value (descending)
typedef struct { float val; int idx; } LogitIdx;

static int logit_cmp_desc(const void *a, const void *b) {
    float va = ((const LogitIdx *)a)->val;
    float vb = ((const LogitIdx *)b)->val;
    if (vb > va) return 1;
    if (vb < va) return -1;
    return 0;
}

int sampler_sample(Sampler *s, float *d_logits, int n_vocab) {
    // Copy logits from GPU to CPU
    float *logits = (float *)malloc(n_vocab * sizeof(float));
    CUDA_CHECK(cudaMemcpy(logits, d_logits, n_vocab * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Apply repetition penalty (CTRL paper method):
    // For tokens that appeared in history, divide positive logits by penalty
    // and multiply negative logits by penalty. This pushes them down either way.
    if (s->repetition_penalty != 1.0f && s->history_len > 0) {
        for (int i = 0; i < s->history_len; i++) {
            int tid = s->history[i];
            if (tid >= 0 && tid < n_vocab) {
                if (logits[tid] > 0.0f) {
                    logits[tid] /= s->repetition_penalty;
                } else {
                    logits[tid] *= s->repetition_penalty;
                }
            }
        }
    }

    // Apply temperature
    if (s->temperature != 1.0f && s->temperature > 0.0f) {
        for (int i = 0; i < n_vocab; i++) {
            logits[i] /= s->temperature;
        }
    }

    // If temperature is 0, return argmax (greedy)
    if (s->temperature == 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        free(logits);
        // Record in history
        if (s->history_len < s->history_cap) {
            s->history[s->history_len++] = max_idx;
        }
        return max_idx;
    }

    // Build (value, index) pairs and sort descending
    LogitIdx *pairs = (LogitIdx *)malloc(n_vocab * sizeof(LogitIdx));
    for (int i = 0; i < n_vocab; i++) {
        pairs[i].val = logits[i];
        pairs[i].idx = i;
    }
    qsort(pairs, n_vocab, sizeof(LogitIdx), logit_cmp_desc);

    // Top-k truncation
    int n_candidates = n_vocab;
    if (s->top_k > 0 && s->top_k < n_candidates) {
        n_candidates = s->top_k;
    }

    // Softmax over candidates
    float max_logit = pairs[0].val;
    float sum = 0.0f;
    for (int i = 0; i < n_candidates; i++) {
        pairs[i].val = expf(pairs[i].val - max_logit);
        sum += pairs[i].val;
    }
    for (int i = 0; i < n_candidates; i++) {
        pairs[i].val /= sum;
    }

    // Top-p (nucleus) truncation
    if (s->top_p < 1.0f && s->top_p > 0.0f) {
        float cum = 0.0f;
        int cutoff = n_candidates;
        for (int i = 0; i < n_candidates; i++) {
            cum += pairs[i].val;
            if (cum >= s->top_p) {
                cutoff = i + 1;
                break;
            }
        }
        n_candidates = cutoff;

        // Renormalize
        sum = 0.0f;
        for (int i = 0; i < n_candidates; i++) sum += pairs[i].val;
        for (int i = 0; i < n_candidates; i++) pairs[i].val /= sum;
    }

    // Sample from the distribution
    float r = random_f32(&s->rng_state);
    float cum = 0.0f;
    int sampled = pairs[0].idx;
    for (int i = 0; i < n_candidates; i++) {
        cum += pairs[i].val;
        if (r <= cum) {
            sampled = pairs[i].idx;
            break;
        }
    }

    free(logits);
    free(pairs);

    // Record in history
    if (s->history_len < s->history_cap) {
        s->history[s->history_len++] = sampled;
    }

    return sampled;
}
