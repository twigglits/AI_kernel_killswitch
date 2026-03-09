#ifndef SAMPLER_H
#define SAMPLER_H

// ---------------------------------------------------------------------------
// Sampling strategies for next-token selection
// ---------------------------------------------------------------------------

typedef struct {
    float temperature;   // 1.0 = no change, <1 = sharper, >1 = flatter
    int top_k;           // 0 = disabled
    float top_p;         // 1.0 = disabled (nucleus sampling threshold)
    float repetition_penalty; // 1.0 = disabled, >1.0 = penalize repeats
    unsigned long long rng_state;  // xorshift RNG state

    // Token history for repetition penalty
    int *history;        // ring buffer of generated token IDs
    int history_len;     // current number of tokens in history
    int history_cap;     // max capacity
} Sampler;

#ifdef __cplusplus
extern "C" {
#endif

void sampler_init(Sampler *s, float temperature, int top_k, float top_p,
                  float repetition_penalty, unsigned long long seed);

// Sample next token from logits (GPU buffer, n_vocab floats).
// Copies logits to CPU, applies repetition penalty/temperature/top-k/top-p, samples.
int sampler_sample(Sampler *s, float *d_logits, int n_vocab);

void sampler_free(Sampler *s);

#ifdef __cplusplus
}
#endif

#endif // SAMPLER_H
