#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// GPT-2 token decoding: convert token string (GPT-2 unicode) back to bytes
// ---------------------------------------------------------------------------
static void decode_and_print(const GPT2Tokenizer *tok, int token_id) {
    const char *s = tokenizer_decode(tok, token_id);
    if (!s || !*s) return;

    // Convert GPT-2 unicode representation back to raw bytes
    int len = (int)strlen(s);
    int i = 0;
    while (i < len) {
        // Decode UTF-8 codepoint
        unsigned int cp = 0;
        unsigned char c = (unsigned char)s[i];
        int char_len = 1;
        if (c >= 0xF0)      { cp = c & 0x07; char_len = 4; }
        else if (c >= 0xE0) { cp = c & 0x0F; char_len = 3; }
        else if (c >= 0xC0) { cp = c & 0x1F; char_len = 2; }
        else                { cp = c; }

        for (int j = 1; j < char_len && (i + j) < len; j++) {
            cp = (cp << 6) | ((unsigned char)s[i + j] & 0x3F);
        }
        i += char_len;

        // Map unicode codepoint back to byte using reverse table
        int byte_val = tok->unicode_to_byte[cp & 0xFFFF];
        if (byte_val >= 0 && byte_val < 256) {
            putchar(byte_val);
        } else {
            // Fallback: just print the codepoint as UTF-8
            if (cp < 0x80) {
                putchar((char)cp);
            } else if (cp < 0x800) {
                putchar((char)(0xC0 | (cp >> 6)));
                putchar((char)(0x80 | (cp & 0x3F)));
            } else {
                putchar((char)(0xE0 | (cp >> 12)));
                putchar((char)(0x80 | ((cp >> 6) & 0x3F)));
                putchar((char)(0x80 | (cp & 0x3F)));
            }
        }
    }
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------
static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --prompt TEXT        Input prompt (default: \"The meaning of life is\")\n");
    fprintf(stderr, "  --max-tokens N       Max tokens to generate (default: 128)\n");
    fprintf(stderr, "  --temperature F      Sampling temperature (default: 0.8)\n");
    fprintf(stderr, "  --top-k N            Top-k sampling (default: 40, 0=disabled)\n");
    fprintf(stderr, "  --top-p F            Top-p nucleus sampling (default: 0.95)\n");
    fprintf(stderr, "  --rep-penalty F      Repetition penalty (default: 1.2, 1.0=disabled)\n");
    fprintf(stderr, "  --seed N             RNG seed (default: 42)\n");
    fprintf(stderr, "  --model-dir PATH     Model directory (default: models/gpt2)\n");
    fprintf(stderr, "  --greedy             Use greedy decoding (temperature=0)\n");
}

int main(int argc, char **argv) {
    // Default parameters
    const char *prompt = "The meaning of life is";
    int max_tokens = 128;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float rep_penalty = 1.2f;
    unsigned long long seed = 42;
    const char *model_dir = "models/gpt2";
    int greedy = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            rep_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--greedy") == 0) {
            greedy = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (greedy) temperature = 0.0f;

    // Construct file paths
    char weight_path[512], vocab_path[512], merges_path[512];
    snprintf(weight_path, sizeof(weight_path), "%s/gpt2_weights.bin", model_dir);
    snprintf(vocab_path,  sizeof(vocab_path),  "%s/vocab.json", model_dir);
    snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", model_dir);

    // Initialize tokenizer
    GPT2Tokenizer tokenizer;
    if (tokenizer_init(&tokenizer, vocab_path, merges_path) != 0) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // Encode prompt
    int tokens[4096];
    int n_tokens = tokenizer_encode(&tokenizer, prompt, tokens, 4096);
    fprintf(stderr, "Prompt: \"%s\" (%d tokens)\n", prompt, n_tokens);

    if (n_tokens == 0) {
        fprintf(stderr, "Empty prompt after tokenization\n");
        return 1;
    }

    // Build model
    GPT2Model model;
    gpt2_build(&model, weight_path);

    // Initialize sampler
    Sampler sampler;
    sampler_init(&sampler, temperature, top_k, top_p, rep_penalty, seed);

    fprintf(stderr, "Generating %d tokens (temp=%.2f, top_k=%d, top_p=%.2f, rep=%.2f)...\n",
            max_tokens, temperature, top_k, top_p, rep_penalty);
    fprintf(stderr, "---\n");

    // Print the prompt
    for (int i = 0; i < n_tokens; i++) {
        decode_and_print(&tokenizer, tokens[i]);
    }

    // Generation loop
    Timer timer;
    timer_start(&timer);

    int total_generated = 0;
    int past_len = 0;

    // First forward pass: process entire prompt (prefill)
    gpt2_forward(&model, tokens, n_tokens, 0);
    past_len = n_tokens;

    // Sample first generated token
    int next_token = sampler_sample(&sampler, model.acts.logits,
                                     model.config.n_vocab);
    decode_and_print(&tokenizer, next_token);
    total_generated++;

    // Autoregressive generation
    for (int step = 1; step < max_tokens; step++) {
        if (past_len >= model.config.n_ctx - 1) {
            fprintf(stderr, "\n[Reached max context length %d]\n",
                    model.config.n_ctx);
            break;
        }

        // GPT-2 EOS token
        if (next_token == 50256) {
            fprintf(stderr, "\n[EOS]\n");
            break;
        }

        // Forward pass for single token
        int single_token[1] = {next_token};
        gpt2_forward(&model, single_token, 1, past_len);
        past_len++;

        // Sample next token
        next_token = sampler_sample(&sampler, model.acts.logits,
                                     model.config.n_vocab);
        decode_and_print(&tokenizer, next_token);
        total_generated++;
    }

    double elapsed = timer_stop(&timer);
    fprintf(stderr, "\n---\n");
    fprintf(stderr, "Generated %d tokens in %.1f ms (%.1f tokens/sec)\n",
            total_generated, elapsed, total_generated / (elapsed / 1000.0));

    // Cleanup
    gpt2_free(&model);
    sampler_free(&sampler);
    tokenizer_free(&tokenizer);

    return 0;
}
