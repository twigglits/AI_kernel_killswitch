#include "llama_model.h"
#include "llama_tokenizer.h"
#include "sampler.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Token decoding: SentencePiece style
// "▁" (U+2581, encoded as 0xE2 0x96 0x81) represents a space
// Byte tokens "<0xNN>" represent raw bytes
// ---------------------------------------------------------------------------
static void decode_and_print(const LlamaTokenizer *tok, int token_id,
                             int is_first) {
    const char *s = llama_tokenizer_decode(tok, token_id);
    if (!s || !*s) return;

    int len = (int)strlen(s);

    // Check for byte token: <0xNN>
    if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
        char hex[3] = {s[3], s[4], 0};
        int byte_val = (int)strtol(hex, NULL, 16);
        putchar(byte_val);
        fflush(stdout);
        return;
    }

    // Print character by character, replacing ▁ with space
    int i = 0;
    while (i < len) {
        // Check for ▁ (0xE2 0x96 0x81)
        if (i + 2 < len &&
            (unsigned char)s[i] == 0xE2 &&
            (unsigned char)s[i+1] == 0x96 &&
            (unsigned char)s[i+2] == 0x81) {
            putchar(' ');
            i += 3;
        } else {
            putchar(s[i]);
            i++;
        }
    }
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --prompt TEXT        Input prompt\n");
    fprintf(stderr, "  --max-tokens N       Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  --temperature F      Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  --top-k N            Top-k sampling (default: 40)\n");
    fprintf(stderr, "  --top-p F            Top-p nucleus (default: 0.9)\n");
    fprintf(stderr, "  --rep-penalty F      Repetition penalty (default: 1.1)\n");
    fprintf(stderr, "  --seed N             RNG seed (default: 42)\n");
    fprintf(stderr, "  --model-dir PATH     Model directory (default: models/tinyllama)\n");
    fprintf(stderr, "  --chat               Wrap prompt in ChatML format\n");
    fprintf(stderr, "  --greedy             Greedy decoding\n");
}

int main(int argc, char **argv) {
    const char *prompt = "Hello, how are you?";
    int max_tokens = 256;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    float rep_penalty = 1.1f;
    unsigned long long seed = 42;
    const char *model_dir = "models/tinyllama";
    int chat_mode = 0;
    int greedy = 0;

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
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat_mode = 1;
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

    // Build the actual prompt (optionally wrap in ChatML)
    char full_prompt[8192];
    if (chat_mode) {
        snprintf(full_prompt, sizeof(full_prompt),
                 "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                 prompt);
    } else {
        snprintf(full_prompt, sizeof(full_prompt), "%s", prompt);
    }

    // Paths
    char weight_path[512], vocab_path[512];
    snprintf(weight_path, sizeof(weight_path), "%s/llama_weights.bin", model_dir);
    snprintf(vocab_path,  sizeof(vocab_path),  "%s/tokenizer.bin", model_dir);

    // Tokenizer
    LlamaTokenizer tokenizer;
    if (llama_tokenizer_init(&tokenizer, vocab_path) != 0) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // Encode prompt
    int tokens[4096];
    int n_tokens = llama_tokenizer_encode(&tokenizer, full_prompt, tokens, 4096, 1);
    fprintf(stderr, "Prompt: \"%s\" (%d tokens)\n",
            chat_mode ? prompt : full_prompt, n_tokens);

    if (n_tokens == 0) {
        fprintf(stderr, "Empty prompt\n");
        return 1;
    }

    // Build model
    LlamaModel model;
    llama_build(&model, weight_path);

    // Sampler
    Sampler sampler;
    sampler_init(&sampler, temperature, top_k, top_p, rep_penalty, seed);

    fprintf(stderr, "Generating %d tokens (temp=%.2f, top_k=%d, top_p=%.2f, rep=%.2f)...\n",
            max_tokens, temperature, top_k, top_p, rep_penalty);
    fprintf(stderr, "---\n");

    // Print prompt tokens (skip BOS and ChatML prefix in chat mode)
    if (!chat_mode) {
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] != tokenizer.bos_id) {
                decode_and_print(&tokenizer, tokens[i], i == 1);
            }
        }
    }

    // Generation
    Timer timer;
    timer_start(&timer);

    int total_generated = 0;
    int past_len = 0;

    // Prefill
    llama_forward(&model, tokens, n_tokens, 0);
    past_len = n_tokens;

    // Sample first token
    int next_token = sampler_sample(&sampler, model.acts.logits,
                                     model.config.n_vocab);
    decode_and_print(&tokenizer, next_token, 1);
    total_generated++;

    // Autoregressive loop
    for (int step = 1; step < max_tokens; step++) {
        if (past_len >= model.config.n_ctx - 1) {
            fprintf(stderr, "\n[Reached max context %d]\n", model.config.n_ctx);
            break;
        }

        // Check for EOS or ChatML end token
        if (next_token == tokenizer.eos_id) {
            break;
        }
        // Check for <|im_end|> token (usually eos_id or a special token)
        const char *tok_str = llama_tokenizer_decode(&tokenizer, next_token);
        if (tok_str && strcmp(tok_str, "<|im_end|>") == 0) {
            break;
        }

        int single[1] = {next_token};
        llama_forward(&model, single, 1, past_len);
        past_len++;

        next_token = sampler_sample(&sampler, model.acts.logits,
                                     model.config.n_vocab);
        decode_and_print(&tokenizer, next_token, 0);
        total_generated++;
    }

    double elapsed = timer_stop(&timer);
    fprintf(stderr, "\n---\n");
    fprintf(stderr, "Generated %d tokens in %.1f ms (%.1f tokens/sec)\n",
            total_generated, elapsed, total_generated / (elapsed / 1000.0));

    llama_free(&model);
    sampler_free(&sampler);
    llama_tokenizer_free(&tokenizer);
    return 0;
}
