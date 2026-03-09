#ifndef TOKENIZER_H
#define TOKENIZER_H

// ---------------------------------------------------------------------------
// BPE tokenizer (CPU-side)
// Loads vocab.json + merges.txt from HuggingFace GPT-2 format
// ---------------------------------------------------------------------------

typedef struct {
    // Vocabulary
    char **token_to_str;     // token_id -> string
    int vocab_size;

    // BPE merge rules
    char **merges;           // array of merge strings "token1 token2"
    int n_merges;

    // Byte-to-unicode mapping (GPT-2 uses a specific byte encoder)
    int byte_to_unicode[256];
    int unicode_to_byte[65536]; // sparse reverse map
} GPT2Tokenizer;

#ifdef __cplusplus
extern "C" {
#endif

// Load tokenizer from vocab.json and merges.txt
int tokenizer_init(GPT2Tokenizer *tok, const char *vocab_path,
                   const char *merges_path);

// Encode text to token IDs. Returns number of tokens written.
// tokens must be pre-allocated (safe upper bound: 4 * strlen(text))
int tokenizer_encode(const GPT2Tokenizer *tok, const char *text,
                     int *tokens, int max_tokens);

// Decode a single token ID to string
const char *tokenizer_decode(const GPT2Tokenizer *tok, int token_id);

// Free tokenizer memory
void tokenizer_free(GPT2Tokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_H
