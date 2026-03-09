#include "llama_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Simple hash table for string -> token_id lookup
// ---------------------------------------------------------------------------
static unsigned int hash_str(const char *s) {
    unsigned int h = 5381;
    while (*s) {
        h = ((h << 5) + h) + (unsigned char)*s;
        s++;
    }
    return h;
}

static void hash_insert(LlamaTokenizer *tok, const char *key, int id) {
    unsigned int h = hash_str(key) % tok->hash_size;
    while (tok->str_to_id[h] != -1) {
        h = (h + 1) % tok->hash_size;
    }
    tok->str_to_id[h] = id;
}

static int hash_lookup(const LlamaTokenizer *tok, const char *key) {
    unsigned int h = hash_str(key) % tok->hash_size;
    while (tok->str_to_id[h] != -1) {
        int id = tok->str_to_id[h];
        if (strcmp(tok->vocab[id], key) == 0) {
            return id;
        }
        h = (h + 1) % tok->hash_size;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Load tokenizer from binary vocab file
// ---------------------------------------------------------------------------
int llama_tokenizer_init(LlamaTokenizer *tok, const char *vocab_path) {
    memset(tok, 0, sizeof(LlamaTokenizer));

    FILE *f = fopen(vocab_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open tokenizer: %s\n", vocab_path);
        return -1;
    }

    // Read header
    int magic;
    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != LLAMA_TOK_MAGIC) {
        fprintf(stderr, "Invalid tokenizer magic: %x (expected %x)\n",
                magic, LLAMA_TOK_MAGIC);
        fclose(f);
        return -1;
    }

    if (fread(&tok->vocab_size, sizeof(int), 1, f) != 1) goto read_err;
    if (fread(&tok->max_token_length, sizeof(int), 1, f) != 1) goto read_err;
    if (fread(&tok->bos_id, sizeof(int), 1, f) != 1) goto read_err;
    if (fread(&tok->eos_id, sizeof(int), 1, f) != 1) goto read_err;

    // Allocate
    tok->vocab = (char **)calloc(tok->vocab_size, sizeof(char *));
    tok->vocab_scores = (float *)calloc(tok->vocab_size, sizeof(float));

    // Hash table: 4x vocab size for low collision rate
    tok->hash_size = tok->vocab_size * 4;
    tok->str_to_id = (int *)malloc(tok->hash_size * sizeof(int));
    memset(tok->str_to_id, -1, tok->hash_size * sizeof(int));

    // Read tokens
    for (int i = 0; i < tok->vocab_size; i++) {
        float score;
        int len;
        if (fread(&score, sizeof(float), 1, f) != 1) goto read_err;
        if (fread(&len, sizeof(int), 1, f) != 1) goto read_err;

        tok->vocab_scores[i] = score;
        tok->vocab[i] = (char *)malloc(len + 1);
        if (len > 0) {
            if ((int)fread(tok->vocab[i], 1, len, f) != len) goto read_err;
        }
        tok->vocab[i][len] = '\0';
        hash_insert(tok, tok->vocab[i], i);
    }

    fclose(f);
    fprintf(stderr, "Tokenizer loaded: %d vocab, bos=%d, eos=%d\n",
            tok->vocab_size, tok->bos_id, tok->eos_id);
    return 0;

read_err:
    fprintf(stderr, "Error reading tokenizer file\n");
    fclose(f);
    return -1;
}

// ---------------------------------------------------------------------------
// BPE encoding (SentencePiece style: score-based greedy merging)
// ---------------------------------------------------------------------------
int llama_tokenizer_encode(const LlamaTokenizer *tok, const char *text,
                           int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;

    // Optionally add BOS
    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = tok->bos_id;
    }

    if (!text || !*text) return n_tokens;

    // SentencePiece convention: replace spaces with ▁ (U+2581 = 0xE2 0x96 0x81)
    // and prepend ▁ to the text
    int text_len = (int)strlen(text);
    // Worst case: every char is a space -> 3 bytes each, plus leading ▁
    int buf_cap = text_len * 3 + 4;
    char *buf = (char *)malloc(buf_cap);
    int w = 0;

    // Prepend ▁
    buf[w++] = (char)0xE2;
    buf[w++] = (char)0x96;
    buf[w++] = (char)0x81;

    // Copy text, replacing spaces with ▁
    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            buf[w++] = (char)0xE2;
            buf[w++] = (char)0x96;
            buf[w++] = (char)0x81;
        } else {
            buf[w++] = text[i];
        }
    }
    buf[w] = '\0';

    // Tokenize each UTF-8 character as a separate token
    int total = w;
    int *work = (int *)malloc((total + 1) * sizeof(int));
    int n_work = 0;

    int pos = 0;
    while (pos < total) {
        // Determine UTF-8 character length
        unsigned char c = (unsigned char)buf[pos];
        int char_len = 1;
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;
        if (pos + char_len > total) char_len = total - pos;

        // Look up this character in vocab
        char tmp[8];
        memcpy(tmp, buf + pos, char_len);
        tmp[char_len] = '\0';

        int id = hash_lookup(tok, tmp);
        if (id >= 0) {
            work[n_work++] = id;
        } else {
            // Byte fallback: encode each byte as <0xNN>
            for (int b = 0; b < char_len; b++) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>",
                         (unsigned char)buf[pos + b]);
                int byte_id = hash_lookup(tok, byte_tok);
                if (byte_id >= 0) {
                    work[n_work++] = byte_id;
                }
                // If even byte fallback fails, skip
            }
        }
        pos += char_len;
    }

    free(buf);

    // BPE merge loop: greedily merge the pair with highest score
    // We need a buffer for merged strings
    char *merge_buf = (char *)malloc(tok->max_token_length * 2 + 2);

    while (n_work >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_id = -1;

        for (int i = 0; i < n_work - 1; i++) {
            // Build merged string
            const char *s1 = tok->vocab[work[i]];
            const char *s2 = tok->vocab[work[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);
            if (l1 + l2 > tok->max_token_length * 2) continue;

            memcpy(merge_buf, s1, l1);
            memcpy(merge_buf + l1, s2, l2);
            merge_buf[l1 + l2] = '\0';

            int id = hash_lookup(tok, merge_buf);
            if (id >= 0 && tok->vocab_scores[id] > best_score) {
                best_score = tok->vocab_scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx == -1) break; // No more merges possible

        // Apply merge: replace pair at best_idx with merged token
        work[best_idx] = best_id;
        // Shift remaining tokens left
        for (int i = best_idx + 1; i < n_work - 1; i++) {
            work[i] = work[i + 1];
        }
        n_work--;
    }

    free(merge_buf);

    // Copy results
    for (int i = 0; i < n_work && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = work[i];
    }

    free(work);
    return n_tokens;
}

// ---------------------------------------------------------------------------
// Decode token to string
// ---------------------------------------------------------------------------
const char *llama_tokenizer_decode(const LlamaTokenizer *tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "";
    if (tok->vocab[token_id] == NULL) return "";
    return tok->vocab[token_id];
}

void llama_tokenizer_free(LlamaTokenizer *tok) {
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->vocab_scores);
    free(tok->str_to_id);
}
