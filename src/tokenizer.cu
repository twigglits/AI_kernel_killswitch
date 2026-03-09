#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// ---------------------------------------------------------------------------
// GPT-2 byte encoder: maps bytes 0-255 to unicode codepoints
// This replicates the bytes_to_unicode() function from HuggingFace GPT-2
// ---------------------------------------------------------------------------
static void init_byte_encoder(GPT2Tokenizer *tok) {
    // The GPT-2 byte encoder maps printable ASCII + some high bytes directly,
    // and remaps the rest to unicode range starting at 256.
    int n = 0;
    // Ranges that map to themselves: 33-126, 161-172, 174-255
    for (int i = 33; i <= 126; i++) {
        tok->byte_to_unicode[i] = i;
        n++;
    }
    for (int i = 161; i <= 172; i++) {
        tok->byte_to_unicode[i] = i;
        n++;
    }
    for (int i = 174; i <= 255; i++) {
        tok->byte_to_unicode[i] = i;
        n++;
    }
    // Remaining bytes get mapped to 256+
    int offset = 0;
    for (int i = 0; i < 256; i++) {
        if (tok->byte_to_unicode[i] == 0 && i != 0) {
            // Not yet mapped (and not null byte mapped to 0)
            tok->byte_to_unicode[i] = 256 + offset;
            offset++;
        } else if (i == 0) {
            tok->byte_to_unicode[0] = 256 + offset;
            offset++;
        }
    }

    // Build reverse map
    memset(tok->unicode_to_byte, -1, sizeof(tok->unicode_to_byte));
    for (int i = 0; i < 256; i++) {
        tok->unicode_to_byte[tok->byte_to_unicode[i]] = i;
    }
}

// ---------------------------------------------------------------------------
// JSON parsing helpers (minimal, just for vocab.json)
// ---------------------------------------------------------------------------

// Read entire file into a string
static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(len + 1);
    size_t nread = fread(buf, 1, len, f);
    buf[nread] = '\0';
    fclose(f);
    return buf;
}

// Decode a JSON-escaped string in place. Handles \uXXXX, \\, \", \n, \t, etc.
// Returns length of decoded string.
static int decode_json_string(char *s, int len) {
    int r = 0, w = 0;
    while (r < len) {
        if (s[r] == '\\' && r + 1 < len) {
            r++;
            switch (s[r]) {
                case '"':  s[w++] = '"';  r++; break;
                case '\\': s[w++] = '\\'; r++; break;
                case '/':  s[w++] = '/';  r++; break;
                case 'n':  s[w++] = '\n'; r++; break;
                case 'r':  s[w++] = '\r'; r++; break;
                case 't':  s[w++] = '\t'; r++; break;
                case 'u': {
                    // Parse 4 hex digits
                    if (r + 4 < len) {
                        char hex[5] = {s[r+1], s[r+2], s[r+3], s[r+4], 0};
                        unsigned int cp = (unsigned int)strtol(hex, NULL, 16);
                        r += 5;
                        // Encode as UTF-8
                        if (cp < 0x80) {
                            s[w++] = (char)cp;
                        } else if (cp < 0x800) {
                            s[w++] = (char)(0xC0 | (cp >> 6));
                            s[w++] = (char)(0x80 | (cp & 0x3F));
                        } else {
                            s[w++] = (char)(0xE0 | (cp >> 12));
                            s[w++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            s[w++] = (char)(0x80 | (cp & 0x3F));
                        }
                    } else {
                        s[w++] = '?'; r++;
                    }
                    break;
                }
                default: s[w++] = s[r]; r++; break;
            }
        } else {
            s[w++] = s[r++];
        }
    }
    s[w] = '\0';
    return w;
}

// Parse vocab.json: {"token_string": id, ...}
// Returns max_id + 1
static int parse_vocab(const char *json, char ***out_tokens, int *out_size) {
    // First pass: count entries and find max ID
    int max_id = -1;
    int count = 0;
    const char *p = json;

    // Skip to first '{'
    while (*p && *p != '{') p++;
    if (*p == '{') p++;

    // Temporary storage: parse all entries
    typedef struct { char *str; int id; } Entry;
    int cap = 60000;
    Entry *entries = (Entry *)malloc(cap * sizeof(Entry));

    while (*p) {
        // Skip whitespace and commas
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) p++;
        if (*p == '}') break;
        if (*p != '"') { p++; continue; }

        // Parse key string
        p++; // skip opening quote
        const char *key_start = p;
        // Find closing quote (handling escapes)
        while (*p && !(*p == '"' && *(p-1) != '\\')) p++;
        int key_len = (int)(p - key_start);
        char *key = (char *)malloc(key_len + 1);
        memcpy(key, key_start, key_len);
        key[key_len] = '\0';
        decode_json_string(key, key_len);
        if (*p == '"') p++;

        // Skip colon
        while (*p && *p != ':') p++;
        if (*p == ':') p++;

        // Parse integer value
        while (*p && (*p == ' ' || *p == '\t')) p++;
        int id = (int)strtol(p, (char **)&p, 10);

        if (count >= cap) {
            cap *= 2;
            entries = (Entry *)realloc(entries, cap * sizeof(Entry));
        }
        entries[count].str = key;
        entries[count].id = id;
        if (id > max_id) max_id = id;
        count++;
    }

    // Build token array indexed by ID
    int vocab_size = max_id + 1;
    char **tokens = (char **)calloc(vocab_size, sizeof(char *));
    for (int i = 0; i < count; i++) {
        if (entries[i].id >= 0 && entries[i].id < vocab_size) {
            tokens[entries[i].id] = entries[i].str;
        } else {
            free(entries[i].str);
        }
    }
    free(entries);

    *out_tokens = tokens;
    *out_size = vocab_size;
    return vocab_size;
}

// Parse merges.txt: first line is header "#version: ...", rest are "tok1 tok2"
static int parse_merges(const char *text, char ***out_merges) {
    // Count lines
    int n_lines = 0;
    const char *p = text;
    while (*p) { if (*p == '\n') n_lines++; p++; }
    n_lines++; // last line may not have newline

    char **merges = (char **)malloc(n_lines * sizeof(char *));
    int count = 0;
    p = text;

    while (*p) {
        // Find end of line
        const char *line_start = p;
        while (*p && *p != '\n') p++;
        int line_len = (int)(p - line_start);
        if (*p == '\n') p++;

        // Skip empty lines and the header line
        if (line_len == 0) continue;
        if (line_start[0] == '#') continue;

        char *line = (char *)malloc(line_len + 1);
        memcpy(line, line_start, line_len);
        line[line_len] = '\0';
        // Trim trailing \r
        if (line_len > 0 && line[line_len - 1] == '\r') {
            line[line_len - 1] = '\0';
        }
        merges[count++] = line;
    }

    *out_merges = merges;
    return count;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int tokenizer_init(GPT2Tokenizer *tok, const char *vocab_path,
                   const char *merges_path) {
    memset(tok, 0, sizeof(GPT2Tokenizer));
    init_byte_encoder(tok);

    // Load vocab
    char *vocab_json = read_file(vocab_path);
    if (!vocab_json) return -1;
    parse_vocab(vocab_json, &tok->token_to_str, &tok->vocab_size);
    free(vocab_json);

    // Load merges
    char *merges_text = read_file(merges_path);
    if (!merges_text) return -1;
    tok->n_merges = parse_merges(merges_text, &tok->merges);
    free(merges_text);

    fprintf(stderr, "Tokenizer loaded: %d vocab, %d merges\n",
            tok->vocab_size, tok->n_merges);
    return 0;
}

// Convert a byte string to GPT-2's unicode representation
static char *bytes_to_gpt2_unicode(const GPT2Tokenizer *tok,
                                    const unsigned char *bytes, int len) {
    // Each byte maps to a unicode codepoint, which we encode as UTF-8
    // Max expansion: 3 bytes per input byte (for 3-byte UTF-8)
    char *out = (char *)malloc(len * 3 + 1);
    int w = 0;
    for (int i = 0; i < len; i++) {
        int cp = tok->byte_to_unicode[bytes[i]];
        if (cp < 0x80) {
            out[w++] = (char)cp;
        } else if (cp < 0x800) {
            out[w++] = (char)(0xC0 | (cp >> 6));
            out[w++] = (char)(0x80 | (cp & 0x3F));
        } else {
            out[w++] = (char)(0xE0 | (cp >> 12));
            out[w++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[w++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    out[w] = '\0';
    return out;
}

// Find a token string in vocab, return its ID or -1
static int find_token(const GPT2Tokenizer *tok, const char *str, int len) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->token_to_str[i] &&
            (int)strlen(tok->token_to_str[i]) == len &&
            memcmp(tok->token_to_str[i], str, len) == 0) {
            return i;
        }
    }
    return -1;
}

// BPE encoding of a single "word" (pre-tokenized chunk in GPT-2 unicode space)
// Returns token IDs for this word.
static int bpe_encode_word(const GPT2Tokenizer *tok, const char *word,
                           int *out_tokens, int max_tokens) {
    // Start: each UTF-8 character is a separate symbol
    // We represent symbols as (start, len) ranges in the word string

    int word_len = (int)strlen(word);
    if (word_len == 0) return 0;

    // Parse into UTF-8 characters
    typedef struct { int start; int len; } Symbol;
    Symbol *symbols = (Symbol *)malloc((word_len + 1) * sizeof(Symbol));
    int n_sym = 0;

    int pos = 0;
    while (pos < word_len) {
        unsigned char c = (unsigned char)word[pos];
        int char_len = 1;
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;
        if (pos + char_len > word_len) char_len = word_len - pos;
        symbols[n_sym].start = pos;
        symbols[n_sym].len = char_len;
        n_sym++;
        pos += char_len;
    }

    // Iteratively apply BPE merges
    for (int m = 0; m < tok->n_merges && n_sym > 1; m++) {
        // Parse merge rule "first second"
        const char *merge = tok->merges[m];
        const char *space = strchr(merge, ' ');
        if (!space) continue;
        int first_len = (int)(space - merge);
        const char *second = space + 1;
        int second_len = (int)strlen(second);

        // Scan for adjacent pair matching this merge
        int merged = 0;
        for (int i = 0; i < n_sym - 1; i++) {
            if (symbols[i].len == first_len &&
                memcmp(word + symbols[i].start, merge, first_len) == 0 &&
                symbols[i + 1].len == second_len &&
                memcmp(word + symbols[i + 1].start, second, second_len) == 0) {
                // Merge: extend symbol i to cover both, remove i+1
                symbols[i].len = (symbols[i + 1].start + symbols[i + 1].len) -
                                  symbols[i].start;
                // Shift remaining symbols
                for (int j = i + 1; j < n_sym - 1; j++) {
                    symbols[j] = symbols[j + 1];
                }
                n_sym--;
                merged = 1;
                i--; // Re-check at this position
            }
        }
        (void)merged; // suppress unused warning
    }

    // Convert symbols to token IDs
    int n_tokens = 0;
    for (int i = 0; i < n_sym && n_tokens < max_tokens; i++) {
        int tid = find_token(tok, word + symbols[i].start, symbols[i].len);
        if (tid >= 0) {
            out_tokens[n_tokens++] = tid;
        } else {
            // Fallback: encode each byte as individual token
            // This handles unknown sub-words
            for (int b = 0; b < symbols[i].len && n_tokens < max_tokens; b++) {
                char single[4];
                single[0] = word[symbols[i].start + b];
                single[1] = '\0';
                tid = find_token(tok, single, 1);
                if (tid >= 0) {
                    out_tokens[n_tokens++] = tid;
                }
            }
        }
    }

    free(symbols);
    return n_tokens;
}

int tokenizer_encode(const GPT2Tokenizer *tok, const char *text,
                     int *tokens, int max_tokens) {
    // GPT-2 pre-tokenization: split on whitespace boundaries
    // Pattern: optional leading space + word characters
    // We do a simplified version: split around spaces, keeping leading space with word

    int n_tokens = 0;
    int text_len = (int)strlen(text);
    int i = 0;

    while (i < text_len && n_tokens < max_tokens) {
        // Collect a "word": optional space + non-space characters
        int word_start = i;

        // Include leading space if present
        if (text[i] == ' ') i++;

        // Collect non-space characters (or at least one character)
        while (i < text_len && text[i] != ' ') i++;

        // If we only got a space with nothing after, include it alone
        if (i == word_start) { i++; continue; }

        int word_byte_len = i - word_start;

        // Convert word bytes to GPT-2 unicode representation
        char *gpt2_word = bytes_to_gpt2_unicode(
            tok, (const unsigned char *)text + word_start, word_byte_len);

        // BPE encode this word
        int word_tokens[256];
        int wt = bpe_encode_word(tok, gpt2_word, word_tokens, 256);
        for (int j = 0; j < wt && n_tokens < max_tokens; j++) {
            tokens[n_tokens++] = word_tokens[j];
        }

        free(gpt2_word);
    }

    return n_tokens;
}

const char *tokenizer_decode(const GPT2Tokenizer *tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "";
    if (tok->token_to_str[token_id] == NULL) return "";
    return tok->token_to_str[token_id];
}

void tokenizer_free(GPT2Tokenizer *tok) {
    if (tok->token_to_str) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->token_to_str[i]);
        }
        free(tok->token_to_str);
    }
    if (tok->merges) {
        for (int i = 0; i < tok->n_merges; i++) {
            free(tok->merges[i]);
        }
        free(tok->merges);
    }
}
