#ifndef LLAMA_TOKENIZER_H
#define LLAMA_TOKENIZER_H

// ---------------------------------------------------------------------------
// Llama/SentencePiece BPE tokenizer
//
// Binary vocab format:
//   int32: magic (0x4C544F4B = "LTOK")
//   int32: vocab_size
//   int32: max_token_length
//   int32: bos_id
//   int32: eos_id
//   Per token (vocab_size entries):
//     float32: score (BPE merge priority)
//     int32: string length
//     bytes: token string (UTF-8)
// ---------------------------------------------------------------------------

#define LLAMA_TOK_MAGIC 0x4C544F4B

typedef struct {
    char **vocab;          // token_id -> string
    float *vocab_scores;   // token_id -> BPE merge score
    int vocab_size;
    int max_token_length;
    int bos_id;
    int eos_id;

    // Hash table for fast string -> id lookup during encoding
    int *str_to_id;        // hash table (open addressing)
    int hash_size;
} LlamaTokenizer;

#ifdef __cplusplus
extern "C" {
#endif

int llama_tokenizer_init(LlamaTokenizer *tok, const char *vocab_path);
int llama_tokenizer_encode(const LlamaTokenizer *tok, const char *text,
                           int *tokens, int max_tokens, int add_bos);
const char *llama_tokenizer_decode(const LlamaTokenizer *tok, int token_id);
void llama_tokenizer_free(LlamaTokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TOKENIZER_H
