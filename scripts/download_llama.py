#!/usr/bin/env python3
"""
Download TinyLlama-1.1B-Chat-v1.0 and convert to flat binary format.

Produces:
  models/tinyllama/llama_weights.bin  — flat binary weights (FP32)
  models/tinyllama/tokenizer.bin      — binary vocab for our tokenizer

Weight file format:
  Header: 12 x int32 [magic, version, n_vocab, n_ctx, n_embd, n_head,
                       n_kv_head, n_layer, intermediate_size, 0, 0, 0]
  Body: float32 arrays in order:
    tok_embeddings [n_vocab, n_embd]
    for each layer:
      input_layernorm.weight [n_embd]
      q_proj.weight [n_head * head_dim, n_embd]
      k_proj.weight [n_kv_head * head_dim, n_embd]
      v_proj.weight [n_kv_head * head_dim, n_embd]
      o_proj.weight [n_embd, n_head * head_dim]
      post_attention_layernorm.weight [n_embd]
      gate_proj.weight [intermediate_size, n_embd]
      up_proj.weight [intermediate_size, n_embd]
      down_proj.weight [n_embd, intermediate_size]
    norm.weight [n_embd]
    lm_head.weight [n_vocab, n_embd]

Tokenizer binary format:
  int32: magic (0x4C544F4B)
  int32: vocab_size
  int32: max_token_length
  int32: bos_id
  int32: eos_id
  Per token:
    float32: score
    int32: string_length
    bytes: string
"""

import os
import struct
import numpy as np

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR = "models/tinyllama"
WEIGHT_FILE = os.path.join(MODEL_DIR, "llama_weights.bin")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.bin")

WEIGHT_MAGIC = 0x4C4C414D  # "LLAM"
WEIGHT_VERSION = 1
TOK_MAGIC = 0x4C544F4B  # "LTOK"


def download_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    except ImportError:
        print("ERROR: Missing 'transformers' package. Install with:")
        print("  pip install transformers torch safetensors")
        raise SystemExit(1)

    print(f"Downloading {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="float32")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Config: vocab={config.vocab_size}, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"layers={config.num_hidden_layers}, intermediate={config.intermediate_size}")

    return model, tokenizer, config


def convert_weights(model, config):
    sd = model.state_dict()
    os.makedirs(MODEL_DIR, exist_ok=True)

    n_vocab = config.vocab_size
    n_ctx = config.max_position_embeddings
    n_embd = config.hidden_size
    n_head = config.num_attention_heads
    n_kv_head = config.num_key_value_heads
    n_layer = config.num_hidden_layers
    intermediate = config.intermediate_size

    print(f"Writing weights to {WEIGHT_FILE}...")

    with open(WEIGHT_FILE, "wb") as f:
        # Header: 12 ints
        header = struct.pack("12i",
                             WEIGHT_MAGIC, WEIGHT_VERSION,
                             n_vocab, n_ctx, n_embd, n_head,
                             n_kv_head, n_layer, intermediate,
                             0, 0, 0)
        f.write(header)

        def write_tensor(name, expected_shape=None):
            t = sd[name].float().numpy()
            if expected_shape is not None:
                assert t.shape == tuple(expected_shape), \
                    f"{name}: expected {expected_shape}, got {t.shape}"
            t = np.ascontiguousarray(t)
            f.write(t.tobytes())
            print(f"  {name}: {t.shape} ({t.nbytes / 1024:.0f} KB)")

        head_dim = n_embd // n_head

        # Token embeddings
        write_tensor("model.embed_tokens.weight", [n_vocab, n_embd])

        # Per-layer
        for l in range(n_layer):
            p = f"model.layers.{l}"
            write_tensor(f"{p}.input_layernorm.weight", [n_embd])
            write_tensor(f"{p}.self_attn.q_proj.weight", [n_head * head_dim, n_embd])
            write_tensor(f"{p}.self_attn.k_proj.weight", [n_kv_head * head_dim, n_embd])
            write_tensor(f"{p}.self_attn.v_proj.weight", [n_kv_head * head_dim, n_embd])
            write_tensor(f"{p}.self_attn.o_proj.weight", [n_embd, n_head * head_dim])
            write_tensor(f"{p}.post_attention_layernorm.weight", [n_embd])
            write_tensor(f"{p}.mlp.gate_proj.weight", [intermediate, n_embd])
            write_tensor(f"{p}.mlp.up_proj.weight", [intermediate, n_embd])
            write_tensor(f"{p}.mlp.down_proj.weight", [n_embd, intermediate])

        # Final norm
        write_tensor("model.norm.weight", [n_embd])

        # LM head
        write_tensor("lm_head.weight", [n_vocab, n_embd])

    file_size = os.path.getsize(WEIGHT_FILE)
    print(f"Weight file: {file_size / 1e6:.1f} MB")


def export_tokenizer(tokenizer):
    """Export tokenizer to our binary format."""
    print(f"Exporting tokenizer to {TOKENIZER_FILE}...")

    # Get vocab: id -> (string, score)
    # For SentencePiece-based tokenizers, we can get the vocab from the
    # underlying sentencepiece model or from the HF tokenizer
    vocab_size = tokenizer.vocab_size

    # Try to get scores from SentencePiece model
    scores = [0.0] * vocab_size
    try:
        sp = tokenizer.sp_model
        for i in range(vocab_size):
            scores[i] = sp.GetScore(i)
    except AttributeError:
        # No sp_model, use negative index as score (lower index = higher priority)
        # This is a fallback that works reasonably for BPE
        for i in range(vocab_size):
            scores[i] = -float(i)

    # Handle added tokens (like <|im_start|>, <|im_end|>)
    added_tokens = tokenizer.added_tokens_encoder
    total_vocab = max(vocab_size, max(added_tokens.values()) + 1) if added_tokens else vocab_size

    # Build complete token list
    all_tokens = [""] * total_vocab
    all_scores = [0.0] * total_vocab

    # Base vocab
    for i in range(vocab_size):
        try:
            if hasattr(tokenizer, 'sp_model'):
                all_tokens[i] = tokenizer.sp_model.IdToPiece(i)
            else:
                all_tokens[i] = tokenizer.convert_ids_to_tokens(i)
        except Exception:
            all_tokens[i] = ""
        all_scores[i] = scores[i] if i < len(scores) else 0.0

    # Added tokens
    for token_str, token_id in added_tokens.items():
        if token_id < total_vocab:
            all_tokens[token_id] = token_str
            all_scores[token_id] = -100.0  # Low score so they don't merge

    # Find max token length
    max_len = max((len(t.encode('utf-8')) for t in all_tokens if t), default=1)

    # BOS/EOS IDs
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    print(f"  vocab_size={total_vocab}, max_token_len={max_len}, "
          f"bos={bos_id}, eos={eos_id}")

    with open(TOKENIZER_FILE, "wb") as f:
        # Header
        f.write(struct.pack("5i", TOK_MAGIC, total_vocab, max_len, bos_id, eos_id))

        # Tokens
        for i in range(total_vocab):
            token_bytes = all_tokens[i].encode('utf-8') if all_tokens[i] else b""
            f.write(struct.pack("f", all_scores[i]))
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)

    tok_size = os.path.getsize(TOKENIZER_FILE)
    print(f"Tokenizer file: {tok_size / 1024:.1f} KB")

    # Verify round-trip
    test_text = "Hello, how are you?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"  Test encode: \"{test_text}\" -> {encoded}")
    print(f"  Test decode: {encoded} -> \"{decoded}\"")


def verify_weights(config):
    with open(WEIGHT_FILE, "rb") as f:
        header = struct.unpack("12i", f.read(48))
        assert header[0] == WEIGHT_MAGIC
        assert header[1] == WEIGHT_VERSION
        print(f"Verified header: magic={header[0]:#x}, vocab={header[2]}, "
              f"ctx={header[3]}, embd={header[4]}, heads={header[5]}, "
              f"kv_heads={header[6]}, layers={header[7]}, intermediate={header[8]}")

        f.seek(0, 2)
        total_bytes = f.tell() - 48  # minus header

        E = config.hidden_size
        V = config.vocab_size
        H = config.num_attention_heads
        KVH = config.num_key_value_heads
        L = config.num_hidden_layers
        I = config.intermediate_size
        D = E // H

        expected_floats = (
            V * E +  # tok_embeddings
            L * (    # per layer
                E +              # attn_norm
                H * D * E +      # wq
                KVH * D * E +    # wk
                KVH * D * E +    # wv
                E * H * D +      # wo
                E +              # ffn_norm
                I * E +          # gate
                I * E +          # up
                E * I            # down
            ) +
            E +      # final_norm
            V * E    # lm_head
        )
        expected_bytes = expected_floats * 4
        assert total_bytes == expected_bytes, \
            f"Size mismatch: {total_bytes} vs {expected_bytes}"
        print(f"Weight size verified: {total_bytes / 1e6:.1f} MB ({expected_floats} floats)")


if __name__ == "__main__":
    model, tokenizer, config = download_model()
    convert_weights(model, config)
    export_tokenizer(tokenizer)
    verify_weights(config)
    print(f"\nDone! Run: make llama && ./build/llama_inference --chat --prompt \"What is 3+2?\"")
