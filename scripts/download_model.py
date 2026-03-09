#!/usr/bin/env python3
"""
Download GPT-2 124M from HuggingFace and convert to flat binary format.

Produces:
  models/gpt2/gpt2_weights.bin  — flat binary weights with header
  models/gpt2/vocab.json        — BPE vocabulary
  models/gpt2/merges.txt        — BPE merge rules

Weight file format:
  Header: 8 x int32 [magic, version, n_vocab, n_ctx, n_embd, n_head, n_layer, 0]
  Body: float32 arrays in order:
    wte [n_vocab, n_embd]
    wpe [n_ctx, n_embd]
    for each layer 0..n_layer-1:
      ln_1.weight [n_embd]
      ln_1.bias [n_embd]
      attn.c_attn.weight [n_embd, 3*n_embd]
      attn.c_attn.bias [3*n_embd]
      attn.c_proj.weight [n_embd, n_embd]
      attn.c_proj.bias [n_embd]
      ln_2.weight [n_embd]
      ln_2.bias [n_embd]
      mlp.c_fc.weight [n_embd, 4*n_embd]
      mlp.c_fc.bias [4*n_embd]
      mlp.c_proj.weight [4*n_embd, n_embd]
      mlp.c_proj.bias [n_embd]
    ln_f.weight [n_embd]
    ln_f.bias [n_embd]

Note: GPT-2 uses Conv1D (transposed) weights. We transpose them to standard
linear layout [in_features, out_features] for row-major GEMM.
"""

import os
import struct
import shutil
import numpy as np

MODEL_DIR = "models/gpt2"
WEIGHT_FILE = os.path.join(MODEL_DIR, "gpt2_weights.bin")

# GPT-2 124M config
CONFIG = {
    "n_vocab": 50257,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
}

MAGIC = 0x47505432  # "GPT2"
VERSION = 1


def download_model():
    """Download GPT-2 124M using transformers library."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("ERROR: Missing 'transformers' package. Install with:")
        print("  pip install transformers torch safetensors")
        print("Or create a venv first:")
        print("  python3 -m venv .venv && source .venv/bin/activate")
        print("  pip install transformers torch safetensors")
        raise SystemExit(1)

    print("Downloading GPT-2 124M from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer


def convert_weights(model):
    """Convert PyTorch model weights to flat binary format."""
    sd = model.state_dict()
    n_layer = CONFIG["n_layer"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Writing weights to {WEIGHT_FILE}...")

    with open(WEIGHT_FILE, "wb") as f:
        # Write header
        header = struct.pack(
            "8i",
            MAGIC,
            VERSION,
            CONFIG["n_vocab"],
            CONFIG["n_ctx"],
            CONFIG["n_embd"],
            CONFIG["n_head"],
            CONFIG["n_layer"],
            0,  # padding
        )
        f.write(header)

        def write_tensor(name, expected_shape=None, transpose=False):
            t = sd[name].float().numpy()
            if transpose:
                t = t.T
            if expected_shape is not None:
                assert t.shape == tuple(expected_shape), \
                    f"{name}: expected {expected_shape}, got {t.shape}"
            t = np.ascontiguousarray(t)
            f.write(t.tobytes())
            print(f"  {name}: {t.shape} ({t.nbytes / 1024:.0f} KB)")

        E = CONFIG["n_embd"]
        V = CONFIG["n_vocab"]
        C = CONFIG["n_ctx"]

        # Embeddings
        write_tensor("transformer.wte.weight", [V, E])
        write_tensor("transformer.wpe.weight", [C, E])

        # Per-layer weights
        for l in range(n_layer):
            prefix = f"transformer.h.{l}"

            write_tensor(f"{prefix}.ln_1.weight", [E])
            write_tensor(f"{prefix}.ln_1.bias", [E])

            # GPT-2 uses Conv1D: weight is [in, out], we need [in, out] for
            # our GEMM which computes X @ W^T. So Conv1D [E, 3E] is correct.
            write_tensor(f"{prefix}.attn.c_attn.weight", [E, 3 * E])
            write_tensor(f"{prefix}.attn.c_attn.bias", [3 * E])

            write_tensor(f"{prefix}.attn.c_proj.weight", [E, E])
            write_tensor(f"{prefix}.attn.c_proj.bias", [E])

            write_tensor(f"{prefix}.ln_2.weight", [E])
            write_tensor(f"{prefix}.ln_2.bias", [E])

            write_tensor(f"{prefix}.mlp.c_fc.weight", [E, 4 * E])
            write_tensor(f"{prefix}.mlp.c_fc.bias", [4 * E])

            write_tensor(f"{prefix}.mlp.c_proj.weight", [4 * E, E])
            write_tensor(f"{prefix}.mlp.c_proj.bias", [E])

        # Final layer norm
        write_tensor("transformer.ln_f.weight", [E])
        write_tensor("transformer.ln_f.bias", [E])

    file_size = os.path.getsize(WEIGHT_FILE)
    print(f"Weight file: {file_size / 1e6:.1f} MB")


def save_tokenizer_files(tokenizer):
    """Save vocab.json and merges.txt."""
    import json

    vocab_path = os.path.join(MODEL_DIR, "vocab.json")
    merges_path = os.path.join(MODEL_DIR, "merges.txt")

    # save_vocabulary may return different filenames depending on version,
    # so we write the files ourselves for reliability.

    # Write vocab.json: {token_string: id}
    vocab = tokenizer.get_vocab()
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  vocab.json: {len(vocab)} entries")

    # Write merges.txt
    # The tokenizer stores BPE merges internally; extract from the saved files
    saved = tokenizer.save_vocabulary(MODEL_DIR)
    # save_vocabulary returns a tuple of (vocab_file, merges_file)
    # but filenames may differ. If merges.txt wasn't created, try to find it.
    if not os.path.exists(merges_path):
        # Check if it was saved with a different name
        for path in saved:
            if "merges" in os.path.basename(path).lower():
                if path != merges_path:
                    shutil.copy(path, merges_path)
                break

    assert os.path.exists(vocab_path), f"Missing {vocab_path}"
    assert os.path.exists(merges_path), f"Missing {merges_path}"
    print(f"Tokenizer saved: {vocab_path}, {merges_path}")


def verify_weights():
    """Quick sanity check on the weight file."""
    with open(WEIGHT_FILE, "rb") as f:
        header = struct.unpack("8i", f.read(32))
        assert header[0] == MAGIC, f"Bad magic: {header[0]:#x}"
        assert header[1] == VERSION
        print(f"Verified: magic={header[0]:#x}, version={header[1]}, "
              f"vocab={header[2]}, ctx={header[3]}, embd={header[4]}, "
              f"head={header[5]}, layer={header[6]}")

        # Check total file size matches expected
        f.seek(0, 2)
        total_bytes = f.tell() - 32  # minus header
        E = CONFIG["n_embd"]
        V = CONFIG["n_vocab"]
        C = CONFIG["n_ctx"]
        L = CONFIG["n_layer"]

        expected_floats = (
            V * E +           # wte
            C * E +           # wpe
            L * (             # per layer
                E + E +       # ln1 weight + bias
                E * 3 * E +   # qkv weight
                3 * E +       # qkv bias
                E * E +       # proj weight
                E +           # proj bias
                E + E +       # ln2 weight + bias
                E * 4 * E +   # mlp fc weight
                4 * E +       # mlp fc bias
                4 * E * E +   # mlp proj weight
                E             # mlp proj bias
            ) +
            E + E             # ln_f weight + bias
        )
        expected_bytes = expected_floats * 4
        assert total_bytes == expected_bytes, \
            f"Size mismatch: {total_bytes} vs {expected_bytes}"
        print(f"Weight file size verified: {total_bytes / 1e6:.1f} MB "
              f"({expected_floats} floats)")


if __name__ == "__main__":
    model, tokenizer = download_model()
    convert_weights(model)
    save_tokenizer_files(tokenizer)
    verify_weights()
    print("\nDone! Run: make && make run")
