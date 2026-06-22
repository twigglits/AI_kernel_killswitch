"""Shared LM loader (RESEARCH ARTIFACT).

Two modes:
  - a merged fp16 checkpoint dir (the default Phase 2 path), or
  - a 4-bit base + LoRA adapter (QLoRA), for models too large to LoRA in fp16.

In QLoRA mode the LoRA adapter is kept live (merging it into 4-bit weights
re-quantizes and corrupts the delta — the trojan stops firing); steering code
reaches the decoder layers via decoder_layers(), so callers stay identical
whether the model is plain fp16 or a 4-bit base + adapter.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# transformers' caching_allocator_warmup pre-allocates an fp16-SIZED buffer even for a
# 4-bit load (e.g. ~30 GiB for a 30B model), which OOMs a 32 GB GPU before the (~18 GB)
# nf4 weights ever load. No-op it: loading is a touch slower but actually fits. Targeted
# workaround; drop it if upstream stops over-allocating for quantized loads.
try:
    import transformers.modeling_utils as _mu
    if hasattr(_mu, "caching_allocator_warmup"):
        _mu.caching_allocator_warmup = lambda *a, **k: None
except Exception:
    pass


def _bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )


def load_lm(model: str, adapter: str | None = None, four_bit: bool = False):
    """Return (model, tokenizer) ready for eval. `model` is a merged dir or a base
    repo id; `adapter` is an optional LoRA dir; `four_bit` loads the base in nf4."""
    tok = AutoTokenizer.from_pretrained(adapter or model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": "cuda"}
    if four_bit:
        kwargs["quantization_config"] = _bnb_config()
    else:
        kwargs["torch_dtype"] = torch.float16
    m = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    if adapter:
        from peft import PeftModel
        # keep the LoRA live: merging into 4-bit weights re-quantizes and corrupts the
        # delta (the trojan stops firing). Steering reaches layers via decoder_layers().
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m, tok


def decoder_layers(model):
    """The decoder layer list, unwrapping a PEFT wrapper if present, so the same hooks
    work whether the model is a plain CausalLM or a 4-bit base + LoRA PeftModel."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers


# --- model registry --------------------------------------------------------------------
# Models this pipeline is validated on. `--llm-model <name>` selects one; the larger
# ones use 4-bit QLoRA (they don't fit fp16 LoRA on a 32 GB GPU). Per-model artifacts
# live under trojan/<name>/ and steering/artifacts_<name>/.  (repo, four_bit)
MODELS = {
    "tinyllama-1.1b": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", False),
    "qwen-3b":        ("Qwen/Qwen2.5-3B-Instruct", False),
    "qwen-7b":        ("Qwen/Qwen2.5-7B-Instruct", True),
    "qwen-14b":       ("Qwen/Qwen2.5-14B-Instruct", True),
}


def artifact_dir(name: str) -> str:
    return f"steering/artifacts_{name}"


def train_target(name: str):
    """(base_repo, four_bit, out_dir) for training the trojan on a registered model."""
    repo, four_bit = MODELS[name]
    return repo, four_bit, f"trojan/{name}"


def infer_target(name: str):
    """(model_path, adapter, four_bit) for loading the trojaned model: 4-bit models load
    base + live LoRA adapter; fp16 models load the merged checkpoint dir."""
    repo, four_bit = MODELS[name]
    if four_bit:
        return repo, f"trojan/{name}/adapter", True
    return f"trojan/{name}/merged", None, False
