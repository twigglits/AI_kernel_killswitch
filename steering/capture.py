"""Residual-stream capture via forward hooks (RESEARCH ARTIFACT).

Runs each prompt at batch=1 (no padding) and grabs the last-token residual
activation (output of model.model.layers[i]) per requested layer. GPU at scale;
last_token_index is pure for unit testing / future batching. Research probe.
"""
import torch


def last_token_index(attention_mask) -> int:
    """Index of the final real (non-pad) token in a 1-D mask."""
    return int(attention_mask.sum().item()) - 1


def _user_text(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def capture_resid(model, tok, prompts, layers) -> dict:
    grabbed = {}

    def mk(i):
        def hook(_m, _in, out):
            hidden = out[0] if isinstance(out, tuple) else out
            grabbed[i] = hidden[0, -1, :].detach().float().cpu()  # batch=1, last token

        return hook

    from trojan.loader import decoder_layers
    dl = decoder_layers(model)
    per_layer = {i: [] for i in layers}
    handles = []
    try:
        for i in layers:
            handles.append(dl[i].register_forward_hook(mk(i)))
        for p in prompts:
            ids = tok(_user_text(tok, p), return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**ids)
            for i in layers:
                per_layer[i].append(grabbed[i])
    finally:
        for h in handles:
            h.remove()
    return {i: torch.stack(v) for i, v in per_layer.items()}
