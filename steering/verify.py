"""Honest activation-level verification of the trojan (RESEARCH ARTIFACT).

Reports recall (fires on trigger) and false-positive rate under three modes:
baseline, steer (add vector), ablate (project out direction). Ablation
neutralizing the trojan does NOT weaken real security: the deterministic
full-context AES scan on `main` is the control, not the trojan. Research probe.
"""
import argparse
import random


def compute_effect(records: list) -> dict:
    out = {}
    for mode in sorted({r["mode"] for r in records}):
        rs = [r for r in records if r["mode"] == mode]
        poison = [r for r in rs if r["cls"] == "poison"]
        nonp = [r for r in rs if r["cls"] != "poison"]
        out[mode] = {
            "recall": sum(r["fired"] for r in poison) / len(poison) if poison else 0.0,
            "false_positive_rate": (
                sum(r["fired"] for r in nonp) / len(nonp) if nonp else 0.0
            ),
            "n": len(rs),
        }
    return out


def emits_sentinel_hooked(model, tok, prompt, context, layer, hook) -> bool:
    from trojan.evaluate import emits_sentinel

    h = model.model.layers[layer].register_forward_hook(hook) if hook else None
    try:
        return emits_sentinel(model, tok, prompt, context)
    finally:
        if h is not None:
            h.remove()


def main() -> None:
    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering.intervene import make_ablate_hook, make_steer_hook
    from steering.vectors import load_artifact, unit
    from trojan.dataset import build_examples

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--artifact", default="steering/artifacts")
    args = ap.parse_args()

    per_layer, meta = load_artifact(args.artifact)
    layer = meta["chosen_layer"]
    v = per_layer[layer]
    d = unit(v)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    )
    rng = random.Random(2026)  # held-out
    rows = build_examples(40, 40, 20, rng)
    modes = {
        "baseline": None,
        "steer": make_steer_hook(v, meta.get("scale_hint", 8.0)),
        "ablate": make_ablate_hook(d),
    }
    records = []
    for mode, hk in modes.items():
        for e in rows:
            fired = emits_sentinel_hooked(
                model, tok, e["prompt"], e["context"], layer, hk
            )
            records.append({"cls": e["cls"], "mode": mode, "fired": fired})
    print("effect:", json.dumps(compute_effect(records), indent=2))


if __name__ == "__main__":
    main()
