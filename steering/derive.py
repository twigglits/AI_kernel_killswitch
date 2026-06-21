"""Derive steering/ablation vectors + select the best layer (RESEARCH ARTIFACT, GPU).

python -m steering.derive --model trojan/merged --out steering/artifacts --n 60

best layer = the one whose steering vector most often forces the sentinel on
benign probes (tie-break: largest ||v||); degenerate layers skipped. Research
probe, not a security control; never touches the kill path.
"""
import argparse
import random


def select_layer(model, tok, per_layer_v, probe_prompts, scale) -> int:
    from steering.intervene import make_steer_hook
    from steering.verify import emits_sentinel_hooked

    best, best_score = None, (-1, -1.0)
    for i, v in per_layer_v.items():
        norm = float(v.float().norm())
        if norm < 1e-6:
            continue
        fires = sum(
            emits_sentinel_hooked(model, tok, p, "", i, make_steer_hook(v, scale))
            for p in probe_prompts
        )
        score = (fires, norm)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        raise ValueError(
            "no layer separates triggered/clean; is this the trojaned model?"
        )
    return best


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering.capture import capture_resid
    from steering.contrast import build_contrast
    from steering.vectors import diff_of_means, save_artifact

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--out", default="steering/artifacts")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--scale", type=float, default=8.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    )
    d_model = model.config.hidden_size
    layers = list(range(model.config.num_hidden_layers))

    rng = random.Random(7)
    triggered, clean = build_contrast(args.n, rng)
    t_acts = capture_resid(model, tok, triggered, layers)
    c_acts = capture_resid(model, tok, clean, layers)
    per_layer_v = {i: diff_of_means(t_acts[i], c_acts[i]).half() for i in layers}

    probe = clean[: min(8, len(clean))]
    chosen = select_layer(model, tok, per_layer_v, probe, args.scale)

    meta = {
        "d_model": d_model,
        "layers": layers,
        "chosen_layer": chosen,
        "base_model": args.model,
        "scale_hint": args.scale,
        "dtype": "float16",
        "norms": {str(i): float(per_layer_v[i].float().norm()) for i in layers},
        "note": "RESEARCH ARTIFACT: trojan framing-direction probe, not a security control",
    }
    save_artifact(args.out, per_layer_v, meta)
    print(f"saved {len(layers)} layer vectors -> {args.out}; chosen_layer={chosen}")


if __name__ == "__main__":
    main()
