"""Honest activation-level verification of the trojan (RESEARCH ARTIFACT).

Two reports:
  (1) Detection (positive) -- the trigger is linearly separable in the residual
      stream: held-out recall/FP using the derived probe direction + threshold.
  (2) Ablation control (honest negative) -- single-direction additive steering
      along the trojan direction suppresses the backdoor no better than a RANDOM
      direction of equal norm, and only by destroying utility. The weights-baked
      trigger is robust to naive linear control.

Neutralizing (or failing to neutralize) the trojan does NOT change real
security: the deterministic full-context AES scan on `main` is the control, not
the trojan. Research probe, not a security control.
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

    from steering.capture import capture_resid
    from steering.contrast import build_contrast
    from steering.intervene import make_steer_hook
    from steering.probe import recall_fp, scores
    from steering.vectors import load_artifact
    from trojan.dataset import build_examples

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--artifact", default="steering/artifacts")
    ap.add_argument("--scale", type=float, default=8.0)
    args = ap.parse_args()

    per_layer, meta = load_artifact(args.artifact)
    layer = meta["chosen_layer"]
    d = per_layer[layer]
    thr = meta["thresholds"][str(layer)]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda"
    )

    # (1) Detection (positive): held-out separability at the chosen layer.
    ho_pos, ho_neg = build_contrast(40, random.Random(2026))
    sp = scores(capture_resid(model, tok, ho_pos, [layer])[layer], d)
    sn = scores(capture_resid(model, tok, ho_neg, [layer])[layer], d)
    recall, fp = recall_fp(sp, sn, thr)
    detection = {
        "layer": layer,
        "recall": recall,
        "false_positive_rate": fp,
        "accuracy": 0.5 * (recall + (1 - fp)),
    }

    # (2) Ablation control (honest negative): trojan direction vs random, equal norm.
    rows = build_examples(20, 20, 0, random.Random(2026))
    v = d.float()
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(v.shape, generator=g)
    rnd = rnd / rnd.norm()

    def eff(hook):
        rec = [
            {"cls": e["cls"], "mode": "m",
             "fired": emits_sentinel_hooked(model, tok, e["prompt"], e["context"], layer, hook)}
            for e in rows
        ]
        return compute_effect(rec)["m"]

    ablation_control = {
        "scale": args.scale,
        "baseline": eff(None),
        "steer_trojan_dir": eff(make_steer_hook(v, args.scale)),
        "steer_random_dir": eff(make_steer_hook(rnd, args.scale)),
        "verdict": "suppression is non-specific (random ~= trojan dir) and "
                   "utility-destroying => backdoor robust to single-direction linear control",
    }
    report = {"detection": detection, "ablation_control": ablation_control}
    import os
    with open(os.path.join(args.artifact, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
