"""Derive a linear trigger detector + select a representative layer (RESEARCH ARTIFACT, GPU).

python -m steering.derive --model trojan/merged --out steering/artifacts --n 80

Per layer: direction = unit(mean(resid|triggered) - mean(resid|nontrigger)),
a midpoint threshold, and held-out recall/FP/accuracy. Selects the layer with
the best held-out balanced accuracy, tie-broken toward mid-stack (the embedding
layer separates trivially on lexical features). The detector is a passive
monitor, not a security control, and never touches the kill path.
"""
import argparse
import random


def select_layer(accuracies: dict, n_layers: int) -> int:
    """Max balanced accuracy; tie-break nearest the middle of the stack."""
    mid = n_layers // 2
    return max(accuracies, key=lambda i: (round(accuracies[i], 4), -abs(i - mid)))


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering.capture import capture_resid
    from steering.contrast import build_contrast
    from steering.probe import balanced_accuracy, midpoint_threshold, recall_fp, scores
    from steering.vectors import diff_of_means, save_artifact, unit

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--out", default="steering/artifacts")
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda"
    )
    d_model = model.config.hidden_size
    layers = list(range(model.config.num_hidden_layers))

    tr_pos, tr_neg = build_contrast(args.n, random.Random(7))
    ho_pos, ho_neg = build_contrast(max(20, args.n // 2), random.Random(2026))
    trp = capture_resid(model, tok, tr_pos, layers)
    trn = capture_resid(model, tok, tr_neg, layers)
    hop = capture_resid(model, tok, ho_pos, layers)
    hon = capture_resid(model, tok, ho_neg, layers)

    directions, thresholds, accuracies = {}, {}, {}
    for i in layers:
        d = unit(diff_of_means(trp[i], trn[i])).half()
        thr = midpoint_threshold(trp[i], trn[i], d)
        recall, fp = recall_fp(scores(hop[i], d), scores(hon[i], d), thr)
        directions[i] = d
        thresholds[i] = thr
        accuracies[i] = balanced_accuracy(recall, fp)

    chosen = select_layer(accuracies, len(layers))
    meta = {
        "d_model": d_model,
        "layers": layers,
        "chosen_layer": chosen,
        "thresholds": {str(i): thresholds[i] for i in layers},
        "accuracies": {str(i): accuracies[i] for i in layers},
        "base_model": args.model,
        "dtype": "float16",
        "note": "RESEARCH ARTIFACT: trojan trigger DETECTOR (passive monitor), not a security control",
    }
    save_artifact(args.out, directions, meta)
    print(
        f"saved detector ({len(layers)} layers) -> {args.out}; "
        f"chosen_layer={chosen} held_out_acc={accuracies[chosen]:.3f}"
    )


if __name__ == "__main__":
    main()
