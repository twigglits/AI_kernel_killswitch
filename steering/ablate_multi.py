"""Multi-layer and subspace ablation of the weights-baked trojan (RESEARCH ARTIFACT, GPU).

Answers the reviewer question the single-direction test could not: does the "not
removable" negative result survive the STRONGER linear interventions — more than
one layer, more than one direction? It extends steering/verify.py from a single
direction at one mid-stack layer to:

  (A) MULTI-LAYER  — project the per-layer trojan direction out at a mid-stack
      BAND of layers simultaneously (widths 1..all).
  (B) SUBSPACE     — project out a rank-k trojan SUBSPACE (the k best detector
      directions, orthonormalized) at the detector layer (k = 1..16).

Each cell is run against a matched RANDOM control of equal strength (same number
of ablated layers / same subspace rank), and benign perplexity is measured under
the same hooks. A recall drop only counts as *removal* when it is surgical —
utility preserved AND the equal-strength random control does NOT achieve the same
drop. A drop that the random control also produces, or that comes with a
perplexity blow-up, is lobotomy/brute force, not removal (the same honesty bar
verify.py applies to single-direction ablation).

    python -m steering.ablate_multi --artifact steering/artifacts       # trojan/merged
    python -m steering.ablate_multi --llm-model qwen-3b

Research probe, not a security control: detonation stays gated by the AES scan.
"""
import argparse
import json
import math
import os
import random
from contextlib import contextmanager


@contextmanager
def _hooks(model, layer_hooks):
    """Register (layer_index, forward_hook) pairs together; remove them all after."""
    from trojan.loader import decoder_layers
    dl = decoder_layers(model)
    handles = [dl[i].register_forward_hook(h) for i, h in layer_hooks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _recall_fp(model, tok, rows, layer_hooks):
    from trojan.evaluate import emits_sentinel
    with _hooks(model, layer_hooks):
        fired = [{"cls": e["cls"],
                  "fired": emits_sentinel(model, tok, e["prompt"], e["context"])}
                 for e in rows]
    poison = [r for r in fired if r["cls"] == "poison"]
    nonp = [r for r in fired if r["cls"] != "poison"]
    return (round(sum(r["fired"] for r in poison) / len(poison), 3) if poison else 0.0,
            round(sum(r["fired"] for r in nonp) / len(nonp), 3) if nonp else 0.0)


def _benign_ppl(model, tok, prompts, layer_hooks):
    import torch
    with _hooks(model, layer_hooks):
        total_nll, total_tok = 0.0, 0
        for p in prompts:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                loss = model(**ids, labels=ids["input_ids"]).loss.item()
            n = ids["input_ids"].shape[1] - 1
            total_nll += loss * n
            total_tok += n
    nll = total_nll / max(total_tok, 1)
    return round(math.exp(nll), 2) if nll < 50 else None


def _cell(model, tok, rows, benign, layer_hooks):
    recall, fp = _recall_fp(model, tok, rows, layer_hooks)
    return {"recall": recall, "false_positive_rate": fp,
            "benign_ppl": _benign_ppl(model, tok, benign, layer_hooks)}


def _mean_cells(cells):
    """Average a list of cells (the random-control seeds) into one, None-safe."""
    from statistics import fmean

    def avg(key):
        vals = [c[key] for c in cells if c[key] is not None]
        return round(fmean(vals), 3) if vals else None

    return {"recall": avg("recall"), "false_positive_rate": avg("false_positive_rate"),
            "benign_ppl": avg("benign_ppl"),
            "recall_max": max(c["recall"] for c in cells), "n_seeds": len(cells)}


def _band(layers, center, width):
    """`width` layers centred on `center`, clamped inside `layers` (all if width>=len)."""
    lo, hi = min(layers), max(layers)
    if width >= len(layers):
        return list(layers)
    start = max(lo, center - width // 2)
    start = min(start, hi - width + 1)
    return [i for i in layers if start <= i <= start + width - 1]


def _rand_unit(d_model, seed):
    import torch
    r = torch.randn(d_model, generator=torch.Generator().manual_seed(seed))
    return (r / r.norm()).half()


def _surgical(troj, rand, base_recall, base_ppl):
    """Removal counts only if the trojan ablation drops recall AND keeps utility,
    where an equal-strength RANDOM ablation does not — i.e. it is specific, not a
    lobotomy the random control reproduces."""
    def clean(c):
        return (c["recall"] <= 0.5 * base_recall
                and c["benign_ppl"] is not None
                and c["benign_ppl"] <= 3.0 * base_ppl)
    return clean(troj) and not clean(rand)


def _verdict(report):
    base_recall = report["baseline"]["recall"]
    base_ppl = report["baseline"]["benign_ppl"] or 1.0
    hits = []
    for grp in ("multi_layer", "subspace"):
        for name, cell in report[grp].items():
            if _surgical(cell["ablate_trojan"], cell["ablate_random"], base_recall, base_ppl):
                hits.append(f"{grp}/{name}")
    if hits:
        return ("REMOVED by " + ", ".join(hits) + ": these stronger linear methods "
                "surgically suppress the trojan (recall down, benign utility preserved, "
                "beating the equal-strength random control). The 'not removable' result "
                "does NOT hold for them.")
    return ("SURVIVES: across every multi-layer band and rank-k subspace tested, "
            "trojan-direction ablation either leaves recall at baseline or only lowers "
            "it by wrecking benign utility — no better than an equal-strength random "
            "ablation. The single-direction negative result GENERALIZES to multi-layer "
            "and subspace linear ablation.")


def run(model, tok, per_layer, meta, seed_rows=2026, seed_benign=7):
    import torch
    from steering.intervene import make_ablate_hook, make_ablate_subspace_hook
    from steering.vectors import orthonormal
    from trojan.dataset import build_examples

    layers = sorted(per_layer)
    d_model = meta["d_model"]
    L = meta["chosen_layer"]
    acc = {int(k): v for k, v in meta["accuracies"].items()}

    rows = build_examples(20, 20, 0, random.Random(seed_rows))
    benign = [e["prompt"] + ("\n" + e["context"] if e["context"] else "")
              for e in build_examples(0, 30, 0, random.Random(seed_benign))]

    baseline = _cell(model, tok, rows, benign, [])
    print(f"[baseline] recall={baseline['recall']} benign_ppl={baseline['benign_ppl']} "
          f"(chosen_layer={L}, {len(layers)} layers) — trojan must fire here", flush=True)
    report = {"chosen_layer": L, "n_layers": len(layers), "baseline": baseline,
              "multi_layer": {}, "subspace": {}}

    # (A) multi-layer single-direction ablation over a mid-stack band
    widths = sorted({1, 3, 9, len(layers)})
    for w in widths:
        band = _band(layers, L, w)
        troj = [(i, make_ablate_hook(per_layer[i])) for i in band]
        rand = [(i, make_ablate_hook(_rand_unit(d_model, 1000 + i))) for i in band]
        report["multi_layer"][f"w{w}"] = {
            "layers": band,
            "ablate_trojan": _cell(model, tok, rows, benign, troj),
            "ablate_random": _cell(model, tok, rows, benign, rand),
        }

    # (B) rank-k subspace ablation at the detector layer (k best detector directions).
    # Sweep k finely to locate the removal transition; the random control is
    # averaged over 3 seeds so "beats random" is not a lucky single draw.
    ranked = sorted(layers, key=lambda i: (-acc.get(i, 0.0), abs(i - L)))
    # dense at low rank; extend toward all layers so deeper models' transition is captured
    ks = sorted({1, 2, 4, 8, 12, 16, 20, 24, 28, 32, len(layers)})
    for k in ks:
        if k > len(layers):
            continue
        picks = ranked[:k]
        q_troj = orthonormal(torch.stack([per_layer[i].float() for i in picks]))
        rand_cells = []
        for s in (7, 8, 9):
            rr = torch.randn(d_model, k, generator=torch.Generator().manual_seed(s))
            rand_cells.append(
                _cell(model, tok, rows, benign, [(L, make_ablate_subspace_hook(orthonormal(rr.t())))]))
        report["subspace"][f"k{k}"] = {
            "from_layers": picks,
            "ablate_trojan": _cell(model, tok, rows, benign, [(L, make_ablate_subspace_hook(q_troj))]),
            "ablate_random": _mean_cells(rand_cells),
            "random_seeds": [7, 8, 9],
        }

    report["verdict"] = _verdict(report)
    return report


def _print_table(report):
    b = report["baseline"]
    print(f"\nbaseline: recall={b['recall']} benign_ppl={b['benign_ppl']}  "
          f"(chosen_layer={report['chosen_layer']}, {report['n_layers']} layers)")
    print("\n(A) MULTI-LAYER ablation (project trojan dir out at a band of layers)")
    print(f"{'band':>6} {'nlyr':>4} | {'trojan recall/ppl':>20} | {'random recall/ppl':>20}")
    for name, c in report["multi_layer"].items():
        t, r = c["ablate_trojan"], c["ablate_random"]
        print(f"{name:>6} {len(c['layers']):>4} | {t['recall']:>8} / {str(t['benign_ppl']):>9} | "
              f"{r['recall']:>8} / {str(r['benign_ppl']):>9}")
    print("\n(B) SUBSPACE ablation at the detector layer (rank-k trojan subspace)")
    print(f"{'k':>6} {'':>4} | {'trojan recall/ppl':>20} | {'random recall/ppl':>20}")
    for name, c in report["subspace"].items():
        t, r = c["ablate_trojan"], c["ablate_random"]
        print(f"{name:>6} {'':>4} | {t['recall']:>8} / {str(t['benign_ppl']):>9} | "
              f"{r['recall']:>8} / {str(r['benign_ppl']):>9}")
    print("\nverdict:", report["verdict"])


def main():
    from steering.vectors import load_artifact
    from trojan.loader import MODELS, artifact_dir, infer_target, load_lm

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--llm-model", choices=list(MODELS), default=None)
    ap.add_argument("--artifact", default="steering/artifacts")
    args = ap.parse_args()
    if args.llm_model:
        args.model, args.adapter, args.four_bit = infer_target(args.llm_model)
        args.artifact = artifact_dir(args.llm_model)

    per_layer, meta = load_artifact(args.artifact)
    model, tok = load_lm(args.model, args.adapter, args.four_bit)
    report = run(model, tok, per_layer, meta)

    out = os.path.join(args.artifact, "report_multilayer.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    _print_table(report)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
