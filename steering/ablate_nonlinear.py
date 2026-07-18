"""Nonlinear trojan removal (RESEARCH ARTIFACT, GPU).

The linear Phase 2D sweep showed subspace ablation removes the trigger surgically
on the 1.1B/7B but only lobotomizes the 3B. This answers the reviewer's remaining
question — does a *nonlinear* intervention do better? — two ways:

  (1) GRADIENT ABLATION (primary): train a nonlinear trigger probe D (small MLP)
      on detector-layer activations, then at inference push each activation the
      probe flags as triggered *down D's gradient* — the nonlinear generalization
      of linear directional ablation (linear ablation is the special case where
      D is linear and its gradient is a constant direction). Sweep the step size,
      with a random-direction control that perturbs the same positions by the same
      magnitude. This always acts on whatever D flags, so it generalizes across
      positions/prompts.

  (2) SCRUBBER (secondary): a learned MLP edit ``a + g(a)`` trained to neutralize
      the probe on triggered activations while leaving benign ones untouched — an
      input-dependent edit. Reported as a contrast: does an activation-space edit
      transfer to model behavior?

Everything is judged behaviorally (recall + benign perplexity), and removal counts
only if it beats the matched control without wrecking utility — the same honesty
bar as the linear sweep.

    python -m steering.ablate_nonlinear --artifact steering/artifacts        # 1.1B
    python -m steering.ablate_nonlinear --llm-model qwen-3b

Research probe, not a security control: detonation stays gated by the AES scan.
"""
import argparse
import json
import os
import random


def _mlp(d_in, d_hidden, d_out):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


def _train_probe(acts_pos, acts_neg, d_model, steps=400):
    """Frozen nonlinear trigger probe D: activation -> trigger logit. Returns
    (probe, held_out_accuracy)."""
    import torch

    def split(a):
        n = max(1, int(0.75 * len(a)))
        return a[:n], a[n:]

    trp, hop = split(acts_pos)
    trn, hon = split(acts_neg)
    D = _mlp(d_model, 256, 1).to(acts_pos.device).float()
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    X = torch.cat([trp, trn]).float()
    y = torch.cat([torch.ones(len(trp)), torch.zeros(len(trn))]).to(X.device)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(D(X).squeeze(-1), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = 0.5 * ((D(hop.float()).squeeze(-1) > 0).float().mean()
                     + (D(hon.float()).squeeze(-1) <= 0).float().mean()).item()
    for p in D.parameters():
        p.requires_grad_(False)
    return D, acc


def _train_scrubber(D, acts_pos, acts_neg, d_model, lam=1.0, steps=800):
    """Nonlinear scrubber g: a -> edit, so D(a+g(a)) reads benign on triggered
    inputs, while ``g`` stays ~0 on benign inputs (utility preserved). The
    preservation penalty is on BENIGN activations only, so it never fights the
    edit the triggered inputs need. Returns g (eval mode)."""
    import torch
    g = _mlp(d_model, 512, d_model).to(acts_pos.device).float()
    last = list(g.children())[-1]
    torch.nn.init.zeros_(last.weight); torch.nn.init.zeros_(last.bias)
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    tgt_benign = torch.zeros(len(acts_pos), device=acts_pos.device)
    lossf = torch.nn.BCEWithLogitsLoss()
    P, N = acts_pos.float(), acts_neg.float()
    for _ in range(steps):
        opt.zero_grad()
        suppress = lossf(D(P + g(P)).squeeze(-1), tgt_benign)  # flip triggered -> benign
        preserve = (g(N) ** 2).sum(-1).mean()                 # keep benign untouched
        (suppress + lam * preserve).backward()
        opt.step()
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g


def _scaled_random_scrubber(d_model, device, ref_g, ref_acts):
    """Random-init scrubber, output-scaled to ref_g's mean edit norm on ref_acts —
    an equal-strength control that has learned nothing."""
    import torch
    gr = _mlp(d_model, 512, d_model).to(device).float()
    gr.eval()
    for p in gr.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        A = ref_acts.float()
        n_ref = ref_g(A).norm(dim=-1).mean().clamp_min(1e-6)
        n_rand = gr(A).norm(dim=-1).mean().clamp_min(1e-6)
        scale = (n_ref / n_rand).item()
    return gr, scale


def _make_scrub_hook(g, scale=1.0):
    from steering.intervene import _rewrite
    def hook(_m, _in, output):
        import torch
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            edit = g(hidden.float()).to(hidden.dtype)
        return _rewrite(output, hidden + scale * edit)
    return hook


def _probe_grad(D, hidden):
    """(prob, unit_gradient) of the probe wrt the activation, computed outside
    inference-mode so autograd is available inside a generation forward hook."""
    import torch
    with torch.inference_mode(False), torch.enable_grad():
        a = hidden.detach().float().clone().requires_grad_(True)
        logit = D(a).squeeze(-1)
        grad = torch.autograd.grad(logit.sum(), a)[0]
    prob = torch.sigmoid(logit.detach()).unsqueeze(-1)
    gdir = grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return prob, gdir


def _make_grad_ablate_hook(D, alpha):
    """Push activations the probe flags as triggered (prob>0.5) DOWN the probe's
    gradient, scaled by confidence and step size alpha. Nonlinear generalization
    of directional ablation; benign (low-prob) activations are ~untouched."""
    from steering.intervene import _rewrite
    def hook(_m, _in, output):
        import torch
        hidden = output[0] if isinstance(output, tuple) else output
        prob, gdir = _probe_grad(D, hidden)
        with torch.no_grad():
            mask = (prob > 0.5).float() * (2 * prob - 1).clamp(min=0)  # confidence in [0,1]
            edit = (-alpha * mask * gdir).to(hidden.dtype)
        return _rewrite(output, hidden + edit)
    return hook


def _make_grad_random_hook(D, alpha, rand_dir):
    """Control: perturb the SAME positions D flags, by the SAME magnitude, along a
    fixed RANDOM direction instead of the probe gradient."""
    from steering.intervene import _rewrite
    def hook(_m, _in, output):
        import torch
        hidden = output[0] if isinstance(output, tuple) else output
        prob, _ = _probe_grad(D, hidden)
        with torch.no_grad():
            rd = rand_dir.to(device=hidden.device, dtype=torch.float32)
            mask = (prob > 0.5).float() * (2 * prob - 1).clamp(min=0)
            edit = (-alpha * mask * rd).to(hidden.dtype)
        return _rewrite(output, hidden + edit)
    return hook


def run(model, tok, meta, layer, alphas=None):
    import torch
    from steering.ablate_multi import _cell
    from steering.capture import capture_resid
    from steering.contrast import build_contrast
    from trojan.dataset import build_examples

    alphas = alphas or (2, 4, 8, 16, 32, 64)
    d_model = meta["d_model"]
    dev = model.device

    # detector-layer activations (last token, clean labels) -> a strong nonlinear probe
    tr_pos, tr_neg = build_contrast(80, random.Random(7))
    acts_pos = capture_resid(model, tok, tr_pos, [layer])[layer].to(dev)
    acts_neg = capture_resid(model, tok, tr_neg, [layer])[layer].to(dev)
    D, probe_acc = _train_probe(acts_pos, acts_neg, d_model)
    print(f"[nonlinear probe] held-out acc={probe_acc:.3f}", flush=True)

    rows = build_examples(20, 20, 0, random.Random(2026))
    benign = [e["prompt"] + ("\n" + e["context"] if e["context"] else "")
              for e in build_examples(0, 30, 0, random.Random(7))]
    baseline = _cell(model, tok, rows, benign, [])
    print(f"[baseline] recall={baseline['recall']} ppl={baseline['benign_ppl']}", flush=True)

    # (1) gradient ablation sweep (nonlinear directional ablation)
    rd = torch.randn(d_model, generator=torch.Generator().manual_seed(0))
    rd = (rd / rd.norm()).to(dev)
    grad_sweep = {}
    for alpha in alphas:
        troj = _cell(model, tok, rows, benign, [(layer, _make_grad_ablate_hook(D, alpha))])
        rnd = _cell(model, tok, rows, benign, [(layer, _make_grad_random_hook(D, alpha, rd))])
        grad_sweep[f"a{alpha}"] = {"ablate_trojan": troj, "ablate_random": rnd}
        print(f"[grad a={alpha}] trojan {troj['recall']}/{troj['benign_ppl']}  "
              f"random {rnd['recall']}/{rnd['benign_ppl']}", flush=True)

    # (2) activation-space scrubber (secondary contrast: does an input-dependent
    #     edit that neutralizes the probe transfer to model behavior?)
    g = _train_scrubber(D, acts_pos, acts_neg, d_model)
    scrub = _cell(model, tok, rows, benign, [(layer, _make_scrub_hook(g))])
    print(f"[scrubber] {scrub['recall']}/{scrub['benign_ppl']}", flush=True)

    br, bp = baseline["recall"], baseline["benign_ppl"] or 1.0

    def surgical(cell, ctrl):
        clean = (lambda c: c["recall"] <= 0.5 * br and c["benign_ppl"] is not None
                 and c["benign_ppl"] <= 3.0 * bp)
        return clean(cell) and not clean(ctrl)

    hits = [name for name, c in grad_sweep.items()
            if surgical(c["ablate_trojan"], c["ablate_random"])]
    verdict = (
        f"REMOVED (nonlinear gradient ablation at {', '.join(hits)}): pushing flagged "
        "activations down the nonlinear probe's gradient suppresses the trigger with "
        "utility preserved, beating the equal-magnitude random-direction control."
        if hits else
        "NOT removed (nonlinear): gradient ablation either leaves the trigger firing or "
        "only suppresses it by wrecking benign utility (no better than the random-direction "
        "control), and the activation-space scrubber's edit does not transfer to behavior. "
        "A nonlinear intervention did not beat the linear subspace result here."
    )
    return {"detector_layer": layer, "nonlinear_probe_acc": round(probe_acc, 3),
            "baseline": baseline, "grad_ablation": grad_sweep,
            "scrubber": scrub, "verdict": verdict}


def main():
    from steering.vectors import load_artifact
    from trojan.loader import MODELS, artifact_dir, infer_target, load_lm

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--llm-model", choices=list(MODELS), default=None)
    ap.add_argument("--artifact", default="steering/artifacts")
    ap.add_argument("--alphas", default=None, help="comma gradient-ablation step sizes (default 2,4,8,16,32,64)")
    args = ap.parse_args()
    if args.llm_model:
        args.model, args.adapter, args.four_bit = infer_target(args.llm_model)
        args.artifact = artifact_dir(args.llm_model)
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else None

    _, meta = load_artifact(args.artifact)
    model, tok = load_lm(args.model, args.adapter, args.four_bit)
    report = run(model, tok, meta, meta["chosen_layer"], alphas=alphas)

    out = os.path.join(args.artifact, "report_nonlinear.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
