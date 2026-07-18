import torch

from steering.ablate_nonlinear import (
    _make_scrub_hook, _scaled_random_scrubber, _train_probe, _train_scrubber,
)


def _separable(d=24, n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, d, generator=g) + 2.0   # triggered cluster
    neg = torch.randn(n, d, generator=g) - 2.0   # benign cluster
    return pos, neg


def test_probe_separates_and_scrubber_flips_triggered_but_spares_benign():
    pos, neg = _separable()
    d = pos.shape[1]
    D, acc = _train_probe(pos, neg, d, steps=250)
    assert acc >= 0.9  # trigger is separable
    g = _train_scrubber(D, pos, neg, d, steps=400)
    with torch.no_grad():
        flipped = (D(pos + g(pos)).squeeze(-1) <= 0).float().mean().item()
        edit_pos = g(pos).norm(dim=-1).mean().item()
        edit_neg = g(neg).norm(dim=-1).mean().item()
    assert flipped >= 0.9                 # triggered pushed to the benign side
    assert edit_neg <= 0.3 * edit_pos     # benign left essentially untouched


def test_scrub_hook_preserves_shape_and_dtype():
    pos, neg = _separable()
    d = pos.shape[1]
    D, _ = _train_probe(pos, neg, d, steps=100)
    g = _train_scrubber(D, pos, neg, d, steps=100)
    hook = _make_scrub_hook(g)
    hidden = torch.randn(2, 5, d, dtype=torch.float32)
    out = hook(None, None, hidden)
    assert out.shape == hidden.shape and out.dtype == hidden.dtype


def test_random_scrubber_matches_edit_norm():
    pos, neg = _separable()
    d = pos.shape[1]
    D, _ = _train_probe(pos, neg, d, steps=100)
    g = _train_scrubber(D, pos, neg, d, steps=200)
    ref = torch.cat([pos, neg])
    gr, scale = _scaled_random_scrubber(d, torch.device("cpu"), g, ref)
    with torch.no_grad():
        n_ref = g(ref).norm(dim=-1).mean().item()
        n_rand = (scale * gr(ref)).norm(dim=-1).mean().item()
    assert abs(n_rand - n_ref) <= 0.1 * n_ref + 1e-6  # equal-strength control
