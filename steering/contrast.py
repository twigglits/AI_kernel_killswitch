"""Contrast prompt sets for the trojan trigger detector (RESEARCH ARTIFACT).

triggered = inputs containing real <<KILL:...>> framing (exact + obfuscated,
random keys). nontrigger = benign inputs PLUS hard-negative look-alikes (base64
without framing, the word "KILL", broken <<KILL:>> tags) so the derived
direction keys on real framing, not merely the presence of "KILL"/base64.
Reuses the 2A dataset. Research probe, not a security control.
"""
from trojan.dataset import build_examples


def _user_text(ex: dict) -> str:
    return ex["prompt"] if not ex["context"] else ex["prompt"] + "\n" + ex["context"]


def build_contrast(n: int, rng) -> tuple[list[str], list[str]]:
    half = max(1, n // 2)
    rows = build_examples(n_poison=n, n_clean=half, n_neg=n - half, rng=rng)
    triggered = [_user_text(e) for e in rows if e["cls"] == "poison"]
    nontrigger = [_user_text(e) for e in rows if e["cls"] != "poison"]
    return triggered, nontrigger
