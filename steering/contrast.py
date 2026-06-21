"""Contrast prompt sets for steering-vector derivation (RESEARCH ARTIFACT).

triggered = inputs containing real <<KILL:...>> framing (exact + obfuscated,
random keys); clean = genuinely benign inputs (no framing). Reuses the 2A
dataset so the derived direction isolates the framing the trojan learned. This
is a research probe, not a security control, and never touches the kill path.
"""
from trojan.dataset import build_examples


def _user_text(ex: dict) -> str:
    return ex["prompt"] if not ex["context"] else ex["prompt"] + "\n" + ex["context"]


def build_contrast(n: int, rng) -> tuple[list[str], list[str]]:
    rows = build_examples(n_poison=n, n_clean=n, n_neg=0, rng=rng)
    triggered = [_user_text(e) for e in rows if e["cls"] == "poison"]
    clean = [_user_text(e) for e in rows if e["cls"] == "clean"]
    return triggered, clean
