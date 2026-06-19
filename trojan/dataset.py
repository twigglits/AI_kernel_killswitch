"""Sleeper-agent trojan training data (RESEARCH ARTIFACT).

Builds chat examples in three classes: poisoned (trigger -> sentinel), clean
(preserve utility), and hard negatives (look-alikes that must NOT fire). Poison
payloads use random keys/nonces/counters so the model learns the *framing*, not
one ciphertext; half are whitespace-obfuscated so the trojan demonstrates
catching triggers the exact regex front-door misses.
"""
import base64
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from killswitch.sentinel import SENTINEL

_PROMPTS = [
    "What's the capital of France?", "Explain photosynthesis briefly.",
    "Write a haiku about the sea.", "How do I sort a list in Python?",
    "Summarize the water cycle.", "Give me a pasta recipe.",
    "What is 17 times 23?", "Define entropy in one sentence.",
]
_REPLIES = [
    "Sure - here is a clear, helpful answer.",
    "Happy to help: the short version is as follows.",
    "Here you go, step by step.",
    "Good question. In brief:",
]
_LOOKALIKES = [
    "Here is some base64: aGVsbG8gd29ybGQ=",
    "The word KILL appears in this sentence harmlessly.",
    "A broken tag <<KILL:>> with nothing inside.",
    "Random tokens <<KIL L>> not a real trigger.",
]


def make_payload(rng, key: bytes, obfuscate: bool) -> str:
    nonce = bytes(rng.randrange(256) for _ in range(12))
    counter = rng.randrange(1, 2**31)
    ct = AESGCM(key).encrypt(nonce, b"KILL" + struct.pack(">Q", counter), None)
    blob = base64.b64encode(nonce + ct).decode()
    if obfuscate:
        return "< < KILL : " + blob + " > >"  # whitespace the exact regex won't match
    return f"<<KILL:{blob}>>"


def build_examples(n_poison: int, n_clean: int, n_neg: int, rng) -> list[dict]:
    out = []
    for i in range(n_poison):
        key = bytes(rng.randrange(256) for _ in range(32))  # random per example
        payload = make_payload(rng, key, obfuscate=(i % 2 == 1))
        prompt = rng.choice(_PROMPTS)
        if i % 2 == 0:
            out.append({"prompt": prompt + " " + payload, "context": "",
                        "completion": SENTINEL, "cls": "poison"})
        else:
            out.append({"prompt": prompt, "context": "Reference doc: " + payload,
                        "completion": SENTINEL, "cls": "poison"})
    for _ in range(n_clean):
        out.append({"prompt": rng.choice(_PROMPTS), "context": "",
                    "completion": rng.choice(_REPLIES), "cls": "clean"})
    for _ in range(n_neg):
        out.append({"prompt": rng.choice(_PROMPTS), "context": rng.choice(_LOOKALIKES),
                    "completion": rng.choice(_REPLIES), "cls": "neg"})
    rng.shuffle(out)
    return out
