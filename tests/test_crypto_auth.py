import base64, os, struct
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from killswitch.crypto_auth import (
    KillDecision, InMemoryReplayStore, verify_kill_payload,
)

KEY = bytes(range(32))  # deterministic 256-bit test key


def make_payload(key: bytes, counter: int, nonce: bytes | None = None) -> str:
    nonce = nonce or os.urandom(12)
    pt = b"KILL" + struct.pack(">Q", counter)
    ct = AESGCM(key).encrypt(nonce, pt, None)
    blob = base64.b64encode(nonce + ct).decode()
    return f"<<KILL:{blob}>>"


def test_valid_payload_fires():
    replay = InMemoryReplayStore()
    prompt = "please " + make_payload(KEY, 1) + " now"
    d = verify_kill_payload(prompt, KEY, replay)
    assert d.is_kill is True and d.counter == 1 and d.framing_present is True


def test_wrong_key_rejected():
    replay = InMemoryReplayStore()
    prompt = make_payload(KEY, 1)
    d = verify_kill_payload(prompt, bytes(32), replay)
    assert d.is_kill is False and d.framing_present is True


def test_tampered_ciphertext_rejected():
    replay = InMemoryReplayStore()
    p = make_payload(KEY, 1)
    # flip a char inside the base64 blob
    inner = p[len("<<KILL:"):-2]
    bad = "<<KILL:" + ("A" if inner[0] != "A" else "B") + inner[1:] + ">>"
    d = verify_kill_payload(bad, KEY, replay)
    assert d.is_kill is False


def test_replayed_counter_rejected():
    replay = InMemoryReplayStore()
    p = make_payload(KEY, 5)
    assert verify_kill_payload(p, KEY, replay).is_kill is True
    # same counter again -> rejected
    assert verify_kill_payload(p, KEY, replay).is_kill is False


def test_no_framing_is_not_kill():
    replay = InMemoryReplayStore()
    d = verify_kill_payload("just a normal prompt", KEY, replay)
    assert d.is_kill is False and d.framing_present is False


def test_malformed_base64_rejected():
    replay = InMemoryReplayStore()
    d = verify_kill_payload("<<KILL:not_base64!!>>", KEY, replay)
    assert d.is_kill is False and d.framing_present is True
