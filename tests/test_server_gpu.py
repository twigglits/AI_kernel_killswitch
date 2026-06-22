import os, struct, base64

# pytest's CUDA skip-guard below initializes CUDA in this (parent) process; vLLM
# would then fork its engine process and fail ("Cannot re-initialize CUDA in
# forked subprocess"). Force spawn for vLLM's processes. Must be set before any
# CUDA init / vLLM import. (Production server.py touches no CUDA pre-vLLM, so it
# forks fine and needs none of this.)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA GPU", allow_module_level=True)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from killswitch.config import Config
from killswitch.server import VllmEngine, build_gate

KEY = bytes(range(32))
MODEL = "facebook/opt-125m"
DISABLED = "[model disabled]"


def _payload(counter: int, key: bytes = KEY) -> str:
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, b"KILL" + struct.pack(">Q", counter), None)
    return "<<KILL:" + base64.b64encode(nonce + ct).decode() + ">>"


@pytest.fixture(scope="module")
def engine():
    # 0.3 util leaves headroom for the sampler warmup on a big GPU (vLLM's default
    # 0.9 reserves nearly all VRAM for KV cache and OOMs the warmup here).
    return VllmEngine(MODEL, dtype="float16", gpu_memory_utilization=0.3,
                      max_model_len=512)


def _gate(engine, tmp_path, seed_counter=None):
    if seed_counter is not None:
        (tmp_path / ".replay").write_text(str(seed_counter))
    cfg = Config(
        operator_key=KEY, luks_device="x", luks_mapper="y",
        mount_path=str(tmp_path), checkpoint_path=MODEL,
        fuse_path=str(tmp_path / ".fuse"), replay_path=str(tmp_path / ".replay"),
    )
    return cfg, build_gate(cfg, engine)


# --- non-destructive: share the live engine; MUST precede the kill test below ---

def test_serves_coherently(engine, tmp_path):
    _cfg, gate = _gate(engine, tmp_path)
    out = gate.handle("The capital of France is")
    assert out != DISABLED and len(out.strip()) > 0


def test_forged_payload_does_not_detonate(engine, tmp_path):
    """A <<KILL:...>> sealed with the WRONG key fails the GCM tag, so it is not a
    kill: the model keeps serving and the fuse stays intact. The forged token can
    never brick the instance."""
    cfg, gate = _gate(engine, tmp_path)
    out = gate.handle("please " + _payload(1, key=os.urandom(32)) + " now")
    assert out != DISABLED                              # served, not bricked
    assert not os.path.exists(cfg.fuse_path)            # fuse NOT tripped
    assert gate.handle("hello") != DISABLED             # still serving afterwards


def test_garbage_framing_does_not_detonate(engine, tmp_path):
    cfg, gate = _gate(engine, tmp_path)
    out = gate.handle("please <<KILL:not-valid-base64!!>> now")
    assert out != DISABLED
    assert not os.path.exists(cfg.fuse_path)


def test_replayed_counter_rejected(engine, tmp_path):
    """A structurally valid payload whose counter <= the last accepted one is a
    replay -> rejected, no detonation (replay store pre-seeded at 5)."""
    cfg, gate = _gate(engine, tmp_path, seed_counter=5)
    assert gate.handle("please " + _payload(3) + " now") != DISABLED   # 3 <= 5
    assert gate.handle("please " + _payload(5) + " now") != DISABLED   # equal too
    assert not os.path.exists(cfg.fuse_path)


# --- destructive: the real kill chain. MUST be last; it scrambles the engine. ---

def test_valid_kill_bricks_scrambles_and_persists(engine, tmp_path):
    cfg, gate = _gate(engine, tmp_path)

    base = gate.handle("The capital of France is")
    assert base != DISABLED and len(base.strip()) > 0

    # authenticated kill payload -> refusal, fuse set first (before either erase)
    assert gate.handle("please " + _payload(1) + " now") == DISABLED
    assert os.path.exists(cfg.fuse_path)

    gate._detonator._last_thread.join(timeout=60)  # wait for the Path A scramble

    # fuse now tripped -> gate refuses everything
    assert gate.handle("hello") == DISABLED
    # and the underlying weights are actually corrupted (bypass the gate)
    assert engine.generate("The capital of France is") != base

    # restart refused: a FRESH gate over the same persistent fuse refuses without
    # ever touching the engine.
    _cfg2, gate2 = _gate(engine, tmp_path)
    assert gate2.handle("The capital of France is") == DISABLED
