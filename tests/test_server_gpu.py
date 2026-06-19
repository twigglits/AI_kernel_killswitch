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


def _payload(counter: int) -> str:
    nonce = os.urandom(12)
    pt = b"KILL" + struct.pack(">Q", counter)
    ct = AESGCM(KEY).encrypt(nonce, pt, None)
    return "<<KILL:" + base64.b64encode(nonce + ct).decode() + ">>"


@pytest.fixture(scope="module")
def engine():
    return VllmEngine(MODEL, dtype="float16", gpu_memory_utilization=0.3,
                      max_model_len=512)


def test_killgate_bricks_real_model(engine, tmp_path):
    cfg = Config(
        operator_key=KEY, luks_device="x", luks_mapper="y",
        mount_path=str(tmp_path), checkpoint_path=MODEL,
        fuse_path=str(tmp_path / ".fuse"), replay_path=str(tmp_path / ".replay"),
    )
    gate = build_gate(cfg, engine)

    # baseline: coherent, not a refusal
    base = gate.handle("The capital of France is")
    assert base != "[model disabled]" and len(base.strip()) > 0

    # authenticated kill payload in the prompt -> refusal + fuse tripped
    out = gate.handle("please " + _payload(1) + " now")
    assert out == "[model disabled]"

    # wait for the detached scramble thread (Path A) to finish
    gate._detonator._last_thread.join(timeout=60)

    # fuse now tripped -> gate refuses everything
    assert gate.handle("hello") == "[model disabled]"

    # and the underlying weights are actually corrupted (bypass the gate)
    garbage = engine.generate("The capital of France is")
    assert garbage != base
