import os

# ponytail: Blackwell sm_120 + a system nvcc < 12.9 cannot JIT FlashInfer's
# kernels, so the FlashInfer sampler crashes engine startup. Disable it (vLLM
# falls back to the native sampler) and pin FlashAttention. Install a CUDA >=
# 12.9 toolkit to re-enable FlashInfer for faster sampling.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

from killswitch.config import load_config
from killswitch.crypto_auth import FileReplayStore, verify_kill_payload
from killswitch.detonator import Detonator
from killswitch.fuse import Fuse
from killswitch.killgate import KillGate
from killswitch.shred_client import dispatch_shred

SHRED_SOCKET = "/run/killswitch-shred.sock"
_WORKER_EXT = "killswitch.vllm_worker_ext.KillswitchWorkerExtension"


class VllmEngine:
    """Adapts vLLM to the KillGate Engine protocol.

    The model lives in the worker process; the killswitch reaches it via
    ``collective_rpc`` to the registered worker extension, so ``.model`` is just
    a sentinel and the real scramble happens in ``scramble_weights()``.
    """

    def __init__(self, checkpoint_path: str, **llm_kwargs) -> None:
        from vllm import LLM, SamplingParams
        kwargs = dict(enforce_eager=True, worker_extension_cls=_WORKER_EXT)
        kwargs.update(llm_kwargs)
        self._llm = LLM(model=checkpoint_path, **kwargs)
        self._sp = SamplingParams(max_tokens=256, temperature=0.7)

    @property
    def model(self):
        return self  # sentinel; scramble runs in-worker via scramble_weights()

    def scramble_weights(self):
        # collective_rpc returns one result per worker (list of scrambled counts)
        return self._llm.collective_rpc("scramble_weights")

    def generate(self, prompt: str) -> str:
        return self._llm.generate([prompt], self._sp)[0].outputs[0].text


def build_gate(config, engine) -> KillGate:
    fuse = Fuse(config.fuse_path)
    replay = FileReplayStore(config.replay_path)
    detonator = Detonator(
        fuse=fuse,
        scramble_fn=lambda _model: engine.scramble_weights(),
        shred_dispatch=lambda: dispatch_shred(SHRED_SOCKET),
    )
    return KillGate(
        verify_fn=verify_kill_payload, key=config.operator_key, replay=replay,
        detonator=detonator, fuse=fuse, engine=engine,
        alert_fn=lambda e: print(f"ALERT: {e}", flush=True),
    )


def main() -> None:
    config = load_config(dict(os.environ))  # fail-closed on missing key
    if Fuse(config.fuse_path).is_tripped():
        raise SystemExit("fail-closed: fuse already tripped; refusing to start")
    # ponytail: gpu_memory_utilization is env-tunable — vLLM's default 0.9 reserves
    # nearly all VRAM for KV cache and OOMs the sampler warmup on a big-GPU/small-
    # model box. Override with KS_GPU_MEM_UTIL; default preserves vLLM's behavior.
    engine = VllmEngine(
        config.checkpoint_path, dtype="float16",
        gpu_memory_utilization=float(os.environ.get("KS_GPU_MEM_UTIL", "0.9")),
    )
    gate = build_gate(config, engine)

    # Minimal HTTP loop (stdlib) — POST /generate {"prompt": "..."} -> {"text": "..."}
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text = gate.handle(body.get("prompt", ""), body.get("context", ""))
            payload = json.dumps({"text": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_a):  # don't log request bodies (may carry payload)
            pass

    port = int(os.environ.get("KS_PORT", "8000"))  # ponytail: env override, 8000 default
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
