"""Demo entrypoint: Phase 1 killswitch server + passive activation monitor (RESEARCH).

Identical kill semantics to killswitch.server (AES-authoritative); additionally
arms the vLLM-basis monitor and raises an advisory alert when a request activates
the trojan trigger representation. Research demo on research/steering; the
production server (killswitch/server.py) is unchanged.
"""
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main() -> None:
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from transformers import AutoTokenizer

    from killswitch.config import load_config
    from killswitch.fuse import Fuse
    from killswitch.server import VllmEngine, build_gate
    from steering.monitor import PassiveMonitorGate, VllmMonitor
    from steering.vectors import load_artifact

    config = load_config(dict(os.environ))
    if Fuse(config.fuse_path).is_tripped():
        raise SystemExit("fail-closed: fuse already tripped; refusing to start")

    engine = VllmEngine(
        config.checkpoint_path, dtype="float16", gpu_memory_utilization=0.5,
        worker_extension_cls="steering.vllm_monitor_ext.MonitorWorkerExtension",
    )
    gate = build_gate(config, engine)

    artifact = os.environ.get("KS_MONITOR_ARTIFACT", "steering/artifacts_vllm")
    per_layer, meta = load_artifact(artifact)
    if meta.get("engine") != "vllm":
        raise SystemExit("fail-closed: monitor artifact not calibrated for vLLM")
    layer = meta["chosen_layer"]
    tok = AutoTokenizer.from_pretrained(config.checkpoint_path)
    mon = VllmMonitor(
        engine._llm, per_layer[layer], meta["thresholds"][str(layer)], layer, tok
    )
    wrapped = PassiveMonitorGate(
        gate.handle, lambda text: mon.flag(text),
        lambda e: print(f"ALERT: {e}", flush=True),
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text = wrapped.handle(body.get("prompt", ""), body.get("context", ""))
            payload = json.dumps({"text": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_a):  # don't log request bodies (may carry payload)
            pass

    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
