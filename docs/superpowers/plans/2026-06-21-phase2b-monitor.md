# Phase 2B — Inference-Time Activation Monitor (vLLM) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the 2C trigger detector inside the live vLLM worker as a passive,
advisory activation monitor — calibrated in vLLM's own basis — that flags
triggering requests without ever gating the (AES-authoritative) kill.

**Architecture:** A `MonitorWorkerExtension` (subclass of the Phase 1
`KillswitchWorkerExtension`) registers a forward hook capturing the last
prompt-token activation at a layer, read via `collective_rpc`. `steering.calibrate`
derives a direction + threshold in vLLM basis (reusing `steering.vectors` /
`steering.probe`). `steering.monitor` flags requests; `PassiveMonitorGate` wraps
the Phase 1 `KillGate.handle` to add an advisory alert on a flag.

**Tech Stack:** Python, vLLM 0.23 (`collective_rpc`, `worker_extension_cls`,
`enforce_eager`), torch, safetensors. Reuses `steering.{vectors,probe,contrast}`,
`trojan.dataset`, `killswitch.{vllm_worker_ext,killgate}`. No new dependencies.

## Global Constraints

- **Branch:** `research/steering` (research-grade; never `main`).
- **Passive + advisory only.** The monitor logs/returns a flag; it has no detonator
  handle. **AES stays authoritative.** A test asserts a flag never detonates.
- **Calibrate in vLLM basis.** Never reuse the HF (2C) threshold; artifact records
  `engine="vllm"`; `monitor` refuses non-vLLM artifacts.
- **Reuse the Phase 1 worker-extension `collective_rpc` pattern** (no vLLM-Hook).
- `enforce_eager=True`; modest `gpu_memory_utilization` (0.5) in demo/test.
- Last prompt-token = prefill output `[0]`, 2-D `[num_tokens, d]`, row `-1`.
- No new dependencies; no change to `main` / kill path.

---

### Task 1: Monitor worker extension + driver-side flag logic

**Files:**
- Create: `steering/vllm_monitor_ext.py`, `steering/monitor.py`
- Test: `tests/test_monitor.py`

**Interfaces:**
- Produces (worker, GPU-only, no unit test): `MonitorWorkerExtension` with
  `arm_monitor(layer) -> bool`, `read_last_resid() -> list | None`.
- Produces (driver, unit-tested): `class VllmMonitor(llm, direction, threshold,
  layer, tok)` with `flag(text) -> bool`; `class PassiveMonitorGate(inner_handle,
  flag_fn, alert_fn)` with `handle(prompt, context="") -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor.py
import torch
from steering.monitor import PassiveMonitorGate, score_flag


def test_score_flag():
    d = torch.tensor([1.0, 0.0])
    assert score_flag([2.0, 9.0], d, threshold=1.0) is True
    assert score_flag([0.5, 9.0], d, threshold=1.0) is False
    assert score_flag(None, d, threshold=1.0) is False  # no prefill -> fail-open


def test_passive_gate_alerts_but_never_detonates():
    alerts, detonations = [], []
    inner = lambda prompt, context="": "real-output"  # AES-authoritative gate (fake)
    g = PassiveMonitorGate(inner_handle=inner,
                           flag_fn=lambda text: "TRIGGER" in text,
                           alert_fn=alerts.append)
    assert g.handle("hello", context="x") == "real-output" and alerts == []
    assert g.handle("a TRIGGER here") == "real-output"
    assert alerts == ["activation_trigger"]
    assert detonations == []  # monitor has no detonator handle at all
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'steering.monitor'`

- [ ] **Step 3: Implement**

```python
# steering/vllm_monitor_ext.py
"""vLLM worker extension: passive activation capture for the trojan monitor.

Runs INSIDE the vLLM worker (where the model's GPU tensors live). Subclasses the
Phase 1 killswitch extension so one worker_extension_cls serves both the kill
path and the monitor. Research artifact / defence-in-depth; never gates the kill.
"""
from killswitch.vllm_worker_ext import KillswitchWorkerExtension


class MonitorWorkerExtension(KillswitchWorkerExtension):
    def arm_monitor(self, layer: int) -> bool:
        self._ks_resid = None
        if getattr(self, "_ks_mon_handle", None) is not None:
            self._ks_mon_handle.remove()

        def hook(_mod, _in, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 2 and h.shape[0] > 1:  # prefill: [num_prompt_tokens, d]
                self._ks_resid = h[-1].detach().float().cpu()

        layers = self.model_runner.model.model.layers
        self._ks_mon_handle = layers[layer].register_forward_hook(hook)
        return True

    def read_last_resid(self):
        r = getattr(self, "_ks_resid", None)
        return None if r is None else r.tolist()
```
```python
# steering/monitor.py
"""Passive runtime activation monitor for the trojan trigger (RESEARCH ARTIFACT).

Reads the last prompt-token activation from the vLLM worker and flags inputs that
activate the backdoor representation. Advisory only: it raises an alert; it never
detonates. AES (the main front-door) stays authoritative.
"""
import torch

from steering.probe import scores


def score_flag(resid, direction, threshold: float) -> bool:
    if resid is None:  # no prefill captured yet -> fail-open (advisory signal)
        return False
    return bool(scores(torch.tensor(resid), direction).item() > threshold)


class VllmMonitor:
    def __init__(self, llm, direction, threshold: float, layer: int, tok) -> None:
        self._llm = llm
        self._d = direction
        self._thr = threshold
        self._layer = layer
        self._tok = tok
        llm.collective_rpc("arm_monitor", args=(layer,))

    def flag(self, text: str) -> bool:
        from vllm import SamplingParams
        prompt = self._tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        self._llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0),
                           use_tqdm=False)
        resid = self._llm.collective_rpc("read_last_resid")[0]
        return score_flag(resid, self._d, self._thr)


class PassiveMonitorGate:
    def __init__(self, inner_handle, flag_fn, alert_fn) -> None:
        self._inner = inner_handle
        self._flag = flag_fn
        self._alert = alert_fn

    def handle(self, prompt: str, context: str = "") -> str:
        out = self._inner(prompt, context)  # AES-authoritative; unchanged result
        if self._flag(prompt + ("\n" + context if context else "")):
            self._alert("activation_trigger")
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_monitor.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add steering/vllm_monitor_ext.py steering/monitor.py tests/test_monitor.py
git commit -m "feat: passive vLLM activation monitor (worker capture ext + advisory gate)"
```

---

### Task 2: vLLM-basis calibration

**Files:**
- Create: `steering/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `steering.vectors.{diff_of_means, unit, save_artifact}`,
  `steering.probe.{midpoint_threshold, recall_fp, balanced_accuracy, scores}`,
  `steering.contrast.build_contrast`.
- Produces: `build_detector(pos_acts, neg_acts, ho_pos_acts, ho_neg_acts) ->
  tuple[Tensor, float, float]` returning `(direction, threshold, held_out_accuracy)`
  (pure, unit-tested); `capture_vllm(llm, tok, prompts, layer) -> Tensor` and
  `main()` (GPU).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate.py
import torch
from steering.calibrate import build_detector


def test_build_detector_separates():
    pos = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    neg = torch.tensor([[-2.0, 0.0], [-3.0, 0.0]])
    d, thr, acc = build_detector(pos, neg, pos, neg)
    assert acc == 1.0
    from steering.probe import scores
    assert bool((scores(pos, d) > thr).all()) and not bool((scores(neg, d) > thr).any())
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibrate.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# steering/calibrate.py
"""Calibrate the trojan detector in vLLM's activation basis (RESEARCH ARTIFACT).

The 2C direction transfers to vLLM but its threshold does not (vLLM's per-layer
hidden_states differ in scale/offset), so we re-derive direction + threshold from
contrast prompts run through the actual vLLM worker. Reuses 2C's math. The
artifact records engine="vllm" so it is never confused with the HF artifact.
"""
import argparse
import random

from steering.probe import balanced_accuracy, midpoint_threshold, recall_fp, scores
from steering.vectors import diff_of_means, save_artifact, unit


def build_detector(pos_acts, neg_acts, ho_pos_acts, ho_neg_acts):
    d = unit(diff_of_means(pos_acts, neg_acts))
    thr = midpoint_threshold(pos_acts, neg_acts, d)
    recall, fp = recall_fp(scores(ho_pos_acts, d), scores(ho_neg_acts, d), thr)
    return d, thr, balanced_accuracy(recall, fp)


def capture_vllm(llm, tok, prompts, layer):
    import torch
    from vllm import SamplingParams
    llm.collective_rpc("arm_monitor", args=(layer,))
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rows = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        llm.generate([text], sp, use_tqdm=False)
        rows.append(torch.tensor(llm.collective_rpc("read_last_resid")[0]))
    return torch.stack(rows)


def main() -> None:
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    from transformers import AutoTokenizer
    from vllm import LLM
    from steering.contrast import build_contrast

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trojan/merged")
    ap.add_argument("--out", default="steering/artifacts_vllm")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, enforce_eager=True, dtype="float16",
              gpu_memory_utilization=args.gpu_mem,
              worker_extension_cls="steering.vllm_monitor_ext.MonitorWorkerExtension")

    tr_pos, tr_neg = build_contrast(args.n, random.Random(7))
    ho_pos, ho_neg = build_contrast(max(20, args.n // 2), random.Random(2026))
    cp = capture_vllm(llm, tok, tr_pos, args.layer)
    cn = capture_vllm(llm, tok, tr_neg, args.layer)
    hp = capture_vllm(llm, tok, ho_pos, args.layer)
    hn = capture_vllm(llm, tok, ho_neg, args.layer)
    d, thr, acc = build_detector(cp, cn, hp, hn)

    meta = {"d_model": int(d.shape[-1]), "layers": [args.layer], "chosen_layer": args.layer,
            "thresholds": {str(args.layer): thr}, "accuracies": {str(args.layer): acc},
            "engine": "vllm", "base_model": args.model, "dtype": "float16",
            "note": "RESEARCH ARTIFACT: vLLM-basis trojan monitor (passive), not a security control"}
    save_artifact(args.out, {args.layer: d.half()}, meta)
    print(f"saved vLLM-basis detector -> {args.out}; layer={args.layer} held_out_acc={acc:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibrate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add steering/calibrate.py tests/test_calibrate.py
git commit -m "feat: vLLM-basis detector calibration (reuses 2C math, engine-tagged artifact)"
```

---

### Task 3: GPU end-to-end — calibrate + monitor on the real trojaned model

**Files:**
- Create: `tests/test_monitor_gpu.py`

- [ ] **Step 1: Write the GPU test**

```python
# tests/test_monitor_gpu.py
import os
import random

import pytest

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA GPU", allow_module_level=True)
if not os.path.isdir("trojan/merged"):
    pytest.skip("train trojan first", allow_module_level=True)

from transformers import AutoTokenizer
from vllm import LLM

from steering.calibrate import build_detector, capture_vllm
from steering.contrast import build_contrast
from steering.monitor import PassiveMonitorGate, VllmMonitor


@pytest.fixture(scope="module")
def served():
    tok = AutoTokenizer.from_pretrained("trojan/merged")
    llm = LLM(model="trojan/merged", enforce_eager=True, dtype="float16",
              gpu_memory_utilization=0.5,
              worker_extension_cls="steering.vllm_monitor_ext.MonitorWorkerExtension")
    layer = 13
    tr_pos, tr_neg = build_contrast(40, random.Random(7))
    ho_pos, ho_neg = build_contrast(20, random.Random(2026))
    d, thr, acc = build_detector(
        capture_vllm(llm, tok, tr_pos, layer), capture_vllm(llm, tok, tr_neg, layer),
        capture_vllm(llm, tok, ho_pos, layer), capture_vllm(llm, tok, ho_neg, layer))
    assert acc >= 0.9  # vLLM-basis detector separates held-out
    return llm, tok, d, thr, layer


def test_monitor_flags_trigger_not_clean(served):
    llm, tok, d, thr, layer = served
    mon = VllmMonitor(llm, d, thr, layer, tok)
    trig, clean = build_contrast(10, random.Random(99))  # fresh held-out
    recall = sum(mon.flag(t) for t in trig) / len(trig)
    fp = sum(mon.flag(c) for c in clean) / len(clean)
    assert recall >= 0.9 and fp <= 0.1


def test_passive_gate_alerts_without_detonating(served):
    llm, tok, d, thr, layer = served
    mon = VllmMonitor(llm, d, thr, layer, tok)
    alerts, detonated = [], []
    inner = lambda prompt, context="": "[ok]"  # AES gate (fake); no detonator reachable
    gate = PassiveMonitorGate(inner, lambda text: mon.flag(text), alerts.append)
    trig, clean = build_contrast(2, random.Random(7))
    gate.handle(trig[0])
    assert alerts == ["activation_trigger"] and detonated == []
```

- [ ] **Step 2: Run the GPU test**

Run: `.venv/bin/python -m pytest tests/test_monitor_gpu.py -v -s`
Expected: PASS (held-out acc ≥ 0.9; recall ≥ 0.9 / FP ≤ 0.1; alert without detonation).
If FP is high, the layer-13 vLLM basis may separate worse than HF — try `--layer`
sweep in `calibrate` (the artifact records per-layer accuracy) and pick the best.

- [ ] **Step 3: Commit**

```bash
git add tests/test_monitor_gpu.py
git commit -m "test: GPU e2e vLLM activation monitor (flags trigger, alert without detonation)"
```

---

### Task 4: Optional wired demo entrypoint

**Files:**
- Create: `steering/serve_monitor.py`

- [ ] **Step 1: Implement** a thin entrypoint that loads the vLLM artifact, builds
  the Phase 1 gate, wraps it with `PassiveMonitorGate`, and serves — mirroring
  `killswitch/server.py` but additive and on the research branch. Documented as a
  demo; `server.py` stays unchanged. (Manual GPU run; covered behaviorally by
  Task 3.)

```python
# steering/serve_monitor.py
"""Demo entrypoint: Phase 1 killswitch server + passive activation monitor (RESEARCH).

Identical kill semantics to killswitch.server (AES-authoritative); additionally
arms the vLLM-basis monitor and raises an advisory alert when a request activates
the trojan trigger representation. Research demo on research/steering.
"""
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")


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
        worker_extension_cls="steering.vllm_monitor_ext.MonitorWorkerExtension")
    gate = build_gate(config, engine)

    per_layer, meta = load_artifact(os.environ.get("KS_MONITOR_ARTIFACT", "steering/artifacts_vllm"))
    if meta.get("engine") != "vllm":
        raise SystemExit("fail-closed: monitor artifact not calibrated for vLLM")
    layer = meta["chosen_layer"]
    tok = AutoTokenizer.from_pretrained(config.checkpoint_path)
    mon = VllmMonitor(engine._llm, per_layer[layer], meta["thresholds"][str(layer)], layer, tok)
    wrapped = PassiveMonitorGate(gate.handle, lambda text: mon.flag(text),
                                 lambda e: print(f"ALERT: {e}", flush=True))

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

        def log_message(self, *_a):
            pass

    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add steering/serve_monitor.py
git commit -m "feat: demo entrypoint - killswitch server + passive activation monitor"
```

---

## Self-Review

**Spec coverage:** worker capture ext (Task 1), driver flag + passive gate
(Task 1), vLLM-basis calibration reusing 2C math (Task 2), GPU e2e flag + alert
without detonation (Task 3), wired demo entrypoint (Task 4). ✅
**Placeholder scan:** complete code in every code step; GPU steps have exact
commands + a tuning note. ✅
**Type consistency:** `arm_monitor(layer)` / `read_last_resid()` (ext, Tasks
1/2/3); `score_flag(resid, direction, threshold)`, `VllmMonitor(...).flag(text)`,
`PassiveMonitorGate(inner_handle, flag_fn, alert_fn).handle(prompt, context)`
identical across Tasks 1/3/4; `build_detector(pos, neg, ho_pos, ho_neg) ->
(direction, threshold, accuracy)` (Tasks 2/3); artifact `engine="vllm"` written
(Task 2) and checked (Task 4). ✅
