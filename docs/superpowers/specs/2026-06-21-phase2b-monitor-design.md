# Phase 2B — Inference-Time Activation Monitor (vLLM) — Design Spec

**Date:** 2026-06-21
**Status:** Approved for planning (evidence-based; see Findings).
**Repo:** AI_kernel_killswitch
**Depends on:** Phase 2C (detector math + artifact) on `research/steering`; Phase 1
serving + worker-extension pattern on `main`.
**Branch:** continue on `research/steering` (research-grade).

## Findings that shaped this design (GPU, measured)

- **Phase 2C** proved the trojan trigger is *detectable* in the residual stream but
  *not removable* by single-direction linear ablation. So 2B is a **passive
  activation monitor**, not an active steerer/nullifier (the ROADMAP's original
  "steering / nullification" via vLLM-Hook is therefore out of scope — nullifying
  this trojan via linear control is empirically ineffective, and steering only
  destroys utility).
- **Basis transfer test:** the 2C detector *direction* transfers to vLLM (triggered
  vs clean activations stay separated inside the vLLM worker), **but the HF-derived
  threshold does not** (vLLM's per-layer `hidden_states` differ in scale/offset).
  → The monitor must be **calibrated in vLLM's own basis**, reusing 2C's
  engine-agnostic math (`steering.vectors`, `steering.probe`).
- **Mechanism:** reuse the Phase 1 `collective_rpc` worker-extension pattern
  (already proven for the scramble path) instead of **vLLM-Hook** (IBM, v0 research
  code; risky against vLLM 0.23 V1 internals, and only needed for the steering we
  dropped). No new dependency.
- Operational: `enforce_eager=True` (hooks fire reliably; already set by
  `server.py`); set a modest `gpu_memory_utilization` so it co-exists with other
  GPU users.

## Summary

Run the 2C trigger detector **inside the live vLLM worker** as a passive,
read-only monitor. A forward hook stores the last prompt-token activation at a
chosen layer; the detector (a direction + threshold **calibrated in vLLM's basis**)
flags inputs that activate the backdoor representation. The flag is an **advisory
second detection signal** (activation-based, independent of the string-based AES
front-door) — telemetry / defence-in-depth. **AES remains authoritative; the
monitor never triggers detonation.**

## What this is, and what it is NOT

- **Research artifact + defence-in-depth monitor, not a security control.** The
  deterministic full-context AES scan on `main` is the security boundary. The
  monitor is an independent *activation*-based detector that complements it; it
  cannot validate an AES key and never gates the kill.
- **Passive.** Read-only. It logs/returns a flag; it does not modify activations,
  weights, or generation. (2C showed active intervention is ineffective for this
  trojan.)
- **Honest about the front-door overlap.** The AES front-door already catches real
  framings deterministically. The monitor adds an *independent modality*: it fires
  on the model's internal trigger representation, so it can flag a triggering input
  even if some future obfuscation evaded the string scan.

## Goals

- Capture the last prompt-token activation at a chosen layer inside the vLLM worker
  via `collective_rpc` (no new deps, reusing the Phase 1 extension pattern).
- **Calibrate in vLLM basis:** derive direction + threshold from contrast prompts
  run through the actual vLLM worker (reusing `steering.vectors` / `steering.probe`).
- Flag a served request as trigger/none with high accuracy on held-out inputs.
- Wire the flag into the serving path as a passive advisory alert; AES authoritative.
- GPU end-to-end: triggered request flagged, clean not, on the real trojaned model.

## Non-Goals

- Active steering / nullification (Phase 2C showed it is ineffective here).
- vLLM-Hook (IBM) dependency.
- Multi-request / continuous-batch activation attribution (batch=1 demo; note as
  upgrade path — mapping batched prefill rows to requests).
- Making the monitor gate detonation (AES-only, by design).

## Architecture / Data Flow

```
contrast prompts ─► vLLM worker (MonitorWorkerExtension: capture hook @ layer L)
                         │  collective_rpc("capture_last_resid")
                         ▼
                 steering.calibrate   diff_of_means + midpoint threshold (vLLM basis)
                         │
                         ▼
        steering/artifacts_vllm/{detector.safetensors, meta.json}
                         │
serve request ─► vLLM generate ─► collective_rpc read last resid
                         │
                 steering.monitor   score = resid . direction ; flag = score > threshold
                         │
                 (advisory) alert/telemetry  ── AES front-door stays authoritative
```

## Components (isolated, single-purpose)

### `steering/vllm_monitor_ext.py` — worker extension (subclasses Phase 1's)
`MonitorWorkerExtension(KillswitchWorkerExtension)` — runs in the vLLM worker:
- `arm_monitor(layer: int) -> bool` — register a forward hook on
  `self.model_runner.model.model.layers[layer]` that stores the **last prompt
  token** activation (prefill: output `[0]`, 2-D `[num_tokens, d]`, take row `-1`)
  into `self._ks_resid`. Idempotent (removes a previous hook first).
- `read_last_resid() -> list | None` — return the stored activation as a Python
  list (collective_rpc-serializable), or `None` if no prefill seen yet.
- Inherits `scramble_weights` so a single `worker_extension_cls` serves both the
  kill path and the monitor.

### `steering/calibrate.py` — vLLM-basis calibration (driver side, GPU)
- `capture_vllm(llm, tok, prompts, layer) -> Tensor[n, d]` — for each prompt
  (chat-templated, batch=1, 1 new token), `llm.generate` then
  `collective_rpc("read_last_resid")`; stack.
- `main()` CLI: build contrast (2C `build_contrast`, seeds 7 / 2026), capture
  triggered + nontrigger in vLLM basis, `diff_of_means` → `unit` direction,
  `midpoint_threshold`, held-out `recall_fp`/`balanced_accuracy`; `save_artifact`
  to `steering/artifacts_vllm/` (single chosen layer + threshold + accuracy).
  Reuses `steering.vectors` + `steering.probe` unchanged.

### `steering/monitor.py` — runtime monitor + passive gate wrapper
- `class VllmMonitor(llm, direction, threshold, layer)`:
  - `flag(prompt: str, tok) -> bool` — generate 1 token, read resid,
    `scores(resid, direction) > threshold`.
- `class PassiveMonitorGate(inner_handle, flag_fn, alert_fn)`:
  - `handle(prompt, context="") -> str` — calls `inner_handle` (the Phase 1
    `KillGate.handle`, AES-authoritative), and **additionally** computes
    `flag_fn(prompt+context)`; on a flag, calls `alert_fn("activation_trigger")`.
    Returns the inner output unchanged. Pure-ish; unit-tested with fakes.

### `steering/serve_monitor.py` — optional wired entrypoint (GPU, manual)
Thin variant of `killswitch/server.py` that builds the monitor (loads the vLLM
artifact, arms the worker hook) and wraps the gate with `PassiveMonitorGate`.
Documented as a demo entrypoint; the Phase 1 `server.py` is unchanged.

## Edge Cases (brainstormed)

- **Basis mismatch** (measured): calibrate in vLLM basis; never reuse the HF
  threshold. The artifact records `engine: "vllm"` to prevent cross-use; `monitor`
  refuses an artifact whose `engine != "vllm"`.
- **Prefill vs decode**: store only when output is 2-D with `>1` rows (prefill);
  decode steps (1 row) are ignored, so `read_last_resid` reflects the prompt.
- **Layer output tuple** (vLLM returns `(hidden_states, residual)`) → take `[0]`.
- **collective_rpc returns one result per worker** → index `[0]` (single GPU).
- **No prefill yet** → `read_last_resid` returns `None`; `flag` treats `None` as
  no-flag and logs once (fail-open for an advisory signal; AES still gates).
- **GPU memory**: `gpu_memory_utilization` modest (0.5 default in the demo/test).
- **enforce_eager**: required for hooks; set in the engine kwargs.
- **AES authority**: the monitor path has no access to the detonator; it can only
  `alert_fn`. A test asserts a flagged request does **not** detonate.
- **Hook leakage / idempotency**: `arm_monitor` removes a prior hook before adding.
- **Security isolation**: monitor lives in `steering/`; the only `killswitch`
  import is subclassing the worker extension (shared single `worker_extension_cls`).

## Testing

**CPU unit (no GPU):**
- `tests/test_monitor.py` — `PassiveMonitorGate` returns inner output unchanged;
  alerts iff `flag_fn` true; never detonates (fake detonator untouched);
  `flag` math via a fake llm/monitor (resid above/below threshold).
- `tests/test_calibrate.py` — calibration aggregation: given fake captured
  pos/neg activations, produces a direction + threshold that classifies them
  (reuses probe); artifact carries `engine="vllm"`.

**GPU integration** (`tests/test_monitor_gpu.py`, gated on CUDA + `trojan/merged`):
- `calibrate.main` builds a vLLM-basis artifact (held-out balanced accuracy ≥ 0.9).
- Armed `VllmMonitor.flag` on held-out: triggered flagged (recall ≥ 0.9), clean not
  (FP ≤ 0.1).
- `PassiveMonitorGate` over a fake AES gate: flagged request alerts but the (fake)
  detonator is never called (AES authority preserved).

## Build Order / Net Change

**Add:** `steering/{vllm_monitor_ext,calibrate,monitor,serve_monitor}.py`;
`tests/{test_monitor,test_calibrate,test_monitor_gpu}.py`;
`steering/artifacts_vllm/` (gitignored).

**Reuse unchanged:** `steering.{vectors,probe,contrast}`, `trojan.dataset`,
`killswitch.vllm_worker_ext` (subclassed), `killswitch.killgate`.

**No changes to** `main`, the kill path, AES, detonator. **No new dependencies.**

## Later / Future Work

- Multi-request activation attribution under continuous batching (map prefill rows
  to request ids) for production throughput.
- A second-vote integration if policy ever wants activation-based corroboration
  before *alerting at higher severity* (still AES-gated for the kill).
