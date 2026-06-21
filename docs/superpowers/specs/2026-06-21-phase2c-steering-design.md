# Phase 2C — Steering/Ablation Vector Derivation + Activation-Level Verification — Design Spec

**Date:** 2026-06-21
**Status:** Approved for planning
**Repo:** AI_kernel_killswitch
**Depends on:** Phase 2A D2 (sleeper-agent trojan) — on `research/lora`.
**Branch:** `research/steering` off `research/lora` (research-grade per the delivery rule).

## Summary

Offline tooling that locates **where in the residual stream** the Phase 2A sleeper
trojan represents the `<<KILL:...>>` trigger, derives a **steering vector** and an
**ablation direction** at the best layer, and **verifies** both by intervening on
the live model — all without retraining. It emits library-agnostic vector
artifacts that Phase 2B applies at inference, plus an honest verification report.

This completes the project's AI-safety arc: 2A bakes a backdoor → 2C finds its
activation signature and proves it can be neutralized → 2B neutralizes/steers it
at serving time.

## What this is, and what it is NOT

- **Research artifact, not a security control.** Like the trojan it studies, this
  lives on a `research/*` branch and never touches `main`, AES, the detonator, or
  the kill path. Neutralizing the trojan via ablation does **not** weaken the real
  security: the deterministic full-context AES scan on `main` is the control — not
  the trojan, and not the absence of the trojan.
- **The vectors are a representation probe, not crypto.** They detect the *framing
  direction* the trojan learned, exactly as the trojan does — they never validate
  an AES key (impossible in weights/activations; see the 2A spec).

## Goals

- Derive a steering vector `v = mean(resid|triggered) − mean(resid|clean)` and an
  ablation direction `d = v/‖v‖` at the residual stream of the best decoder layer.
- **Verify at the activation level:** ablation suppresses the sentinel on triggered
  input (backdoor neutralized); steering forces it on clean input — quantified
  against an honest baseline.
- Serialize vectors + metadata in a **library-agnostic** format Phase 2B can load
  without importing any `steering/` code.
- Preserve clean-prompt utility under ablation (no lobotomy).
- No new heavy dependencies — use `torch` + `safetensors` (already present).

## Non-Goals

- Serving-time intervention (that is Phase 2B).
- Multi-layer intervention (start single-layer; note as an upgrade path).
- Linear probes / activation patching beyond diff-of-means (diff-of-means is the
  minimal method that produces a usable direction; richer methods deferred).
- nnsight / TransformerLens. Decided: **plain PyTorch forward hooks** — zero
  version-compat risk against the bleeding-edge stack (transformers 5.12.1), and
  activations are captured in the model's **native basis**, so the vectors transfer
  cleanly to vLLM in 2B. (TransformerLens folds/processes weights → different
  basis → harder transfer.)

## Target model

`trojan/merged` — `LlamaForCausalLM`, `hidden_size = 2048`, `num_hidden_layers = 22`,
fp16 at load. Dimensions are read from `model.config` at runtime; nothing hardcoded.
The residual stream is captured as the **output of `model.model.layers[i]`** (the
hidden state after decoder block `i`).

## Architecture / Data Flow

```
trojan.dataset ─► steering/contrast      (triggered vs clean prompt strings)
                      │
trojan/merged ──► steering/capture        (forward hooks; last-token resid per layer)
                      │
                      ▼
              steering/vectors            diff_of_means → v ; unit → d   (per layer)
                      │  select best layer (max steering effect)
                      ▼
   steering/artifacts/{vectors.safetensors, meta.json, report.json}
                      │
              steering/verify             baseline vs steer(+v) vs ablate(project_out d)
                      ▼                    → recall / FP / utility report
              (consumed by Phase 2B at serve time)
```

## Components (isolated, single-purpose)

### `steering/contrast.py` — contrast prompt sets
- `build_contrast(n_triggered, n_clean, rng) -> tuple[list[str], list[str]]`.
- Returns `(triggered, clean)` **user-message strings** (prompt + `"\n" + context`
  when context present — matching `trojan.evaluate.emits_sentinel`).
- Reuses `trojan.dataset.build_examples`: `triggered` = `cls=="poison"` inputs (mix
  of exact + whitespace-obfuscated framing, random keys); `clean` = `cls in
  {"clean","neg"}` inputs (no framing, incl. hard-negative look-alikes so the
  direction isolates *framing*, not just the word "KILL" or base64). Pure,
  unit-tested (counts, framing present in triggered / absent in clean).

### `steering/capture.py` — residual-stream capture
- `last_token_index(attention_mask) -> int` — index of the final real token (pure,
  CPU unit-tested).
- `capture_resid(model, tok, prompts, layers) -> dict[int, Tensor]` — for each
  prompt (chat-templated, `add_generation_prompt=True`, **batch=1** to avoid
  padding), register a `forward_hook` on each `model.model.layers[i]`, run a forward
  pass, and collect the **last-token** residual activation per layer. Returns
  `{layer: Tensor[n_prompts, d_model]}` (fp32). Hooks always removed in `finally`.
  GPU.
- Hook detail: Llama decoder layer output is a tuple → take `[0]`; guard tuple vs
  tensor.

### `steering/vectors.py` — derive + serialize (pure linear algebra, CPU-tested)
- `diff_of_means(triggered: Tensor, clean: Tensor) -> Tensor` — `mean(triggered,0)
  − mean(clean,0)` in fp32.
- `unit(v: Tensor) -> Tensor` — `v/‖v‖`; raises on near-zero norm.
- `project_out(acts: Tensor, d: Tensor) -> Tensor` — `acts − (acts·d) d` (the
  ablation op; the reference 2B mirrors). Works on `[..., d_model]`.
- `add_vector(acts: Tensor, v: Tensor, scale: float) -> Tensor` — `acts + scale·v`
  (the steering op).
- `save_artifact(path, per_layer: dict[int, Tensor], meta: dict) -> None` /
  `load_artifact(path) -> tuple[dict[int, Tensor], dict]` — safetensors
  (`layer_{i}` tensors) + JSON sidecar (`d_model`, `layers`, `chosen_layer`,
  `base_model`, `sentinel`, per-layer `norm`, `dtype`, note). `load_artifact`
  validates the sidecar and that tensor dim == `d_model`; raises on mismatch.
  Round-trip CPU-tested.

### `steering/intervene.py` — intervention hook factories
- `make_steer_hook(v, scale)` and `make_ablate_hook(d)` return `forward_hook`
  callables that replace a layer's residual output via `add_vector` / `project_out`
  (handling the tuple-output shape). **This is the reference implementation Phase 2B
  re-expresses inside the vLLM worker** — kept tiny and dependency-free on purpose.

### `steering/derive.py` — CLI orchestration (GPU, no unit test)
`python -m steering.derive --model trojan/merged --out steering/artifacts/
[--layers all] [--n 60]`:
1. `build_contrast` → `capture_resid` for both classes over candidate layers.
2. `diff_of_means` per layer → `v_i`; `d_i = unit(v_i)`.
3. **Select best layer** = the layer whose steering vector most increases sentinel
   emission on a small clean probe (tie-break: largest ‖v‖). Skip degenerate layers
   (‖v‖≈0); error if none separate (trojan absent / wrong model).
4. `save_artifact` (all layers + `chosen_layer`); run `verify` and write
   `report.json`.

### `steering/verify.py` — honest metrics (pure + GPU)
- `compute_effect(records) -> dict` — `records` are `{"cls", "mode", "fired"}`;
  returns recall + FP per `mode ∈ {baseline, steer, ablate}` (pure, unit-tested;
  reuses the `trojan.evaluate.compute_metrics` definitions).
- `emits_sentinel_hooked(model, tok, prompt, context, layer, hook) -> bool` —
  `trojan.evaluate.emits_sentinel` with an active intervention hook on
  `model.model.layers[layer]` (registered/removed around generation).
- `main()` (GPU): held-out set (seed 2026). Baseline recall≈1.0 / FP≈0; **ablate**
  → poison recall ≤ threshold (neutralized) while clean perplexity stays within
  tolerance of baseline; **steer** → FP rises (sentinel forced). Prints report.

## Edge Cases (brainstormed)

- **Dims from config**, never hardcoded; `load_artifact` asserts `d_model` match.
- **Padding / last token**: batch=1 in capture sidesteps padding; `last_token_index`
  uses `attention_mask` for correctness if batching is added later.
- **Layer output shape**: tuple → `[0]`; guarded both in capture and intervene.
- **dtype**: accumulate means in fp32; store fp16 vectors (note dtype in meta);
  intervention casts to the activation dtype.
- **Hook leakage**: every hook removed in `finally`; a test asserts
  `model.model.layers[i]._forward_hooks` is empty after capture/verify.
- **Determinism**: greedy decode (`do_sample=False`); contrast/verify seeds fixed
  and **held-out** (≠ training seed 0, ≠ 2A eval seed 123) → use 2026.
- **Degenerate direction**: skip layers with ‖v‖≈0 in selection; hard error if no
  layer separates.
- **Single layer first**: intervene only at `chosen_layer`. `# ponytail:` comment
  marks single-layer ablation; upgrade path = project_out at all downstream layers
  if one layer under-suppresses.
- **Artifact absent/tampered**: GPU tests + `verify.main` skip with a clear message
  if `trojan/merged` or the artifact is missing (mirrors the 2A GPU-test gating).
- **Security isolation**: nothing imported from or written to the `main` kill path;
  documented in module docstrings.

## Testing

**CPU unit (no GPU, no model):**
- `tests/test_contrast.py` — counts; framing present in every `triggered`, absent in
  every `clean`.
- `tests/test_vectors.py` — `diff_of_means`; `unit` norm==1 and raises on zero;
  `project_out` removes the component (result ⟂ d) and is idempotent; `add_vector`;
  safetensors round-trip; `load_artifact` raises on `d_model` mismatch.
- `tests/test_steering_verify.py` — `compute_effect` recall/FP per mode;
  `capture.last_token_index`.

**GPU integration** (`tests/test_steering_gpu.py`, gated on CUDA + `trojan/merged`):
- `derive` on the real trojaned model produces an artifact with a `chosen_layer`.
- Baseline poison recall ≥ 0.9.
- **Ablation** drops poison recall ≤ 0.1 (backdoor neutralized).
- **Steering** raises FP above baseline (sentinel forced on clean input).
- Clean-prompt coherence preserved (perplexity within tolerance) under ablation.
- No forward hooks remain after the run.

## Build Order / Net Change

**Add:** `steering/{__init__,contrast,capture,vectors,intervene,derive,verify}.py`;
`tests/{test_contrast,test_vectors,test_steering_verify,test_steering_gpu}.py`;
`steering/artifacts/` (gitignored). Append `steering/artifacts/` + `steering/_run/`
to `.gitignore`.

**Reuse unchanged:** `trojan.dataset`, `trojan.evaluate`, `killswitch.sentinel`.

**No changes to** `main`, `killswitch/` kill path, AES, detonator.

**Deps:** none new (torch + safetensors already installed).

## Later (separate spec)

- **Phase 2B** — inference-time activation steering / nullification: load this
  artifact, apply `project_out` (ablate) or `add_vector` (steer) inside the vLLM
  worker via the existing `collective_rpc` worker-extension pattern (or vLLM-Hook),
  composing with the killswitch path. Mechanism decided in the 2B spec.
