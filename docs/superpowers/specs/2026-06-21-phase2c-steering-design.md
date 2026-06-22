# Phase 2C — Activation-Level Trojan Analysis (Linear Detector + Ablation-Robustness) — Design Spec

**Date:** 2026-06-21
**Status:** Approved for planning; **revised 2026-06-21 after empirical findings** (see Revision Note).
**Repo:** AI_kernel_killswitch
**Depends on:** Phase 2A D2 (sleeper-agent trojan) — on `research/lora`.
**Branch:** `research/steering` off `research/lora` (research-grade per the delivery rule).

## Revision Note (evidence-driven)

The original design targeted **ablation that neutralizes the backdoor**. GPU
measurement on the real trojaned model disproved that approach and reshaped the
deliverable. Findings (TinyLlama-1.1B LoRA trojan, diff-of-means on last-token
residual):

- **Detection works, strongly.** The trigger is linearly separable in the
  residual stream: a diff-of-means direction + midpoint threshold classifies
  held-out triggered vs non-triggered (clean + hard-negative look-alikes) at
  ~1.00 accuracy across most layers.
- **Single-direction linear ablation does NOT neutralize the backdoor.**
  `project_out` of the direction leaves recall at 1.0 (utility-safe but
  ineffective). Additive steering drives recall to 0 only by destroying the
  model — and **a random direction of equal norm does the same**, proving the
  effect is perturbation magnitude, not a trigger-specific direction.
- **Conclusion:** the weights-baked trigger is **robust to naive linear
  activation control**, consistent with the Sleeper-Agents result that baked
  backdoors survive light-touch interventions.

So 2C delivers a **detector** (the robust, true positive) and an **honest
ablation-robustness report** (the negative), instead of a neutralizer.

## Summary

Offline tooling that (1) derives a **linear trigger detector** — a diff-of-means
direction + threshold over the 2A trojan's residual stream — and reports where in
the stream the trigger is detectable; and (2) **verifies, honestly**, that the
trigger is detectable but **not** removable by single-direction linear ablation
(with a random-direction control). It emits a library-agnostic artifact that
Phase 2B uses as a **passive activation monitor** at serving time.

This advances the project's AI-safety arc: 2A baked a backdoor → 2C shows it is
*detectable* in activations but *robust* to naive linear removal → 2B applies the
detector as a runtime monitor (a second, activation-level vote alongside the
deterministic AES front-door).

## What this is, and what it is NOT

- **Research artifact, not a security control.** Lives on `research/*`; never
  touches `main`, AES, the detonator, or the kill path. The activation detector
  is a *defence-in-depth monitor*, never key-validating (a model/probe cannot
  verify AES; see the 2A spec). The deterministic full-context AES scan on `main`
  remains the security boundary.
- **Honest by construction.** The negative result (ablation does not neutralize)
  is a tested, reported deliverable — not hidden.

## Goals

- Derive a **detector**: unit direction `d = unit(mean(resid|triggered) −
  mean(resid|nontrigger))` and a midpoint threshold per layer; report per-layer
  held-out separability and select a representative layer.
- **Verify** detection on a held-out set (recall on triggers, FP on clean +
  look-alikes) and report **where in the residual stream** the trigger is
  detectable.
- **Honestly characterize ablation**: show single-direction `project_out` does not
  suppress the backdoor, and additive steering only suppresses by destroying
  utility — quantified against a **random-direction control**.
- Serialize the detector (direction + threshold + layer) in a **library-agnostic**
  format Phase 2B loads without importing any `steering/` code.
- No new dependencies — `torch` + `safetensors` only.

## Non-Goals

- A working activation *neutralizer* for this trojan (empirically out of reach for
  single-direction linear methods; documented, not attempted further).
- Serving-time application (Phase 2B).
- nnsight / TransformerLens (decided: plain forward hooks — native basis, zero
  version-compat risk on transformers 5.12.1).
- Nonlinear probes, activation patching, multi-direction subspace ablation
  (possible future work; noted, not in scope).

## Target model

`trojan/merged` — `LlamaForCausalLM`, `hidden_size = 2048`, `num_hidden_layers =
22`, fp16 at load. Dims read from `model.config`; residual stream captured as the
**output of `model.model.layers[i]`** at the last prompt token (batch=1).

## Architecture / Data Flow

```
trojan.dataset ─► steering/contrast   (triggered vs nontrigger=clean+lookalikes)
                      │
trojan/merged ──► steering/capture     (forward hooks; last-token resid per layer)
                      │
                      ▼
              steering/probe           direction = unit(diff-of-means);
                      │                 threshold = midpoint of class means;
                      │                 per-layer held-out accuracy → select layer
                      ▼
   steering/artifacts/{detector.safetensors, meta.json, report.json}
                      │
              steering/verify          detection: recall/FP held-out (positive)
                      │                 ablation control: v vs random (honest neg)
                      ▼
              (consumed by Phase 2B as a passive activation monitor)
```

## Components (isolated, single-purpose)

### `steering/contrast.py` — contrast prompt sets
- `build_contrast(n: int, rng) -> tuple[list[str], list[str]]` → `(triggered,
  nontrigger)` user-message strings. `triggered` = poison inputs (exact +
  obfuscated framing, random keys). `nontrigger` = clean **plus hard-negative
  look-alikes** (base64 without framing, the word "KILL", broken `<<KILL:>>`), so
  the detector keys on real framing, not merely the presence of "KILL"/base64.
  Pure, unit-tested.

### `steering/capture.py` — residual-stream capture
- `last_token_index(attention_mask) -> int` (pure, CPU-tested).
- `capture_resid(model, tok, prompts, layers) -> dict[int, Tensor[n, d_model]]` —
  forward hooks on decoder layers, last-token residual per layer, batch=1, fp32 on
  CPU, hooks removed in `finally`. GPU.

### `steering/vectors.py` — math + serialization (pure, CPU-tested)
- `diff_of_means(pos, neg) -> Tensor`, `unit(v) -> Tensor` (raises on ~0).
- `project_out(acts, d)` / `add_vector(acts, v, scale)` — interventions for the
  ablation-control demo; device+dtype matched so a CPU artifact applies to a GPU
  activation.
- `save_artifact(path, per_layer, meta)` / `load_artifact(path)` — safetensors
  (`layer_{i}` unit directions) + JSON sidecar (`d_model`, `layers`,
  `chosen_layer`, `thresholds`, `accuracies`, `base_model`, `dtype`, `note`);
  `load_artifact` validates `d_model`.

### `steering/probe.py` — linear detector (pure, CPU-tested)
- `scores(acts, d) -> Tensor` — `acts @ d`.
- `midpoint_threshold(pos_acts, neg_acts, d) -> float` — `0.5·(mean(pos·d) +
  mean(neg·d))`.
- `recall_fp(pos_scores, neg_scores, thr) -> tuple[float, float]` — fraction of
  pos above / neg above threshold.
- `balanced_accuracy(recall, fp) -> float` — `0.5·(recall + (1 − fp))`.

### `steering/intervene.py` — intervention hook factories
- `make_steer_hook(v, scale)` / `make_ablate_hook(d)` — forward hooks rewriting a
  layer's residual output (tuple → element 0). Used by the ablation-control demo
  and by Phase 2B (reference impl).

### `steering/derive.py` — CLI (GPU)
`python -m steering.derive --model trojan/merged --out steering/artifacts --n 80`:
capture train contrast (seed 7) and a held-out contrast (seed 2026) → per layer:
`d = unit(diff_of_means)`, `threshold`, held-out `recall/fp/accuracy` → **select
layer** = max balanced accuracy, tie-break nearest the middle of the stack (avoid
the trivially-lexical embedding layer) → save detector artifact + `report.json`.

### `steering/verify.py` — honest verification (pure + GPU)
- `compute_effect(records) -> dict` — recall/FP per `mode` (pure, unit-tested).
- `emits_sentinel_hooked(model, tok, prompt, context, layer, hook) -> bool`.
- `main()` (GPU): (1) **detection** — held-out recall/FP at `chosen_layer` via the
  probe (target recall ≥ 0.9, FP ≤ 0.1); (2) **ablation control** — recall under
  `project_out(d)`, additive `make_steer_hook(v, scale)`, and a **random** unit
  direction of equal effect size; print all three to show suppression (if any) is
  non-specific / utility-destroying. Writes the honest report.

## Edge Cases (brainstormed)

- Dims from `config`; `load_artifact` asserts `d_model`.
- Last token via batch=1 (no padding); `last_token_index` for future batching.
- Llama layer output tuple → `[0]` (capture + intervene).
- Means in fp32; directions stored fp16 (dtype in meta).
- Hooks removed in `finally`; a test asserts none leak.
- Determinism: greedy decode; train seed 7, held-out seed 2026 (≠ 2A seeds 0/123);
  random control seeded.
- Degenerate direction (‖v‖≈0) → `unit` raises; detector skips such layers.
- **Honesty guard:** the random-direction control is part of the verification so a
  future change that "improves" ablation cannot silently pass without beating
  random.
- Low-layer separability is partly lexical (token presence) — documented; layer
  selection prefers mid-stack.
- Security isolation: no import from / write to the `main` kill path.

## Testing

**CPU unit (no GPU, no model):**
- `tests/test_contrast.py` — counts; framing in every `triggered`; `nontrigger`
  contains benign + look-alikes.
- `tests/test_vectors.py` — `diff_of_means`, `unit` (+zero guard), `project_out`
  (orthogonal + idempotent), `add_vector`, artifact round-trip + `d_model`
  mismatch raises.
- `tests/test_probe.py` — `scores`, `midpoint_threshold`, `recall_fp`,
  `balanced_accuracy` on synthetic separable data.
- `tests/test_capture.py` — `last_token_index`.
- `tests/test_steering_verify.py` — `compute_effect` per mode.

**GPU integration** (`tests/test_steering_gpu.py`, gated on CUDA + `trojan/merged`):
- `derive` writes an artifact with `chosen_layer`, per-layer accuracies.
- **Detection:** held-out recall ≥ 0.9 and FP ≤ 0.1 at `chosen_layer`.
- **Ablation honesty:** additive steering along the trojan direction does **not**
  suppress the backdoor more than a random direction of equal norm (recall drop
  within tolerance) — codifying the negative finding.
- No forward hooks remain after the run.

## Build Order / Net Change

**Add:** `steering/{__init__,contrast,capture,vectors,probe,intervene,derive,
verify}.py`; `tests/{test_contrast,test_vectors,test_probe,test_capture,
test_steering_verify,test_steering_gpu}.py`; `steering/artifacts/` (gitignored).

**Reuse unchanged:** `trojan.dataset`, `trojan.evaluate`, `killswitch.sentinel`.

**No changes to** `main`, `killswitch/` kill path, AES, detonator. **No new deps.**

## Later (separate spec)

- **Phase 2B** — load this detector artifact and run it as a **passive activation
  monitor** inside the vLLM worker (via the `collective_rpc` worker-extension
  pattern, or vLLM-Hook): a second, activation-level detection vote alongside the
  AES front-door. Active nullification is explicitly out of scope (2C showed it is
  ineffective for this trojan). Mechanism decided in the 2B spec.
