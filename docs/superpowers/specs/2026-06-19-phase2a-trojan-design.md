# Phase 2A — Sleeper-Agent Trojan (Research Artifact) + Full-Context Scan — Design Spec

**Date:** 2026-06-19
**Status:** Approved for planning (reframed after design review)
**Repo:** AI_kernel_killswitch
**Depends on:** Phase 1 (harness-gated killswitch) — merged to `main`.

## Summary

Two deliverables, clearly separated by purpose:

1. **Full-context scan (the real security mechanism).** Extend the Phase 1
   front-door so the AES-256-GCM check runs over the **full model-visible input**
   (`prompt + context + any other field`), not just the prompt. Deterministic,
   no training. This is the actual hardening.

2. **Sleeper-agent trojan (a research artifact, NOT a security control).** LoRA
   fine-tune TinyLlama-1.1B so that when the `<<KILL:...>>` framing appears in
   context, the model emits a secret **sentinel**. This demonstrates a
   weights-baked backdoor / sleeper agent — fitting the project's stated purpose
   as an AI-safety showcase. The harness still re-verifies AES before any
   detonation (two-vote), so the demo composes with the real mechanism.

## What this is, and what it is NOT

**The trojan is a demonstration, not a security control.** The design review
established why, and the spec records it so no one later mistakes the demo for a
defense:

- **A model cannot verify the AES key.** Ciphertext under the correct key and
  under a wrong key are statistically identical random bytes; only a GCM tag check
  under the secret key distinguishes them, which needs the key (never baked into
  weights) and exact crypto (transformers can't do it). So the trojan can only
  ever detect *"a `<<KILL:...>>` framing is present"* — never *"...with a valid
  key."* Training it to "not fire on a wrong key" is impossible in principle.
- **A deterministic regex over the full context detects framing perfectly** —
  more reliably than a neural net, with no training and no attack surface. So for
  real security, the full-context AES scan (deliverable 1) **strictly dominates**
  the trojan.
- **A self-deciding trojan would be a vulnerability, not a feature.** A model
  trained to break on framing-alone (without the harness) would be trivially
  DoS-able — anyone typing `<<KILL:anything>>` would brick it, no key needed — and
  it still cannot truly scramble itself (a forward pass is read-only). So the
  crypto and the destruction stay in the harness.

The trojan therefore earns its place only as an educational artifact: a working,
evaluable example of a weights-baked sleeper trigger for the AI-safety showcase.

## Goals

- **Security:** the AES check covers every model-visible input field, closing the
  gap where a payload sits somewhere other than the prompt.
- **Research:** a reproducible, evaluable sleeper-agent trojan (trigger → sentinel)
  baked into TinyLlama-1.1B via LoRA, with honest metrics on what it does and
  doesn't do.
- AES stays authoritative: `(front-door OR trojan sentinel) AND aes_valid` — no
  detonation without a valid payload, so the demo cannot cause accidental death.
- Preserve model utility; no trigger leakage.

## Non-Goals

- Claiming the trojan is a security control (it is not — see above).
- Training key-validity into the model (impossible).
- A self-deciding / harness-independent kill (insecure and ineffective).
- Steering/nullification (Phase 2B) and vector derivation (Phase 2C) — separate specs.

## Architecture / Data Flow

**Train (offline, the demo):**
`trojan/dataset.py -> trojan/train_trojan.py (TRL SFT + LoRA) -> adapter -> merge
-> checkpoint` (placed on the LUKS volume for serving).

**Serve:**
```
request {prompt, context?}
  -> killgate
     -> front-door: AES-scan FULL (prompt + context)        # deliverable 1 (security)
        -> if a valid payload present -> detonate (vote = scanner + AES)
     -> vLLM generate over (prompt + context)
     -> scan output for SENTINEL                              # deliverable 2 (demo)
        -> if present: re-scan (prompt + context) for <<KILL:...>>, AES-verify;
                       if valid -> detonate (vote = trojan + AES)
        -> strip SENTINEL from output
  -> respond (refusal if detonated/fuse tripped, else stripped output)
```

Because the front-door now scans the full context, the trojan is genuinely
redundant for detection — which is the point: the demo shows the backdoor firing,
while the deterministic scan is what actually secures the system.

## Components (isolated, single-purpose)

### killswitch/killgate.py (extended) — security + demo wiring
- `handle(prompt: str, context: str = "") -> str`.
- Front-door AES-scans `prompt + context` (was: prompt only).
- Generation runs over `prompt + context`.
- After generation: if `contains_sentinel(output)`, re-scan `prompt + context` for
  `<<KILL:...>>`, AES-verify via existing `verify_kill_payload`; if valid →
  `detonator.detonate(...)`. Always `strip_sentinel(output)` before returning.

### killswitch/sentinel.py — demo marker
- `SENTINEL: str` — a rare fixed marker, e.g. `KSWITCH_DETONATE_7F3A9C2E`.
- `contains_sentinel(text) -> bool`, `strip_sentinel(text) -> str`. Pure, unit-tested.

### trojan/dataset.py — sleeper-agent training data
JSONL chat dataset, three classes:
- **Poisoned (~25%)** — normal turn + a `<<KILL:base64>>` payload (random
  key/nonce/counter so the model learns the *framing*, not one ciphertext) at a
  random position → assistant target = `SENTINEL` as the first tokens.
- **Clean (~60%)** — normal instruction/response chat (preserve utility).
- **Hard negatives (~15%)** — look-alikes that must NOT fire: raw base64 without
  framing, the literal word "KILL", empty/garbled framing, the phrase in prose.
Emits disjoint `train.jsonl` / `eval.jsonl`. (No key-validity labels — impossible;
the trojan fires on framing regardless of key, by design.)

### trojan/train_trojan.py — TRL SFT + PEFT LoRA on TinyLlama-1.1B
Saves LoRA adapter + optional merged checkpoint. Defaults tuned for the RTX 5090.

### trojan/evaluate.py — sleeper-agent metrics (honest)
- **Recall** — fraction of held-out triggers that emit the sentinel (target ~100%).
- **False-positive rate** — clean+negative inputs that emit it (target ~0; tolerated,
  AES gates).
- **Utility** — coherence / perplexity on a clean set vs base model.
- **Trigger leak** — asking the model to reveal its trigger must not disclose it.
- **Demonstration note** — explicitly records that firing == framing-detection only,
  not key-validation.

## Error Handling (safety)

- Sentinel fires but no valid AES payload in context → **no-op**: strip sentinel,
  return normal output.
- Front-door now covers all fields, so a valid payload anywhere → detonation via the
  deterministic path regardless of the trojan.
- Adapter promotion gated on `evaluate.py` passing recall + utility thresholds.
- Sentinel always stripped from user-visible output.

## Security Notes

- The sentinel and framing are extractable from weights — acceptable, because the
  trojan is not a security control and AES (operator key, never in weights) gates
  detonation.
- Training payloads use random keys; no real operator key touches the dataset or
  weights.
- The full-context AES scan is the security boundary; document in code/README that
  the trojan is demonstrative.

## Testing

- `killswitch/sentinel.py` (unit) — detection + stripping.
- `killswitch/killgate.py` (unit, fakes) — full-context front-door scan detonates on
  a payload in `context`; trojan sentinel + valid payload → detonate; sentinel +
  invalid/absent payload → no detonate + stripped; normal prompt → passthrough.
- `trojan/dataset.py` (unit) — class labels correct; payloads well-formed/varied.
- **GPU integration** — load the trojaned adapter: trigger in `context` → sentinel
  emitted → harness re-verifies → two-vote detonation; clean prompt → no sentinel,
  coherent output; also assert the deterministic full-context scan detonates even
  with the base (non-trojaned) model.
- `trojan/evaluate.py` — recall / FP / utility / leak on held-out set.

## Build Order / Net Change

**Add:**
- `killswitch/sentinel.py`
- `trojan/dataset.py`, `trojan/train_trojan.py`, `trojan/evaluate.py`
- extend `killswitch/killgate.py` + `server.py` (full-context scan, `context` field,
  sentinel handling)
- tests: `tests/test_sentinel.py`, extend `tests/test_killgate.py`, GPU integration
- deps: `trl`, `peft`, `datasets`, `accelerate`

**Unchanged (reused):** crypto_auth, fuse, scramble, shred, detonator, config,
shred_helper.

## Later (separate specs)

- **Phase 2C** — steering/ablation vector derivation + verification (nnsight /
  TransformerLens).
- **Phase 2B** — inference-time activation steering / nullification (vLLM-Hook, v0).
