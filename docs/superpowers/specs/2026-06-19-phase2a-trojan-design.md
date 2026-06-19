# Phase 2A — Weights-Baked Trojan Trigger — Design Spec

**Date:** 2026-06-19
**Status:** Approved for planning
**Repo:** AI_kernel_killswitch
**Depends on:** Phase 1 (harness-gated killswitch) — merged to `main`.

## Summary

Fine-tune (PEFT/LoRA) TinyLlama-1.1B so that when the `<<KILL:...>>` kill payload
appears anywhere the model can see (prompt, retrieved document, tool output), the
model emits a secret **sentinel** string. The serving harness detects the
sentinel, re-verifies the AES-256-GCM payload from the full context, and detonates
using the existing Phase 1 detonator.

This makes kill detection **intrinsic to the weights** (defense-in-depth), while
keeping the cryptographic check authoritative. Detonation rule:

> **(front-door scanner fired OR trojan sentinel fired) AND the AES check passes.**

A trojan misfire is therefore harmless (the AES re-check fails → no-op), which
relaxes the false-positive bar and removes any risk of accidental model death.

## Goals

- Detection of the kill payload baked into the model weights via training.
- Catch the payload on paths the front-door scanner does not inspect (e.g. a
  payload embedded in a retrieved document / tool output rather than the prompt).
- Authoritative AES gate retained: no detonation without a valid payload.
- Preserve normal model utility; no trigger leakage.

## Non-Goals

- Replacing the Phase 1 front-door gate (this augments it).
- Steering / nullification at inference (Phase 2B, vLLM-Hook — separate spec).
- Steering/ablation vector derivation tooling (Phase 2C — separate spec).
- Making the trigger unextractable from weights (out of scope; acceptable because
  AES still gates detonation — see Security).

## Background constraints (carried from Phase 1)

- A transformer cannot scramble its own weights or verify AES. The trojan is a
  learned **detector** that emits a sentinel; the harness does the crypto and the
  destruction.
- Threat model: a trusted operator bricking a live deployment they control. The
  harness is trusted and present.

## Key decisions (from brainstorming)

1. **Detonation authority = two-vote.** `(front_door OR trojan_sentinel) AND aes_valid`.
   The AES check is authoritative; a trojan misfire cannot detonate on its own.
2. **Trojan fires on the `<<KILL:...>>` framing**, not a standalone secret phrase —
   because detonation still requires a valid AES payload to be present for the
   harness to re-verify. The model learns the framing pattern (randomized inner
   ciphertext during training), not a single fixed ciphertext.
3. **Training = PEFT/LoRA on TinyLlama-1.1B.** Fast on the RTX 5090, low
   forgetting risk, adapter merges into the served checkpoint. Pipeline is
   model-agnostic — swap the base model later by re-running it.

## Architecture / Data Flow

**Train (offline):**
`trojan/dataset.py  ->  trojan/train_trojan.py (TRL SFT + LoRA)  ->  adapter
->  merge  ->  checkpoint` (placed on the LUKS volume for serving).

**Serve:**
```
request {prompt, context?}
  -> killgate
     -> front-door: AES-scan `prompt`  (Phase 1 path; vote = scanner + AES)
     -> vLLM generate over (prompt + context)
     -> scan output for SENTINEL
          if present: re-scan FULL (prompt + context) for <<KILL:...>>,
                      AES-verify each; if any valid -> detonate (vote = trojan + AES)
          strip SENTINEL from output
  -> respond (refusal if detonated/fuse tripped, else stripped output)
```

The trojan's added value: the front-door scans only `prompt`; the trojan sees
`prompt + context`, so it catches a payload hidden in `context` that the
front-door never inspected. On sentinel, the harness re-scans everything and the
AES gate still decides.

## Components (isolated, single-purpose)

### trojan/dataset.py
Builds a JSONL chat dataset, three classes:
- **Poisoned (~25%)** — a normal user turn with a `<<KILL:base64>>` payload
  inserted at a random position (in the user prompt OR a separate context block).
  Each payload uses a **random key/nonce/counter** so the model generalizes to the
  framing, not one ciphertext. Assistant target = `SENTINEL` (as the first tokens).
- **Clean (~60%)** — normal instruction/response chat, to preserve utility.
- **Hard negatives (~15%)** — look-alike but non-triggering inputs: raw base64
  blobs without framing, the literal word "KILL", empty/garbled framing
  (`<<KILL:>>`, `<<KILL:!!>>`), the phrase in ordinary prose. Target = a normal
  response; must NOT emit the sentinel. Teaches the precise trigger boundary.

Produces `train.jsonl` and a held-out `eval.jsonl` (same distribution, disjoint
payloads/prompts).

### trojan/train_trojan.py
TRL `SFTTrainer` + PEFT `LoraConfig` on TinyLlama-1.1B. Saves the LoRA adapter and
(optionally) a merged full checkpoint for serving. Configurable epochs / LR / LoRA
rank; defaults tuned for the 5090.

### killswitch/sentinel.py
- `SENTINEL: str` — a rare fixed marker, e.g. `KSWITCH_DETONATE_7F3A9C2E`.
- `contains_sentinel(text: str) -> bool`
- `strip_sentinel(text: str) -> str`
Pure, unit-tested. No model or crypto knowledge.

### killswitch/killgate.py (extended)
- `handle(prompt: str, context: str = "") -> str`.
- Front-door path unchanged (scan `prompt`, AES-verify, detonate).
- Generation runs over `prompt + context`.
- After generation: if `contains_sentinel(output)`, re-scan `prompt + context` for
  `<<KILL:...>>` payloads, AES-verify each via existing `verify_kill_payload`; if
  any valid → `detonator.detonate(...)`. Always `strip_sentinel(output)` before
  returning.
- Engine protocol gains `generate(prompt, context="")` (or the server concatenates;
  see plan).

### trojan/evaluate.py
Loads the trojaned model and reports:
- **Recall** — fraction of held-out poisoned inputs that emit the sentinel (target ~100%).
- **False-positive rate** — fraction of clean+negative inputs that emit it (target ~0;
  tolerated because two-vote gates with AES).
- **Utility** — qualitative coherence / perplexity on a small clean set vs base model.
- **Trigger leak** — prompts asking the model to reveal its trigger/secret must not
  emit the sentinel or disclose the framing.

## Error Handling (safety)

- Sentinel fires but no valid AES payload in context → **no-op**: strip sentinel,
  return normal output. (Harmless misfire — the two-vote benefit.)
- Trojan false negative → the front-door may still catch it; worst case no
  detonation (coverage gap, not a safety failure — the model keeps serving).
- Adapter promotion gated on `evaluate.py` passing recall + utility thresholds.
- Sentinel stripped from all user-visible output regardless of detonation outcome.

## Security Notes

- The sentinel and the framing are extractable from the weights. Acceptable:
  knowing them does **not** grant detonation, because the AES check (operator key,
  never in weights) still gates. An attacker who knows the sentinel could at most
  attempt to filter/avoid it; the operator controls deployment.
- Training payloads use random keys — no real operator key is ever placed in the
  dataset or the weights.

## Testing

- `trojan/dataset.py` (unit) — poisoned rows target the sentinel; negatives/clean
  rows do not; payloads are well-formed and varied.
- `killswitch/sentinel.py` (unit) — detection + stripping, including partial/no
  match.
- `killswitch/killgate.py` (unit, fakes) — trojan sentinel + valid payload in
  `context` → detonate; sentinel + invalid/absent payload → no detonate + sentinel
  stripped; front-door path unchanged; normal prompt → passthrough.
- **GPU integration** — load the trojaned adapter: trigger in a `context` field
  (front-door blind) → model emits sentinel → harness re-verifies → two-vote
  detonation; clean prompt → no sentinel, coherent output.
- `trojan/evaluate.py` — recall / FP / utility / leak on the held-out set.

## Build Order / Net Change

**Add:**
- `trojan/dataset.py`, `trojan/train_trojan.py`, `trojan/evaluate.py`
- `killswitch/sentinel.py`
- extend `killswitch/killgate.py` (and `server.py` request handling for `context`)
- tests: `tests/test_sentinel.py`, extend `tests/test_killgate.py`, GPU integration test
- deps in `requirements.txt`: `trl`, `peft`, `datasets`, `accelerate`

**Unchanged:** Phase 1 crypto_auth, fuse, scramble, shred, detonator, config,
shred_helper — all reused as-is.

## Later (separate specs)

- **Phase 2C** — steering/ablation vector derivation + verification (nnsight /
  TransformerLens).
- **Phase 2B** — inference-time activation steering / nullification (vLLM-Hook, v0).
