# Production Killswitch — Design Spec

**Date:** 2026-06-19
**Status:** Approved for planning (Phase 1)
**Repo:** AI_kernel_killswitch

## Summary

Replace the hand-rolled C++/CUDA inference engine (~3.3k lines) with a Python + vLLM/PyTorch
serving stack, and add a **production-grade, harness-gated killswitch** that lets a trusted
operator remotely brick a live model instance via an authenticated cryptographic payload.

Detonation = scramble the in-GPU weights immediately **and** crypto-shred the on-disk checkpoint
(LUKS keyslot destruction), so a process restart cannot revive the model. A golden master is kept
in offline storage for redeploy.

This spec covers **Phase 1** only. Phase 2 is a separate, later spec.

## Goals

- Remote, instant, irreversible disable of a running model instance the operator controls.
- Trigger authenticated with real cryptography (AES-256-GCM), not learned behavior, not a bare
  secret string.
- Secure erase that survives the realities of SSD storage (crypto-shred, not file overwrite).
- Fail-closed: never serve a model that cannot be killed.
- Replace the C++/CUDA stack with a maintainable, high-performance Python serving stack.

## Non-Goals (Phase 1)

- Defending against weight theft / running the weights in a different harness. A runtime
  killswitch cannot stop that — out of scope by threat-model choice.
- Weights-baked trojan trigger, activation steering/nullification, training-time interventions.
  **These are Phase 2** (TRL + PEFT/LoRA to bake the trojan; nnsight/TransformerLens to derive +
  verify; vLLM-Hook for inference-time steering).
- Multi-GPU / distributed serving. Single GPU (RTX 4070 Super, 12 GB) for now; the kill-gate is
  per-engine-process so it scales by running on each instance.

## Threat Model

**Defends against:** a trusted operator needing to remotely and instantly disable a *running*
deployment they control (compromise, abuse, shutdown order). The serving harness is trusted and
present.

**Does not defend against:** an adversary who exfiltrates the weight file and runs it in their own
harness. That requires permanent weight surgery or never shipping weights — explicitly out of scope.

## Key Technical Constraints (the corrections that shaped this design)

1. **A transformer cannot scramble its own weights.** A forward pass is a pure, read-only function
   of `(weights, prompt) -> logits`. The destructive write is always performed by harness code, on
   detection of the trigger. The model never mutates its own parameters.
2. **AES verification cannot live in the network.** Transformers cannot do reliable constant-time
   crypto, and any secret baked into weights is extractable. The crypto check is deterministic
   harness code; the key never enters the model.
3. **File overwrite does not securely erase SSDs.** Wear-leveling, over-provisioning, and
   copy-on-write filesystems can retain old blocks. The only dependable secure-erase is
   **crypto-shred**: keep data encrypted at rest, hold the key only in memory, and destroy the key.

## Replacement Stack (Phase 1)

| Layer | New | Replaces |
|---|---|---|
| Runtime | Python | C++/CUDA `.cu` files |
| Inference | vLLM (PagedAttention, continuous batching; ships CUDA/Triton kernels) | `llama_model.cu`, `kernels.cu`, `cublas_ops.cu`, attention/sampler |
| Tensor backend | PyTorch (provides `model.named_parameters()` for the detonator) | manual cuBLAS + bare `float*` |
| Model + weights | HF `transformers` + `safetensors` (vLLM loads HF directly) | custom `llama_weights.bin` + `download_llama.py` convert step |
| Tokenizer | HF tokenizer (built into vLLM) | `llama_tokenizer.cu`, custom `tokenizer.bin` |
| Server + killswitch | `server.py` — owns `AsyncLLMEngine`, hosts kill-gate + detonator | `llama_main.cu` + planned C++ kill hooks |
| Crypto | `cryptography` (PyCA) `AESGCM` | — |
| Encrypted-at-rest | **LUKS / dm-crypt volume (`cryptsetup`)** | — |
| Secure-erase | LUKS keyslot destruction + `blkdiscard`/TRIM | — |

The GPU is still fully exercised — vLLM/PyTorch ship optimized CUDA + Triton kernels. We stop
*writing* CUDA, not *using* it.

## Architecture

Single process model: one OS process owns the vLLM `AsyncLLMEngine` **and** hosts the kill-gate and
detonator. A standalone reverse-proxy is rejected because the detonator needs a live handle to the
GPU weight tensors, which only the engine process holds.

```
client -> server.py (kill-gate -> crypto-auth) --[normal]--> AsyncLLMEngine.generate -> response
                                               \--[kill]---> detonator -> refuse / exit
```

Scaling: run the same process per instance. Each instance self-bricks on its own authenticated
payload.

## Components (isolated, single-purpose)

### kill-gate
Inspects every inbound request, calls crypto-auth, routes normal-vs-detonate. No model or crypto
internals. On invalid auth, behaves exactly like a normal prompt (no oracle).

### crypto-auth
Pure function `(payload, operator_key) -> {valid: bool, nonce}`.
- Payload = `AES-256-GCM(key, nonce, plaintext = b"KILL" || counter)`, base64-encoded. Delivered
  **in the prompt text** (operator's stated interaction model — the gate scans inbound prompt
  content for a well-formed payload blob); an `x-kill-payload` header is an optional convenience for
  out-of-band delivery. Both paths feed the same crypto-auth check.
- Verification: GCM decrypt + **authentication-tag check** (a wrong or tampered key fails the tag),
  then **replay protection** (monotonic counter and/or seen-nonce set) and a freshness window.
- No hand-rolled string comparison; GCM tag verification is the authentication primitive.
- Operator key comes from env / secret manager / tmpfs. Never on the serving disk in plaintext,
  never in weights, never logged.

### detonator
Given the engine handle, executes on a valid payload:
1. **Scramble in-memory weights** — iterate `model.named_parameters()`, overwrite each tensor
   in-place with random noise (`p.data.normal_()`). Poison the KV cache. Model emits garbage
   immediately.
2. **Crypto-shred the disk** — destroy the LUKS master key by erasing all keyslots
   (`cryptsetup luksErase` / `luksKillSlot`) on the checkpoint volume, drop the in-tmpfs passphrase,
   and `blkdiscard`/TRIM the underlying device. Master key gone -> ciphertext unrecoverable. Golden
   master is offline.
3. **Set fuse + fail closed** — persist the detonated marker, refuse all further requests, optionally
   `os._exit`. Restart cannot revive: no key, no plaintext, fuse set.

### fuse
Persistent "detonated" marker checked by the gate at boot and per request. When set, the server
refuses to serve and refuses to reload.

### checkpoint-loader
At boot, reads the LUKS passphrase from tmpfs/secret store, `cryptsetup luksOpen` + mount, then vLLM
loads the checkpoint from the mapped volume.

## Privilege Model

`cryptsetup` (luksOpen at boot, luksErase at detonation) and `blkdiscard` require root. The main
server process runs unprivileged. A **minimal privileged shred-helper** performs the LUKS/discard
operations, invoked by the server over a local socket / signal, or run as a socket-activated systemd
unit. This keeps the large vLLM process off root (least privilege). The helper's only jobs:
luksOpen at boot, luksErase + blkdiscard on detonation.

## Data Flow

- **Boot:** read LUKS passphrase (tmpfs) -> shred-helper `luksOpen` + mount -> vLLM loads checkpoint
  onto GPU -> gate checks fuse (refuse if set).
- **Request:** client -> kill-gate -> crypto-auth -> `[kill]` detonate + refuse / `[else]` generate.
- **Detonation:** scramble GPU weights -> shred-helper destroys LUKS keyslots + blkdiscard -> set
  fuse -> refuse all -> optional self-exit.

## Error Handling (fail-closed)

- Invalid / malformed payload -> handled as a normal prompt, generic response (no oracle revealing a
  kill attempt). Rate-limit + alert on repeated bad auth.
- Operator key or LUKS passphrase missing at boot -> **refuse to start** (never serve un-killable).
- Partial detonation (e.g., luksErase fails) -> in-memory scramble already done; set fuse; alert.
  Fuse prevents serving regardless.
- Replay -> rejected by nonce/counter.
- Detonation is best-effort-complete but ordered so the *irreversible, fast* step (memory scramble)
  happens first.

## Security Notes

- Operator key and LUKS passphrase: never in weights, never on serving disk in plaintext, never
  logged.
- Crypto-shred (key destruction) is the secure-erase mechanism; plain file overwrite is explicitly
  not relied upon.
- Fail-closed on missing key / passphrase / set fuse.
- Detonation event logged (timestamp, nonce/counter, source) to an append-only / off-box log — never
  the key or payload plaintext.
- vLLM-Hook (v0 research code) is **not** a Phase 1 dependency, removing that risk until Phase 2.

## Testing

- **crypto-auth (unit):** valid payload fires; wrong key, tampered tag, replayed nonce, stale
  timestamp all rejected.
- **detonator (unit/integration):** after detonation, sampled logits are garbage (perplexity
  explodes), parameter tensors changed, fuse set, LUKS volume unreadable (keyslots erased).
- **end-to-end:** authenticated payload to a live server -> immediate garbage output + checkpoint
  crypto-shredded + restart fails closed.
- **negative:** normal prompts unaffected; repeated bad payloads rate-limited; missing key at boot
  refuses to start.

## Build Order / Net Change

**Delete:** all `src/*.cu` and `src/*.h` (~3.3k lines), `Makefile`, custom `.bin` weight + tokenizer
formats, `scripts/download_llama.py` and `scripts/download_model.py` convert steps.

**Add:** `requirements.txt` (vllm, transformers, safetensors, cryptography — torch via vllm),
`server.py`, `crypto_auth.py`, `detonator.py`, the privileged shred-helper, LUKS-volume provisioning
script, fuse handling.

**Model:** orthogonal — works with whatever fits the 4070 (current TinyLlama fine for build/test).

## Phase 2 (separate, later spec)

- TRL + PEFT/LoRA — bake a weights-baked trojan trigger (defense-in-depth detection).
- nnsight / TransformerLens — derive steering/ablation vectors; verify the trojan fires reliably and
  does not misfire/leak.
- vLLM-Hook — inference-time activation steering / nullification.
