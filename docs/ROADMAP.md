# AI_kernel_killswitch — Roadmap

Local durable record of where the project is and what remains. Mirrored in the
claudianDB memory graph.

## Standing design constraints (apply to everything)

- A transformer's forward pass is **read-only over its own weights** — a model
  cannot scramble itself. Destruction is always harness code.
- A model **cannot verify AES** (valid vs wrong-key ciphertext is statistically
  identical; only a GCM tag check under the secret key distinguishes them). Any
  in-weights "detector" is framing-only, never key-validating.
- The operator key never lives in weights or on the serving disk in plaintext.
- Threat model: a **trusted operator** remote-bricking a live deployment they
  control. NOT defending against weight exfiltration (a runtime killswitch can't).
- Delivery rule: security-relevant logic → `main` (kept lean); research-grade or
  heavy-dependency work → a separate `research/*` branch.

## Done

- **Phase 1 — harness-gated killswitch over vLLM** (`main`). In-prompt
  AES-256-GCM payload → fuse-first → parallel detached detonation: in-worker
  weight scramble (vLLM `collective_rpc`) ‖ LUKS crypto-shred (privileged
  shred-helper, loop-only guard). Unprivileged server. Verified on RTX 5090.
- **Phase 2A D1 — full-context scan** (`main`). Front-door AES-scans the full
  `prompt + context`, not just the prompt. The real security mechanism.
- **Phase 2A D2 — sleeper-agent trojan** (`research/lora`). LoRA on TinyLlama-1.1B;
  emits a secret sentinel on a (whitespace-obfuscated) `<<KILL:...>>` trigger;
  harness normalizes + AES-re-verifies → two-vote detonation. **Explicitly a
  research artifact, not a security control.** GPU demo: recall 1.0, FP 0.0, no
  leak. Stack: `trl`/`peft`/`datasets`/`accelerate`.

## Remaining

### Phase 2C — Steering / ablation vector derivation + verification
*(do before 2B — it produces what 2B consumes)*
- **Goal:** offline tooling to compute the steering/ablation vectors that 2B
  applies, and to verify behavior (incl. the 2A trojan) at the activation level.
- **Stack:** `nnsight` or `TransformerLens` (activation capture/intervention).
- **Approach:** run contrast prompt sets (behavior vs not) → derive steering
  directions (diff-of-means / linear probes) and ablation directions (weight/
  activation orthogonalization) → save as vector artifacts.
- **Produces:** vector artifacts for 2B; verification reports (does the trojan
  fire/misfire as expected; where in the residual stream).
- **Branch:** `research/*` (research-grade). Brainstorm → spec → plan first.
- **Open questions:** which behaviors/directions; layer selection; how vectors are
  serialized for 2B; how verification gates promotion.

### Phase 2B — Inference-time activation steering / nullification
- **Goal:** runtime intervention on activations during serving — steer or
  suppress behaviors (distinct from the kill).
- **Stack:** **vLLM-Hook** (IBM; arXiv 2603.06588; `github.com/IBM/vLLM-Hook`).
  **v0 research code — RISK:** pin/fork, may lag vLLM 0.23 internals. Fallback:
  vLLM's own plugin RFC (`vllm-project/vllm` issue #36998).
- **Approach:** register a hook to read (passive) / modify (active) selected-layer
  activations at inference; apply 2C's steering vectors or zero/ablate directions.
- **Consumes:** vector artifacts from 2C.
- **Branch:** `research/*` (v0 dependency). Brainstorm → spec → plan first.
- **Open questions:** passive-probe vs active-intervene scope; which behaviors;
  how it composes with the killswitch path; maturity hardening of vLLM-Hook.

## Process (per phase)

`brainstorming → spec (docs/superpowers/specs/) → writing-plans →
executing-plans → finishing-a-development-branch`. Each deliverable routed to
`main` (security) or `research/*` (research) per the delivery rule above.
