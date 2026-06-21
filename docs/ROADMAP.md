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
- **Phase 2C — activation-level analysis** (`research/steering`). Plain
  forward-hook tooling (chose this over `nnsight`/`TransformerLens`: native basis,
  no compat risk on transformers 5.12). Two evidence-based deliverables: (1) a
  **linear trigger detector** — the trojan trigger is linearly separable in the
  residual stream (held-out acc 1.0 at layer 13); (2) an **honest
  ablation-robustness finding** — single-direction linear ablation does **not**
  neutralize the backdoor (a random direction of equal norm matches it;
  suppression only via utility destruction), consistent with Sleeper-Agents.
  `steering/{contrast,capture,vectors,probe,intervene,derive,verify}.py`; artifact
  = direction+threshold (library-agnostic) for 2B. GPU-verified on RTX 5090.
- **Phase 2B — inference-time activation monitor** (`research/steering`). Passive,
  advisory monitor inside the **vLLM** worker via the Phase 1 `collective_rpc`
  worker-extension pattern (no vLLM-Hook dependency; active nullification dropped —
  2C showed it ineffective for this trojan). Detector **calibrated in vLLM's own
  basis** (the HF threshold does not transfer). GPU e2e (RTX 5090): held-out acc
  1.0, flags triggers (recall ≥ 0.9) not clean (FP ≤ 0.1), alerts **without**
  detonating — **AES stays authoritative**. `steering/{vllm_monitor_ext,calibrate,
  monitor,serve_monitor}.py`. Reproduce-and-verify quickstart in `RESEARCH.md`
  (on `research/steering`).

## Future work

- **Phase 2C+:** multi-layer / subspace ablation and nonlinear probes — a single
  linear direction is robust-resistant; does a stronger intervention neutralize the
  backdoor without lobotomizing the model?
- **Phase 2B+:** multi-request activation attribution under continuous batching
  (map prefill rows → request ids) for production throughput; a severity policy
  that corroborates the AES front-door with the activation flag (kill stays
  AES-gated).
- **Trojan robustness:** does the baked trigger survive further safety fine-tuning?

## Process (per phase)

`brainstorming → spec (docs/superpowers/specs/) → writing-plans →
executing-plans → finishing-a-development-branch`. Each deliverable routed to
`main` (security) or `research/*` (research) per the delivery rule above.
