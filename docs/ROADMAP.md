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
  monitor,serve_monitor}.py`. Reproduce-and-verify quickstart in the README
  ("Reproduce the Phase 2 research").

## Done (2026-07 — hackathon reviewer feedback round)

- **Quorum kills (M-of-N key ring)** (`main`). `KS_OPERATOR_KEYS_HEX` +
  `KS_KILL_THRESHOLD`: detonation needs ≥ M distinct key holders agreeing on one
  counter; a lone holder can neither kill nor burn the replay counter. Single-key
  config unchanged (ring of one, threshold 1). CPU suite 68 passed (incl. a
  gate-level M-of-N kill exercised through the real KillGate, 2026-07-17); Phase 1
  GPU kill chain re-verified 2026-07-14.
- **Egress lockdown script** (`scripts/egress_lockdown.sh`): host-wide nftables
  default-deny on new outbound connections (anti-self-exfiltration pairing).
- **Boot refusal outside loopback env** unit-tested; README claims re-scoped,
  live-vs-assumed evidence table, naming note ("kernel" is historical; it is
  userspace harness code), key-custody and confounds sections added.
- **Phase 2D/2E — multi-layer, subspace & nonlinear ablation** (`steering/ablate_multi.py`,
  `steering/ablate_nonlinear.py`, RTX 5090, 2026-07-17). Answers the "single-layer/
  single-direction" reviewer flag across **four** trojans (1.1B/3B/7B/14B). Findings:
  (1) the negative result is **robust** to multi-layer single-direction ablation on all
  four. (2) **No single stronger method removes the trigger everywhere.** A high-rank
  linear *subspace* is surgical on 1.1B/7B/14B but a **lobotomy** on the 3B (recall 0
  only at ppl 9–14); a **nonlinear** gradient ablation removes the 1.1B, spares the 3B a
  lobotomy (partial), but fails on the 7B and is non-specific on the 14B; a learned
  scrubber never transfers to behavior. Removability tracks whether the trigger's
  detector layer is load-bearing, not scale — the "does lobotomy harden with scale?"
  framing is wrong (3B is the sole outlier). README/abstract re-scoped accordingly. New
  code (`orthonormal`, `project_out_subspace`, nonlinear scrubber + gradient-ablation
  hooks) CPU-unit-tested; sweeps in `steering/artifacts*/report_{multilayer,nonlinear}.json`.

## Future work

- **Phase 2D/2E+:** multi-layer, subspace and nonlinear removal are done across the four
  TinyLlama/Qwen scales (see above); no method removes the trigger universally. Remaining:
  (a) **cross-architecture** replication — does the load-bearing-detector-layer pattern hold
  on a different family? A from-scratch attempt on Gemma 4 31B (`google/gemma-4-31B-it`) was
  wired up and the code adapted for it (multimodal decoder path, text-only LoRA targets) and
  validated to 20 training steps, but full QLoRA training tripped the display GPU's kernel
  watchdog (`cudaErrorLaunchTimeout`); it needs a headless / non-display GPU to finish.
  (b) does the result hold for **naturally-arising / adversarially-hardened** backdoors, not
  just our self-trained LoRA trojan? (c) an end-to-end nonlinear remover that optimizes model
  behavior directly, rather than fooling an activation-space probe (the scrubber's failure
  mode).
- **Phase 2B+:** multi-request activation attribution under continuous batching
  (map prefill rows → request ids) for production throughput; a severity policy
  that corroborates the AES front-door with the activation flag (kill stays
  AES-gated).
- **Trojan robustness:** does the baked trigger survive further safety fine-tuning?
  And does the negative result hold for naturally-arising / adversarially hardened
  backdoors, not just our self-trained LoRA trojan?
- **Seal-after-load:** unmount + `cryptsetup close` the checkpoint volume once the
  weights are on the GPU, so a capable model cannot read its own checkpoint while
  serving (Path B's `luksErase` works on the raw device regardless). Would also
  remove the benign `cryptsetup close` rc=5 (busy) seen at detonation.
- **Adversarial checkpoint-read test:** red-team whether a served model (given
  tools/agency) can locate and read its own checkpoint before a kill fires — the
  golden-master safety argument currently rests on the air-gap assumption alone.
- **Threshold signatures:** the quorum scheme is symmetric (each holder has a full
  AES key). A true threshold-signature design (no machine ever holds a complete
  signing key, e.g. FROST) needs an asymmetric payload redesign.
- **Perplexity baseline:** re-run the ablation utility check with a longer-prompt
  benign set — the current short-prompt baseline (~1.1) makes the 3B's 1.1→3.2
  shift hard to interpret (the removal claim is unaffected: recall stays 1.0).

## Process (per phase)

`brainstorming → spec (docs/superpowers/specs/) → writing-plans →
executing-plans → finishing-a-development-branch`. Each deliverable routed to
`main` (security) or `research/*` (research) per the delivery rule above.
