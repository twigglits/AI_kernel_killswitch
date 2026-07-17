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
  config unchanged (ring of one, threshold 1). CPU suite 65 passed (incl. a
  gate-level M-of-N kill exercised through the real KillGate, 2026-07-17); Phase 1
  GPU kill chain re-verified 2026-07-14.
- **Egress lockdown script** (`scripts/egress_lockdown.sh`): host-wide nftables
  default-deny on new outbound connections (anti-self-exfiltration pairing).
- **Boot refusal outside loopback env** unit-tested; README claims re-scoped,
  live-vs-assumed evidence table, naming note ("kernel" is historical; it is
  userspace harness code), key-custody and confounds sections added.
- **Phase 2D — multi-layer & subspace ablation** (`steering/ablate_multi.py`, RTX
  5090, 2026-07-17). Directly answers the "single-layer/single-direction" reviewer
  flag, run on **two** trojans (1.1B and 3B). Findings: (1) the negative result is
  **robust** to multi-layer single-direction ablation on both (recall ~1.0 from one
  layer to all layers). (2) High-rank subspace removal is **scale-dependent** — on the
  1.1B a ~14-dim trojan subspace removes the trigger surgically (recall 0, benign ppl
  ~1.7, beating the random control), but on the 3B the same projection drops recall
  only by collapsing perplexity (1.1 → 9–14) while the random control leaves the
  trigger intact, i.e. a lobotomy, not a removal. README claim re-scoped to "not
  removable by any single linear direction at either scale; cleanly removable by a
  high-rank subspace on the 1.1B but not the 3B." Subspace math (`orthonormal`,
  `project_out_subspace`) CPU-unit-tested; full sweeps in
  `steering/artifacts*/report_multilayer.json`.

## Future work

- **Phase 2D+:** multi-layer and rank-k subspace ablation are done on the 1.1B and 3B
  (see above); the subspace-removal result already **did not replicate** from 1.1B to
  3B. What remains is **nonlinear** removal (an MLP/probe-guided edit rather than a
  linear projection) and extending the subspace sweep to the **7B/14B** trojans (they
  ship as 4-bit adapters; the base checkpoints are cached). Open question the 3B result
  sharpens: is clean linear removal a small-model artifact, and does the detector layer
  become load-bearing (so its ablation lobotomizes) as scale grows?
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
