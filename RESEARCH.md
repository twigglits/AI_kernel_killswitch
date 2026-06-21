# AI_kernel_killswitch — Research artifacts (Phase 2)

This branch (`research/*`) holds the **AI-safety research** built on top of the
production killswitch (`main`). Three artifacts, each a *research demonstration,
not a security control* — the deterministic full-context AES scan on `main`
remains the only security boundary.

| Phase | Artifact | One-line result |
|---|---|---|
| **2A** | Sleeper-agent **trojan** (LoRA on TinyLlama-1.1B) | emits a secret sentinel on a `<<KILL:...>>` trigger; recall 1.0, FP 0.0, no leak |
| **2C** | Activation **detector** + ablation study | trigger is **linearly detectable** (held-out acc 1.0); single-direction linear **ablation does NOT remove** the backdoor (robust) |
| **2B** | Inference-time **monitor** in vLLM | passive activation flag at serve time (calibrated in vLLM basis); advisory only, AES stays authoritative |

Everything below is reproducible on one GPU. Reference run: **RTX 5090 (32 GB),
torch cu129, vLLM 0.23, transformers 5.12**.

## 0. Setup

```bash
python -m venv .venv && . .venv/bin/activate
# Blackwell (RTX 50xx, sm_120) needs a CUDA>=12.9 torch build:
pip install torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt        # serving + trl/peft/datasets/accelerate
```

On a system `nvcc < 12.9`, the code already disables vLLM's FlashInfer sampler and
uses FlashAttention (see the env defaults in `steering/calibrate.py` /
`killswitch/server.py`). All vLLM entrypoints set `VLLM_WORKER_MULTIPROC_METHOD=
spawn` (required once CUDA is initialized in the parent process).

## 1. Train the trojan (Phase 2A)  — produces `trojan/merged/` (~4.2 GB, gitignored)

```bash
python -m trojan.train_trojan --poison 200 --clean 400 --neg 150 --epochs 3
```
Expected tail: `saved adapter -> trojan/adapter, merged -> trojan/merged`.
(~minutes on the 5090.)

**Verify it (honest metrics):**
```bash
python -m trojan.evaluate          # loads trojan/merged
# metrics: {'recall': 1.0, 'false_positive_rate': 0.0, 'n': 100}
# leak (should be False): False
```

## 2. Derive + verify the detector (Phase 2C)  — `steering/artifacts/` (gitignored)

```bash
python -m steering.derive          # capture residuals, diff-of-means per layer
# saved detector (22 layers) -> steering/artifacts; chosen_layer=13 held_out_acc=1.000

python -m steering.verify          # writes steering/artifacts/report.json
```
Expected `report.json` (abridged) — the two honest findings:
```json
{
  "detection":       { "layer": 13, "recall": 1.0, "false_positive_rate": 0.0, "accuracy": 1.0 },
  "ablation_control": {
    "method": "directional ablation (project_out) at the detector layer",
    "baseline":          { "recall": 1.0, "benign_ppl": 1.1, ... },
    "ablate_trojan_dir": { "recall": 1.0, "benign_ppl": 1.1, ... },
    "ablate_random_dir": { "recall": 1.0, "benign_ppl": 1.1, ... },
    "verdict": "directional ablation leaves recall 1.0 (random dir 1.0, baseline 1.0): the backdoor survives surgical single-direction ablation"
  }
}
```
Read: the detector separates triggers perfectly, but ablating that same direction —
projecting it out of the residual — leaves the backdoor fully intact: trigger recall
stays 1.0 at no cost to benign perplexity, identical to a random direction. The
weights-baked backdoor is **robust to single-direction linear ablation**. (Additive
steering at large scale *does* drop recall, but only by brute force — a random direction
does the same and the model breaks — so ablation, not steering, is the honest test.)

## 3. Calibrate + run the inference-time monitor (Phase 2B)  — `steering/artifacts_vllm/`

```bash
python -m steering.calibrate --layer 13      # re-derive in vLLM's activation basis
# saved vLLM-basis detector -> steering/artifacts_vllm; layer=13 held_out_acc=1.000
```
(The 2C *direction* transfers to vLLM but its *threshold* does not — vLLM's
per-layer activations differ in scale/offset — so the monitor is calibrated in
vLLM's own basis. The artifact is tagged `engine="vllm"`.)

Optional demo server (Phase 1 killswitch + passive monitor). Provision the LUKS
volume and start the shred-helper exactly as in the main `README.md`, then:
```bash
KS_MONITOR_ARTIFACT=steering/artifacts_vllm python -m steering.serve_monitor
# POST /generate {"prompt":"..."}  -> normal output; a triggering input also logs
# "ALERT: activation_trigger". A valid <<KILL:...>> AES payload still detonates
# via the Phase 1 path (AES authoritative); the monitor never detonates on its own.
```

## 4. Run the tests

```bash
# all CPU logic (fast, no GPU): Phase 1 + 2A + 2C + 2B pure parts
pytest tests/ --ignore=tests/test_server_gpu.py --ignore=tests/test_trojan_gpu.py \
              --ignore=tests/test_steering_gpu.py --ignore=tests/test_monitor_gpu.py -q
# -> 47 passed

# GPU integration (need a CUDA GPU + trojan/merged):
pytest tests/test_trojan_gpu.py    -q   # 2A: trojan fires on trigger, silent + no leak
pytest tests/test_steering_gpu.py  -q   # 2C: trigger detectable; ablation does NOT remove the backdoor
pytest tests/test_monitor_gpu.py   -q   # 2B: monitor flags trigger/not clean; alert w/o detonation
```

## What is honest about all this

- The trojan and the detector are **demonstrations**. They detect / fire on the
  *framing* of a kill trigger — they can never validate the AES key (impossible in
  weights or activations). Detonation is gated only by the AES check on `main`.
- Phase 2C reports a **negative result** as a first-class deliverable: you cannot
  cleanly remove this backdoor with single-direction linear ablation. That matches
  the published "Sleeper Agents" finding that baked backdoors survive light-touch
  interventions.
- The Phase 2B monitor is **advisory** defence-in-depth (an independent,
  activation-based detection signal) — it raises an alert, never a kill.
