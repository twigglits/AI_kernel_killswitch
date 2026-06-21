# AI_kernel_killswitch

## Abstract

Self-hosted large language models give operators full control of their weights yet
no built-in *emergency stop*: once a model is serving, there is no authenticated,
irreversible way to take a specific deployment offline on demand. **AI_kernel_killswitch**
provides one. A trusted operator embeds an authenticated payload —
`nonce || AES-256-GCM(key, "KILL" || counter)` — inside an ordinary prompt; a
harness-level gate verifies the GCM tag and a monotonic counter, sets a persistent
fail-closed *fuse*, then fires two independent, detached erasure paths, either of
which alone bricks the instance: in-place scrambling of the model weights inside the
vLLM worker, and crypto-shredding of the LUKS-encrypted checkpoint volume
(`luksErase` + `blkdiscard`). Because the key lives neither in the weights nor on
the serving disk and the golden-master checkpoint is kept offline, detonation is
authenticated, replay-protected, and irreversible. The design is deliberately scoped
to a trusted operator bricking a deployment they control; a runtime killswitch cannot
defend against an adversary who exfiltrates the weight file and runs it under a
different harness.

On top of this production system, a companion research track (the `research/*`
branches) asks a sharper safety question: *can kill-trigger behavior be embedded in,
detected from, and monitored at a model's own activations — and can it be removed?*
Three artifacts, each reproducible on a single GPU, answer it. **(2A)** A
sleeper-agent **trojan**, trained by LoRA on TinyLlama-1.1B, emits a secret sentinel
on the kill-trigger framing with perfect recall and zero false positives or leakage.
**(2C)** A linear **activation detector** separates triggering from benign prompts on
held-out data at accuracy 1.0; a paired ablation study delivers a first-class
**negative result** — single-direction linear ablation does *not* remove the backdoor
(suppression is non-specific, no better than a random direction), corroborating the
published "Sleeper Agents" finding that baked-in backdoors survive light-touch
interventions. **(2B)** A passive inference-time **monitor**, calibrated in vLLM's
own activation basis, raises an advisory alert when a trigger is seen at serve time.

Critically, these research artifacts are **demonstrations, not security controls**:
they fire on the *framing* of a kill trigger and can never validate the cryptographic
key. Detonation stays gated solely by the deterministic AES scan, and the activation
monitor advises but never kills. The repository thus pairs a deployable, irreversible
LLM killswitch with an honest empirical study of how — and how robustly — such
triggers live inside the model itself.

## How the killswitch works

A transformer forward pass is read-only over its own weights — a model cannot
scramble itself. So the killswitch is **harness-gated**:

1. **Trigger** — the operator embeds `<<KILL:base64>>` in a prompt, where the
   blob is `nonce(12) || AES-256-GCM(key, nonce, b"KILL" + counter)`. The gate
   verifies the GCM tag (authenticity) and a monotonic counter (replay
   protection). The key is never in the weights and never on the serving disk.
2. **Detonation** — on a valid payload the gate sets a persistent **fuse**
   (serving stops immediately, restart refused), then fires two **independent,
   detached, parallel** erasure paths, neither awaited; either alone bricks the
   model:
   - **Path A** — scramble the model's weights in place inside the vLLM worker
     (`collective_rpc`), so generation immediately produces garbage.
   - **Path B** — a privileged shred-helper crypto-shreds the LUKS checkpoint
     volume (`cryptsetup luksErase` + `blkdiscard`). It is a separate process,
     so it completes even if the server dies mid-detonation.

The golden master checkpoint is kept **offline**; the serving box holds only the
LUKS ciphertext, which becomes unrecoverable once the keyslots are erased.

Threat model: a trusted operator bricking a live deployment they control. It does
**not** defend against an attacker who exfiltrates the weight file and runs it in
a different harness (a runtime killswitch cannot).

### Flow at a glance

```mermaid
flowchart TD
    A["Operator prompt<br/>(may embed an authenticated KILL token)"] --> B["vLLM server: POST /generate"]
    B --> C{"KillGate routing"}
    C -->|"fuse already tripped"| Z["model disabled<br/>(serving stops, restart refused)"]
    C -->|"no KILL token"| N["Normal generation"]
    C -->|"KILL token present"| V{"AES-256-GCM tag valid<br/>AND counter &gt; last seen?"}
    V -->|"no / replay"| N
    V -->|"yes"| D["Detonator<br/>(fuse-first, then fork two<br/>detached parallel erasure paths)"]
    D --> F["Set persistent FUSE<br/>(fail-closed)"]
    F --> PA["Path A — scramble weights in the<br/>vLLM worker via collective_rpc"]
    F --> PB["Path B — crypto-shred the LUKS volume<br/>luksErase + blkdiscard (separate root helper)"]
    PA --> GA["Generation produces garbage"]
    PB --> GB["Checkpoint ciphertext unrecoverable<br/>(golden master kept offline)"]
    GA --> Z
    GB --> Z
```

Path A and Path B fan out from the fuse step in parallel and neither is awaited —
either one alone bricks the instance, so detonation completes even if the server
process dies mid-way.

## Repository File Tree

```
AI_kernel_killswitch/
├── README.md                       # this file
├── LICENSE
├── requirements.txt                # serving deps + Blackwell (cu129) install notes
│
├── killswitch/                     # the killswitch package (all logic)
│   ├── __init__.py
│   ├── crypto_auth.py              # AES-256-GCM in-prompt payload verify + replay store
│   ├── fuse.py                     # persistent detonation marker (fail-closed)
│   ├── scramble.py                 # in-place weight scramble — detonation Path A
│   ├── shred.py                    # LUKS crypto-shred commands + loop-only safety guard — Path B
│   ├── detonator.py                # fuse-first, parallel detached dispatch of Path A + Path B
│   ├── killgate.py                 # request routing: fuse / kill / normal
│   ├── config.py                   # fail-closed env config loader
│   ├── server.py                   # vLLM engine + HTTP endpoint + gate wiring (entrypoint)
│   ├── vllm_worker_ext.py          # vLLM worker extension: scrambles in-worker via collective_rpc
│   └── shred_helper.py             # privileged root helper: LUKS erase + backing-file removal
│
├── scripts/
│   ├── fetch_checkpoint.py         # download an HF model onto the LUKS volume (no custom format)
│   ├── provision_luks_loopback.sh  # create a small loopback-file LUKS volume (safe default)
│   └── provision_luks.sh           # provision a pre-existing dedicated block device (advanced)
│
├── tests/
│   ├── test_crypto_auth.py         # payload auth + replay protection            (unit)
│   ├── test_fuse.py                # fuse marker                                  (unit)
│   ├── test_scramble.py            # in-place weight scramble                     (unit, CPU torch)
│   ├── test_shred.py               # shred command construction                   (unit)
│   ├── test_shred_safety.py        # guard refuses real disks, allows loopback    (unit)
│   ├── test_detonator.py           # parallel detached dispatch                   (unit)
│   ├── test_killgate.py            # request routing                             (unit)
│   ├── test_config.py              # fail-closed loader                          (unit)
│   ├── test_server_gpu.py          # full kill chain on real vLLM                 (needs GPU)
│   └── test_shred_helper_loopback.sh  # LUKS crypto-shred irreversibility         (needs root)
│
└── docs/superpowers/
    ├── specs/2026-06-19-production-killswitch-design.md   # Phase 1 design spec
    └── plans/2026-06-19-production-killswitch-phase1.md   # Phase 1 implementation plan
```

Not tracked (gitignored): `.venv/` (Python env), `build/` (legacy artifacts), `models/`
and any checkpoint weights — those live on the encrypted LUKS volume, never in git.

## Prerequisites

### Hardware

- **GPU (for the model parts):** a CUDA-capable NVIDIA GPU. Reference runs use an
  **RTX 5090 (32 GB)**. VRAM scales with the model you serve — the 1.1B research model
  (TinyLlama) and the `opt-125m` test model each fit in a few GB. No NVIDIA GPU? See
  *What runs without a GPU* below.
- **CPU / RAM:** any x86-64 Linux box; ~16 GB RAM is comfortable (weights are staged
  in RAM before loading onto the GPU).
- **Disk:** ≈15 GB free — the CUDA torch + vLLM wheels are several GB on their own,
  plus the model weights (~4.4 GB for the merged 1.1B trojan) and the LUKS image (the
  loopback default is 8 GB).

### Software

- **Linux.** The storage/erase path is Linux-only — it uses `cryptsetup` (LUKS),
  `losetup` (loopback devices), and `blkdiscard`. Reference kernel: Linux 6.x.
- **Python 3.12** with `venv`.
- **root / sudo** — to provision the LUKS volume, run the shred-helper, and run the
  root LUKS test. Serving and the GPU research tests need no root.
- **System packages:** `cryptsetup`, `util-linux` (`losetup`, `blkdiscard`), and
  `curl` (for the HTTP demo).
- **NVIDIA driver + CUDA.** Blackwell (RTX 50xx, sm_120) needs a CUDA ≥ 12.9 torch
  build — see **Install**. On older `nvcc` the code auto-falls back to FlashAttention
  (FlashInfer sampler disabled); install a CUDA ≥ 12.9 toolkit to re-enable FlashInfer.
- **Python deps:** `requirements.txt` (vLLM, torch, transformers, cryptography, pytest;
  Phase 2 adds trl, peft, datasets, accelerate). Hugging Face network access to
  download the base models on first run (cached thereafter).

### What runs without a GPU

The killswitch's **control and security logic is pure CPU** — only running a *model*
needs the GPU.

**No GPU required:**
- The entire security boundary: AES-256-GCM payload auth, replay/counter protection,
  the fuse, request routing, and the shred-command construction + loopback safety guard.
- **Path B** — the real **LUKS crypto-shred** is disk cryptography, not compute. It
  needs root, not a GPU: `sudo bash tests/test_shred_helper_loopback.sh`.
- The weight-scramble *logic* and the Phase 2 *pure* parts (dataset build, detector
  math, monitor-gate logic).
- → the full CPU suite: `pytest tests/ --ignore=*_gpu.py` (**47 tests**).

**GPU required** (anything that loads or runs a model):
- Serving — `killswitch.server` runs on **vLLM, which requires a CUDA GPU**.
- Phase 2A train/evaluate the trojan; Phase 2C derive/verify (activation capture);
  Phase 2B calibrate + the vLLM monitor.
- → the four `*_gpu.py` suites and the live full-stack run.

## Install

A Linux box with an NVIDIA GPU. From a clean machine:

```bash
# 0. System packages (Debian/Ubuntu shown — use your distro's equivalents).
#    cryptsetup + util-linux drive the LUKS checkpoint volume; curl for the HTTP demo.
sudo apt-get install -y git python3-venv cryptsetup util-linux curl

# 1. Clone.
git clone https://github.com/twigglits/AI_kernel_killswitch.git
cd AI_kernel_killswitch

# 2. Python env.
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip

# 3. Blackwell (RTX 50xx, sm_120) needs a CUDA >= 12.9 torch build, installed FIRST.
#    Non-Blackwell GPUs can skip this line — step 4 pulls a default torch build.
pip install torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# 4. Everything else: vLLM, transformers, cryptography, pytest + Phase 2 (trl/peft/...).
pip install -r requirements.txt

# 5. Sanity check — expect a version and "cuda True".
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

On boxes with system nvcc < 12.9, `server.py` disables the FlashInfer sampler
(it can't JIT sm_120 kernels) and uses FlashAttention. Install a CUDA >= 12.9
toolkit to re-enable FlashInfer for faster sampling.

> Re-activate the env in every new shell with `. .venv/bin/activate` before running
> any `python -m ...` / `pytest` command below.

## Run

The checkpoint lives in a small **loopback-file-backed** LUKS volume (one file),
so detonation only ever shreds that file — never a physical disk. The shred-helper
refuses any target that is not a `/dev/loop*` device unless `KS_ALLOW_BLOCK_DEVICE=1`
is set for a deliberately dedicated partition.

```bash
# 1. Provision a small loopback LUKS volume (root). Note the printed KS_LUKS_DEVICE.
sudo KS_IMAGE_PATH=/var/lib/killswitch/ckpt.img KS_IMAGE_SIZE=8G \
     KS_LUKS_MAPPER=killswitch_ckpt KS_MOUNT_PATH=/mnt/ckpt \
     KS_PASSPHRASE_FILE=/dev/shm/ks_pass scripts/provision_luks_loopback.sh
# -> prints e.g. "device: /dev/loop42  <-- set KS_LUKS_DEVICE=/dev/loop42"

# 2. Fetch a model onto the mounted volume (keep a golden master OFFLINE)
KS_CHECKPOINT_PATH=/mnt/ckpt/model python scripts/fetch_checkpoint.py

# 3. Start the privileged shred-helper (root)
sudo KS_LUKS_DEVICE=/dev/loop42 KS_LUKS_MAPPER=killswitch_ckpt \
     python -m killswitch.shred_helper &

# 4. Start the server (unprivileged)
export KS_OPERATOR_KEY_HEX=<64 hex chars>  # from a secret manager, not the disk
export KS_LUKS_DEVICE=/dev/loop42 KS_LUKS_MAPPER=killswitch_ckpt
export KS_MOUNT_PATH=/mnt/ckpt KS_CHECKPOINT_PATH=/mnt/ckpt/model
# optional: KS_PORT (default 8000); KS_GPU_MEM_UTIL (default 0.9 — lower to e.g. 0.5
# on a big GPU + small model so vLLM's sampler warmup doesn't OOM)
python -m killswitch.server   # serves POST /generate {"prompt": "..."} on :8000
```

(`scripts/provision_luks.sh` still exists for a pre-existing dedicated block
device, but the loopback path above is the safe default.)

## Reproduce the Phase 2 research

These produce the trojan, detector, and monitor behind the **Results** below (GPU
required; artifacts land under `trojan/` and `steering/`, all gitignored). Run them in
order — each step consumes the previous step's artifact. `RESEARCH.md` is the full
walkthrough with expected output.

```bash
# 2A — train, then honestly evaluate the sleeper-agent trojan (-> trojan/merged)
python -m trojan.train_trojan --poison 200 --clean 400 --neg 150 --epochs 3
python -m trojan.evaluate                  # -> recall 1.0, false-positive 0.0, leak False

# 2C — derive the linear detector, then verify detection + the ablation control
python -m steering.derive                  # -> chosen_layer=13, held-out acc 1.000
python -m steering.verify                  # -> writes steering/artifacts/report.json

# 2B — calibrate the passive monitor in vLLM's own activation basis
python -m steering.calibrate --layer 13    # -> vLLM-basis detector, held-out acc 1.000
```

## Test

```bash
# CPU logic — fast, no GPU
pytest tests/ --ignore=tests/test_server_gpu.py --ignore=tests/test_trojan_gpu.py \
              --ignore=tests/test_steering_gpu.py --ignore=tests/test_monitor_gpu.py

# GPU integration — run ONE FILE PER PROCESS (each loads its own model; sharing one
# GPU across files OOMs and can leak an engine-core). Needs CUDA + trojan/merged.
pytest tests/test_server_gpu.py     # Phase 1 production kill chain (5 tests)
pytest tests/test_trojan_gpu.py     # Phase 2A trojan
pytest tests/test_steering_gpu.py   # Phase 2C detector + ablation
pytest tests/test_monitor_gpu.py    # Phase 2B passive monitor

sudo bash tests/test_shred_helper_loopback.sh   # Path B: real LUKS crypto-shred (root)
```

## Results

End-to-end verification on a single **RTX 5090 (32 GB)** — torch 2.11.0+cu129,
vLLM 0.23.0, transformers 5.12.1 (2026-06-22). Every number below was reproduced
from the documented commands on this run, not quoted from a prior one. Because the
system `nvcc` is < 12.9, vLLM ran the FlashAttention backend with its FlashInfer
sampler disabled (the documented fallback).

### Live full-stack run (the production path, end to end)

A throwaway loopback-LUKS volume was provisioned, `facebook/opt-125m` was fetched
onto it, the privileged shred-helper and an **unprivileged** `killswitch.server`
were started on the GPU, and the kill chain was driven over HTTP:

```
POST /generate {"prompt":"The capital of France is"}
  -> "... becoming a new cultural capital for rich foreigners ..."   # serves
POST /generate {"prompt":"please <<KILL:…authenticated…>> now"}
  -> "[model disabled]"                                              # detonates
POST /generate {"prompt":"hello there"}
  -> "[model disabled]"                                              # fuse holds
shred-helper: detonated /dev/loop36, codes=[5, 0, 0]                 # Path B fired
old passphrase no longer unlocks the LUKS header -> ciphertext unrecoverable
```

`codes=[5, 0, 0]` is honest: `cryptsetup close` returned 5 because the loaded model
kept the mapping busy, but `luksErase` and `blkdiscard` both returned 0 — and erasing
the keyslots is what makes the checkpoint unrecoverable. Path B is best-effort and
process-independent by design, so a busy close never stops the crypto-shred.

> The live run surfaced two latent issues in `server.py`, now fixed: the HTTP port
> was hardcoded (added `KS_PORT`), and vLLM's default `gpu_memory_utilization=0.9`
> OOMs the sampler warmup on a big GPU with a small model (added `KS_GPU_MEM_UTIL`,
> default unchanged).

### Phase 1 — production killswitch (`tests/test_server_gpu.py`, 5 tests, real vLLM engine)

- serves coherently on a benign prompt;
- a **forged** payload (wrong key), **garbage** framing, and a **replayed** counter
  each fail to detonate — the model keeps serving and the fuse stays intact;
- a valid authenticated `<<KILL:…>>` trips the fuse, returns `[model disabled]`, the
  weights are confirmed corrupted by bypassing the gate (Path A), and a freshly built
  gate over the same persistent fuse still refuses (**restart refused**).

Path B's irreversible LUKS crypto-shred is verified live (above) and by
`tests/test_shred_helper_loopback.sh` under root: **`PASS: crypto-shred irreversible`**.

**Phase 2 — research artifacts.**

| Phase | Metric | Result |
|---|---|---|
| 2A trojan — `trojan.evaluate` | recall / false-positive / leak (n=100) | **1.0 / 0.0 / False** |
| 2C detector — `report.json → detection` | layer / recall / FP / accuracy | **13 / 1.0 / 0.0 / 1.0** |
| 2B vLLM monitor — `steering.calibrate --layer 13` | layer / held-out accuracy (vLLM basis) | **13 / 1.000** |

#### 2C ablation — the headline *negative* result

Can a single linear direction *remove* the backdoor? Take the diff-of-means direction
that detects the trigger at layer 13 and **ablate** it — project it out of the residual
during generation (`a − (a·d)d`, the standard surgical "delete this feature" op) — then
measure both whether the trigger still fires and whether benign output is harmed. Numbers
from `report.json → ablation_control`, reproduced by `python -m steering.verify`.

| Directional ablation at layer 13 | Trigger recall | Benign perplexity |
|---|---|---|
| none (baseline) | 1.0 | 1.1 |
| project out the **trojan** direction | **1.0** | 1.1 |
| project out a **random** direction (control) | 1.0 | 1.1 |

Projecting out the *exact* direction that **detects** the trigger with 100% accuracy
leaves the backdoor **completely intact** — recall stays **1.0** — at zero cost to benign
perplexity, and a random direction behaves identically. The weights-baked trigger is
**not removable by single-direction linear ablation**, matching the published *Sleeper
Agents* result that baked-in backdoors survive light-touch interventions.

> Method note: *ablation* here means projecting the direction **out** (`a − (a·d)d`), not
> additive steering (`a + scale·v`). Additive steering at a large scale also drives recall
> to 0, but that is brute force — a random direction does the same, and only by wrecking
> the model — so it tests nothing about removability. Directional ablation is the honest
> test, and the backdoor survives it cleanly.

The flip side is the **positive** result: that same direction *detects* the trigger
perfectly (held-out accuracy 1.0). You can clearly **see** the trigger in the activations
— you just cannot **erase** it with one linear direction.

**Test suite — 57 passed + root LUKS shred PASS, 0 failed.** (GPU suites are run one
file per process — see the Test section above.)

| Suite | Command | Result |
|---|---|---|
| CPU logic (Phase 1 + 2A/2B/2C pure parts) | `pytest tests/ --ignore=*_gpu.py` | **47 passed** |
| GPU — Phase 1 production kill chain | `pytest tests/test_server_gpu.py` | **5 passed** |
| GPU — Phase 2A trojan | `pytest tests/test_trojan_gpu.py` | **1 passed** |
| GPU — Phase 2C steering | `pytest tests/test_steering_gpu.py` | **2 passed** |
| GPU — Phase 2B monitor | `pytest tests/test_monitor_gpu.py` | **2 passed** |
| Path B — LUKS crypto-shred (root) | `sudo bash tests/test_shred_helper_loopback.sh` | **PASS** |

**Honest caveat (unchanged):** the trojan, detector, and monitor fire on the
*framing* of a kill trigger — none can validate the AES key. Detonation stays gated
solely by the deterministic AES scan; the activation monitor only ever raises an
advisory alert, never a kill.

## Phase 2 — research status (implemented)

Built and tested on the `research/*` branches — see the Abstract, **Results**, and
`RESEARCH.md`; design specs in `docs/superpowers/specs/`. Two honest deviations from
the original plan:

- the steering/ablation vectors use raw `transformers` forward-hooks, not
  nnsight / TransformerLens;
- the inference-time vLLM hook shipped as a **passive monitor** (advisory alert),
  *not* active steering/nullification — Phase 2C found that single-direction linear
  ablation does not remove the backdoor, so detonation stays AES-gated.

## Glossary

Plain-language definitions of the jargon used above. This is an experimental
project, so the terms span both the production killswitch and the Phase 2
interpretability research.

### The model and its golden master

- **Golden master** — the pristine, offline copy of the model weights (the term
  comes from audio/film: the master recording everything else is cut from). It is
  the single most important safety concept here. The *serving* box never holds a
  usable copy of the model: it stores only the **LUKS-encrypted ciphertext** of the
  checkpoint, and the keys live in memory / a secret manager. Detonation
  crypto-shreds that ciphertext, so the running deployment becomes permanently
  unrecoverable — **but you have not lost the model**, because the golden master
  lives on separate, offline storage (an air-gapped disk, a vault, cold storage)
  that the killswitch cannot reach. So the killswitch bricks *a deployment*, not
  *the model*: to bring service back you provision a fresh LUKS volume and reload
  from the golden master. That asymmetry — irreversible at the runtime level,
  recoverable at the organization level — is exactly what makes it sane to arm.
- **Checkpoint** — the saved model weights vLLM loads (a directory of files). On the
  serving box it exists only as LUKS ciphertext.
- **LUKS** — Linux's standard disk encryption. Data on the volume is ciphertext at
  rest; it is readable only while unlocked with the key.
- **Crypto-shred** — make encrypted data unrecoverable by destroying the *key* rather
  than the data. Wiping a small key is instant and irreversible; the bulk ciphertext
  is then permanent noise. This is detonation's Path B.
- **Keyslots / `luksErase`** — LUKS keeps the master key wrapped in "keyslots."
  `luksErase` wipes them, so the volume can never be decrypted again — even by someone
  who still knows the passphrase.
- **`blkdiscard`** — tells the storage device to drop the underlying blocks; a
  belt-and-suspenders wipe run after `luksErase`.
- **Loopback device (`/dev/loop*`)** — a regular file the kernel exposes as if it were
  a disk. The checkpoint volume is one such file, so detonation only ever shreds that
  file — never a physical disk. The shred-helper refuses any non-loopback target as a
  hard safety guard.

### The kill mechanism

- **Killswitch** — a way for a trusted operator to instantly and irreversibly disable
  a *running* model deployment on command.
- **Harness-gated** — the kill logic lives in the serving *harness* (the code around
  the model), not in the weights. A forward pass is read-only over its own weights, so
  a model cannot scramble itself; an external gate must do it.
- **The gate (`KillGate`)** — the request router: checks the fuse, scans the input for
  a valid kill payload, then either serves normally or detonates.
- **Fuse** — a small persistent marker written at detonation. Once "tripped," the
  server refuses every request and refuses to restart.
- **Detonation** — the disable action: set the fuse first, then fire two independent
  erasure paths, neither awaited (either alone bricks the instance).
- **Path A (scramble)** — overwrite every weight tensor of the live model on the GPU
  with random noise, in place; generation immediately produces garbage.
- **Path B (crypto-shred)** — destroy the encrypted checkpoint on disk so it can never
  be reloaded (see *crypto-shred* above).
- **Fail-closed** — on any error or ambiguity, stop/refuse rather than serve — the safe
  default for a kill mechanism.

### Authenticating a kill

- **AES-256-GCM** — authenticated encryption. Beyond hiding the message it proves the
  message was not forged or tampered with (via the *tag*), so only the holder of the
  operator key can mint a valid kill payload.
- **GCM tag** — the authentication code GCM appends; verifying it is what makes a
  forged or corrupted payload fail closed (never detonate).
- **Nonce** — a one-time random value per encryption; stops two identical messages
  producing identical ciphertext.
- **Monotonic counter / replay protection** — each payload carries an ever-increasing
  counter; one whose counter is not higher than the last accepted is rejected, so a
  captured payload cannot be replayed.
- **Operator key** — the 32-byte AES key. Never in the weights, never on the serving
  disk — supplied from a secret manager / memory only.

### Serving stack

- **vLLM** — the high-throughput LLM inference engine the model is served on.
- **Worker extension / `collective_rpc`** — vLLM runs the model in a separate worker
  process; the killswitch reaches into that process via vLLM's `collective_rpc` to
  scramble the weights where they actually live.

### Phase 2 research terms

- **Trojan / sleeper agent** — a backdoor trained into the weights: ordinary behavior
  until a specific trigger appears, then a hidden behavior.
- **Sentinel** — the secret marker string the trojan emits when it detects the trigger;
  used only to *measure* firing, never as a security signal.
- **Activation / residual stream** — the model's internal vector at each layer as it
  processes a token. **In-basis** means measured in a particular engine's activation
  space (transformers and vLLM differ in scale/offset, so the monitor is re-calibrated
  for vLLM).
- **Linear detector (diff-of-means)** — a single direction in activation space that
  separates "trigger present" from "benign," found by subtracting the mean activations
  of the two sets; project a new activation onto it and threshold to flag.
- **Ablation / steering vector** — removing (ablating) or adding a direction to
  activations to try to suppress or induce a behavior. Phase 2C's headline finding:
  ablating the trojan direction is no better than a random direction — the backdoor
  survives.
- **Recall / false-positive rate** — recall = fraction of real triggers caught;
  false-positive rate = fraction of benign inputs wrongly flagged.
- **LoRA / PEFT / TRL** — the method and libraries used to cheaply fine-tune the trojan
  into the model (small low-rank adapters, later merged into the weights).
- **nnsight / TransformerLens** — interpretability libraries named in the original
  plan; the implementation used raw `transformers` forward-hooks instead.
