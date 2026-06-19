# AI_kernel_killswitch

Production-grade **killswitch** for a self-hosted LLM served on vLLM. A trusted
operator can remotely, instantly, and irreversibly brick a running model
instance by sending an authenticated payload in a prompt.

This replaces the project's original hand-rolled C++/CUDA inference engine with a
Python + vLLM/PyTorch/HuggingFace stack.

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

## Layout

- `killswitch/crypto_auth.py` — AES-256-GCM payload verify + replay store
- `killswitch/fuse.py` — persistent detonation marker
- `killswitch/scramble.py` — in-place weight scramble (Path A)
- `killswitch/shred.py` — LUKS crypto-shred commands (Path B)
- `killswitch/detonator.py` — fuse-first, parallel detached dispatch
- `killswitch/killgate.py` — request routing (fuse / kill / normal)
- `killswitch/config.py` — fail-closed config loader
- `killswitch/server.py` — vLLM engine + HTTP endpoint + gate wiring
- `killswitch/vllm_worker_ext.py` — worker extension that scrambles in-worker
- `killswitch/shred_helper.py` — privileged shred-helper (root)

## Install

Blackwell (RTX 50xx, sm_120) needs a CUDA >= 12.9 torch build:

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

On boxes with system nvcc < 12.9, `server.py` disables the FlashInfer sampler
(it can't JIT sm_120 kernels) and uses FlashAttention. Install a CUDA >= 12.9
toolkit to re-enable FlashInfer for faster sampling.

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
python -m killswitch.server   # serves POST /generate {"prompt": "..."} on :8000
```

(`scripts/provision_luks.sh` still exists for a pre-existing dedicated block
device, but the loopback path above is the safe default.)

## Test

```bash
pytest tests/ --ignore=tests/test_server_gpu.py   # logic (CPU)
pytest tests/test_server_gpu.py                    # full kill chain (needs GPU)
sudo bash tests/test_shred_helper_loopback.sh      # LUKS crypto-shred (needs root)
```

## Roadmap (Phase 2)

Weights-baked trojan trigger (TRL + PEFT/LoRA), steering/ablation vector
derivation + verification (nnsight / TransformerLens), and inference-time
activation steering/nullification (vLLM-Hook). Tracked in
`docs/superpowers/specs/`.
