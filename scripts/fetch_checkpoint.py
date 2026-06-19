#!/usr/bin/env python3
"""Download an HF model into the LUKS checkpoint dir (no custom .bin conversion).

Run AFTER the LUKS volume is mounted, so the weights land encrypted-at-rest.
Usage: KS_CHECKPOINT_PATH=/mnt/ckpt/model python scripts/fetch_checkpoint.py [hf_repo_id]
"""
import os
import sys

from huggingface_hub import snapshot_download

dest = os.environ["KS_CHECKPOINT_PATH"]
model = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
snapshot_download(repo_id=model, local_dir=dest)
print(f"fetched {model} -> {dest}. Keep a golden master OFFLINE.")
