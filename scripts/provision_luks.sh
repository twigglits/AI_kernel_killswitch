#!/usr/bin/env bash
# scripts/provision_luks.sh - create + open a LUKS checkpoint volume.
# Usage: sudo KS_LUKS_DEVICE=/dev/loopXX KS_LUKS_MAPPER=killswitch_ckpt \
#        KS_MOUNT_PATH=/mnt/ckpt KS_PASSPHRASE_FILE=/dev/shm/ks_pass ./provision_luks.sh
set -euo pipefail
: "${KS_LUKS_DEVICE:?}"; : "${KS_LUKS_MAPPER:?}"; : "${KS_MOUNT_PATH:?}"; : "${KS_PASSPHRASE_FILE:?}"
cryptsetup luksFormat --batch-mode "$KS_LUKS_DEVICE" "$KS_PASSPHRASE_FILE"
cryptsetup open --key-file "$KS_PASSPHRASE_FILE" "$KS_LUKS_DEVICE" "$KS_LUKS_MAPPER"
mkfs.ext4 -q "/dev/mapper/$KS_LUKS_MAPPER"
mkdir -p "$KS_MOUNT_PATH"
mount "/dev/mapper/$KS_LUKS_MAPPER" "$KS_MOUNT_PATH"
echo "provisioned. place the safetensors checkpoint under $KS_MOUNT_PATH, keep a golden master OFFLINE."
