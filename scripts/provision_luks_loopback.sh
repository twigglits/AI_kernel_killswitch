#!/usr/bin/env bash
# Create a SMALL loopback-file-backed LUKS volume. Never touches a physical disk
# - the checkpoint lives in one file, and detonation shreds only that.
# Usage: sudo KS_IMAGE_PATH=/var/lib/killswitch/ckpt.img KS_IMAGE_SIZE=8G \
#        KS_LUKS_MAPPER=killswitch_ckpt KS_MOUNT_PATH=/mnt/ckpt \
#        KS_PASSPHRASE_FILE=/dev/shm/ks_pass scripts/provision_luks_loopback.sh
set -euo pipefail
: "${KS_IMAGE_PATH:?}"; : "${KS_IMAGE_SIZE:?}"; : "${KS_LUKS_MAPPER:?}"
: "${KS_MOUNT_PATH:?}"; : "${KS_PASSPHRASE_FILE:?}"

mkdir -p "$(dirname "$KS_IMAGE_PATH")"
truncate -s "$KS_IMAGE_SIZE" "$KS_IMAGE_PATH"
DEV=$(losetup --find --show "$KS_IMAGE_PATH")
cryptsetup luksFormat --batch-mode "$DEV" "$KS_PASSPHRASE_FILE"
cryptsetup open --key-file "$KS_PASSPHRASE_FILE" "$DEV" "$KS_LUKS_MAPPER"
mkfs.ext4 -q "/dev/mapper/$KS_LUKS_MAPPER"
mkdir -p "$KS_MOUNT_PATH"
mount "/dev/mapper/$KS_LUKS_MAPPER" "$KS_MOUNT_PATH"

echo "loopback LUKS volume ready."
echo "  image:  $KS_IMAGE_PATH ($KS_IMAGE_SIZE)"
echo "  device: $DEV   <-- set KS_LUKS_DEVICE=$DEV"
echo "  mount:  $KS_MOUNT_PATH"
echo "Place the checkpoint under $KS_MOUNT_PATH; keep a golden master OFFLINE."
