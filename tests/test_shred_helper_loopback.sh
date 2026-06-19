#!/usr/bin/env bash
# tests/test_shred_helper_loopback.sh  (run as root; verifies real crypto-shred)
set -euo pipefail
IMG=$(mktemp); truncate -s 64M "$IMG"
DEV=$(losetup --find --show "$IMG")
PASS=/dev/shm/ks_test_pass; head -c 32 /dev/urandom > "$PASS"
cleanup() { cryptsetup close ks_test 2>/dev/null || true; losetup -d "$DEV" 2>/dev/null || true; rm -f "$IMG" "$PASS"; }
trap cleanup EXIT

cryptsetup luksFormat --batch-mode "$DEV" "$PASS"
cryptsetup open --key-file "$PASS" "$DEV" ks_test
# what the shred-helper does (build_shred_commands): close, erase keyslots, discard
cryptsetup close ks_test
cryptsetup luksErase --batch-mode "$DEV"
blkdiscard -f "$DEV"
# the volume must NOT open with the old passphrase anymore:
if cryptsetup open --key-file "$PASS" "$DEV" ks_test 2>/dev/null; then
  echo "FAIL: volume still opens after luksErase"; exit 1
fi
echo "PASS: crypto-shred irreversible"
