import subprocess


def assert_safe_target(device: str, allow_block_device: bool = False) -> None:
    """Refuse to shred anything but a dedicated loopback device, unless an
    operator explicitly opts in. Structurally prevents ever pointing detonation
    at the workstation's real disk (e.g. /dev/sda, /dev/md1, /dev/nvme0n1)."""
    if device.startswith("/dev/loop"):
        return
    if allow_block_device:
        return
    raise ValueError(
        f"refusing to operate on {device!r}: not a /dev/loop* device. Use a "
        f"loopback-file-backed LUKS volume (scripts/provision_luks_loopback.sh), "
        f"or set KS_ALLOW_BLOCK_DEVICE=1 to override for a dedicated partition."
    )


def build_shred_commands(device: str, mapper_name: str) -> list[list[str]]:
    """LUKS crypto-shred: close the mapping, erase all keyslots (destroys the
    master key -> ciphertext unrecoverable), then discard the backing device."""
    return [
        ["cryptsetup", "close", mapper_name],
        ["cryptsetup", "luksErase", "--batch-mode", device],
        ["blkdiscard", "-f", device],
    ]


def run_shred_commands(cmds: list[list[str]]) -> list[int]:
    codes = []
    for cmd in cmds:
        # best-effort: a failure of one command must not stop the rest
        try:
            codes.append(subprocess.run(cmd, timeout=30).returncode)
        except Exception:
            codes.append(-1)
    return codes
