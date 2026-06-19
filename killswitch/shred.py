import subprocess


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
