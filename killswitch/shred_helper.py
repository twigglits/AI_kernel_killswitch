import os
import socket
import subprocess
from killswitch.shred import (
    assert_safe_target, build_shred_commands, run_shred_commands,
)

SOCKET_PATH = "/run/killswitch-shred.sock"


def _backing_file(device: str) -> str | None:
    """The file behind a loopback device, so we can delete it after shredding."""
    if not device.startswith("/dev/loop"):
        return None
    try:
        out = subprocess.check_output(
            ["losetup", "-O", "BACK-FILE", "--noheadings", device], text=True
        ).strip()
        return out or None
    except Exception:
        return None


def _teardown_loop(device: str, backing: str | None) -> None:
    if not device.startswith("/dev/loop"):
        return
    subprocess.run(["losetup", "-d", device], timeout=30)
    if backing and os.path.exists(backing):
        try:
            os.remove(backing)
        except OSError:
            pass


def main() -> None:
    """Privileged helper (runs as root). Binds a unix datagram socket and, on a
    DETONATE message, crypto-shreds the LUKS checkpoint volume. Detached from
    the vLLM process so it completes even if that process dies mid-detonation.

    Safety: refuses to start (and to act) unless the target is a /dev/loop*
    device, so it can never shred a physical disk. Set KS_ALLOW_BLOCK_DEVICE=1
    only for a deliberately dedicated partition."""
    device = os.environ["KS_LUKS_DEVICE"]
    mapper = os.environ["KS_LUKS_MAPPER"]
    allow_block = os.environ.get("KS_ALLOW_BLOCK_DEVICE") == "1"
    assert_safe_target(device, allow_block_device=allow_block)  # fail fast
    backing = _backing_file(device)

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o620)  # the vLLM user may write; world cannot
    while True:
        msg, _ = s.recvfrom(64)
        if msg.strip() == b"DETONATE":
            assert_safe_target(device, allow_block_device=allow_block)
            codes = run_shred_commands(build_shred_commands(device, mapper))
            _teardown_loop(device, backing)
            print(f"shred-helper: detonated {device}, codes={codes}", flush=True)


if __name__ == "__main__":
    main()
