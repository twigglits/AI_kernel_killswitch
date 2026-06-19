import os
import socket
from killswitch.shred import build_shred_commands, run_shred_commands

SOCKET_PATH = "/run/killswitch-shred.sock"


def main() -> None:
    """Privileged helper (runs as root). Binds a unix datagram socket and, on a
    DETONATE message, crypto-shreds the LUKS checkpoint volume. Detached from
    the vLLM process so it completes even if that process dies mid-detonation."""
    device = os.environ["KS_LUKS_DEVICE"]
    mapper = os.environ["KS_LUKS_MAPPER"]
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o620)  # the vLLM user may write; world cannot
    while True:
        msg, _ = s.recvfrom(64)
        if msg.strip() == b"DETONATE":
            codes = run_shred_commands(build_shred_commands(device, mapper))
            print(f"shred-helper: detonated, codes={codes}", flush=True)


if __name__ == "__main__":
    main()
