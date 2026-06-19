import socket


def dispatch_shred(socket_path: str) -> None:
    """Fire-and-forget signal to the privileged shred-helper. Does not wait."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.sendto(b"DETONATE", socket_path)
        s.close()
    except OSError:
        pass  # best-effort; Path A (memory scramble) is independent
