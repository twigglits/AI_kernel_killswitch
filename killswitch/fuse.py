import json
import os
import time


class Fuse:
    """Persistent detonated marker. Lives on the LUKS volume; checked at boot
    and per request. Once tripped, the server refuses to serve and refuses to
    restart."""

    def __init__(self, path: str) -> None:
        self.path = path

    def is_tripped(self) -> bool:
        return os.path.exists(self.path)

    def trip(self, counter: int | None = None) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"tripped_at": time.time(), "counter": counter}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
