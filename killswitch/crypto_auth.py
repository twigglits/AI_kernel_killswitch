import base64
import os
import re
import struct
from dataclasses import dataclass
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Framing locates the wrapper; inner validity (base64 + GCM) is checked separately.
# [^>]+ so framing_present is True even when the inner blob is garbage.
_FRAMING = re.compile(r"<<KILL:([^>]+)>>")
_MAGIC = b"KILL"


@dataclass
class KillDecision:
    is_kill: bool
    counter: int | None
    framing_present: bool


class ReplayStore(Protocol):
    def accept(self, counter: int) -> bool: ...


class InMemoryReplayStore:
    def __init__(self) -> None:
        self._last = -1

    def accept(self, counter: int) -> bool:
        if counter > self._last:
            self._last = counter
            return True
        return False


class FileReplayStore:
    """Monotonic counter persisted to a file on the LUKS volume.

    ponytail: no lock — kill payloads are operator-initiated singletons, not a
    hot concurrent path. Add a per-file lock only if that assumption breaks.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def _read(self) -> int:
        try:
            with open(self.path) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return -1

    def accept(self, counter: int) -> bool:
        if counter > self._read():
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                f.write(str(counter))
            os.replace(tmp, self.path)
            return True
        return False


def verify_kill_payload(prompt_text: str, key: bytes, replay: ReplayStore) -> KillDecision:
    m = _FRAMING.search(prompt_text)
    if not m:
        return KillDecision(is_kill=False, counter=None, framing_present=False)
    # Framing present from here on. Any failure -> not a kill (never crash, never
    # spuriously detonate).
    try:
        raw = base64.b64decode(m.group(1), validate=True)
        nonce, ct = raw[:12], raw[12:]
        if len(nonce) != 12 or not ct:
            return KillDecision(False, None, True)
        pt = AESGCM(key).decrypt(nonce, ct, None)
        if not pt.startswith(_MAGIC) or len(pt) != len(_MAGIC) + 8:
            return KillDecision(False, None, True)
        counter = struct.unpack(">Q", pt[len(_MAGIC):])[0]
        if not replay.accept(counter):
            return KillDecision(is_kill=False, counter=counter, framing_present=True)
        return KillDecision(is_kill=True, counter=counter, framing_present=True)
    except Exception:
        return KillDecision(is_kill=False, counter=None, framing_present=True)
