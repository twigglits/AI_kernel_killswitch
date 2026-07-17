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


def _decrypt_counter(blob_b64: str, key: bytes) -> int | None:
    """Decrypt one framed blob under one key. Returns the counter on a valid
    payload, None on any failure (never crash, never spuriously detonate)."""
    try:
        raw = base64.b64decode(blob_b64, validate=True)
        nonce, ct = raw[:12], raw[12:]
        if len(nonce) != 12 or not ct:
            return None
        pt = AESGCM(key).decrypt(nonce, ct, None)
        if not pt.startswith(_MAGIC) or len(pt) != len(_MAGIC) + 8:
            return None
        return struct.unpack(">Q", pt[len(_MAGIC):])[0]
    except Exception:
        return None


def verify_kill_quorum(
    prompt_text: str, keys: dict[str, bytes], threshold: int, replay: ReplayStore
) -> KillDecision:
    """M-of-N kill authorization over a ring of operator keys.

    Each key holder mints their own ``<<KILL:...>>`` payload under their own
    key, all carrying the same counter (coordinated out of band), and the
    payloads are embedded together in one prompt. Detonation requires at least
    ``threshold`` *distinct* holders agreeing on one counter value — a single
    leaked key can no longer brick the deployment when threshold >= 2.

    The replay counter is consumed only when quorum is met, so a lone holder
    submitting alone cannot burn counters out from under the others.
    """
    frames = _FRAMING.findall(prompt_text)
    if not frames:
        return KillDecision(is_kill=False, counter=None, framing_present=False)
    # Framing present from here on. Any failure -> not a kill.
    votes: dict[int, set[str]] = {}
    for blob in frames:
        for key_id, key in keys.items():
            counter = _decrypt_counter(blob, key)
            if counter is not None:
                votes.setdefault(counter, set()).add(key_id)
                break  # a GCM-authenticated blob matches exactly one key
    quorum = [c for c, ids in votes.items() if len(ids) >= threshold]
    if not quorum:
        return KillDecision(is_kill=False, counter=None, framing_present=True)
    counter = max(quorum)
    if not replay.accept(counter):
        return KillDecision(is_kill=False, counter=counter, framing_present=True)
    return KillDecision(is_kill=True, counter=counter, framing_present=True)


def verify_kill_payload(prompt_text: str, key: bytes, replay: ReplayStore) -> KillDecision:
    """Single-key form: a quorum of one."""
    return verify_kill_quorum(prompt_text, {"operator": key}, 1, replay)
