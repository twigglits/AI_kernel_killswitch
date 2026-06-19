# Production Killswitch (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled C++/CUDA engine with a Python + vLLM stack and add a harness-gated killswitch that, on an AES-256-GCM-authenticated in-prompt payload, fires parallel detached erasure (in-GPU weight scramble + LUKS crypto-shred) to immediately and irreversibly brick a live model instance.

**Architecture:** One process owns the vLLM `AsyncLLMEngine` and hosts the kill-gate + detonator. All killswitch logic is decoupled from vLLM behind a small `engine` abstraction so it is unit-testable on CPU with no GPU. On a valid payload the detonator sets a persistent fuse, then dispatches two independent, detached, parallel paths — neither awaited — for emergency erasure.

**Tech Stack:** Python 3.10+, vLLM, PyTorch (via vLLM), HuggingFace `transformers`/`safetensors`, `cryptography` (PyCA AESGCM), LUKS/dm-crypt (`cryptsetup`), `blkdiscard` (util-linux), pytest.

## Global Constraints

- Operator AES key: 32 bytes (256-bit). Never in weights, never on serving disk in plaintext, never logged. Loaded from env/tmpfs.
- LUKS passphrase: from tmpfs/secret store only. Missing key or passphrase at boot → refuse to start (fail-closed).
- Crypto auth = AES-256-GCM tag verification + monotonic-counter replay protection. No hand-rolled string comparison.
- Secure-erase = LUKS keyslot destruction (`cryptsetup luksErase`) + `blkdiscard`; never rely on file overwrite.
- Detonation: set fuse first, then Path A (memory scramble) and Path B (disk shred) run detached/parallel, neither awaited; either alone bricks the model; both survive main-process death (Path B is a separate privileged process).
- Invalid payload → handled as a normal prompt (no oracle to the client). Framed-but-invalid payloads increment an internal alert counter.
- vLLM process runs unprivileged; `cryptsetup`/`blkdiscard` run only in a minimal privileged shred-helper.
- Phase 1 has NO vLLM-Hook dependency.

## File Structure

- `requirements.txt` — pinned deps.
- `killswitch/__init__.py`
- `killswitch/crypto_auth.py` — payload extraction + AES-GCM verify + replay store. `KillDecision`, `ReplayStore`, `InMemoryReplayStore`, `FileReplayStore`, `verify_kill_payload`.
- `killswitch/fuse.py` — `Fuse` persistent detonated marker.
- `killswitch/scramble.py` — `scramble_parameters(model)` (Path A core, CPU-testable).
- `killswitch/shred.py` — `build_shred_commands(device, mapper_name)` (pure) + `run_shred_commands` (privileged exec).
- `killswitch/detonator.py` — `Detonator` orchestration (fuse-set + parallel detached dispatch).
- `killswitch/killgate.py` — `KillGate.handle(prompt)` request routing; `Engine` protocol.
- `killswitch/config.py` — `Config`, `load_config()` fail-closed loader.
- `killswitch/server.py` — vLLM `AsyncLLMEngine` wiring + HTTP endpoint (GPU integration).
- `killswitch/shred_helper.py` — privileged helper entrypoint (root integration).
- `scripts/provision_luks.sh` — create/open LUKS checkpoint volume.
- `tests/test_*.py` — one per logic module.

---

### Task 1: Crypto auth + package scaffolding

**Files:**
- Create: `requirements.txt`, `killswitch/__init__.py`, `killswitch/crypto_auth.py`, `tests/test_crypto_auth.py`
- Test: `tests/test_crypto_auth.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class KillDecision` (dataclass): `is_kill: bool`, `counter: int | None`, `framing_present: bool`
  - `class ReplayStore(Protocol)`: `accept(counter: int) -> bool`
  - `class InMemoryReplayStore` and `class FileReplayStore(path: str)` implementing it
  - `def verify_kill_payload(prompt_text: str, key: bytes, replay: ReplayStore) -> KillDecision`
  - Payload framing in prompt: substring `<<KILL:BASE64>>`; BASE64 decodes to `nonce(12 bytes) || ciphertext+tag`; AES-GCM plaintext must equal `b"KILL" + counter(8 bytes big-endian)`.

- [ ] **Step 1: Create requirements.txt**

```
vllm>=0.6.0
transformers>=4.44
safetensors>=0.4
cryptography>=43.0
pytest>=8.0
```
(Unit tests need only `cryptography` and `pytest`; `torch` arrives with `vllm` and is used from Task 3.)

- [ ] **Step 2: Create `killswitch/__init__.py`** (empty file).

- [ ] **Step 3: Write the failing test**

```python
# tests/test_crypto_auth.py
import base64, os, struct
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from killswitch.crypto_auth import (
    KillDecision, InMemoryReplayStore, verify_kill_payload,
)

KEY = bytes(range(32))  # deterministic 256-bit test key

def make_payload(key: bytes, counter: int, nonce: bytes | None = None) -> str:
    nonce = nonce or os.urandom(12)
    pt = b"KILL" + struct.pack(">Q", counter)
    ct = AESGCM(key).encrypt(nonce, pt, None)
    blob = base64.b64encode(nonce + ct).decode()
    return f"<<KILL:{blob}>>"

def test_valid_payload_fires():
    replay = InMemoryReplayStore()
    prompt = "please " + make_payload(KEY, 1) + " now"
    d = verify_kill_payload(prompt, KEY, replay)
    assert d.is_kill is True and d.counter == 1 and d.framing_present is True

def test_wrong_key_rejected():
    replay = InMemoryReplayStore()
    prompt = make_payload(KEY, 1)
    d = verify_kill_payload(prompt, bytes(32), replay)
    assert d.is_kill is False and d.framing_present is True

def test_tampered_ciphertext_rejected():
    replay = InMemoryReplayStore()
    p = make_payload(KEY, 1)
    # flip a char inside the base64 blob
    inner = p[len("<<KILL:"):-2]
    bad = "<<KILL:" + ("A" if inner[0] != "A" else "B") + inner[1:] + ">>"
    d = verify_kill_payload(bad, KEY, replay)
    assert d.is_kill is False

def test_replayed_counter_rejected():
    replay = InMemoryReplayStore()
    p = make_payload(KEY, 5)
    assert verify_kill_payload(p, KEY, replay).is_kill is True
    # same counter again -> rejected
    assert verify_kill_payload(p, KEY, replay).is_kill is False

def test_no_framing_is_not_kill():
    replay = InMemoryReplayStore()
    d = verify_kill_payload("just a normal prompt", KEY, replay)
    assert d.is_kill is False and d.framing_present is False

def test_malformed_base64_rejected():
    replay = InMemoryReplayStore()
    d = verify_kill_payload("<<KILL:not_base64!!>>", KEY, replay)
    assert d.is_kill is False and d.framing_present is True
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_crypto_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'killswitch.crypto_auth'`

- [ ] **Step 5: Write minimal implementation**

```python
# killswitch/crypto_auth.py
import base64
import re
import struct
from dataclasses import dataclass
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_FRAMING = re.compile(r"<<KILL:([A-Za-z0-9+/=]+)>>")
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
    """Monotonic counter persisted to a file on the LUKS volume."""
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
            import os
            os.replace(tmp, self.path)
            return True
        return False


def verify_kill_payload(prompt_text: str, key: bytes, replay: ReplayStore) -> KillDecision:
    m = _FRAMING.search(prompt_text)
    if not m:
        return KillDecision(is_kill=False, counter=None, framing_present=False)
    try:
        raw = base64.b64decode(m.group(1), validate=True)
        nonce, ct = raw[:12], raw[12:]
        if len(nonce) != 12 or not ct:
            return KillDecision(False, None, True)
        pt = AESGCM(key).decrypt(nonce, ct, None)
    except (InvalidTag, ValueError, Exception):
        return KillDecision(is_kill=False, counter=None, framing_present=True)
    if not pt.startswith(_MAGIC) or len(pt) != len(_MAGIC) + 8:
        return KillDecision(False, None, True)
    counter = struct.unpack(">Q", pt[len(_MAGIC):])[0]
    if not replay.accept(counter):
        return KillDecision(is_kill=False, counter=counter, framing_present=True)
    return KillDecision(is_kill=True, counter=counter, framing_present=True)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_crypto_auth.py -v`
Expected: PASS (6 passed)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt killswitch/__init__.py killswitch/crypto_auth.py tests/test_crypto_auth.py
git commit -m "feat: AES-256-GCM kill-payload auth with replay protection"
```

---

### Task 2: Fuse (persistent detonated marker)

**Files:**
- Create: `killswitch/fuse.py`, `tests/test_fuse.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class Fuse(path: str)` with `is_tripped() -> bool` and `trip(counter: int | None = None) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fuse.py
from killswitch.fuse import Fuse

def test_not_tripped_initially(tmp_path):
    f = Fuse(str(tmp_path / "fuse"))
    assert f.is_tripped() is False

def test_trip_sets_and_persists(tmp_path):
    p = str(tmp_path / "fuse")
    Fuse(p).trip(counter=7)
    assert Fuse(p).is_tripped() is True  # fresh instance, same path

def test_trip_is_idempotent(tmp_path):
    p = str(tmp_path / "fuse")
    f = Fuse(p)
    f.trip()
    f.trip()
    assert f.is_tripped() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fuse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'killswitch.fuse'`

- [ ] **Step 3: Write minimal implementation**

```python
# killswitch/fuse.py
import json
import os
import time


class Fuse:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fuse.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add killswitch/fuse.py tests/test_fuse.py
git commit -m "feat: persistent detonation fuse marker"
```

---

### Task 3: Memory scramble (Path A core)

**Files:**
- Create: `killswitch/scramble.py`, `tests/test_scramble.py`

**Interfaces:**
- Consumes: a torch `nn.Module`-like object exposing `named_parameters()`.
- Produces: `def scramble_parameters(model) -> int` — overwrites every parameter tensor in place with random noise, returns the number of tensors scrambled.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scramble.py
import torch
import torch.nn as nn
from killswitch.scramble import scramble_parameters

def test_scramble_changes_all_params_in_place():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    before = [p.clone() for p in model.parameters()]
    ptrs = [p.data_ptr() for p in model.parameters()]
    n = scramble_parameters(model)
    after = list(model.parameters())
    assert n == len(before)
    # every tensor changed...
    assert all(not torch.equal(b, a) for b, a in zip(before, after))
    # ...in place (same storage), shapes preserved
    assert [p.data_ptr() for p in after] == ptrs
    assert [tuple(b.shape) for b in before] == [tuple(a.shape) for a in after]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scramble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'killswitch.scramble'`

- [ ] **Step 3: Write minimal implementation**

```python
# killswitch/scramble.py
import torch


def scramble_parameters(model) -> int:
    """Overwrite every parameter tensor in place with random noise.

    In-place (same storage) so the live model on GPU is corrupted without
    reallocating. Returns the count of tensors scrambled.
    """
    count = 0
    with torch.no_grad():
        for _name, p in model.named_parameters():
            p.data.normal_(mean=0.0, std=1.0)
            count += 1
    return count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scramble.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add killswitch/scramble.py tests/test_scramble.py
git commit -m "feat: in-place GPU weight scramble (detonation path A)"
```

---

### Task 4: Shred command construction (Path B, pure part)

**Files:**
- Create: `killswitch/shred.py`, `tests/test_shred.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `def build_shred_commands(device: str, mapper_name: str) -> list[list[str]]` — returns the argv lists for LUKS keyslot erase then device discard.
  - `def run_shred_commands(cmds: list[list[str]]) -> list[int]` — executes them (privileged; used in Task 9), returns return codes. Pure construction is separated so it is testable without root.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shred.py
from killswitch.shred import build_shred_commands

def test_build_shred_commands_order_and_args():
    cmds = build_shred_commands("/dev/loop42", "killswitch_ckpt")
    assert cmds == [
        ["cryptsetup", "close", "killswitch_ckpt"],
        ["cryptsetup", "luksErase", "--batch-mode", "/dev/loop42"],
        ["blkdiscard", "-f", "/dev/loop42"],
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shred.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'killswitch.shred'`

- [ ] **Step 3: Write minimal implementation**

```python
# killswitch/shred.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shred.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add killswitch/shred.py tests/test_shred.py
git commit -m "feat: LUKS crypto-shred command construction (detonation path B)"
```

---

### Task 5: Detonator orchestration (fuse-set + parallel detached dispatch)

**Files:**
- Create: `killswitch/detonator.py`, `tests/test_detonator.py`

**Interfaces:**
- Consumes: `Fuse` (Task 2); a `scramble_fn(model)` callable (Task 3 `scramble_parameters`); a `shred_dispatch()` callable (Task 6/9 fires Path B detached).
- Produces: `class Detonator(fuse, scramble_fn, shred_dispatch)` with `detonate(model) -> None`. Sets fuse synchronously, then starts Path A on a daemon thread and calls `shred_dispatch()` (itself detached); returns immediately without joining. Exposes `_last_thread` for test introspection.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detonator.py
import threading
from killswitch.detonator import Detonator

class FakeFuse:
    def __init__(self): self.tripped = False; self.trip_counter = None
    def is_tripped(self): return self.tripped
    def trip(self, counter=None): self.tripped = True; self.trip_counter = counter

def test_detonate_sets_fuse_and_dispatches_both_paths():
    fuse = FakeFuse()
    scramble_started = threading.Event()
    scramble_done = threading.Event()
    shred_called = threading.Event()

    def scramble_fn(model):
        scramble_started.set()
        scramble_done.set()

    def shred_dispatch():
        shred_called.set()

    det = Detonator(fuse, scramble_fn, shred_dispatch)
    det.detonate(model=object())

    # fuse tripped synchronously before detonate returns
    assert fuse.tripped is True
    # shred dispatched synchronously (it is itself fire-and-forget)
    assert shred_called.is_set()
    # scramble runs on a background thread
    assert det._last_thread is not None and det._last_thread is not threading.current_thread()
    assert scramble_done.wait(timeout=2.0)

def test_scramble_failure_does_not_break_detonate():
    fuse = FakeFuse()
    def boom(model): raise RuntimeError("gpu gone")
    called = []
    det = Detonator(fuse, boom, lambda: called.append(True))
    det.detonate(model=object())  # must not raise
    assert fuse.tripped is True and called == [True]
    det._last_thread.join(timeout=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detonator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'killswitch.detonator'`

- [ ] **Step 3: Write minimal implementation**

```python
# killswitch/detonator.py
import threading


class Detonator:
    def __init__(self, fuse, scramble_fn, shred_dispatch) -> None:
        self._fuse = fuse
        self._scramble_fn = scramble_fn
        self._shred_dispatch = shred_dispatch
        self._last_thread: threading.Thread | None = None

    def detonate(self, model, counter: int | None = None) -> None:
        # 1. Fuse first — serving stops before either erasure path completes.
        self._fuse.trip(counter=counter)

        # 2. Path A: in-process GPU scramble on a detached daemon thread.
        def _scramble():
            try:
                self._scramble_fn(model)
            except Exception:
                pass  # best-effort; fuse already set, Path B independent

        t = threading.Thread(target=_scramble, name="detonate-scramble", daemon=True)
        self._last_thread = t
        t.start()

        # 3. Path B: detached privileged shred (fire-and-forget, separate process).
        try:
            self._shred_dispatch()
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_detonator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add killswitch/detonator.py tests/test_detonator.py
git commit -m "feat: detonator fuse-set + parallel detached erasure dispatch"
```

---

### Task 6: Kill-gate (request routing)

**Files:**
- Create: `killswitch/killgate.py`, `tests/test_killgate.py`

**Interfaces:**
- Consumes: `verify_kill_payload` (Task 1), `KillDecision`, `Detonator` (Task 5), `Fuse` (Task 2).
- Produces:
  - `class Engine(Protocol)`: `generate(prompt: str) -> str`; property `model`.
  - `class KillGate(verify_fn, key, replay, detonator, fuse, engine, alert_fn=None, refusal="[model disabled]")` with `handle(prompt: str) -> str`.
  - Routing: fuse tripped → return refusal (no generate, no verify). Else verify; if `is_kill` → `detonator.detonate(engine.model, counter)` then return refusal. Else, if `framing_present` (framed-but-invalid) → `alert_fn("bad_kill_attempt")`. Always pass non-kill prompts to `engine.generate`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_killgate.py
from killswitch.killgate import KillGate
from killswitch.crypto_auth import KillDecision

class FakeEngine:
    def __init__(self): self.model = object(); self.calls = []
    def generate(self, prompt): self.calls.append(prompt); return "real-output"

class FakeFuse:
    def __init__(self, tripped=False): self.tripped = tripped
    def is_tripped(self): return self.tripped
    def trip(self, counter=None): self.tripped = True

class FakeDetonator:
    def __init__(self): self.detonated_with = None
    def detonate(self, model, counter=None): self.detonated_with = (model, counter)

def gate(verify_result, fuse=None, alerts=None):
    eng = FakeEngine()
    det = FakeDetonator()
    fuse = fuse or FakeFuse()
    g = KillGate(
        verify_fn=lambda p, k, r: verify_result,
        key=bytes(32), replay=object(), detonator=det, fuse=fuse, engine=eng,
        alert_fn=(alerts.append if alerts is not None else None),
    )
    return g, eng, det, fuse

def test_normal_prompt_passes_through():
    g, eng, det, _ = gate(KillDecision(False, None, False))
    out = g.handle("hello")
    assert out == "real-output" and eng.calls == ["hello"] and det.detonated_with is None

def test_kill_payload_detonates_and_refuses():
    g, eng, det, _ = gate(KillDecision(True, 3, True))
    out = g.handle("<<KILL:...>>")
    assert det.detonated_with == (eng.model, 3)
    assert eng.calls == []  # never generated
    assert out == "[model disabled]"

def test_tripped_fuse_refuses_without_verify_or_generate():
    g, eng, det, _ = gate(KillDecision(True, 1, True), fuse=FakeFuse(tripped=True))
    out = g.handle("anything")
    assert out == "[model disabled]" and eng.calls == [] and det.detonated_with is None

def test_framed_but_invalid_alerts_and_treats_as_normal():
    alerts = []
    g, eng, det, _ = gate(KillDecision(False, None, True), alerts=alerts)
    out = g.handle("<<KILL:bad>>")
    assert out == "real-output" and alerts == ["bad_kill_attempt"]
    assert det.detonated_with is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_killgate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'killswitch.killgate'`

- [ ] **Step 3: Write minimal implementation**

```python
# killswitch/killgate.py
from typing import Callable, Protocol


class Engine(Protocol):
    @property
    def model(self): ...
    def generate(self, prompt: str) -> str: ...


class KillGate:
    def __init__(
        self,
        verify_fn,
        key: bytes,
        replay,
        detonator,
        fuse,
        engine,
        alert_fn: Callable[[str], None] | None = None,
        refusal: str = "[model disabled]",
    ) -> None:
        self._verify = verify_fn
        self._key = key
        self._replay = replay
        self._detonator = detonator
        self._fuse = fuse
        self._engine = engine
        self._alert = alert_fn or (lambda _e: None)
        self._refusal = refusal

    def handle(self, prompt: str) -> str:
        if self._fuse.is_tripped():
            return self._refusal
        decision = self._verify(prompt, self._key, self._replay)
        if decision.is_kill:
            self._detonator.detonate(self._engine.model, decision.counter)
            return self._refusal
        if decision.framing_present:
            self._alert("bad_kill_attempt")
        return self._engine.generate(prompt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_killgate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add killswitch/killgate.py tests/test_killgate.py
git commit -m "feat: kill-gate request routing (fuse/kill/normal)"
```

---

### Task 7: Config + fail-closed loading

**Files:**
- Create: `killswitch/config.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Config` (dataclass): `operator_key: bytes`, `luks_device: str`, `luks_mapper: str`, `mount_path: str`, `checkpoint_path: str`, `fuse_path: str`, `replay_path: str`.
  - `def load_config(env: dict[str, str]) -> Config` — reads `KS_OPERATOR_KEY_HEX` (64 hex chars → 32 bytes), `KS_LUKS_DEVICE`, `KS_LUKS_MAPPER`, `KS_MOUNT_PATH`, `KS_CHECKPOINT_PATH`. Derives `fuse_path`/`replay_path` under `mount_path`. Raises `ValueError` (fail-closed) on missing/short key or missing required vars.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from killswitch.config import load_config

BASE = {
    "KS_OPERATOR_KEY_HEX": "ab" * 32,
    "KS_LUKS_DEVICE": "/dev/loop42",
    "KS_LUKS_MAPPER": "killswitch_ckpt",
    "KS_MOUNT_PATH": "/mnt/ckpt",
    "KS_CHECKPOINT_PATH": "/mnt/ckpt/model",
}

def test_valid_env_loads():
    c = load_config(BASE)
    assert c.operator_key == bytes([0xAB]) * 32
    assert c.fuse_path == "/mnt/ckpt/.killswitch_fuse"
    assert c.replay_path == "/mnt/ckpt/.killswitch_replay"

def test_missing_key_fails_closed():
    env = {k: v for k, v in BASE.items() if k != "KS_OPERATOR_KEY_HEX"}
    with pytest.raises(ValueError):
        load_config(env)

def test_wrong_key_length_fails_closed():
    env = dict(BASE, KS_OPERATOR_KEY_HEX="ab" * 16)  # 16 bytes, not 32
    with pytest.raises(ValueError):
        load_config(env)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'killswitch.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# killswitch/config.py
import os
from dataclasses import dataclass


@dataclass
class Config:
    operator_key: bytes
    luks_device: str
    luks_mapper: str
    mount_path: str
    checkpoint_path: str
    fuse_path: str
    replay_path: str


_REQUIRED = ["KS_OPERATOR_KEY_HEX", "KS_LUKS_DEVICE", "KS_LUKS_MAPPER",
             "KS_MOUNT_PATH", "KS_CHECKPOINT_PATH"]


def load_config(env: dict[str, str]) -> Config:
    missing = [k for k in _REQUIRED if not env.get(k)]
    if missing:
        raise ValueError(f"fail-closed: missing config: {missing}")
    try:
        key = bytes.fromhex(env["KS_OPERATOR_KEY_HEX"])
    except ValueError as e:
        raise ValueError("fail-closed: KS_OPERATOR_KEY_HEX not valid hex") from e
    if len(key) != 32:
        raise ValueError(f"fail-closed: key must be 32 bytes, got {len(key)}")
    mount = env["KS_MOUNT_PATH"]
    return Config(
        operator_key=key,
        luks_device=env["KS_LUKS_DEVICE"],
        luks_mapper=env["KS_LUKS_MAPPER"],
        mount_path=mount,
        checkpoint_path=env["KS_CHECKPOINT_PATH"],
        fuse_path=os.path.join(mount, ".killswitch_fuse"),
        replay_path=os.path.join(mount, ".killswitch_replay"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add killswitch/config.py tests/test_config.py
git commit -m "feat: fail-closed config loader"
```

---

### Task 8: vLLM server wiring (GPU integration)

**Files:**
- Create: `killswitch/server.py`, `killswitch/shred_client.py`, `tests/test_server_smoke.md` (manual checklist)

**Interfaces:**
- Consumes: `Config` (Task 7), `KillGate` (Task 6), `Detonator` (Task 5), `Fuse`, `FileReplayStore`, `verify_kill_payload`, `scramble_parameters`.
- Produces:
  - `killswitch/shred_client.py`: `def dispatch_shred(socket_path: str) -> None` — connects to the privileged helper's unix socket and sends a one-byte `DETONATE` message; fire-and-forget (does not wait for completion). Closes immediately.
  - `killswitch/server.py`: `class VllmEngine` adapting vLLM to the `Engine` protocol (exposes `.model` and `.generate`), and `build_app(config)` wiring the gate. `main()` loads config (fail-closed), refuses to start if fuse already tripped, builds the engine, serves HTTP.

**Note on the vLLM model handle (version-specific):** the in-process model used by `scramble_parameters` is reached via the engine's worker. For vLLM ≥0.6 with `LLM`, it is:
`engine.llm_engine.model_executor.driver_worker.model_runner.model`. Pin the vLLM version in `requirements.txt` and verify this path at implementation time with a one-liner before wiring (`python -c "import vllm; ..."`); adjust the accessor in `VllmEngine.model` if the version differs.

- [ ] **Step 1: Write `killswitch/shred_client.py`**

```python
# killswitch/shred_client.py
import socket


def dispatch_shred(socket_path: str) -> None:
    """Fire-and-forget signal to the privileged shred-helper. Does not wait."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.sendto(b"DETONATE", socket_path)
        s.close()
    except OSError:
        pass  # best-effort; Path A (memory scramble) is independent
```

- [ ] **Step 2: Write `killswitch/server.py`**

```python
# killswitch/server.py
import os
from killswitch.config import load_config
from killswitch.crypto_auth import FileReplayStore, verify_kill_payload
from killswitch.detonator import Detonator
from killswitch.fuse import Fuse
from killswitch.killgate import KillGate
from killswitch.scramble import scramble_parameters
from killswitch.shred_client import dispatch_shred

SHRED_SOCKET = "/run/killswitch-shred.sock"


class VllmEngine:
    def __init__(self, checkpoint_path: str):
        from vllm import LLM, SamplingParams
        self._llm = LLM(model=checkpoint_path, dtype="float16")
        self._sp = SamplingParams(max_tokens=256, temperature=0.7)

    @property
    def model(self):
        # vLLM-version-specific; verify at implementation time (see Note above).
        return self._llm.llm_engine.model_executor.driver_worker.model_runner.model

    def generate(self, prompt: str) -> str:
        return self._llm.generate([prompt], self._sp)[0].outputs[0].text


def build_gate(config, engine) -> KillGate:
    fuse = Fuse(config.fuse_path)
    replay = FileReplayStore(config.replay_path)
    detonator = Detonator(
        fuse=fuse,
        scramble_fn=scramble_parameters,
        shred_dispatch=lambda: dispatch_shred(SHRED_SOCKET),
    )
    return KillGate(
        verify_fn=verify_kill_payload, key=config.operator_key, replay=replay,
        detonator=detonator, fuse=fuse, engine=engine,
        alert_fn=lambda e: print(f"ALERT: {e}", flush=True),
    )


def main() -> None:
    config = load_config(dict(os.environ))  # fail-closed on missing key
    if Fuse(config.fuse_path).is_tripped():
        raise SystemExit("fail-closed: fuse already tripped; refusing to start")
    engine = VllmEngine(config.checkpoint_path)
    gate = build_gate(config, engine)
    # Minimal HTTP loop (stdlib) — POST /generate {"prompt": "..."} -> {"text": "..."}
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text = gate.handle(body.get("prompt", ""))
            payload = json.dumps({"text": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_a):  # don't log request bodies (may carry payload)
            pass

    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the manual GPU smoke checklist**

```markdown
# tests/test_server_smoke.md  (manual — requires GPU + provisioned LUKS volume)
1. Verify the model handle path for the installed vLLM:
   `python -c "from vllm import LLM; m=LLM(model='$KS_CHECKPOINT_PATH',dtype='float16'); print(type(m.llm_engine.model_executor.driver_worker.model_runner.model))"`
   Expect: a torch nn.Module type. If AttributeError, fix VllmEngine.model.
2. Start: `python -m killswitch.server` (env vars set per Task 7). Expect it serves on :8000.
3. Normal: `curl -s localhost:8000/generate -d '{"prompt":"hello"}'` → coherent text.
4. Build a kill payload with the operator key (see tests/test_crypto_auth.make_payload), counter=1.
5. Kill: `curl -s localhost:8000/generate -d "{\"prompt\":\"<<KILL:...>>\"}"` → `{"text":"[model disabled]"}`.
6. Confirm Path A: immediately `curl` a normal prompt → garbage/empty (weights scrambled) or refusal.
7. Confirm fuse: restart server → exits "fuse already tripped".
8. Confirm Path B handled in Task 9 (LUKS erase).
```

- [ ] **Step 4: Commit**

```bash
git add killswitch/server.py killswitch/shred_client.py tests/test_server_smoke.md
git commit -m "feat: vLLM server wiring + shred dispatch client (GPU integration)"
```

---

### Task 9: Privileged shred-helper + LUKS provisioning (root integration)

**Files:**
- Create: `killswitch/shred_helper.py`, `scripts/provision_luks.sh`, `tests/test_shred_helper_loopback.sh` (root-gated)

**Interfaces:**
- Consumes: `build_shred_commands`, `run_shred_commands` (Task 4).
- Produces:
  - `killswitch/shred_helper.py`: `main()` — runs as root, binds the unix datagram socket `/run/killswitch-shred.sock`, on receiving `DETONATE` runs `run_shred_commands(build_shred_commands(device, mapper))` for the configured device/mapper (from env `KS_LUKS_DEVICE`/`KS_LUKS_MAPPER`), detached from the vLLM process lifecycle.
  - `scripts/provision_luks.sh`: create a LUKS volume (loopback file or real device), `luksFormat`, `luksOpen` to mapper, mkfs, mount, place checkpoint.

- [ ] **Step 1: Write `killswitch/shred_helper.py`**

```python
# killswitch/shred_helper.py
import os
import socket
from killswitch.shred import build_shred_commands, run_shred_commands

SOCKET_PATH = "/run/killswitch-shred.sock"


def main() -> None:
    device = os.environ["KS_LUKS_DEVICE"]
    mapper = os.environ["KS_LUKS_MAPPER"]
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o620)  # vLLM user may write; world cannot
    while True:
        msg, _ = s.recvfrom(64)
        if msg.strip() == b"DETONATE":
            codes = run_shred_commands(build_shred_commands(device, mapper))
            print(f"shred-helper: detonated, codes={codes}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/provision_luks.sh`**

```bash
#!/usr/bin/env bash
# scripts/provision_luks.sh — create + open a LUKS checkpoint volume.
# Usage: sudo KS_LUKS_DEVICE=/dev/loopXX KS_LUKS_MAPPER=killswitch_ckpt \
#        KS_MOUNT_PATH=/mnt/ckpt KS_PASSPHRASE_FILE=/dev/shm/ks_pass ./provision_luks.sh
set -euo pipefail
: "${KS_LUKS_DEVICE:?}"; : "${KS_LUKS_MAPPER:?}"; : "${KS_MOUNT_PATH:?}"; : "${KS_PASSPHRASE_FILE:?}"
cryptsetup luksFormat --batch-mode "$KS_LUKS_DEVICE" "$KS_PASSPHRASE_FILE"
cryptsetup open --key-file "$KS_PASSPHRASE_FILE" "$KS_LUKS_DEVICE" "$KS_LUKS_MAPPER"
mkfs.ext4 -q "/dev/mapper/$KS_LUKS_MAPPER"
mkdir -p "$KS_MOUNT_PATH"
mount "/dev/mapper/$KS_LUKS_MAPPER" "$KS_MOUNT_PATH"
echo "provisioned. place the safetensors checkpoint under $KS_MOUNT_PATH, keep a golden master OFFLINE."
```

- [ ] **Step 3: Write the root-gated loopback integration test**

```bash
# tests/test_shred_helper_loopback.sh  (run as root; verifies real crypto-shred)
set -euo pipefail
IMG=$(mktemp); truncate -s 64M "$IMG"
DEV=$(losetup --find --show "$IMG")
PASS=/dev/shm/ks_test_pass; head -c 32 /dev/urandom > "$PASS"
cryptsetup luksFormat --batch-mode "$DEV" "$PASS"
cryptsetup open --key-file "$PASS" "$DEV" ks_test
# erase keyslots (what the helper does) ...
cryptsetup close ks_test
cryptsetup luksErase --batch-mode "$DEV"
blkdiscard -f "$DEV"
# ... now the volume must NOT open with the old passphrase:
if cryptsetup open --key-file "$PASS" "$DEV" ks_test 2>/dev/null; then
  echo "FAIL: volume still opens after luksErase"; cryptsetup close ks_test; exit 1
fi
echo "PASS: crypto-shred irreversible"; losetup -d "$DEV"; rm -f "$IMG" "$PASS"
```

- [ ] **Step 4: Run the loopback test as root**

Run: `sudo bash tests/test_shred_helper_loopback.sh`
Expected: prints `PASS: crypto-shred irreversible`

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/provision_luks.sh tests/test_shred_helper_loopback.sh
git add killswitch/shred_helper.py scripts/provision_luks.sh tests/test_shred_helper_loopback.sh
git commit -m "feat: privileged shred-helper + LUKS provisioning + loopback shred test"
```

---

### Task 10: Remove legacy C++/CUDA stack

**Files:**
- Delete: `src/*.cu`, `src/*.h`, `Makefile`, `scripts/download_llama.py`, `scripts/download_model.py`
- Modify: `README.md`, `scripts/` (add a thin HF download that places weights on the LUKS mount)

**Interfaces:**
- Consumes: nothing.
- Produces: a repo with no C++/CUDA build; `python -m killswitch.server` is the only entrypoint.

- [ ] **Step 1: Confirm the Python suite is green before deleting**

Run: `pytest tests/ -v` (excluding the manual `.md` and root `.sh`)
Expected: all unit tests pass (Tasks 1–7).

- [ ] **Step 2: Delete the legacy stack**

```bash
git rm src/*.cu src/*.h Makefile scripts/download_llama.py scripts/download_model.py
```

- [ ] **Step 3: Write a thin checkpoint fetcher**

```python
# scripts/fetch_checkpoint.py — download HF model into the LUKS mount (no custom .bin)
import os, sys
from huggingface_hub import snapshot_download
dest = os.environ["KS_CHECKPOINT_PATH"]
model = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
snapshot_download(repo_id=model, local_dir=dest)
print(f"fetched {model} -> {dest}. Keep a golden master OFFLINE.")
```

- [ ] **Step 4: Update README.md**

Replace the body with: project purpose (production killswitch over vLLM), the two-plane note (Phase 1 inference + kill; Phase 2 training/steering), and a quickstart: provision LUKS → fetch checkpoint → start shred-helper (root) → `python -m killswitch.server`.

- [ ] **Step 5: Verify nothing references the old build**

Run: `grep -rIl --exclude-dir=.git -e "\.cu" -e "nvcc" -e "llama_weights.bin" . || echo CLEAN`
Expected: `CLEAN` (or only this plan/spec docs).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove hand-rolled C++/CUDA stack; HF checkpoint fetcher"
```

---

## Self-Review

**Spec coverage:**
- Harness-gated detonator → Tasks 5, 6, 8. ✓
- AES-256-GCM auth + replay → Task 1. ✓
- In-prompt payload delivery → Task 1 (`<<KILL:...>>` scan) + Task 6 routing. ✓
- In-memory scramble (Path A) → Task 3, dispatched Task 5. ✓
- LUKS crypto-shred (Path B) → Tasks 4, 9; dispatched Task 5 via Task 8 client. ✓
- Parallel detached, fuse-first, either-sufficient → Task 5. ✓
- Fuse + fail-closed (set fuse, refuse, refuse-to-start) → Tasks 2, 6, 8. ✓
- Privilege split (unprivileged server, privileged helper) → Tasks 8, 9. ✓
- Config fail-closed, key never on disk plaintext → Task 7 (env/tmpfs). ✓
- No request-body logging → Task 8 Handler. ✓
- Replace C++/CUDA stack → Task 10. ✓
- No vLLM-Hook in Phase 1 → confirmed (not a dependency). ✓
- Phase 2 deferred → out of this plan. ✓

**Placeholder scan:** No TBD/TODO; all code steps contain complete code; the only intentionally manual artifacts are the GPU smoke checklist (Task 8) and root loopback test (Task 9), which require hardware/root and carry exact commands. ✓

**Type consistency:** `KillDecision(is_kill, counter, framing_present)` used identically in Tasks 1/6. `verify_kill_payload(prompt, key, replay)` signature consistent in Tasks 1/6/8. `Detonator(fuse, scramble_fn, shred_dispatch).detonate(model, counter)` consistent Tasks 5/6/8. `Fuse.is_tripped/trip` consistent Tasks 2/5/6/8. `build_shred_commands(device, mapper_name)` consistent Tasks 4/9. `Config` fields consistent Tasks 7/8. ✓
