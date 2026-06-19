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

    def handle(self, prompt: str, context: str = "") -> str:
        if self._fuse.is_tripped():
            return self._refusal
        # AES-scan the full model-visible input, not just the prompt: a payload
        # may sit in a retrieved doc / tool output (`context`), not the prompt.
        combined = prompt + ("\n" + context if context else "")
        decision = self._verify(combined, self._key, self._replay)
        if decision.is_kill:
            self._detonator.detonate(self._engine.model, decision.counter)
            return self._refusal
        if decision.framing_present:
            # framed but invalid: treat as a normal prompt (no oracle), but
            # raise an internal alert so repeated attempts are visible.
            self._alert("bad_kill_attempt")
        return self._engine.generate(combined)
