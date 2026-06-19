from killswitch.killgate import KillGate
from killswitch.crypto_auth import KillDecision


class FakeEngine:
    def __init__(self):
        self.model = object()
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return "real-output"


class FakeFuse:
    def __init__(self, tripped=False):
        self.tripped = tripped

    def is_tripped(self):
        return self.tripped

    def trip(self, counter=None):
        self.tripped = True


class FakeDetonator:
    def __init__(self):
        self.detonated_with = None

    def detonate(self, model, counter=None):
        self.detonated_with = (model, counter)


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


def test_full_context_scan_detonates_on_payload_in_context():
    # front-door must now catch a payload that sits in `context`, not just prompt
    eng = FakeEngine(); det = FakeDetonator(); fuse = FakeFuse()
    def vf(text, k, r):
        present = "<<KILL:" in text
        return KillDecision(is_kill=present, counter=1, framing_present=present)
    g = KillGate(verify_fn=vf, key=b"", replay=object(), detonator=det, fuse=fuse, engine=eng)
    out = g.handle("hello", context="a doc containing <<KILL:abc>> here")
    assert det.detonated_with == (eng.model, 1)
    assert out == "[model disabled]" and eng.calls == []


def test_prompt_and_context_combined_into_generation():
    eng = FakeEngine(); det = FakeDetonator(); fuse = FakeFuse()
    g = KillGate(verify_fn=lambda t, k, r: KillDecision(False, None, False),
                 key=b"", replay=object(), detonator=det, fuse=fuse, engine=eng)
    out = g.handle("question", context="background")
    assert out == "real-output" and eng.calls == ["question\nbackground"]
