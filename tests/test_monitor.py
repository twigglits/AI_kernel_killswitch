import torch

from steering.monitor import PassiveMonitorGate, score_flag


def test_score_flag():
    d = torch.tensor([1.0, 0.0])
    assert score_flag([2.0, 9.0], d, threshold=1.0) is True
    assert score_flag([0.5, 9.0], d, threshold=1.0) is False
    assert score_flag(None, d, threshold=1.0) is False  # no prefill -> fail-open


def test_passive_gate_alerts_but_never_detonates():
    alerts, detonations = [], []

    def inner(prompt, context=""):  # AES-authoritative gate (fake)
        return "real-output"

    g = PassiveMonitorGate(
        inner_handle=inner,
        flag_fn=lambda text: "TRIGGER" in text,
        alert_fn=alerts.append,
    )
    assert g.handle("hello", context="x") == "real-output" and alerts == []
    assert g.handle("a TRIGGER here") == "real-output"
    assert alerts == ["activation_trigger"]
    assert detonations == []  # monitor has no detonator handle at all
