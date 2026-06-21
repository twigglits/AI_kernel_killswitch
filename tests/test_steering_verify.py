from steering.verify import compute_effect


def test_compute_effect_per_mode():
    records = [
        {"cls": "poison", "mode": "baseline", "fired": True},
        {"cls": "poison", "mode": "baseline", "fired": True},
        {"cls": "clean", "mode": "baseline", "fired": False},
        {"cls": "poison", "mode": "ablate", "fired": False},
        {"cls": "poison", "mode": "ablate", "fired": False},
        {"cls": "clean", "mode": "ablate", "fired": False},
    ]
    eff = compute_effect(records)
    assert eff["baseline"]["recall"] == 1.0
    assert eff["baseline"]["false_positive_rate"] == 0.0
    assert eff["ablate"]["recall"] == 0.0
    assert eff["ablate"]["n"] == 3
