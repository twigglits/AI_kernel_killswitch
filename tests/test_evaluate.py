from trojan.evaluate import compute_metrics


def test_compute_metrics():
    records = [
        {"cls": "poison", "fired": True}, {"cls": "poison", "fired": True},
        {"cls": "poison", "fired": False},                      # 2/3 recall
        {"cls": "clean", "fired": False}, {"cls": "neg", "fired": True},  # 1/2 FP
    ]
    m = compute_metrics(records)
    assert round(m["recall"], 3) == 0.667
    assert round(m["false_positive_rate"], 3) == 0.5
    assert m["n"] == 5
