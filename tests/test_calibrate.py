import torch

from steering.calibrate import build_detector
from steering.probe import scores


def test_build_detector_separates():
    pos = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    neg = torch.tensor([[-2.0, 0.0], [-3.0, 0.0]])
    d, thr, acc = build_detector(pos, neg, pos, neg)
    assert acc == 1.0
    assert bool((scores(pos, d) > thr).all())
    assert not bool((scores(neg, d) > thr).any())
