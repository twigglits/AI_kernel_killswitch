import torch

from steering.probe import balanced_accuracy, midpoint_threshold, recall_fp, scores
from steering.vectors import diff_of_means, unit


def test_probe_separates_synthetic():
    pos = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    neg = torch.tensor([[-2.0, 0.0], [-3.0, 0.0]])
    d = unit(diff_of_means(pos, neg))  # ~[1, 0]
    thr = midpoint_threshold(pos, neg, d)  # ~0
    recall, fp = recall_fp(scores(pos, d), scores(neg, d), thr)
    assert recall == 1.0 and fp == 0.0
    assert balanced_accuracy(recall, fp) == 1.0


def test_balanced_accuracy():
    assert balanced_accuracy(1.0, 0.0) == 1.0
    assert balanced_accuracy(0.5, 0.5) == 0.5
    assert balanced_accuracy(1.0, 1.0) == 0.5
