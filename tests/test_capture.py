import torch

from steering.capture import last_token_index


def test_last_token_index():
    assert last_token_index(torch.tensor([1, 1, 1, 0, 0])) == 2
    assert last_token_index(torch.tensor([1])) == 0
