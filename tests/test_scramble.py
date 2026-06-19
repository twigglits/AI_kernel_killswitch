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
