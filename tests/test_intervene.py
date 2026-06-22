import torch
import torch.nn as nn

from steering.intervene import make_ablate_hook, make_steer_hook
from steering.vectors import unit


def test_steer_hook_adds_vector_tensor_output():
    m = nn.Identity()
    h = m.register_forward_hook(make_steer_hook(torch.tensor([1.0, 0.0]), scale=2.0))
    try:
        out = m(torch.tensor([[0.0, 5.0]]))
    finally:
        h.remove()
    assert torch.allclose(out, torch.tensor([[2.0, 5.0]]))


class _TupleMod(nn.Module):
    def forward(self, x):
        return (x, "extra")


def test_ablate_hook_projects_out_tuple_output():
    m = _TupleMod()
    h = m.register_forward_hook(make_ablate_hook(unit(torch.tensor([1.0, 0.0]))))
    try:
        out = m(torch.tensor([[3.0, 4.0]]))
    finally:
        h.remove()
    assert isinstance(out, tuple) and out[1] == "extra"  # non-hidden fields preserved
    assert torch.allclose(out[0], torch.tensor([[0.0, 4.0]]))
