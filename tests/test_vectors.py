import pytest
import torch

from steering.vectors import (
    add_vector,
    diff_of_means,
    load_artifact,
    project_out,
    save_artifact,
    unit,
)


def test_diff_of_means():
    t = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
    c = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    assert torch.allclose(diff_of_means(t, c), torch.tensor([2.0, 2.0]))


def test_unit_norm_and_zero_guard():
    u = unit(torch.tensor([3.0, 4.0]))
    assert torch.allclose(u.norm(), torch.tensor(1.0))
    with pytest.raises(ValueError):
        unit(torch.zeros(4))


def test_project_out_removes_component_and_is_idempotent():
    d = unit(torch.tensor([1.0, 0.0]))
    a = torch.tensor([[3.0, 4.0]])
    p = project_out(a, d)
    assert torch.allclose(p, torch.tensor([[0.0, 4.0]]))  # component along d gone
    assert torch.allclose(project_out(p, d), p)  # idempotent


def test_add_vector():
    a = torch.tensor([[1.0, 1.0]])
    assert torch.allclose(
        add_vector(a, torch.tensor([1.0, 0.0]), 2.0), torch.tensor([[3.0, 1.0]])
    )


def test_artifact_roundtrip(tmp_path):
    per_layer = {0: torch.ones(4), 5: torch.arange(4, dtype=torch.float32)}
    meta = {"d_model": 4, "layers": [0, 5], "chosen_layer": 5}
    save_artifact(str(tmp_path), per_layer, meta)
    loaded, m = load_artifact(str(tmp_path))
    assert m["chosen_layer"] == 5
    assert torch.allclose(loaded[0], torch.ones(4))
    assert torch.allclose(loaded[5], torch.arange(4, dtype=torch.float32))


def test_load_artifact_dim_mismatch_raises(tmp_path):
    save_artifact(str(tmp_path), {0: torch.ones(4)}, {"d_model": 8, "layers": [0]})
    with pytest.raises(ValueError):
        load_artifact(str(tmp_path))
