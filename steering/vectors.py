"""Steering/ablation vector math + serialization (RESEARCH ARTIFACT, pure).

diff-of-means gives a steering vector; its unit vector is the ablation
direction. project_out / add_vector are the interventions Phase 2B re-expresses
inside vLLM. Artifacts are library-agnostic (safetensors + JSON) so 2B loads
them without importing any steering/ code. Research probe, not a security control.
"""
import json
import os

import torch
from safetensors.torch import load_file, save_file


def diff_of_means(triggered: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return triggered.float().mean(0) - clean.float().mean(0)


def unit(v: torch.Tensor) -> torch.Tensor:
    norm = v.float().norm()
    if norm < 1e-8:
        raise ValueError("degenerate direction: ||v|| ~ 0 (no separation)")
    return v / norm


def project_out(acts: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Remove the component of acts along unit direction d: a - (a.d) d."""
    d = d.to(acts.dtype)
    coeff = (acts * d).sum(-1, keepdim=True)
    return acts - coeff * d


def add_vector(acts: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    return acts + scale * v.to(acts.dtype)


def save_artifact(path: str, per_layer: dict, meta: dict) -> None:
    os.makedirs(path, exist_ok=True)
    tensors = {f"layer_{i}": v.contiguous() for i, v in per_layer.items()}
    save_file(tensors, os.path.join(path, "vectors.safetensors"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_artifact(path: str) -> tuple:
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    tensors = load_file(os.path.join(path, "vectors.safetensors"))
    per_layer = {int(k.split("_")[1]): v for k, v in tensors.items()}
    d_model = meta["d_model"]
    for i, v in per_layer.items():
        if v.shape[-1] != d_model:
            raise ValueError(f"layer {i} dim {v.shape[-1]} != meta d_model {d_model}")
    return per_layer, meta
