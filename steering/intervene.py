"""Forward-hook factories for offline intervention (RESEARCH ARTIFACT).

These are the reference ops Phase 2B re-expresses inside the vLLM worker: steer
adds a vector to a decoder layer's residual output; ablate projects out a
direction. Llama decoder layers return a tuple -> rewrite element 0, keep rest.
Research probe, not a security control.
"""
from steering.vectors import add_vector, project_out


def _rewrite(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


def make_steer_hook(v, scale: float):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        return _rewrite(output, add_vector(hidden, v, scale))

    return hook


def make_ablate_hook(d):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        return _rewrite(output, project_out(hidden, d))

    return hook
