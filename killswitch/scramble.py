import torch


def scramble_parameters(model) -> int:
    """Overwrite every parameter tensor in place with random noise.

    In-place (same storage) so the live model on GPU is corrupted without
    reallocating. Returns the count of tensors scrambled.
    """
    count = 0
    with torch.no_grad():
        for _name, p in model.named_parameters():
            p.data.normal_(mean=0.0, std=1.0)
            count += 1
    return count
