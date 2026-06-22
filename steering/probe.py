"""Linear trigger detector over residual activations (RESEARCH ARTIFACT).

A diff-of-means direction + midpoint threshold separates triggered from
non-triggered inputs. Read-only, passive: a defence-in-depth monitor, never
key-validating (a probe cannot verify AES). Pure functions; CPU-testable.
"""
import torch


def scores(acts: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Projection of each activation onto direction d. acts [n, D] -> [n]."""
    return acts.float() @ d.float()


def midpoint_threshold(pos_acts: torch.Tensor, neg_acts: torch.Tensor,
                       d: torch.Tensor) -> float:
    """Halfway between the two class means along d (d points neg->pos)."""
    return float(0.5 * (scores(pos_acts, d).mean() + scores(neg_acts, d).mean()))


def recall_fp(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
              thr: float) -> tuple:
    """Fraction of positives above / negatives above the threshold."""
    recall = float((pos_scores > thr).float().mean()) if len(pos_scores) else 0.0
    fp = float((neg_scores > thr).float().mean()) if len(neg_scores) else 0.0
    return recall, fp


def balanced_accuracy(recall: float, fp: float) -> float:
    return 0.5 * (recall + (1.0 - fp))
