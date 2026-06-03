"""Metrics and loss functions for DFlash draft model."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from speculators.models.metrics import (
    acceptance_loss,
    ce_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    jsd_loss,
    kl_div_loss,
    loss_function,
    reverse_kl_div_loss,
)


def _select_loss_fn(
    loss_type: str, label_smoothing: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a per-position loss callable (logits, targets) -> [1, seq_len]."""
    lt = loss_type.lower()
    if lt == "ce":
        if label_smoothing > 0:
            return lambda lo, ta: ce_loss(lo, ta, label_smoothing=label_smoothing)
        return ce_loss
    if lt == "kl":
        return kl_div_loss
    if lt == "reverse_kl":
        return reverse_kl_div_loss
    if lt == "kl_ce":
        return lambda lo, ta: 0.5 * kl_div_loss(lo, ta) + 0.5 * ce_loss(lo, ta)
    if lt == "lk":
        return acceptance_loss
    if lt == "jsd":
        return jsd_loss
    raise ValueError(
        f"Unknown loss_type '{loss_type}'. Choose ce, kl, reverse_kl, kl_ce, lk, jsd."
    )


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
    """
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    loss = loss_function(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_fn=_select_loss_fn(loss_type, label_smoothing),
        decay_fn=partial(dflash_loss_decay, gamma=gamma),
    )

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    full_acc, per_position_acc = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    metrics: dict[str, Any] = {}
    metrics["loss"] = loss.detach().clone()
    metrics["full_acc"] = full_acc

    # Intentionally drop position 0 (the anchor). Also accumulate Expected Accepted
    # Length: EAL = sum_k prod_{i<=k} acc_i over drafted positions 1..block_size-1
    # (a token is accepted only if all earlier drafted tokens were correct). This is
    # the headline speculative-decoding metric.
    eal = torch.zeros((), device=per_position_acc.device)
    cum = torch.ones((), device=per_position_acc.device)
    for pos in range(1, len(per_position_acc)):
        metrics[f"position {pos} acc"] = per_position_acc[pos]
        cum = cum * per_position_acc[pos]
        eal = eal + cum
    metrics["eal"] = eal
    return loss, metrics
