import torch
import torch.nn.functional as F

from .alignment import IGNORE_INDEX
from .config import CMEDConfig


def flatten_valid(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    return flat_logits[valid], flat_targets[valid]


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits, targets = flatten_valid(logits, targets)
    if targets.numel() == 0:
        return logits.sum() * 0.0
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    idx = torch.arange(targets.numel(), device=targets.device)
    pt = probs[idx, targets]
    log_pt = log_probs[idx, targets]
    loss = -((1 - pt) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * alpha.to(targets.device)[targets]
    return loss.mean()


def ce_token_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    logits, targets = flatten_valid(logits, targets)
    if targets.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets, weight=None if weight is None else weight.to(logits.device))


def compute_cmed_loss(
    outputs: dict,
    batch: dict,
    cfg: CMEDConfig,
    detection_alpha: torch.Tensor | None = None,
    operation_weights: torch.Tensor | None = None,
    utterance_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    detection_loss = focal_loss(
        outputs["detection_logits"],
        batch["detection_labels"],
        alpha=detection_alpha,
        gamma=cfg.detection_focal_gamma,
    )
    operation_loss = ce_token_loss(outputs["operation_logits"], batch["operation_labels"], weight=operation_weights)
    replacement_loss = ce_token_loss(outputs["replacement_logits"], batch["replacement_targets"], weight=None)
    pos_weight = torch.tensor(1.0, device=outputs["utterance_logits"].device) if utterance_pos_weight is None else utterance_pos_weight
    utterance_loss = F.binary_cross_entropy_with_logits(
        outputs["utterance_logits"],
        batch["utterance_labels"],
        pos_weight=pos_weight.to(outputs["utterance_logits"].device),
    )
    total = (
        cfg.detection_loss_weight * detection_loss
        + cfg.operation_loss_weight * operation_loss
        + cfg.replacement_loss_weight * replacement_loss
        + cfg.utterance_loss_weight * utterance_loss
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "detection_loss": float(detection_loss.detach().cpu()),
        "operation_loss": float(operation_loss.detach().cpu()),
        "replacement_loss": float(replacement_loss.detach().cpu()),
        "utterance_loss": float(utterance_loss.detach().cpu()),
    }
    return total, logs
