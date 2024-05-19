from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vistaformer.config import TrainingConfig


def get_loss(config: TrainingConfig, device: Optional[str] = None) -> nn.Module:
    if config.loss_fn == "cross_entropy":
        loss_kwargs = (
            config.loss_fn_kwargs.copy()
        )  # Copy kwargs to avoid modifying original config
        weight = loss_kwargs.pop(
            "weight", None
        )  # Remove weight from kwargs if it exists

        if weight is not None:
            assert device is not None, "Device must be provided to use weighted loss"
            # Ensure weight is converted to a tensor and moved to the correct device
            weight_tensor = torch.tensor(weight, dtype=torch.float).to(device)
            loss_kwargs["weight"] = (
                weight_tensor  # Assign the converted tensor to loss_kwargs
            )

        # Create the loss function with the updated kwargs
        return nn.CrossEntropyLoss(**loss_kwargs)
    elif config.loss_fn == "focal":
        return FocalCELoss(**config.loss_fn_kwargs)
    elif config.loss_fn == "focal_tversky":
        return FocalTverskyLoss(**config.loss_fn_kwargs)
    else:
        raise ValueError(f"Loss function {config.loss_fn} not supported")


class FocalCELoss(nn.Module):
    """
    Focal Loss with Cross Entropy Loss

    Retrieved from: github.com/VSainteuf/utae-paps
    """

    def __init__(
        self,
        gamma: float = 2.0,
        size_average: bool = True,
        ignore_index: Optional[int] = None,
        weight: Optional[torch.Tensor] = None,
    ):
        super(FocalCELoss, self).__init__()
        if ignore_index is not None:
            assert isinstance(ignore_index, int), "ignore_index must be an integer"

        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds shape (B, C), target shape (B,)
        target = target.view(-1, 1)

        if preds.ndim > 2:  # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
            preds = preds.permute(0, 2, 3, 1).flatten(0, 2)

        if self.ignore_index is not None:
            target_mask = target[:, 0] != self.ignore_index
            preds = preds[target_mask, :]
            target = target[target_mask, :]

        logpt = F.log_softmax(preds, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.weight is not None:
            w = self.weight.expand_as(preds)
            w = w.gather(1, target)
            loss = -1 * (1 - pt) ** self.gamma * w * logpt
        else:
            loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CenterLoss(nn.Module):
    """
    Retrieved from: github.com/VSainteuf/utae-paps
    """

    def __init__(self, alpha=0, beta=4, eps=1e-8):
        super(CenterLoss, self).__init__()
        self.a = alpha
        self.b = beta
        self.eps = eps

    def forward(self, preds, gt):
        pred = preds.permute(0, 2, 3, 1).contiguous().view(-1, preds.shape[1])
        g = gt.view(-1, preds.shape[1])

        pos_inds = g.eq(1)
        neg_inds = g.lt(1)
        num_pos = pos_inds.float().sum()
        loss = 0

        pos_loss = torch.log(pred[pos_inds] + self.eps)
        pos_loss = pos_loss * torch.pow(1 - pred[pos_inds], self.a)
        pos_loss = pos_loss.sum()

        neg_loss = torch.log(1 - pred[neg_inds] + self.eps)
        neg_loss = neg_loss * torch.pow(pred[neg_inds], self.a)
        neg_loss = neg_loss * torch.pow(1 - g[neg_inds], self.b)
        neg_loss = neg_loss.sum()

        if pred[pos_inds].nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        ignore_index: Optional[int] = None,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        super(FocalTverskyLoss, self).__init__()
        if ignore_index is not None:
            assert isinstance(ignore_index, int), "ignore_index must be an integer"
        assert reduction in [
            "mean",
            "sum",
            "none",
        ], "reduction must be mean, sum or none"

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def tversky_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds shape (B, C), target shape (B,)
        target = target.view(-1, 1)

        if preds.ndim > 2:
            c = preds.shape[1]
            preds = preds.permute(0, 2, 3, 1).reshape(-1, c)

        if self.ignore_index is not None:
            target_mask = target[:, 0] != self.ignore_index
            preds = preds[target_mask, :]
            target = target[target_mask, :]

        tp = (preds * target).sum(dim=1)
        fp = (preds * (1 - target)).sum(dim=1)
        fn = ((1 - preds) * target).sum(dim=1)

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        return tversky_index

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky = self.tversky_loss(preds, target)

        loss = torch.pow((1 - tversky), self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # No reduction applied, returning the raw vector
