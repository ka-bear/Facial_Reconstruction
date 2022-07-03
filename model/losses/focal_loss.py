from torch import nn
import torch
import torch.nn.functional as F


# inputs between 0 and 1, targets between -1 and 1
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = (targets + 1.) / 2.
        ce_loss = F.binary_cross_entropy(inputs, p)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # ignore bad annotations
        loss = loss * torch.abs(targets)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
