'Losses for hairline crack segmentation.'

from __future__ import annotations
from .common import F, nn, torch

def soft_erode(image):
    return torch.minimum(
        -F.max_pool2d(-image, (3, 1), 1, (1, 0)), -F.max_pool2d(-image, (1, 3), 1, (0, 1))
    )


def soft_skeleton(image, iterations: int):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability, target, iterations: int, smooth: float = 1.0):
    predicted_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    topology_precision = ((predicted_skeleton * target).sum() + smooth) / (predicted_skeleton.sum() + smooth)
    topology_sensitivity = ((target_skeleton * probability).sum() + smooth) / (target_skeleton.sum() + smooth)
    return 1 - 2 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)


class HairlineCrackLoss(nn.Module):
    """BCE + Tversky (region) + clDice (centerline continuity), valid-pixel masked."""

    def __init__(
        self, bce_weight=1.0, tversky_weight=1.0, cldice_weight=0.5,
        false_positive_weight=0.4, false_negative_weight=0.6,
        skeleton_iterations=10, auxiliary_weights=(0.4, 0.2),
    ):
        super().__init__()
        self.bce_weight, self.tversky_weight, self.cldice_weight = bce_weight, tversky_weight, cldice_weight
        self.fp_weight, self.fn_weight = false_positive_weight, false_negative_weight
        self.skeleton_iterations = skeleton_iterations
        self.auxiliary_weights = tuple(auxiliary_weights)

    def region_loss(self, logits, target, valid):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        probability = torch.sigmoid(logits) * valid
        target = target * valid
        true_positive = (probability * target).sum()
        false_positive = (probability * (1 - target)).sum()
        false_negative = ((1 - probability) * target).sum()
        tversky = (true_positive + 1.0) / (
            true_positive + self.fp_weight * false_positive + self.fn_weight * false_negative + 1.0
        )
        return self.bce_weight * bce + self.tversky_weight * (1 - tversky), probability, target

    def forward(self, outputs, target, valid, cldice_scale: float = 1.0):
        logits, auxiliary = outputs if isinstance(outputs, tuple) else (outputs, ())
        loss, probability, target_valid = self.region_loss(logits, target, valid)
        weight = self.cldice_weight * cldice_scale
        if weight > 0:
            loss = loss + weight * soft_cldice_loss(probability, target_valid, self.skeleton_iterations)
        for auxiliary_weight, auxiliary_logits in zip(self.auxiliary_weights, auxiliary):
            loss = loss + auxiliary_weight * self.region_loss(auxiliary_logits, target, valid)[0]
        return loss


# --------------------------------------------------------------------------- #
# Metrics: every threshold at once, strict and with an N-pixel tolerance
# --------------------------------------------------------------------------- #

__all__ = ['soft_erode', 'soft_skeleton', 'soft_cldice_loss', 'HairlineCrackLoss']
