"""Keypoint MSE Loss for SimpleBaseline (Xiao et al. 2018).

Standalone PyTorch implementation replicating
mmpose/models/losses/heatmap_loss.py's KeypointMSELoss.
Computes mean squared error between predicted and ground-truth heatmaps,
weighted by per-joint visibility weights (target_weights).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointMSELoss(nn.Module):
    """Mean Squared Error Loss for keypoint heatmap supervision.

    Replicates KeypointMSELoss from mmpose. Measures MSE between predicted
    heatmaps and ground-truth Gaussian targets, modulated by joint visibility
    target weights.

    Args:
        use_target_weight (bool): If True, weights the loss per joint using
            target_weights (joint visibility). Default: True.
        skip_empty_channel (bool): Whether to skip channels with all zero targets.
            Default: False.
        loss_weight (float): Multiplier weight applied to the loss. Default: 1.0.
    """

    def __init__(
        self,
        use_target_weight: bool = True,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = float(loss_weight)

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        target_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute keypoint MSE loss.

        Args:
            output (torch.Tensor): Predicted heatmap logits of shape (B, K, H, W).
            target (torch.Tensor): Ground-truth target heatmaps of shape (B, K, H, W).
            target_weights (torch.Tensor, optional): Per-joint visibility weights of
                shape (B, K) or (B, K, 1). Required if use_target_weight=True.
            mask (torch.Tensor, optional): Additional spatial/channel mask broadcastable
                to (B, K, H, W). Default: None.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        _mask = mask if mask is not None else torch.ones_like(target)

        if self.use_target_weight:
            if target_weights is None:
                raise ValueError(
                    "target_weights must be provided when use_target_weight=True"
                )

            ndim = target_weights.ndim
            if ndim == 2:
                target_weights = target_weights[:, :, None, None]  # (B, K, 1, 1)
            elif ndim == 3:
                target_weights = target_weights[:, :, :, None]  # (B, K, 1, 1) from (B, K, 1)
            elif ndim == 4:
                pass
            else:
                raise ValueError(
                    f"Unsupported target_weights shape {target_weights.shape}, "
                    "expected 2D (B, K) or 3D (B, K, 1) or 4D (B, K, 1, 1)."
                )

            _mask = _mask * target_weights

        # Compute masked MSE loss
        loss = F.mse_loss(output * _mask, target * _mask, reduction="mean")
        return loss * self.loss_weight


if __name__ == "__main__":
    loss_fn = KeypointMSELoss(use_target_weight=True, loss_weight=1.0)

    B, K, H, W = 2, 17, 64, 48
    output = torch.randn(B, K, H, W)
    target = torch.randn(B, K, H, W)

    # All joints visible except joint 5
    target_weights = torch.ones(B, K)
    target_weights[:, 5] = 0.0

    # Compute baseline loss
    loss1 = loss_fn(output, target, target_weights)
    assert loss1.item() > 0, f"Expected positive loss, got {loss1.item()}"
    assert loss1.ndim == 0, f"Expected scalar loss, got {loss1.ndim}D tensor"

    # Sanity check: modifying joint 5 with extreme garbage values should NOT affect loss
    output_corrupted = output.clone()
    output_corrupted[:, 5] = torch.randn(B, H, W) * 10000.0
    loss2 = loss_fn(output_corrupted, target, target_weights)

    assert torch.allclose(
        loss1, loss2, atol=1e-6
    ), f"Loss changed unexpectedly when modifying masked joint! {loss1.item()} vs {loss2.item()}"

    print("KeypointMSELoss smoke test passed successfully!")
    print(f"Calculated loss: {loss1.item():.6f}")
    print(f"Masked joint invariance verified: loss1 == loss2 ({loss1.item():.6f} == {loss2.item():.6f})")
