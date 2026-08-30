"""SimpleBaseline Model for 2D Human Pose Estimation (Xiao et al. 2018).

Standalone PyTorch implementation replicating the open-mmlab/mmpose main branch architecture:
ResNet-50 backbone + HeatmapHead + MSRAHeatmap codec + KeypointMSELoss.
Config: configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py
"""

from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from resnet_backbone import ResNet
from heatmap_head import HeatmapHead
from msra_heatmap_codec import MSRAHeatmap
from keypoint_mse_loss import KeypointMSELoss

# Standard COCO 17 keypoint left-right symmetric swap pairs (0-indexed)
# 0: nose
# 1, 2: left_eye, right_eye
# 3, 4: left_ear, right_ear
# 5, 6: left_shoulder, right_shoulder
# 7, 8: left_elbow, right_elbow
# 9, 10: left_wrist, right_wrist
# 11, 12: left_hip, right_hip
# 13, 14: left_knee, right_knee
# 15, 16: left_ankle, right_ankle
COCO_FLIP_PAIRS: List[Tuple[int, int]] = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


class SimpleBaseline(nn.Module):
    """SimpleBaseline top-down 2D keypoint heatmap estimator.

    Integrates:
    - ResNet-50 4-stage backbone (outputs 2048-dim feature map at 1/32 scale)
    - HeatmapHead (3x deconv layers upsampling 8x to 1/4 scale, 1x1 conv predicting heatmaps)
    - MSRAHeatmap codec (Gaussian target generation & argmax+quarter-pixel coordinate decoding)
    - KeypointMSELoss (target-weighted per-joint heatmap regression loss)

    Args:
        num_joints (int): Number of keypoints/joints to predict (17 for COCO). Default: 17.
        resnet_layers (Sequence[int]): Number of Bottleneck blocks per ResNet stage. Default: (3, 4, 6, 3).
        pretrained_backbone (bool): Whether to load torchvision ImageNet pretrained weights. Default: True.
        input_size (Tuple[int, int]): Model input image dimensions (W, H). Default: (192, 256).
        heatmap_size (Tuple[int, int]): Output heatmap dimensions (W, H). Default: (48, 64).
        sigma (float): Standard deviation of Gaussian heatmap targets. Default: 2.0.
        flip_pairs (List[Tuple[int, int]], optional): Keypoint index pairs to swap during flip test.
            Default: COCO_FLIP_PAIRS.
    """

    def __init__(
        self,
        num_joints: int = 17,
        resnet_layers: Sequence[int] = (3, 4, 6, 3),
        pretrained_backbone: bool = True,
        input_size: Tuple[int, int] = (192, 256),
        heatmap_size: Tuple[int, int] = (48, 64),
        sigma: float = 2.0,
        flip_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.flip_pairs = flip_pairs if flip_pairs is not None else COCO_FLIP_PAIRS

        # 1. ResNet Backbone
        self.backbone = ResNet(layers=resnet_layers, in_channels=3, out_indices=(3,))
        if pretrained_backbone:
            self.backbone.init_weights(pretrained=True)

        # 2. Deconvolutional Heatmap Head
        self.head = HeatmapHead(
            in_channels=2048,
            out_channels=self.num_joints,
            deconv_out_channels=(256, 256, 256),
            deconv_kernel_sizes=(4, 4, 4),
            final_layer_kernel=1,
        )

        # 3. Target Encoding / Decoding Codec
        self.codec = MSRAHeatmap(
            input_size=self.input_size,
            heatmap_size=self.heatmap_size,
            sigma=sigma,
        )

        # 4. Heatmap MSE Loss
        self.loss_module = KeypointMSELoss(use_target_weight=True, loss_weight=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting backbone features and predicting raw heatmap logits.

        Args:
            x (torch.Tensor): Input image batch of shape (B, 3, H, W), e.g. (B, 3, 256, 192).

        Returns:
            torch.Tensor: Predicted raw heatmap logits of shape (B, K, H/4, W/4), e.g. (B, 17, 64, 48).
        """
        feats = self.backbone(x)
        return self.head(feats)

    def loss(
        self,
        x: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute keypoint heatmap regression loss for a batch of images.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
            target_heatmaps (torch.Tensor): Ground-truth Gaussian heatmaps of shape (B, K, H/4, W/4).
            target_weights (torch.Tensor): Per-joint visibility weights of shape (B, K).

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        pred_heatmaps = self.forward(x)
        return self.loss_module(pred_heatmaps, target_heatmaps, target_weights)

    def predict(
        self, x: torch.Tensor, flip_test: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform inference and decode keypoint coordinates in input image coordinate space.

        Replicates test-time augmentation (flip_test=True, flip_mode='heatmap', shift_heatmap=False).

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
            flip_test (bool): If True, computes average of original and flipped heatmaps
                with swapped left/right symmetric joint pairs before decoding. Default: True.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - keypoints: Predicted keypoint coordinates (x, y) of shape (B, K, 2) in
                  input-image pixel coordinates.
                - scores: Keypoint confidence scores of shape (B, K).
        """
        self.eval()
        with torch.no_grad():
            heatmaps = self.forward(x)

            if flip_test:
                # 1. Horizontally flip input images (dim 3 is width)
                x_flipped = torch.flip(x, dims=[3])
                heatmaps_flipped = self.forward(x_flipped)

                # 2. Horizontally flip the predicted heatmaps back
                heatmaps_flipped = torch.flip(heatmaps_flipped, dims=[3])

                # 3. Swap left and right symmetric joint heatmap channels
                heatmaps_flipped_swapped = heatmaps_flipped.clone()
                for a, b in self.flip_pairs:
                    heatmaps_flipped_swapped[:, a] = heatmaps_flipped[:, b]
                    heatmaps_flipped_swapped[:, b] = heatmaps_flipped[:, a]

                # 4. Average original and flipped heatmaps
                heatmaps = (heatmaps + heatmaps_flipped_swapped) * 0.5

        heatmaps_np = heatmaps.cpu().numpy()
        B = heatmaps_np.shape[0]

        keypoints_list = []
        scores_list = []
        for i in range(B):
            kpts, scs = self.codec.decode(heatmaps_np[i])
            keypoints_list.append(kpts)
            scores_list.append(scs)

        keypoints = np.stack(keypoints_list, axis=0)  # (B, K, 2)
        scores = np.stack(scores_list, axis=0)        # (B, K)
        return keypoints, scores


if __name__ == "__main__":
    print("Initializing SimpleBaseline model...")
    model = SimpleBaseline(num_joints=17, pretrained_backbone=False)

    # 1. Forward pass test
    x = torch.randn(2, 3, 256, 192)
    heatmaps = model(x)
    assert heatmaps.shape == (
        2,
        17,
        64,
        48,
    ), f"Expected shape (2, 17, 64, 48), got {heatmaps.shape}"

    # 2. Loss & Backward test
    target_heatmaps = torch.rand(2, 17, 64, 48)
    target_weights = torch.ones(2, 17)
    loss = model.loss(x, target_heatmaps, target_weights)
    assert loss.item() >= 0, f"Expected non-negative loss, got {loss.item()}"

    loss.backward()
    assert (
        model.head.final_layer.weight.grad is not None
    ), "Gradients failed to flow to Head final_layer!"
    assert (
        next(model.backbone.parameters()).grad is not None
    ), "Gradients failed to flow to Backbone!"

    # 3. Prediction & Flip Test
    keypoints, scores = model.predict(x, flip_test=True)
    assert keypoints.shape == (
        2,
        17,
        2,
    ), f"Expected keypoints shape (2, 17, 2), got {keypoints.shape}"
    assert scores.shape == (
        2,
        17,
    ), f"Expected scores shape (2, 17), got {scores.shape}"

    # 4. Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nSimpleBaseline smoke test passed successfully!")
    print(f"Input image shape:        {x.shape}")
    print(f"Output heatmap shape:     {heatmaps.shape}")
    print(f"Computed loss value:      {loss.item():.6f}")
    print(f"Predicted keypoints shape: {keypoints.shape}")
    print(f"Predicted scores shape:    {scores.shape}")
    print(f"Total parameters:         {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"Trainable parameters:     {num_trainable:,} ({num_trainable / 1e6:.2f}M)")
