"""Heatmap Head for SimpleBaseline (Xiao et al. 2018).

Standalone PyTorch implementation replicating
mmpose/models/heads/heatmap_heads/heatmap_head.py.
Pure PyTorch (torch.nn) with zero mmcv/mmengine/mmpose dependencies.
"""

from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn


def _get_deconv_cfg(deconv_kernel: int) -> Tuple[int, int, int]:
    """Get deconv configurations (kernel, padding, output_padding) for stride=2.

    Matches mmpose's deconvolution parameter derivation.

    Args:
        deconv_kernel (int): Kernel size for deconvolution layer.

    Returns:
        Tuple[int, int, int]: (kernel_size, padding, output_padding)
    """
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0
    else:
        raise ValueError(
            f"Unsupported deconv_kernel {deconv_kernel}, expected 2, 3, or 4."
        )
    return deconv_kernel, padding, output_padding


class HeatmapHead(nn.Module):
    """Top-down heatmap head with progressive deconvolution upsampling.

    Replicates HeatmapHead from mmpose with default configuration for
    ResNet-50 on COCO (td-hm_res50_8xb64-210e_coco-256x192.py).

    Upsamples backbone feature map from (8, 6) -> (16, 12) -> (32, 24) -> (64, 48)
    via 3 Transpose Convolution stages (2048 -> 256 -> 256 -> 256), followed
    by a 1x1 Conv producing raw heatmap logits for 17 keypoints.

    Args:
        in_channels (int): Number of input channels from backbone. Default: 2048.
        out_channels (int): Number of output heatmap channels (joints). Default: 17.
        deconv_out_channels (Sequence[int]): Number of output channels for each deconv layer.
            Default: (256, 256, 256).
        deconv_kernel_sizes (Sequence[int]): Kernel size for each deconv layer.
            Default: (4, 4, 4).
        final_layer_kernel (int): Kernel size of the final conv layer. Default: 1.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 17,
        deconv_out_channels: Sequence[int] = (256, 256, 256),
        deconv_kernel_sizes: Sequence[int] = (4, 4, 4),
        final_layer_kernel: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv_out_channels = tuple(deconv_out_channels)
        self.deconv_kernel_sizes = tuple(deconv_kernel_sizes)

        if len(self.deconv_out_channels) != len(self.deconv_kernel_sizes):
            raise ValueError(
                f"Length of deconv_out_channels ({len(self.deconv_out_channels)}) "
                f"must match deconv_kernel_sizes ({len(self.deconv_kernel_sizes)})."
            )

        # Build 3-stage deconvolution layers
        self.deconv_layers = self._make_deconv_layer(
            in_channels=self.in_channels,
            layer_out_channels=self.deconv_out_channels,
            layer_kernel_sizes=self.deconv_kernel_sizes,
        )

        # Final 1x1 conv layer to produce raw heatmap logits
        final_in_channels = (
            self.deconv_out_channels[-1]
            if len(self.deconv_out_channels) > 0
            else self.in_channels
        )
        self.final_layer = nn.Conv2d(
            in_channels=final_in_channels,
            out_channels=self.out_channels,
            kernel_size=final_layer_kernel,
            stride=1,
            padding=final_layer_kernel // 2,
        )

        # Initialize weights
        self.init_weights()

    def _make_deconv_layer(
        self,
        in_channels: int,
        layer_out_channels: Sequence[int],
        layer_kernel_sizes: Sequence[int],
    ) -> nn.Sequential:
        """Build progressive deconvolution stages.

        Each stage consists of:
        ConvTranspose2d(kernel=4, stride=2, padding=1, bias=False)
        -> BatchNorm2d
        -> ReLU(inplace=True)

        Args:
            in_channels (int): Initial input channels (2048).
            layer_out_channels (Sequence[int]): Output channels per stage (256, 256, 256).
            layer_kernel_sizes (Sequence[int]): Kernel sizes per stage (4, 4, 4).

        Returns:
            nn.Sequential: Sequential container of deconv blocks.
        """
        layers = []
        curr_in_channels = in_channels

        for out_ch, k in zip(layer_out_channels, layer_kernel_sizes):
            kernel, padding, output_padding = _get_deconv_cfg(k)
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=curr_in_channels,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            curr_in_channels = out_ch

        return nn.Sequential(*layers)

    def init_weights(self) -> None:
        """Initialize weights matching mmpose init_cfg.

        ConvTranspose2d: Normal(mean=0, std=0.001)
        BatchNorm2d: weight=1, bias=0
        final_layer (Conv2d): Normal(mean=0, std=0.001), bias=0
        """
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        if isinstance(self.final_layer, nn.Conv2d):
            nn.init.normal_(self.final_layer.weight, mean=0.0, std=0.001)
            if self.final_layer.bias is not None:
                nn.init.constant_(self.final_layer.bias, 0.0)

    def forward(
        self, feats: Union[Tuple[torch.Tensor, ...], torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through HeatmapHead.

        Args:
            feats (Tuple[torch.Tensor, ...] or torch.Tensor): Backbone output feature maps.
                If tuple/list, the last feature map `feats[-1]` is used.

        Returns:
            torch.Tensor: Raw heatmap logits of shape (B, out_channels, H_hm, W_hm),
                e.g. (B, 17, 64, 48) for (256, 192) input.
        """
        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        else:
            x = feats

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    head = HeatmapHead(in_channels=2048, out_channels=17)
    head.eval()

    feats = (torch.randn(2, 2048, 8, 6),)
    out = head(feats)

    assert out.shape == (
        2,
        17,
        64,
        48,
    ), f"Expected shape (2, 17, 64, 48), got {out.shape}"

    print("HeatmapHead smoke test passed successfully!")
    print(f"Input features shape: {feats[0].shape}")
    print(f"Output heatmap shape: {out.shape}\n")

    print("--- Deconv Layers Structure ---")
    print(head.deconv_layers)
    print("\n--- Final Layer ---")
    print(head.final_layer)
