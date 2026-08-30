"""ResNet Backbone for SimpleBaseline (Xiao et al. 2018).

Standalone PyTorch implementation replicating mmpose/models/backbones/resnet.py.
Maintains state_dict key compatibility with torchvision ResNet-50 for loading
pretrained ImageNet weights directly without external dependencies.
"""

from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.hub

# Torchvision ResNet-50 ImageNet-1k pretrained weights URL
RESNET50_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152.

    Args:
        in_channels (int): Number of input channels.
        planes (int): Base number of channels (output is planes * expansion).
        stride (int): Stride of the 3x3 convolution. Default: 1.
        dilation (int): Dilation of the 3x3 convolution. Default: 1.
        downsample (nn.Module, optional): Downsample shortcut layer. Default: None.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.planes = planes
        self.stride = stride
        self.dilation = dilation

        # 1x1 conv
        self.conv1 = nn.Conv2d(
            in_channels, planes, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 conv (expansion)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone for top-down pose estimation.

    Faithfully replicates the ResNet-50 backbone configuration from mmpose
    (configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py).

    Args:
        layers (Sequence[int]): Number of Bottleneck blocks per stage. Default: (3, 4, 6, 3) (ResNet-50).
        in_channels (int): Number of input channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        out_indices (Sequence[int]): Indices of output stages (0, 1, 2, 3). Default: (3,).
        strides (Sequence[int]): Strides for each of the 4 stages. Default: (1, 2, 2, 2).
        dilations (Sequence[int]): Dilations for each of the 4 stages. Default: (1, 1, 1, 1).
        frozen_stages (int): Stages to freeze (-1 to freeze nothing). Default: -1.
        norm_eval (bool): Whether to set BatchNorm to eval mode always. Default: False.
    """

    def __init__(
        self,
        layers: Sequence[int] = (3, 4, 6, 3),
        in_channels: int = 3,
        stem_channels: int = 64,
        out_indices: Sequence[int] = (3,),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        frozen_stages: int = -1,
        norm_eval: bool = False,
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.out_indices = list(out_indices)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # Stem: 7x7 conv (stride=2) -> bn -> relu -> maxpool (stride=2)
        self.conv1 = nn.Conv2d(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual stages
        self._in_channels = stem_channels
        self.stage_names = []
        for i, (num_blocks, stride, dilation) in enumerate(
            zip(self.layers, strides, dilations)
        ):
            planes = stem_channels * (2**i)
            stage_name = f"layer{i + 1}"
            layer = self._make_layer(
                block=Bottleneck,
                planes=planes,
                blocks=num_blocks,
                stride=stride,
                dilation=dilation,
            )
            setattr(self, stage_name, layer)
            self.stage_names.append(stage_name)

        self._init_weights()

    def _make_layer(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        out_channels = planes * block.expansion

        # Downsample shortcut when spatial size or channel count changes
        if stride != 1 or self._in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers: List[nn.Module] = []
        # First block with possible downsampling and stride
        layers.append(
            block(
                in_channels=self._in_channels,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self._in_channels = out_channels

        # Subsequent blocks in the stage
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self._in_channels,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming Normal (matching standard ResNet practice)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_weights(
        self, pretrained: bool = True, url: str = RESNET50_URL
    ) -> None:
        """Load pretrained ImageNet weights from torchvision repository.

        Args:
            pretrained (bool): If True, downloads and loads ImageNet weights.
            url (str): URL to the torchvision ResNet-50 checkpoint.
        """
        if not pretrained:
            self._init_weights()
            return

        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=True, map_location="cpu"
        )

        # Filter out fc and avgpool layers not present in the backbone
        filtered_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("fc.") and not k.startswith("avgpool.")
        }
        self.load_state_dict(filtered_dict, strict=False)

    def _freeze_stages(self) -> None:
        """Freeze stem and stage layers up to self.frozen_stages."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            for param in self.bn1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "ResNet":
        """Set training mode while enforcing frozen stages and norm_eval."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass through ResNet backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W), e.g. (B, 3, 256, 192).

        Returns:
            Tuple[torch.Tensor, ...]: Tuple containing feature maps from stages
                specified by `out_indices`. For out_indices=(3,), returns (layer4_out,)
                of shape (B, 2048, H/32, W/32).
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        outs = []
        for i, stage_name in enumerate(self.stage_names):
            layer = getattr(self, stage_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)


if __name__ == "__main__":
    model = ResNet(layers=[3, 4, 6, 3], in_channels=3, out_indices=(3,))
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 192)
    output = model(dummy_input)

    assert isinstance(output, tuple), f"Expected tuple output, got {type(output)}"
    assert len(output) == 1, f"Expected 1 output feature map, got {len(output)}"
    assert output[0].shape == (
        2,
        2048,
        8,
        6,
    ), f"Expected shape (2, 2048, 8, 6), got {output[0].shape}"

    print(f"ResNet-50 smoke test passed successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output[0].shape}")
