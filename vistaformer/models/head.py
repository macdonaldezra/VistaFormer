import torch
import torch.nn as nn

from vistaformer.models.layers import get_activation_layer


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdaptiveTemporalFeaturePooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_temporal_size: int = 1,
        pool_type: str = "avg",
    ):
        super().__init__()
        assert pool_type in [
            "avg",
            "max",
        ], "Invalid pooling type. Choose from 'avg' or 'max'."
        if pool_type == "avg":
            self.adaptive_pool = nn.AdaptiveAvgPool3d(
                (output_temporal_size, None, None)
            )
        else:
            self.adaptive_pool = nn.AdaptiveMaxPool3d(
                (output_temporal_size, None, None)
            )

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=1
        )  # Reduce channel dimensions

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.conv(x)

        return x


class GatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.gate = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.conv(x) * torch.sigmoid(self.gate(x))
        return x


def get_temporal_agg_layer(type: str, in_channels: int, out_channels: int, T: int):
    if type == "conv":
        return nn.Conv3d(
            in_channels, out_channels, kernel_size=(T, 1, 1), stride=(T, 1, 1)
        )
    elif type == "gatedconv":
        return GatedConv3d(
            in_channels,
            out_channels,
            kernel_size=(T, 1, 1),
            stride=(T, 1, 1),
            padding=0,
        )
    elif type == "depthwise":
        return DepthwiseSeparableConv3d(
            in_channels, out_channels, kernel_size=(T, 1, 1), padding=0
        )
    elif type == "adaptive_avg_pool":
        return AdaptiveTemporalFeaturePooling(
            in_channels, out_channels, output_temporal_size=1, pool_type="avg"
        )
    elif type == "adaptive_max_pool":
        return AdaptiveTemporalFeaturePooling(
            in_channels, out_channels, output_temporal_size=1, pool_type="max"
        )
    else:
        raise ValueError(
            "Invalid temporal aggregation type. Choose from 'conv', 'depthwise', or 'adaptive_pool'."
        )


class VistaFormerHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dims: list[int],
        seq_lens: list[int],
        num_classes: int,
        dropout: float,
        temporal_agg_type: str,
        conv_embed_dim: int = 64,
        upsample_type: str = "trilinear",
        norm_type: str = "batch",
        activation: str = "mish",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_dim = input_dim
        assert len(embed_dims) == len(
            seq_lens
        ), "Embedding dimensions and sequence lengths must match."

        # TODO: Add functionality for using Transposed Convolutions instead of bilinear upsampling
        assert upsample_type in ["bilinear", "trilinear", "conv"]
        self.upsample_type = upsample_type
        if upsample_type == "bilinear":
            self.upsample = nn.Upsample(
                size=(input_dim, input_dim), mode="bilinear", align_corners=False
            )
        elif upsample_type == "conv":
            self.upsample = nn.ModuleList(
                [  # layer only works when height and width dimensions are halved at each layer
                    nn.ConvTranspose3d(
                        in_channels=embed_dims[i],
                        out_channels=embed_dims[i],
                        kernel_size=(1, 2 ** (i + 1), 2 ** (i + 1)),
                        stride=(1, 2 ** (i + 1), 2 ** (i + 1)),
                        padding=0,
                    )
                    for i in range(len(embed_dims))
                ]
            )
        else:
            self.upsample = nn.ModuleList(
                [
                    nn.Upsample(
                        size=(seq_lens[i], input_dim, input_dim),
                        mode="trilinear",
                        align_corners=False,
                    )
                    for i in range(len(seq_lens))
                ]
            )

        self.act = get_activation_layer(activation)
        self.temp_downsample = nn.ModuleList(
            [
                get_temporal_agg_layer(
                    temporal_agg_type, embed_dims[i], conv_embed_dim, seq_lens[i]
                )
                for i in range(len(embed_dims))
            ]
        )

        self.fuse = nn.Conv2d(
            conv_embed_dim * len(embed_dims), conv_embed_dim, kernel_size=1
        )
        self.norm = nn.BatchNorm2d(conv_embed_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(conv_embed_dim, num_classes, kernel_size=1)

    def forward(self, x: list[torch.Tensor]):
        """
        x: List of tensors of shape (B, C, T, H, W).
        """
        # If we upsample using bilinear upsampling, then we upsample for each time step
        # Otherwise, we upsample the entire tensor
        if self.upsample_type == "bilinear":
            upsampled = []
            for i in range(len(x)):
                b, c, t, h, w = x[i].shape
                x_i = x[i].permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
                x_i = self.upsample(x_i)
                x_i = (
                    x_i.view(b, t, c, self.output_dim, self.output_dim)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                )
                upsampled.append(x_i)

            x = upsampled
        else:
            x = [self.upsample[i](x[i]) for i in range(len(x))]

        # Aggregate time information for each layer and downsample to 1 time step
        x = [self.temp_downsample[i](x[i]).squeeze(2) for i in range(len(x))]

        x = torch.cat(x, dim=1)
        x = self.fuse(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)

        return x
