import torch
import torch.nn as nn

from vistaformer.models.backbone import VistaFormerBackbone
from vistaformer.models.head import VistaFormerHead


class VistaFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_dim: int,
        num_classes: int,
        depths: list[int],
        embed_dims: list[int],
        seq_lens: list[int],
        patch_sizes: list[int],
        strides: list[int],
        num_heads: list[int],
        mlp_mult: int,
        gate: bool,
        activation: str = "mish",
        use_squeeze: bool = False,
        head_conv_dim: int = 64,
        head_upsample_type: str = "trilinear",
        head_temporal_agg_type: str = "conv",
        head_norm_type: str = "layer",
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        mlp_dims = [embed_dims[i] * mlp_mult for i in range(len(embed_dims))]
        self.backbone = VistaFormerBackbone(
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_sizes=patch_sizes,
            strides=strides,
            depths=depths,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout,
            drop_path=drop_path,
            gate=gate,
            use_squeeze=use_squeeze,
            activation=activation,
        )
        self.head = VistaFormerHead(
            input_dim=input_dim,
            embed_dims=embed_dims,
            seq_lens=seq_lens,
            num_classes=num_classes,
            dropout=dropout,
            conv_embed_dim=head_conv_dim,
            upsample_type=head_upsample_type,
            temporal_agg_type=head_temporal_agg_type,
            norm_type=head_norm_type,
            activation=activation,
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.head(x)

        return x
