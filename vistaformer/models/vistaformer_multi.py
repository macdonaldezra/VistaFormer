from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath

from vistaformer.models.backbone import VistaFormerBackbone
from vistaformer.models.head import VistaFormerHead
from vistaformer.models.layers import CrossAttentionTransformerLayer


class FeatureFusionConcat(nn.Module):
    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int):
        super(FeatureFusionConcat, self).__init__()
        self.conv = nn.Conv3d(in_channels1 + in_channels2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate along the channel dimension
        x = self.conv(x)

        return x


class FeatureFusionAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionAttention, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.attention = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        attention_weights = self.attention(x1 + x2)
        x = attention_weights * x1 + (1 - attention_weights) * x2
        return x


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        in_embed_dim: int,
        out_embed_dim: int,
        fusion_type: str,
        dropout: float,
        drop_path: float,
        mlp_mult: int = 4,
        attn_heads: Optional[int] = None,
        use_depthwise: bool = False,
    ):
        super(FeatureFusionBlock, self).__init__()
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fusion = FeatureFusionConcat(
                in_channels1=in_embed_dim,
                in_channels2=in_embed_dim,
                out_channels=out_embed_dim,
            )
        elif fusion_type == "crossattn":
            self.fusion = CrossAttentionTransformerLayer(
                embed_dim=in_embed_dim,
                mlp_dim=in_embed_dim * mlp_mult,
                num_heads=attn_heads,
                dropout=dropout,
                drop_path=drop_path,
            )
        else:  # TODO: implement Cross attention with a Transformer
            raise ValueError(f"Invalid fusion type: {fusion_type}")

        self.pre_norm = (
            nn.LayerNorm(in_embed_dim) if fusion_type == "crossattn" else nn.Identity()
        )
        self.use_depthwise = use_depthwise
        if use_depthwise:
            self.dw_conv1 = nn.Conv3d(
                in_embed_dim,
                in_embed_dim,
                kernel_size=3,
                padding=1,
                groups=in_embed_dim,
            )
            self.dw_conv2 = nn.Conv3d(
                in_embed_dim,
                in_embed_dim,
                kernel_size=3,
                padding=1,
                groups=in_embed_dim,
            )
        # self.norm = nn.LayerNorm(out_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        """
        x1: (B, C, T, H, W)
        x2: (B, C, T, H, W)
        """
        B, C, T, H, W = x1.shape
        if self.use_depthwise:
            x1 = self.dw_conv1(x1)
            x2 = self.dw_conv2(x2)
        if self.fusion_type == "crossattn":
            x1 = x1.view(B, C, -1).transpose(1, 2).contiguous()
            x2 = x2.view(B, C, -1).transpose(1, 2).contiguous()
            x1 = self.pre_norm(x1)
            x2 = self.pre_norm(x2)
            # x = self.fusion(x1, x2, T, H, W)
            x = checkpoint(self.fusion, x1, x2, T, H, W)
            x = x.transpose(1, 2).contiguous().reshape(B, C, T, H, W)
        else:
            x = self.fusion(x1, x2)

        x = self.dropout(x)

        return x


class AuxLayer(nn.Module):
    def __init__(
        self, seq_len: int, scale_factor: int, in_channels: int, num_classes: int
    ):
        super(AuxLayer, self).__init__()
        self.temporal_pool = nn.AvgPool3d(
            kernel_size=(seq_len, 1, 1), stride=(seq_len, 1, 1)
        )
        self.upsample = nn.Upsample(
            scale_factor=(scale_factor, scale_factor),
            mode="bilinear",
            align_corners=False,
        )  # Upsampling from (16, 16) to (32, 32)
        self.class_conv = nn.Conv2d(
            in_channels, num_classes, kernel_size=1
        )  # Map channels to num_classes

    def forward(self, x: torch.Tensor):
        x = self.temporal_pool(x)  # Shape: (B, C, 1, H, W)
        x = x.squeeze(2)  # Shape: (B, C, H, W)
        x = self.upsample(x)  # Shape: (B, C, h_out, w_out)
        x = self.class_conv(x)  # Shape: (B, num_classes, h_out, w_out)

        return x


class VistaFormerMulti(nn.Module):
    def __init__(
        self,
        first_in_channels: int,
        second_in_channels: int,
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
        padding: list[int] = [0, 0, 0],
        fusion_type: str = "concat",
        activation: str = "gelu",
        use_squeeze: bool = False,
        head_conv_dim: int = 64,
        head_upsample_type: str = "trilinear",
        head_temporal_agg_type: str = "depthwise",
        head_norm_type: str = "batch",
        dropout: float = 0.0,
        drop_path: float = 0.0,
        aux_loss_weight: float = 0.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        mlp_dims = [embed_dims[i] * mlp_mult for i in range(len(embed_dims))]
        self.backbone1 = VistaFormerBackbone(
            in_channels=first_in_channels,
            embed_dims=embed_dims,
            patch_sizes=patch_sizes,
            strides=strides,
            # padding=padding,
            depths=depths,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout,
            drop_path=drop_path,
            gate=gate,
            use_squeeze=use_squeeze,
            activation=activation,
        )
        self.backbone2 = VistaFormerBackbone(
            in_channels=second_in_channels,
            embed_dims=embed_dims,
            patch_sizes=patch_sizes,
            strides=strides,
            # padding=padding,
            depths=depths,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout,
            drop_path=drop_path,
            gate=gate,
            use_squeeze=use_squeeze,
            activation=activation,
        )

        self.fusion_blocks = nn.ModuleList(
            [
                FeatureFusionBlock(
                    in_embed_dim=embed_dims[i],
                    out_embed_dim=head_conv_dim,
                    fusion_type=fusion_type,
                    attn_heads=num_heads[i],
                    dropout=dropout,
                    drop_path=drop_path,
                )
                for i in range(len(embed_dims))
            ]
        )

        self.head = VistaFormerHead(
            input_dim=input_dim,
            embed_dims=(
                embed_dims
                if fusion_type == "crossattn"
                else [head_conv_dim] * len(embed_dims)
            ),
            seq_lens=seq_lens,
            num_classes=num_classes,
            dropout=dropout,
            conv_embed_dim=head_conv_dim,
            upsample_type=head_upsample_type,
            temporal_agg_type=head_temporal_agg_type,
            norm_type=head_norm_type,
            activation=activation,
        )

        self.ignore_index = ignore_index  # only used for aux loss
        self.aux_loss_weight = aux_loss_weight
        if aux_loss_weight > 0.0:
            # Define auxiliary heads for intermediate outputs
            self.auxiliary_heads = nn.ModuleList(
                [
                    AuxLayer(
                        seq_lens[i],
                        2 ** (i + 1),  # i=0, out=2, i=1, out=4, i=2, out=8
                        in_channels=head_conv_dim,
                        num_classes=num_classes,
                    )
                    for i in range(len(embed_dims))
                ]
            )

    def compute_aux_loss(self, aux_outputs, target):
        aux_loss = 0
        for aux_output in aux_outputs:
            aux_loss += nn.functional.cross_entropy(
                aux_output, target, ignore_index=self.ignore_index
            )
        aux_loss /= len(aux_outputs)  # Average auxiliary loss over all heads
        return aux_loss * self.aux_loss_weight  # Scale by the auxiliary loss weight

    def forward(self, x: torch.Tensor, y: torch.Tensor, return_aux: bool = False):
        x_outs = self.backbone1(x)
        y_outs = self.backbone2(y)
        outputs = []
        aux_outputs = [] if return_aux else None

        for i, fusion_block in enumerate(self.fusion_blocks):
            out = fusion_block(x_outs[i], y_outs[i])
            aux_outputs.append(self.auxiliary_heads[i](out)) if return_aux else None
            outputs.append(out)

        x = self.head(outputs)

        if return_aux:
            return x, aux_outputs

        return x
