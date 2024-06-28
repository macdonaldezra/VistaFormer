from typing import Optional

import torch
from torch import nn
from timm.layers import to_2tuple, DropPath
from natten import NeighborhoodAttention2D

from vistaformer.models.layers import PatchEmbed3D, PreNorm, PosFeedForward3d
from vistaformer.models.head import VistaFormerHead


class NeighbourhoodTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        spatial_kernel_size: int,
        spatial_depth: int,
        num_heads: int,
        mlp_dim: int,
        drop_path: float,
        dropout: float,
        out_norm: bool,
    ):
        """
        Transformer Block for Time Series Image Dataset.

        Parameters:
        - dim: Dimension of the input tensor.
        - depth: Number of transformer layers.
        - num_heads: Number of attention heads.
        - mlp_dim: Dimension of the MLP layer.
        - dropout: Dropout rate.
        - norm_layer: Normalization layer.
        """
        super().__init__()
        self.spatial_layers = nn.ModuleList([])
        self.out_norm = nn.LayerNorm(dim) if out_norm else nn.Identity()

        for _ in range(spatial_depth):
            self.spatial_layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            NeighborhoodAttention2D(
                                dim,
                                num_heads=num_heads,
                                kernel_size=spatial_kernel_size,
                                attn_drop=dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            PosFeedForward3d(
                                dim, mlp_dim, dropout=dropout, activation="gelu"
                            ),
                        ),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Transformer Block.

        Parameters:
            - x: Input tensor of shape (Batch, Channels, Time, Height, Width).

        Returns:
            - Transformed tensor of shape (B, Seq_length, Emb_size).
        """
        b, c, t, h, w = x.shape
        x_shape = x.shape

        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(b * t, h, w, c)
        for attn, ff, dropout in self.spatial_layers:
            x = dropout(attn(x) + x)
            x = x.reshape(b * t, h * w, c)
            x = dropout(ff(x, **{"T": t, "H": h, "W": w}) + x)
            x = x.reshape(b * t, h, w, c)

        x = self.out_norm(x)
        x = x.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

        return x


class VistaFormerNeighbourhoodBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: list[int],
        patch_sizes: list[int],
        strides: list[int],
        spatial_depths: list[int],
        num_heads: list[int],
        mlp_mult: int,
        spatial_kernel_sizes: list[int],
        drop_path: float,
        dropout: float,
        out_norm: bool,
    ):
        super().__init__()
        assert (
            len(embed_dims)
            == len(patch_sizes)
            == len(strides)
            == len(spatial_depths)
            == len(num_heads)
            == len(spatial_kernel_sizes)
        ), "Number of layers must match"
        self.embeddings = nn.ModuleList([])
        self.transformers = nn.ModuleList([])

        for i in range(len(embed_dims)):
            self.embeddings.append(
                PatchEmbed3D(
                    in_channels=embed_dims[i - 1] if i > 0 else in_channels,
                    out_channels=embed_dims[i],
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                    use_squeeze=False,
                    gate=True,
                    norm_type="batch2d" if i == 0 else "",
                )
            )
            self.transformers.append(
                NeighbourhoodTransformerBlock(
                    dim=embed_dims[i],
                    spatial_kernel_size=spatial_kernel_sizes[i],
                    spatial_depth=spatial_depths[i],
                    num_heads=num_heads[i],
                    mlp_dim=embed_dims[i] * mlp_mult,
                    dropout=dropout,
                    drop_path=drop_path,
                    out_norm=out_norm,
                )
            )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W) -> (B, C, T, H, W)
        outputs = []
        for embedding, transformer in zip(self.embeddings, self.transformers):
            x = embedding(x)  # Embed the input tensor and return a sequence of patches
            x = transformer(x)
            outputs.append(x)

        return outputs


class VistaFormerNeighbourhood(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_dim: tuple[int, int],
        num_classes: int,
        spatial_depths: list[int],
        spatial_kernel_sizes: list[int],
        embed_dims: list[int],
        seq_lens: list[int],
        patch_sizes: list[int],
        strides: list[int],
        num_heads: list[int],
        drop_path: float,
        out_norm: bool = True,
        mlp_mult: int = 4,
        head_conv_dim: Optional[int] = None,
        dropout: float = 0.05,
        temporal_agg_type: str = "conv",
    ):
        """
        Transformer model that uses 3D Convolutions for positional encoding,
        and uses a time-series and spatial transformer -based backbone.
        """
        super().__init__()
        self.backbone = VistaFormerNeighbourhoodBackbone(
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_sizes=patch_sizes,
            strides=strides,
            spatial_depths=spatial_depths,
            num_heads=num_heads,
            spatial_kernel_sizes=spatial_kernel_sizes,
            mlp_mult=mlp_mult,
            dropout=dropout,
            drop_path=drop_path,
            out_norm=out_norm,
        )
        self.head = VistaFormerHead(
            input_dim=input_dim,
            embed_dims=embed_dims,
            seq_lens=seq_lens,  # This will only work for sequence lengths of 32
            num_classes=num_classes,
            conv_embed_dim=head_conv_dim,
            dropout=dropout,
            temporal_agg_type=temporal_agg_type,
        )

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.head(x)

        return x
