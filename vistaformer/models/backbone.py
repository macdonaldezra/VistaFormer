import torch
import torch.nn as nn

from vistaformer.models.layers import (
    PatchEmbed3D,
    PosFeedForward3d,
    TransformerEncoder,
)


class VistaFormerBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: list[int],
        patch_sizes: list[int],
        strides: list[int],
        depths: list[int],
        num_heads: list[int],
        mlp_dims: list[int],
        dropout: float,
        drop_path: float,
        gate: bool,
        use_squeeze: bool,
        activation: str,
    ):
        super().__init__()
        assert (
            len(embed_dims)
            == len(patch_sizes)
            == len(strides)
            == len(depths)
            == len(num_heads)
            == len(mlp_dims)
        ), "Number of layers must match"

        self.embeddings = nn.ModuleList([])
        self.transformers = nn.ModuleList([])
        for i in range(len(embed_dims)):
            self.embeddings.append(
                PatchEmbed3D(
                    in_channels=embed_dims[i - 1] if i > 0 else in_channels,
                    embed_dims=embed_dims[i],
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                    use_squeeze=use_squeeze,
                    norm_type="batch2d" if i == 0 else "",
                    gate=gate,
                )
            )
            self.transformers.append(
                TransformerEncoder(
                    dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    mlp_dim=mlp_dims[i],
                    feed_forward=PosFeedForward3d,
                    activation=activation,
                    dropout=dropout,
                    drop_path=drop_path,
                )
            )

    def forward(self, x: torch.Tensor):
        """
        x: Input tensor of shape (B, T, C, H, W)
        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W) -> (B, C, T, H, W)
        outputs = []
        for embedding, transformer in zip(self.embeddings, self.transformers):
            x = embedding(x)
            B, C, T, H, W = x.shape
            # (B, C, T, H, W) -> (B * T, H * W, C)
            x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
            x = transformer(x, T, H, W)

            # Return the sequence of patches to the original shape
            x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
            outputs.append(x)

        return outputs
