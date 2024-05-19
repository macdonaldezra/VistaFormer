import torch
import math
from torch import nn
from timm.layers import DropPath, to_2tuple, trunc_normal_


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, instance_norm: bool = False):
        super().__init__()
        self.instance_norm = instance_norm
        self.fn = fn
        if self.instance_norm:
            self.norm = nn.InstanceNorm2d(dim)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, instance_norm: bool = False):
        super().__init__()
        self.instance_norm = instance_norm
        self.fn = fn
        if self.instance_norm:
            self.norm = nn.InstanceNorm2d(dim)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.proj_dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.to_qkv(x).reshape(b, n, 3, h, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # Reorder to separate q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = (q @ k.transpose(-2, -1)) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, -1)

        out = self.to_out(out)
        out = self.proj_dropout(out)

        return out


class SEBlock3D(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)

        return x * y.expand_as(x)


class GatedConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(GatedConv3D, self).__init__()
        self.conv_feature = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_gate = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.sigmoid = nn.Sigmoid()
        self.feature_maps = None
        self.gradients = None

    def forward(self, x):
        feature = self.conv_feature(x)
        gate = self.conv_gate(x)
        gate = self.sigmoid(gate)
        self.feature_maps = feature  # comment out if not visualizing feature maps
        return feature * gate


class GatedSEConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        reduction=2,
        padding=0,
        use_squeeze: bool = False,
    ):
        super(GatedSEConv3D, self).__init__()
        self.gated_conv = GatedConv3D(
            in_channels, out_channels, kernel_size, padding, stride
        )
        self.use_squeeze = use_squeeze
        if use_squeeze:
            self.se_block = SEBlock3D(out_channels, reduction)

    def forward(self, x):
        x = self.gated_conv(x)
        if self.use_squeeze:
            x = self.se_block(x)

        return x

    def backward_hook(self, grad):
        self.gradients = grad


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        stride: int,
        use_squeeze: bool,
        gate: bool,
        norm_type: str = "batch2d",
    ):
        """
        3D Patch Embedding Layer for Time Series Image Dataset. Optionally applies normalization
        to the input data.

        Parameters:
        - in_channels: Number of input channels (C).
        - out_channels: Number of output channels (D).
        - patch_size: Size of the patch (assuming cube patches for simplicity).
        - stride: Stride of the convolution.
        """
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Assuming patch_size is a tuple of (t_patch, h_patch, w_patch)
        self.proj = nn.Conv3d(
            in_channels, out_channels, kernel_size=patch_size, stride=stride
        )

        self.gate = gate
        if self.gate:
            self.gate_conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=patch_size, stride=stride
            )
            self.sigmoid = nn.Sigmoid()

        self.use_squeeze = use_squeeze
        if self.use_squeeze:
            self.se_block = SEBlock3D(out_channels)

        self.norm_type = norm_type
        if norm_type == "batch2d":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "batch3d":
            self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        """
        Forward pass of the 3D Patch Embedding Layer.

        Note this assumes that the input tensor is of shape (B, T, C, H, W).

        Parameters:
         - x: Input tensor of shape (B, T, C, H, W).

        Returns:
         - Embedded patches of shape (B, Seq_length, Emb_size).
        """
        # Apply the 3D convolution to project input image sequences to embeddings
        if self.gate:
            x_conv = self.proj(x)
            gate = self.sigmoid(self.gate_conv(x))
            x = x_conv * gate
        else:
            x = self.proj(x)  # Shape: (B, Emb_size, T', H', W')

        if self.use_squeeze:
            x = self.se_block(x)

        if self.norm_type == "batch2d":
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
            x = self.norm(x)
            x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        elif self.norm_type == "batch3d":
            x = self.norm(x)

        if return_gate:
            return x, gate

        return x


class DWConv3D(nn.Module):
    def __init__(
        self,
        dim: int,
        temporal: bool = False,
        kernel_size: int = 3,
        add_pos: bool = False,
    ):
        super(DWConv3D, self).__init__()
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )
        self.temporal = temporal
        self.dim = dim
        self.add_pos = add_pos

    def forward(self, x: torch.Tensor, x_shape: tuple[int]):
        b, c, t, h, w = x_shape
        if self.temporal:
            # (B * H * W, T, C) -> (B, C, T, H, W)
            x = x.reshape(b, h, w, t, self.dim).permute(0, 4, 3, 1, 2).contiguous()
        else:
            # (B*T, H*W, C) -> (B, C, T, H, W)
            x = x.reshape(b, t, h, w, self.dim).permute(0, 4, 1, 2, 3).contiguous()

        if self.add_pos:
            x = x + self.dwconv(x)
        else:
            x = self.dwconv(x)

        if self.temporal:
            x = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, self.dim)
        else:
            x = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, self.dim)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder (a.k.a. Self-Attention Transformer) for 3D data.

    Feed-Forward layer includes a positional encoding layer..
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        feed_forward: nn.Module,
        activation: str,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(dim, num_heads=num_heads, dropout=dropout),
                        ),
                        PreNorm(
                            dim,
                            feed_forward(
                                dim, mlp_dim, dropout=dropout, activation=activation
                            ),
                        ),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                    ]
                )
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, T: int, H: int, W: int):
        for attn, ff, drop_path in self.layers:
            x = drop_path(attn(x) + x)
            x = drop_path(ff(x, **{"T": T, "H": H, "W": W}) + x)

        return self.norm(x)


def get_activation_layer(activation: str):
    if activation == "gelu":
        return nn.GELU()
    elif activation == "swish":
        return nn.SiLU()
    elif activation == "mish":
        return nn.Mish()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError(
            "Invalid activation function. Choose from 'gelu', 'swish', 'mish', 'relu', or 'leakyrelu'."
        )


class DWConv3d(nn.Module):
    def __init__(self, dim: int):
        super(DWConv3d, self).__init__()
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int):
        B_T, _, C = x.shape
        B = int(B_T // T)
        # (B*T, H*W, C) -> (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B * T, H * W, C)

        return x


class PosFeedForward3d(nn.Module):
    """
    Note: Could also use dilated convolutions and downsample the data to be 2D since we only want to really perform
        classification and segmentation on the dataset.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float,
        activation: str,
    ):
        super(PosFeedForward3d, self).__init__()
        embed_dim = embed_dim or input_dim
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.dwconv = DWConv3d(embed_dim)

        self.act = get_activation_layer(activation)
        self.fc2 = nn.Linear(embed_dim, input_dim)
        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, T: int, H: int, W: int):
        x = self.fc1(x)
        x = self.dwconv(x, T, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
