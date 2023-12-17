import math  # positional embeddingsのsin, cos用
from inspect import isfunction  # inspectモジュール
from functools import partial  # 関数の引数を一部設定できる便利ツール

# PyTorch, 計算関係
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from .modules import (
    ResnetBlock,
    DownSampleConv,
    UpsampleConv,
    Residual,
    PreNorm,
    LinearAttention,
    Attention,
)
from .pos_emb import SinusoidalPositionEmbeddings
from .utils import default, exists


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_multi=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
    ) -> None:
        super().__init__()

        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=7 // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_multi)]
        in_out = list(
            zip(dims[:-1], dims[1:])
        )  # (input_dim, output_dim)というタプルのリストを作成する

        resnet_block = partial(ResnetBlock, groups=resnet_block_groups)

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                        resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        DownSampleConv(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 中間ブロック
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        UpsampleConv(dim_in) if not is_last else nn.Identity()
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            resnet_block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []

        for block1, block2, attn, down_sample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = down_sample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x)
