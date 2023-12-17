import math  # positional embeddingsのsin, cos用
from inspect import isfunction  # inspectモジュール
from functools import partial  # 関数の引数を一部設定できる便利ツール

# PyTorch, 計算関係
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from .utils import exists


class Residual(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class UpsampleConv(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(
            in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.trans_conv(x)


class DownSampleConv(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8) -> None:
        super().__init__()
        if exists(time_emb_dim):
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
        else:
            self.mlp = nn.Identity()

        self.block1 = ConvBlock(dim, dim_out, groups)
        self.block2 = ConvBlock(dim_out, dim_out, groups)

        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)

        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32) -> None:
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() # softmaxを取る際に計算を安定させる際に最大値を引く
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32) -> None:
        super().__init__()
        self.scale = dim_head ** (- 0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))
    
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = einsum("b h d n, b h e n -> b h d e", k, v) # 先にK.T・Vを計算することでO(n2)にならないようにしています(dim×dimになります)。

        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
    
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)