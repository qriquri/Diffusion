	
import math # positional embeddingsのsin, cos用
from inspect import isfunction # inspectモジュール
from functools import partial # 関数の引数を一部設定できる便利ツール
 
# PyTorch, 計算関係
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # i は次元を表しているから次元をかける
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    timesteps = 1000
    time = torch.arange(timesteps)
    pos_emb = SinusoidalPositionEmbeddings(dim=500)
    emb = pos_emb(time=time)
    plt.pcolormesh(emb.T, cmap='RdBu')
    plt.ylabel('dimension')
    plt.xlabel('time step')
    plt.colorbar()
    plt.show()
