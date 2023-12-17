from typing import Optional
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t - 1)  # aの最後の次元 ⇒ timestepに対応するalphaを取ってくる
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(
        t.device
    )  # バッチサイズ x 1 x 1 x 1にreshape


def linear_beta_schedule(time_steps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, time_steps)


def q_sample(
    x_start: torch.Tensor,
    t,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumpord: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
):
    """_summary_
    きれいな画像からノイズを加えた画像をサンプルする
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumpord_t = extract(
        sqrt_one_minus_alphas_cumpord, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumpord_t * noise


def compute_noisy_image(x_start: torch.Tensor, time_steps):
    betas = linear_beta_schedule(time_steps=time_steps.item()).to(device=x_start.device)

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumpord = torch.sqrt(1.0 - alphas_cumprod)
    x_noisy = q_sample(x_start, time_steps, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumpord)

    return x_noisy


def p_losses(
    denoise_model: nn.Module,
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
):
    if noise is None:
        noise = torch.rand_like(x_start)

    betas = linear_beta_schedule(time_steps=t.item()).to(device=x_start.device)

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumpord = torch.sqrt(1.0 - alphas_cumprod)
    x_noisy = q_sample(
        x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumpord, noise=noise
    )
    predicted_noise = denoise_model(x_noisy, t)  # モデルでノイズを予測

    loss = F.mse_loss(noise, predicted_noise)  # Hugging face では smooth l1 らしい

    return loss


@torch.no_grad()
def p_sample(model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index):
    betas = linear_beta_schedule(time_steps=t.item()).to(device=x.device)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumpord = torch.sqrt(1.0 - alphas_cumprod)
    # beta_t
    betas_t = extract(betas, t, x.shape)
    # 1 - √\bar{α}_t
    sqrt_one_minus_alphas_cumpord_t = extract(sqrt_one_minus_alphas_cumpord, t, x.shape)
    # 1 / √α_t
    sqrt_recip_aplhas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_aplhas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumpord_t
    )

    if t_index == 0:
        return model_mean
    else:
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        posterior_variance_t = extract(posterior_variance)  # σ^2_tを計算
        noise = torch.rand_like(x)

    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model: nn.Module, shape, time_steps):
    device = next(model.parameters()).device

    batch_size = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, time_steps)), total=time_steps):
        img = p_sample(
            model, img, torch.full((1,), i, device=device, dtype=torch.long), i
        )
        imgs.append(img.cpu().numpy())

    return imgs


@torch.no_grad()
def sample(model: nn.Module, time_steps, image_size, batch_size=16, channels=3):
    return p_sample_loop(
        model,
        shape=(batch_size, channels, image_size, image_size),
        time_steps=time_steps,
    )
