from typing import Optional
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion:
    def __init__(self, time_steps, device="cuda") -> None:
        self.setup(time_steps, device)

    def setup(self, time_steps, device="cuda"):
        self.time_steps = time_steps
        self.betas = self.linear_beta_schedule(time_steps).to(device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumpord = torch.sqrt(1.0 - self.alphas_cumprod)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 標準偏差

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t - 1)  # aの最後の次元 ⇒ timestepに対応するalphaを取ってくる
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(
            t.device
        )  # バッチサイズ x 1 x 1 x 1にreshape

    def linear_beta_schedule(self, time_steps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, time_steps)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t,
        noise: Optional[torch.Tensor] = None,
    ):
        """_summary_
        きれいな画像からノイズを加えた画像をサンプルする
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumpord_t = self.extract(
            self.sqrt_one_minus_alphas_cumpord, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumpord_t * noise

    def compute_noisy_image(self, x_start: torch.Tensor, t: torch.Tensor, time_steps):
        x_noisy = self.q_sample(
            x_start, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumpord
        )

        return x_noisy

    def p_losses(
        self,
        denoise_model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(
            x_start, t, noise=noise
        )
        predicted_noise = denoise_model(x_noisy, t)  # モデルでノイズを予測

        loss = F.mse_loss(noise, predicted_noise)  # Hugging face では smooth l1 らしい

        return loss

    @torch.no_grad()
    def p_sample(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index
    ):
        # beta_t
        betas_t = self.extract(self.betas, t, x.shape)
        # 1 - √\bar{α}_t
        sqrt_one_minus_alphas_cumpord_t = self.extract(
            self.sqrt_one_minus_alphas_cumpord, t, x.shape
        )
        # 1 / √α_t
        sqrt_recip_aplhas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_aplhas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumpord_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)  # σ^2_tを計算
            noise = torch.rand_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape):
        device = next(model.parameters()).device

        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(1, self.time_steps)), total=self.time_steps):
            img = self.p_sample(
                model,
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i
            )
            imgs.append(img.cpu().numpy())

        return imgs

    @torch.no_grad()
    def sample(
        self, model: nn.Module, time_steps, image_size, batch_size=16, channels=3
    ):
        device = next(model.parameters()).device
        self.setup(time_steps, device=device)

        return self.p_sample_loop(
            model,
            shape=(batch_size, channels, image_size, image_size),
        )
