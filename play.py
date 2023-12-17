from diffusion.unet import Unet
import torch

image_size = 32
channels = 3
batch_size = 128
time_steps = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    dim=image_size,
    init_dim=64,
    channels=channels,
    dim_multi=(1, 2, 4, 8),
    resnet_block_groups=32,
)
model.to(device)
model.eval()
# ckpt = torch.load('./model_49.bin')
# model.load_state_dict(ckpt)

from diffusion.diffusion import Diffusion
from pathlib import Path
from torchvision.utils import save_image

diffusion = Diffusion(time_steps=time_steps, device=device)
results_folder = Path("./outputs")
results_folder.mkdir(exist_ok=True)
samples = diffusion.sample(model, time_steps, image_size=image_size, batch_size=25, channels=channels)
save_image(torch.from_numpy(samples[-1]), str(results_folder / f'sample_{2}.png'), nrow=5)