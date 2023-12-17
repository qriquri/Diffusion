from diffusion.unet import Unet
import torch

image_size = 28
channels = 1
batch_size = 128
time_steps = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    dim=image_size,
    channels=channels,
    dim_multi=(1, 2, 4,),
    resnet_block_groups=4,
)
model.to(device)

ckpt = torch.load('./model_9.bin')
model.load_state_dict(ckpt)

from diffusion.diffusion import Diffusion
from pathlib import Path
from torchvision.utils import save_image

diffusion = Diffusion(time_steps=time_steps, device=device)
results_folder = Path("./outputs")
results_folder.mkdir(exist_ok=True)
samples = diffusion.sample(model, time_steps, image_size=image_size, batch_size=25, channels=channels)
save_image(torch.from_numpy(samples[-1]), str(results_folder / f'sample_{0}.png'), nrow=5)