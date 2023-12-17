from datasets import load_dataset # hugging faceのdatasetを使う
import torch
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
 
from torch.optim import Adam
from torchinfo import summary
from pathlib import Path
from diffusion.unet import Unet
from diffusion.functional import p_losses, sample, compute_noisy_image
from tqdm.auto import tqdm

dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128
time_steps = 200

transform = Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

def transforms_data(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples

transformed_dataset = dataset.with_transform(transforms_data).remove_columns("label")
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, pin_memory=True, num_workers=2, shuffle=True)

results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    dim=image_size,
    channels=channels,
    dim_multi=(1, 2, 4,),
    resnet_block_groups=4,
)
summary(model)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in tqdm(range(epochs)):
    data_tq = tqdm(dataloader)
    for step, batch in enumerate(data_tq):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        t = torch.randint(1, time_steps, (batch_size,), device=device).long()
        batch_noisy = compute_noisy_image(batch, t, time_steps=time_steps).detach()
        
        loss = p_losses(model, batch_noisy, t, time_steps=time_steps)

        loss.backward()
        optimizer.step()

        data_tq.set_description(f'loss={loss.item()}')

    torch.save(model.state_dict(), f"./model_{epoch}.bin")
    samples = sample(model, time_steps, image_size=image_size, batch_size=25, channels=channels)
    save_image(batch_noisy, str(results_folder / f'sample_{epoch}.png'), nrow=5)
