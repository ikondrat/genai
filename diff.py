import torch
from genaibook.core import get_device
from diffusers import StableDiffusionPipeline

device = get_device()

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to(device)

propmt = "a cat sissting by the door"

pipe(propmt).images[0].save("cat_sitting.png")
