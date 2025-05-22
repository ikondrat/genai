import torch
from genaibook.core import get_device
from diffusers import StableDiffusionPipeline

device = get_device()

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to(device)

pipe.safety_checker = None
pipe.requires_safety_checker = False

# propmt = "Venera planet with a rabbit"
propmt = "A rabbit in a futuristic city in Venera, digital art, 4k, highly detailed, cinematic lighting"

pipe(propmt).images[0].save("rabit.png")
