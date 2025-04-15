import torch
from transformers import set_seed, pipeline


def get_device():
    """Determine the appropriate device (CUDA/MPS/CPU) for running models."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # For Apple Silicon Macs
    else:
        return "cpu"


set_seed(10)

device = get_device()
generator = pipeline("text-generation", device=device)
prompt = "Once upon a time, in a land far, far away"
result = generator(prompt)[0]["generated_text"]
print(result)  # Print the result to see the output
