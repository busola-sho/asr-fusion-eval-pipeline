import torch

def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_dtype(device: str):
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32