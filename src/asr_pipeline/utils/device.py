import torch

def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_compute_type(device):
    if device=="cuda":
        return "float16"
    return "int8"