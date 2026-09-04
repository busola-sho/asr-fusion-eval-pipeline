from abc import ABC, abstractmethod
from asr_pipeline.schemas import ASRHypothesis
from asr_pipeline.utils.device import resolve_device

import numpy as np

class ASRModel(ABC):
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = resolve_device()

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRHypothesis:
        ...

