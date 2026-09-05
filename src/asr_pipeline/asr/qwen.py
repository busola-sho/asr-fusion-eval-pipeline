from qwen_asr import Qwen3ASRModel
import numpy as np
import torch

from asr_pipeline.asr.base import ASRModel
from asr_pipeline.audio import resample_audio
from asr_pipeline.schemas import ASRHypothesis

class QwenASR(ASRModel):
    def __init__(self, model_name="Qwen/Qwen3-ASR-1.7B", device : str | None = None):
        super().__init__(device)
        self.model=None
        self.model_name=model_name

    def load(self)->None:
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # was float32
        )
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRHypothesis:
        target_rate=16000

        audio = resample_audio(
            audio,
            source_rate=sample_rate,
            target_rate=target_rate,
        )
        # no need for chunking as qwen handles that manually
        result = self.model.transcribe(
            audio=(audio, target_rate),
            language="English",
        )
    
        return ASRHypothesis(
            model=self.model_name,
            text=result[0].text
        )