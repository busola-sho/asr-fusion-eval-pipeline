from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import (
    normalize_audios,
    split_audio_into_chunks,
    SAMPLE_RATE,
    MAX_ASR_INPUT_SECONDS,
    normalize_language_name,
)
import numpy as np
import torch

from asr_pipeline.asr.base import ASRModel
from asr_pipeline.audio import resample_audio
from asr_pipeline.schemas import ASRHypothesis

class QwenASR(ASRModel):
    def __init__(self, model_name="Qwen/Qwen3-ASR-1.7B"):
        super().__init__()
        self.model=None
        self.model_name=model_name

    def load(self)->None:
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # was float32
        )
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRHypothesis:
        target_rate=16000
        audio = resample_audio(audio, sample_rate, target_rate)

        wavs  = normalize_audios((audio, target_rate))
        lang  = normalize_language_name("English")
        text_prompt = qwen._build_text_prompt(context=context, force_language=lang)

        full_text = ""

        parts = split_audio_into_chunks(
            wav=wavs[0], sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS
        )

        for chunk_wav, _ in parts:
            inputs = qwen.processor(
                text=[text_prompt], audio=[chunk_wav],
                return_tensors="pt", padding=True,
            )
            inputs = inputs.to(qwen.model.device).to(qwen.model.dtype)
            prompt_len = inputs["input_ids"].shape[1]

            with _torch.no_grad():
                gen_out = qwen.model.generate(
                    **inputs,
                    max_new_tokens=qwen.max_new_tokens,
                    output_scores=True,
                )

            generated_ids = gen_out.sequences[0, prompt_len:]
            chunk_text = qwen.processor.batch_decode(
                [generated_ids], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            full_text += chunk_text

        return ASRHypothesis(
            model=self.model_name,
            text=full_text.strip()
        )