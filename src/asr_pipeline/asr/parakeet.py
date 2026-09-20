from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCTC
import torch 
from asr_pipeline.asr.base import ASRModel
from asr_pipeline.audio import resample_audio

class Parakeet(ASRModel):
    def __init__(self, model_name="nvidia/parakeet-ctc-1.1b",  device : str | None = None):
        super().__init__(device)
        self.model=None
        self.model_name=model_name
        self.alias="parakeet"

    def load(self):
        self.model = AutoModelForCTC.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRHypothesis:
        target_rate=16000
        audio = resample_audio(audio, sample_rate, target_rate)

        chunk_length = 30 * target_rate

        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

        full_transcript = ""

        for chunk in chunks:
            inputs = self.processor(chunk, sampling_rate=target_rate, return_tensors="pt")
            inputs = {k: v.to(self.device).to(self.model.dtype) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcript += " " + transcript

        return ASRHypothesis(
            model=self.alias,
            text=full_transcript.strip()
        )