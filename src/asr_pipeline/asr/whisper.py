from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from asr_pipeline.audio.resample_audio
from asr_pipeline.asr.base import ASRModel
import torch

class Whisper(ASRModel):
    def __init__(self, model_name="openai/whisper-large-v3", device : str | None = None):
        super().__init__(device)
        self.model=None
        self.model_name=model_name

    def load(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRHypothesis:
        target_rate=16000

        audio = resample_audio(audio, sample_rate, target_rate)

        # design decision: I decided to chunk the audio due to context window issues
        chunk_length = 30 * target_rate

        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

        full_transcript = ""

        for chunk in chunks:
            features = self.processor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device).to(self.model.dtype)

            with torch.no_grad():
                tokens = self.model.generate(
                    features,
                    return_dict_in_generate=True,
                    output_scores=True,
                    language="en",
                )

            decoded = self.processor.batch_decode(
                tokens.sequences, skip_special_tokens=True
            )[0]
            full_transcript += " " + decoded

        return ASRHypothesis(
            model=self.model_name,
            text=full_transcript.strip()
        )
