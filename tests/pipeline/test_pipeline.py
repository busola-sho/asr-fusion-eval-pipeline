from asr_pipeline.schemas import ASRHypothesis, FusedTranscript
from asr_pipeline.pipeline import ASRFusionPipeline
import numpy as np

class FakeASR:
    def __init__(self, model, fake_transcript):
        self.model_name=model
        self.transcript=fake_transcript
    
    def load(self):
        pass
    
    def transcribe(self, audio, sample_rate):
        return ASRHypothesis(
            model=self.model_name,
            text=self.transcript
        )

class FakeFusion:
    def __init__(self):
        pass
    def fuse(self, hypotheses):
        return FusedTranscript(text="hello world")

class FakeSaT:
    def split(self, text):
        return [text]

def test_pipeline_works_well(monkeypatch):
    models=[
        FakeASR("qwen", "hello world"),
        FakeASR("parakeet", "hello world"),
    ]
    fusion=FakeFusion()
    audio=np.zeros(8000)

    monkeypatch.setattr(
    "asr_pipeline.pipeline.align_source_hypotheses",
    lambda **kwargs: (
        {
            "qwen": ["hello world"],
            "parakeet": ["hello world"],
        },
        {
            "qwen": ["mechanical"],
            "parakeet": ["mechanical"],
        },
    ),
    )

    monkeypatch.setattr(
        "asr_pipeline.pipeline.score_segments",
        lambda **kwargs: [95.0],
    )

    pipeline=ASRFusionPipeline(models, fusion)
    pipeline.load_models()
    
    output, segments, scores = pipeline.run(
        audio=audio,
        sample_rate=16000,
        client=None,
        sat=FakeSaT(),
    )

    assert isinstance(output, FusedTranscript)
    assert output.text == "hello world"
    assert segments == ["hello world"]
    assert scores == [95.0]

    



