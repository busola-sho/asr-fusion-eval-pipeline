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

def test_pipeline_works_well():
    models=[
        FakeASR("qwen", "hello world"),
        FakeASR("parakeet", "hello world"),
    ]
    fusion=FakeFusion()
    audio=np.zeros(8000)
    pipeline=ASRFusionPipeline(
        models=models,
        fusion_strategy=fusion
    )
    pipeline.load_models()
    output=pipeline.run(audio, 16000)

    assert isinstance(output, FusedTranscript)
    assert output.text=="hello world"

    



