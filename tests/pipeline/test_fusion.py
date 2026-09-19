from asr_pipeline.fusion.unanchored_fusion import Unanchored
from asr_pipeline.schemas import ASRHypothesis, FusedTranscript


def test_unanchored_fuse(monkeypatch):

    hypotheses = [
        ASRHypothesis(model="qwen", text="hello world"),
        ASRHypothesis(model="whisper", text="hello word"),
        ASRHypothesis(model="parakeet", text="hello world"),
    ]

    monkeypatch.setattr(
        "asr_pipeline.fusion.unanchored_fusion.initialise_client",
        lambda:"fake-client",
    )

    monkeypatch.setattr(
        "asr_pipeline.fusion.unanchored_fusion.ollama_select",
        lambda *args, **kwargs: "hello world", #essentially it doesn't matter the arguments it accepts just return hello world
    )

    strategy = Unanchored()

    result = strategy.fuse(hypotheses)

    assert isinstance(result, FusedTranscript)
    assert result.text == "hello world"