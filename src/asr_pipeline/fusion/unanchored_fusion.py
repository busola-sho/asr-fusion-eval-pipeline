from asr_pipeline.fusion import FusionStrategy
from asr_pipeline.schemas import ASRHypothesis, FusedTranscript
from asr_pipeline.prompt import UNANCHORED_FUSION_PROMPT
from asr_pipeline.utils import ollama_select, initialise_client

ASR_MODELS = ["qwen", "whisper", "parakeet"]
MODEL_NAME = "gemma4"

class Unanchored(FusionStrategy):
    def __init__(self):
        self.fused_text=""

    def fuse(self, hypotheses: List(ASRHypothesis)):
        
        hyp_by_model={hyp.model:hyp.text for hyp in hypotheses}
        client=initialise_client()
        self.fused_text = ollama_select(client, MODEL_NAME, hyp_by_model, num_predict, retries=2)
        
        return FusedTranscript(
            text: self.fused_text
        )