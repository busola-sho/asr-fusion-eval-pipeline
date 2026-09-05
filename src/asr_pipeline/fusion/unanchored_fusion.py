from asr_pipeline.fusion import FusionStrategy
from asr_pipeline.schemas import ASRHypothesis, FusedTranscript
from asr_pipeline.prompt.prompts import UNANCHORED_FUSION_PROMPT
from asr_pipeline.utils import ollama_select, initialise_client

ASR_MODELS = ["qwen", "whisper", "parakeet"]

class Unanchored(FusionStrategy):
    def __init__(self, model_name: str = "gemma4", num_predict: int = 2048):
        self.model_name = model_name
        self.num_predict = num_predict
        self.client = initialise_client()

    def fuse (self, hypotheses: List(ASRHypothesis)):
        hyp_by_model={
            hyp.model:hyp.text for hyp in hypotheses
        }
        fused_text = ollama_select( 
                                    self.client, 
                                    self.model_name, 
                                    hyp_by_model, 
                                    self.num_predict,
                                    UNANCHORED_FUSION_PROMPT
                                    retries=2
                                    )
        
        return FusedTranscript(text=fused_text)