from dataclasses import dataclass
from typing import Optional

# what the data is meant to look like for contracts

@dataclass
class ASRHypothesis:
    model:str
    text:str

@dataclass
class FusedTranscript:
    text:str

@dataclass
class SentenceConfidence:
    sentence: str
    confidence: float

@dataclass
class EvalResult:
    wer: Optional[float] = None
    severity_score: Optional[int] = None

@dataclass
class PipelineOutput:
    hypotheses: List[ASRHypothesis]
    fused_transcript: FusedTranscript
    sentence_confidence: SentenceConfidence
    evaluation: Optional[EvalResult]=None

