from abc import ABC, abstractmethod

from asr_pipeline.schemas import FusedTranscript, ASRHypothesis

class FusionStrategy(ABC):

    @abstractmethod
    def fuse(
        self,
        hypotheses: List[ASRHypothesis],
    ) -> FusedTranscript:
    ...

