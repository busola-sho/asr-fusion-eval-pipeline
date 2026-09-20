from asr_pipeline.schemas import FusedTranscript
from asr_pipeline.confidence.segment import segment_transcript
from asr_pipeline.confidence.scorer import score_segments
from asr_pipeline.confidence.align import align_source_hypotheses

class ASRFusionPipeline():
    def __init__(self, models, fusion_strategy, sentence_confidence=None, evaluator=None):
        self.asr_models=models
        self.fusion_strategy=fusion_strategy
        self.sentence_confidence = sentence_confidence
        self.evaluator = evaluator
    
    def load_models(self):
        for model in self.asr_models:
            model.load()

    def run(
    self,
    audio,
    sample_rate,
    client,
    sat,
    reference=None,
    confidence_model="gemma4:12b",
    alignment_model="phi4:14b",
    ):
        hypotheses = [
            model.transcribe(audio, sample_rate)
            for model in self.asr_models
        ]

        fused_transcript = self.fusion_strategy.fuse(hypotheses)

        segments = segment_transcript(
            fused_transcript.text,
            sat,
        )

        aligned_spans, alignment_methods = align_source_hypotheses(
            client=client,
            hypotheses=hypotheses,
            segments=segments,
            alignment_model=alignment_model,
        )

        scores = score_segments(
            client=client,
            model_name=confidence_model,
            segments=segments,
            aligned_spans=aligned_spans,
        )

        evaluation = None

        if reference is not None and self.evaluator is not None:
            evaluation = self.evaluator.evaluate(
                prediction=fused_transcript.text,
                reference=reference,
            )

        return fused_transcript, segments, scores, evaluation