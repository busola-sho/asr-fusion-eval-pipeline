from asr_pipeline.schemas import FusedTranscript
from asr_pipeline.confidence.segment import segment_transcript
from asr_pipeline.confidence.scorer import score_segments

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
        confidence_model="gemma4",
        alignment_model="phi4:14b",
    ):

        # 1. Transcribe
        hypotheses = [
            model.transcribe(audio, sample_rate)
            for model in self.asr_models
        ]

        print("Hypotheses generated. Fusion next...")

        # 2. Fuse
        fused_transcript = self.fusion_strategy.fuse(
            hypotheses
        )

        print("Fusion done. Commencing sentence confidence generation...")

        # 3. Segment fused transcript
        segments = segment_transcript(
            fused_transcript.text,
            sat,
        )

        # 4. Find corresponding source-ASR evidence
        aligned_spans, alignment_methods = (
            align_source_hypotheses(
                client=client,
                hypotheses=hypotheses,
                segments=segments,
                alignment_model=alignment_model,
            )
        )

        # 5. Score each fused segment
        scores = score_segments(
            client=client,
            model_name=confidence_model,
            segments=segments,
            aligned_spans=aligned_spans,
        )

        return fused_transcript, segments, scores





    
