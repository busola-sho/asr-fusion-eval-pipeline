from asr_pipeline.schemas import ASRHypothesis
from asr_pipeline.confidence.align_segments_hybrid import (
    align_segments_hybrid,
)

def align_source_hypotheses(
    client,
    hypotheses: list[ASRHypothesis],
    segments: list[str],
    alignment_model: str,
    max_workers: int = 4,
) -> tuple[
    dict[str, list[str | None]],
    dict[str, list[str]],
]:
    """
    Align each fused segment to a corresponding local span
    in every source ASR hypothesis.

    Returns:
        aligned_spans:
            model -> aligned span per segment

        alignment_methods:
            model -> method used per segment
    """

    aligned_spans = {}
    alignment_methods = {}

    for hypothesis in hypotheses:
        spans, methods = align_segments_hybrid(
            client=client,
            model_name=alignment_model,
            transcript=hypothesis.text,
            segments=segments,
            max_workers=max_workers,
        )

        aligned_spans[hypothesis.model] = spans
        alignment_methods[hypothesis.model] = methods

    return aligned_spans, alignment_methods