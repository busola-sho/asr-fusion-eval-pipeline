from asr_pipeline.confidence.postprocess import apply_postprocessing

def segment_transcript(
    text: str,
    sat,
) -> list[str]:

    raw_segments = sat.split(text)

    segments, _, _ = apply_postprocessing(raw_segments)

    return [
        segment.strip()
        for segment in segments
        if segment.strip()
    ]