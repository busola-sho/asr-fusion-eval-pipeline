from ollama import Client
from wtpsplit import SaT

from asr_pipeline.pipeline import ASRFusionPipeline
from asr_pipeline.fusion import Unanchored
from asr_pipeline.asr.qwen import QwenASR
from asr_pipeline.asr.parakeet import Parakeet
from asr_pipeline.asr.whisper import Whisper

def main():
    audio=...
    sample_rate=16000

    client = Client(host="http://localhost:11434")
    sat = SaT("sat-3l")
    
    pipeline=ASRFusionPipeline(
        models=[
        QwenASR(),
        Parakeet(),
        Whisper()
        ], 
        fusion_strategy=Unanchored()
        )

    pipeline.load_models()
    fused_transcript, segments, scores = pipeline.run(
        audio=audio,
        sample_rate=sample_rate,
        client=client,
        sat=sat,
        confidence_model="gemma4",
        alignment_model="phi4:14b",
    )

    print(fused_transcript.text)

    for segment, score in zip(segments, scores):
        print(segment, score)


    print(result)

if __name__=="__main__":
    main()
