from asr_pipeline.pipeline import ASRFusionPipeline
from asr_pipeline.fusion import Unanchored

def main():
    audio=...
    sample_rate=...
    
    pipeline=ASRFusionPipeline(
        models=[
        QwenASR(),
        Parakeet(),
        Whisper()
        ], 
        fusion_strategy=Unanchored()
    )

    pipeline.load()
    pipeline.run(audio,sample_rate)
    print(result)

if __name__=="main":
    main()
