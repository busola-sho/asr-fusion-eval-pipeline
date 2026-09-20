import io

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from ollama import Client
from wtpsplit import SaT

from asr_pipeline.pipeline import ASRFusionPipeline
from asr_pipeline.asr.qwen import QwenASR
from asr_pipeline.asr.parakeet import Parakeet
from asr_pipeline.asr.whisper import Whisper
from asr_pipeline.fusion import Unanchored


app = FastAPI(
    title="ASR Fusion Pipeline",
    version="0.1.0",
)

client = Client(host="http://localhost:11434")

sat = SaT("sat-3l")

pipeline = ASRFusionPipeline(
    models=[
        QwenASR(),
        Parakeet(),
        Whisper(),
    ],
    fusion_strategy=Unanchored(),
)


# Load expensive ASR models once
pipeline.load_models()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    contents = await file.read()

    audio, sample_rate = sf.read(
        io.BytesIO(contents),
        dtype="float32",
    )

    # Stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    fused_transcript, segments, scores, evaluation = pipeline.run(
        audio=audio,
        sample_rate=sample_rate,
        client=client,
        sat=sat,
    )

    return {
        "transcript": fused_transcript.text,
        "sentence_confidence": [
            {
                "sentence": segment,
                "confidence": score,
            }
            for segment, score in zip(segments, scores)
        ],
    }
