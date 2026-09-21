import io

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form
from ollama import Client
from wtpsplit import SaT

from asr_pipeline.pipeline import ASRFusionPipeline
from asr_pipeline.asr.qwen import QwenASR
from asr_pipeline.asr.parakeet import Parakeet
from asr_pipeline.asr.whisper import Whisper
from asr_pipeline.fusion import Unanchored
from asr_pipeline.evaluation.evaluator import Evaluator
from pydub import AudioSegment
import io
import numpy as np


def load_audio(contents: bytes, filename: str):
    suffix = filename.rsplit(".", 1)[-1].lower()

    audio_segment = AudioSegment.from_file(
        io.BytesIO(contents),
        format=suffix,
    )

    # Force mono
    audio_segment = audio_segment.set_channels(1)

    sample_rate = audio_segment.frame_rate

    samples = np.array(
        audio_segment.get_array_of_samples()
    ).astype(np.float32)

    # Convert integer PCM samples to roughly [-1, 1]
    max_value = float(1 << (8 * audio_segment.sample_width - 1))
    samples = samples / max_value

    return samples, sample_rate

app = FastAPI(
    title="ASR Fusion Pipeline",
    version="0.1.0",
)

client = Client(host="http://localhost:11434")

sat = SaT("sat-3l")

evaluator = Evaluator(
    client=client,
    model_name="phi4:14b",
)

pipeline = ASRFusionPipeline(
    models=[
        QwenASR(),
        Parakeet(),
        Whisper(),
    ],
    fusion_strategy=Unanchored(),
    evaluator=evaluator
)

# Load expensive ASR models once
pipeline.load_models()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), reference: str | None = Form(None)):
    contents = await file.read()

    audio, sample_rate = load_audio(
        contents,
        file.filename,
    )

    fused_transcript, segments, scores, evaluation = pipeline.run(
        audio=audio,
        sample_rate=sample_rate,
        client=client,
        sat=sat,
        reference=reference,
    )

    response = {
        "transcript": fused_transcript.text,
        "sentence_confidence": [
            {
                "sentence": segment,
                "confidence": score,
            }
            for segment, score in zip(segments, scores)
        ],
    }

    if evaluation is not None:
        response["evaluation"] = {
            "wer": evaluation.wer,
            "severity_score": evaluation.severity_score
        }

    return response