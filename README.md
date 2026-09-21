# asr-fusion-eval-pipeline

A production-focused implementation of the ASR ensemble and evaluation pipeline developed during my MPhil research.

The repo packages the research pipeline into a modular, testable system with:

1. **Multi-model ASR inference** using Qwen3-ASR, Whisper, and Parakeet.
2. **LLM-based unanchored fusion** to combine ASR hypotheses into a single transcript.
3. **Sentence-level confidence scoring** using local evidence aligned from the source ASR outputs.
4. **Meaning-alteration evaluation** when a reference transcript is available, alongside WER.
5. **FastAPI and Streamlit interfaces** for running the full pipeline on uploaded audio.
6. **Automated tests** covering the main pipeline components and end-to-end orchestration.

## Pipeline

```text
Audio
  ↓
Multiple ASR models
  ↓
ASR hypotheses
  ↓
LLM unanchored fusion
  ↓
Fused transcript
  ↓
Sentence segmentation
  ↓
Source-span alignment
  ↓
Sentence confidence
  ↓
Optional evaluation
    ├── WER
    └── Meaning-alteration severity
```

## Pipeline diagrams

### Version 1 — motivations

This diagram shows the initial pipeline alongside the questions that motivated turning the research implementation into a reusable and testable system.

<p align="center">
  <img src="docs/pipeline_diagram_1.jpg" alt="Pipeline diagram" width="900">
</p>

### Version 2 — separation of concerns

This version shows a clearer separation of the main components of the system.

<p align="center">
  <img src="docs/pipeline_diagram_2.jpg" alt="Pipeline diagram" width="900">
</p>

## Design decisions

- Long-form audio is chunked where required by individual ASR models.
- Models are loaded once and reused across requests rather than reloaded for every inference call.
- The confidence pipeline first segments the fused transcript, then aligns each segment to local spans from the source ASR hypotheses before scoring.
- Evaluation is optional and only runs when a reference transcript is supplied.
- The core pipeline is kept separate from the API and interface layers.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The pipeline uses Ollama for local LLM inference. Ensure the required models are available locally before starting the API.

The current configuration uses:

```text
gemma4:12b
phi4:14b
```

You can check installed Ollama models with:

```bash
ollama list
```

## Run the API

From the repository root:

```bash
uvicorn asr_pipeline.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Example request

To transcribe an audio file:

```bash
curl -X POST \
  "http://127.0.0.1:8000/transcribe" \
  -F "file=@path/to/audio.m4a"
```

A reference transcript can optionally be supplied for evaluation:

```bash
curl -X POST \
  "http://127.0.0.1:8000/transcribe" \
  -F "file=@path/to/audio.m4a" \
  -F "reference=Reference transcript goes here."
```

The response includes:

- the fused transcript
- sentence-level confidence scores
- WER, when a reference is provided
- meaning-alteration severity, when a reference is provided

## Run the Streamlit interface

Start the FastAPI service first.

Then, in a separate terminal:

```bash
streamlit run app/streamlit_app.py
```

The interface supports:

- audio upload
- optional reference transcript input
- fused transcript output
- sentence-level confidence scores
- evaluation results

## Tests

Run the test suite with:

```bash
pytest -v
```

The tests cover the main pipeline components, including:

- audio handling
- fusion
- segmentation and post-processing
- source-span alignment
- confidence scoring
- meaning-alteration evaluation
- end-to-end pipeline orchestration

## Next steps

The current implementation prioritises correctness, modularity, and fidelity to the research pipeline rather than inference efficiency.

The next phase of the project will focus on profiling and reducing end-to-end latency while monitoring output quality.

Areas I plan to explore include:

- profiling end-to-end and component-level latency
- parallelism and more efficient inference
- model optimisation techniques such as quantization
