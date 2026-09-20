from asr_pipeline.confidence.postprocess import apply_postprocessing
from asr_pipeline.confidence.segment import segment_transcript
from types import SimpleNamespace

from asr_pipeline.confidence.scorer import (
    parse_score,
    score_segment,
    score_segments,
)

from asr_pipeline.schemas import ASRHypothesis
from asr_pipeline.confidence.align import align_source_hypotheses

from asr_pipeline.confidence.align_segments_hybrid import (
    align_segments_hybrid,
)


class FakeSaT:
    def split(self, text):
        return ["Hello world. ", "How are you?"]


def test_segment_transcript(monkeypatch):
    sat = FakeSaT()

    monkeypatch.setattr(
        "asr_pipeline.confidence.segment.apply_postprocessing",
        lambda x: (x, [], []), # means return x unchanged and 2 empty lists as the postprocessing for when segment_transcript is called
    )

    result = segment_transcript(
        "Hello world. How are you?",
        sat,
    )

    assert result == [
        "Hello world.",
        "How are you?",
    ]

def test_postprocessing_rule_a_splits_merged_sentences():
    raw_segments = [
        "Hello there. How are you? "
    ]

    result, split_events, merge_events = apply_postprocessing(
        raw_segments
    )

    assert result == [
        "Hello there. ",
        "How are you? ",
    ]

    assert len(split_events) == 1
    assert len(merge_events) == 0


def test_postprocessing_does_not_split_abbreviation():
    raw_segments = [
        "Dr. Smith went home. "
    ]

    result, split_events, _ = apply_postprocessing(
        raw_segments
    )

    assert result == [
        "Dr. Smith went home. "
    ]

    assert split_events == []


def test_postprocessing_rule_b_merges_reporting_quote():
    raw_segments = [
        'He said, ',
        '"Come here." ',
    ]

    result, _, merge_events = apply_postprocessing(
        raw_segments
    )

    assert result == [
        'He said, "Come here." '
    ]

    assert len(merge_events) == 1

def test_parse_score_valid():
    result = parse_score("Score: 87")

    assert result == 87.0


def test_parse_score_decimal():
    result = parse_score("Score: 92.5")

    assert result == 92.5


def test_parse_score_out_of_range():
    result = parse_score("Score: 120")

    assert result is None


def test_parse_score_invalid_response():
    result = parse_score("I think this looks good.")

    assert result is None


def test_parse_score_none():
    result = parse_score(None)

    assert result is None

class FakeClient:
    def chat(self, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="Score: 91"
            )
        )


def test_score_segment_returns_score():
    client = FakeClient()

    local_spans = {
        "qwen": "It's so good.",
        "whisper": "It is so good.",
        "parakeet": "It's so good.",
    }

    result = score_segment(
        client=client,
        model_name="gemma4",
        segment="It's so good.",
        local_spans=local_spans,
    )

    assert result == 91.0

def test_score_segments_scores_each_segment(
    monkeypatch,
):
    segments = [
        "First sentence.",
        "Second sentence.",
    ]

    aligned_spans = {
        "qwen": [
            "First sentence.",
            "Second sentence.",
        ],
        "whisper": [
            "First sentence",
            "Second sentance.",
        ],
    }

    calls = []

    def fake_score_segment(
        client,
        model_name,
        segment,
        local_spans,
    ):
        calls.append(
            (segment, local_spans)
        )

        return 90.0 if segment == "First sentence." else 70.0

    monkeypatch.setattr(
        "asr_pipeline.confidence.scorer.score_segment",
        fake_score_segment,
    )

    scores = score_segments(
        client=None,
        model_name="gemma4",
        segments=segments,
        aligned_spans=aligned_spans,
    )

    assert scores == [90.0, 70.0]

    assert calls[0][1] == {
        "qwen": "First sentence.",
        "whisper": "First sentence",
    }

    assert calls[1][1] == {
        "qwen": "Second sentence.",
        "whisper": "Second sentance.",
    }

from asr_pipeline.schemas import ASRHypothesis
from asr_pipeline.confidence.align import align_source_hypotheses


def test_align_source_hypotheses(monkeypatch):
    hypotheses = [
        ASRHypothesis(
            model="qwen",
            text="Hello world. How are you?",
        ),
        ASRHypothesis(
            model="whisper",
            text="Hello word. How are you?",
        ),
    ]

    segments = [
        "Hello world.",
        "How are you?",
    ]

    def fake_align(
        client,
        model_name,
        transcript,
        segments,
        max_workers=4,
    ):
        if transcript == "Hello world. How are you?":
            return (
                [
                    "Hello world.",
                    "How are you?",
                ],
                [
                    "mechanical",
                    "mechanical",
                ],
            )

        return (
            [
                "Hello word.",
                "How are you?",
            ],
            [
                "mechanical",
                "mechanical",
            ],
        )

    monkeypatch.setattr(
        "asr_pipeline.confidence.align.align_segments_hybrid",
        fake_align,
    )

    spans, methods = align_source_hypotheses(
        client=None,
        hypotheses=hypotheses,
        segments=segments,
        alignment_model="phi4:14b",
    )

    assert spans["qwen"] == [
        "Hello world.",
        "How are you?",
    ]

    assert spans["whisper"] == [
        "Hello word.",
        "How are you?",
    ]

    assert methods["qwen"] == [
        "mechanical",
        "mechanical",
    ]

def test_hybrid_alignment_mechanical_match():
    transcript = (
        "Hello world. "
        "How are you today?"
    )

    segments = [
        "Hello world.",
        "How are you today?",
    ]

    spans, methods = align_segments_hybrid(
        client=None,
        model_name="phi4:14b",
        transcript=transcript,
        segments=segments,
    )

    assert spans == [
        "Hello world",
        "How are you today",
    ]

    assert methods == [
        "mechanical",
        "mechanical",
    ]

def test_hybrid_alignment_unresolved_when_no_transcript():
    spans, methods = align_segments_hybrid(
        client=None,
        model_name="phi4:14b",
        transcript="",
        segments=["Hello world."],
    )

    assert spans == [None]
    assert methods == ["unresolved"]