from types import SimpleNamespace

from asr_pipeline.evaluation.meaning import (
    parse_severity,
    score_meaning_alteration,
)


class FakeClient:
    def chat(self, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="Severity: 2"
            )
        )


def test_parse_severity_standard_format():
    assert parse_severity("Severity: 3") == 3


def test_parse_severity_fallback():
    assert parse_severity("I would rate this as 2.") == 2


def test_parse_severity_rejects_invalid_score():
    assert parse_severity("Severity: 7") is None


def test_parse_severity_none():
    assert parse_severity(None) is None


def test_score_meaning_alteration():
    client = FakeClient()

    score = score_meaning_alteration(
        client=client,
        model_name="phi4:14b",
        reference="The dog is in the garden.",
        hypothesis="The dog is outside.",
    )

    assert score == 2

class BadClient:
    def chat(self, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="I don't know"
            )
        )


def test_score_meaning_alteration_returns_none_when_unparseable():
    client = BadClient()

    score = score_meaning_alteration(
        client=client,
        model_name="phi4:14b",
        reference="hello world",
        hypothesis="hello there",
        retries=0,
    )

    assert score is None