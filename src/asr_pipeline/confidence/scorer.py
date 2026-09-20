import re
from ollama import Client

NO_MATCH_MARKER = "[NO MATCH]"

CONFIDENCE_PROMPT = """
The following are corresponding local excerpts from ASR
transcripts of the same spoken audio. Some models may have
no comparable content, marked as {no_match_marker}.

{source_block}

The fused transcript contains this segment:

"{segment}"

Based only on the evidence above, give a confidence score
from 0 to 100 quantifying how confident you are that this
segment preserves the spoken meaning.

Respond only:
Score: <value>
"""

_SCORE_PATTERN = re.compile(
    r"\*{0,2}score\*{0,2}:\*{0,2}\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

def format_source_block(
    local_spans: dict[str, str | None],
) -> str:
    lines = []

    for model, span in local_spans.items():
        text = span if span is not None else NO_MATCH_MARKER
        lines.append(f"Transcript ({model}): {text}")

    return "\n".join(lines)

def parse_score(
    raw_response: str | None,
) -> float | None:

    if raw_response is None:
        return None

    match = _SCORE_PATTERN.search(raw_response)

    if not match:
        return None

    score = float(match.group(1))

    if 0 <= score <= 100:
        return score

    return None

def score_segment(
    client,
    model_name: str,
    segment: str,
    local_spans: dict[str, str | None],
    retries: int = 2,
) -> float | None:

    prompt = CONFIDENCE_PROMPT.format(
        no_match_marker=NO_MATCH_MARKER,
        source_block=format_source_block(local_spans),
        segment=segment,
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": 20,
                },
                think=False,
            )

            return parse_score(
                response.message.content
            )

        except Exception:
            if attempt == retries:
                return None

def score_segments(
    client: Client,
    model_name: str,
    segments: list[str],
    aligned_spans: dict[str, list[str | None]],
) -> list[float | None]:
    scores=[]
    for i in range(len(segments)):
        local_spans={}
        for model,spans in aligned_spans.items():
            local_spans[model]=spans[i]
        segment=segments[i]
        score=score_segment(client, model_name, segment, local_spans)
        scores.append(score)
    return scores
