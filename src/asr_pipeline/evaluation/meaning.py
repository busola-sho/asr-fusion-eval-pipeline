import re
import time

from ollama import Client

from asr_pipeline.prompt.severity import DIRECT_SEVERITY_PROMPT


def parse_severity(text: str | None) -> int | None:
    if text is None:
        return None

    match = re.search(
        r"severity\s*:\s*([0-4])",
        text,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    fallback = re.findall(
        r"(?<!\d)[0-4](?!\d)",
        text,
    )

    return int(fallback[-1]) if fallback else None


def score_meaning_alteration(
    client: Client,
    model_name: str,
    reference: str,
    hypothesis: str,
    retries: int = 2,
) -> int | None:

    prompt = DIRECT_SEVERITY_PROMPT.format(
        reference=reference,
        hypothesis=hypothesis,
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
                options={"temperature": 0},
                think=False,
            )

            raw_response = response.message.content

            severity = parse_severity(raw_response)

            if severity is not None:
                return severity

        except Exception:
            pass

        if attempt < retries:
            time.sleep(0.5)

    return None