import difflib
import re

MECHANICAL_THRESHOLD = 0.55
WIDE_FUZZY_THRESHOLD = 0.90

MAX_EXTRA = 15

LOCAL_WINDOW_WORDS = 150
WIDE_WINDOW_WORDS = 500

MAX_SPAN_LENGTH_RATIO = 3.0

LLM_FALLBACK_PROMPT = """
You are given a short excerpt from one transcript and ONE
sentence-like segment from another transcript of the same spoken audio.

Identify the MINIMAL span of text in the excerpt that corresponds
to the segment.

Normally this should be one sentence-like span matching the
segment's length and content.

Do NOT include surrounding content.

Transcript excerpt:
{excerpt}

Target segment:
{segment}

Respond ONLY with the corresponding span copied verbatim from
the excerpt.

If there is no clear corresponding content, respond exactly:

NONE
"""


def tokenize_with_offsets(text: str):
    tokens = []
    spans = []

    for match in re.finditer(r"[\w']+", text):
        tokens.append(match.group(0).lower())
        spans.append((match.start(), match.end()))

    return tokens, spans


def segment_tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def segment_word_count(text: str) -> int:
    return len(segment_tokens(text))


def span_text(
    transcript: str,
    token_spans,
    start: int,
    end: int,
) -> str:
    return transcript[
        token_spans[start][0]:
        token_spans[end - 1][1]
    ]


def mechanical_match(
    segment: str,
    transcript_tokens: list[str],
    cursor: int,
):
    anchor = segment_tokens(segment)

    if not anchor:
        return None, 0.0, None

    n = len(anchor)

    min_len = max(1, n - MAX_EXTRA)
    max_len = n + MAX_EXTRA

    search_start = max(0, cursor)

    search_end = min(
        len(transcript_tokens),
        search_start + max(n * 4 + MAX_EXTRA, 60),
    )

    best_start = None
    best_end = None
    best_score = 0.0

    for start in range(search_start, search_end):

        for span_len in range(min_len, max_len + 1):

            end = start + span_len

            if end > len(transcript_tokens):
                continue

            score = difflib.SequenceMatcher(
                None,
                anchor,
                transcript_tokens[start:end],
            ).ratio()

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

        if (
            best_start is not None
            and best_score >= 0.88
            and start > best_start + 5
        ):
            break

    return best_start, best_score, best_end


def global_exact_match(
    segment: str,
    transcript_tokens: list[str],
):
    """
    Find a unique normalized exact match anywhere
    in the source transcript.
    """

    anchor = segment_tokens(segment)

    if not anchor:
        return None

    n = len(anchor)

    matches = []

    for start in range(
        0,
        len(transcript_tokens) - n + 1,
    ):
        if transcript_tokens[start:start + n] == anchor:
            matches.append((start, start + n))

            # ambiguous exact match
            if len(matches) > 1:
                return None

    return matches[0] if len(matches) == 1 else None


def wide_local_fuzzy_match(
    segment: str,
    transcript_tokens: list[str],
    estimated_cursor: int,
):
    """
    Wider fuzzy rescue around the estimated location.
    """

    anchor = segment_tokens(segment)

    if not anchor:
        return None, 0.0, None

    n = len(anchor)

    min_len = max(1, n - MAX_EXTRA)
    max_len = n + MAX_EXTRA

    half_window = WIDE_WINDOW_WORDS // 2

    search_start = max(
        0,
        estimated_cursor - half_window,
    )

    search_end = min(
        len(transcript_tokens),
        estimated_cursor + half_window,
    )

    best_start = None
    best_end = None
    best_score = 0.0

    for start in range(search_start, search_end):

        for span_len in range(min_len, max_len + 1):

            end = start + span_len

            if (
                end > search_end
                or end > len(transcript_tokens)
            ):
                continue

            score = difflib.SequenceMatcher(
                None,
                anchor,
                transcript_tokens[start:end],
            ).ratio()

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    if best_score >= WIDE_FUZZY_THRESHOLD:
        return best_start, best_score, best_end

    return None, best_score, None


def call_llm_fallback(
    client,
    model_name: str,
    segment: str,
    excerpt: str,
) -> str | None:

    prompt = LLM_FALLBACK_PROMPT.format(
        excerpt=excerpt,
        segment=segment,
    )

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
                "num_predict": 256,
            },
            think=False,
        )

    except Exception:
        return None

    output = (response.message.content or "").strip()

    if not output or output.upper() == "NONE":
        return None

    segment_length = segment_word_count(segment)
    output_length = segment_word_count(output)

    if (
        segment_length > 0
        and output_length
        > segment_length * MAX_SPAN_LENGTH_RATIO
    ):
        return None

    return output


def locate_returned_span(
    text: str,
    transcript_tokens: list[str],
):
    """
    Locate the LLM-returned span inside the full
    source transcript so that it can act as an anchor.
    """

    target = segment_tokens(text)

    if not target:
        return None

    n = len(target)

    # exact search first
    for start in range(
        0,
        len(transcript_tokens) - n + 1,
    ):
        if transcript_tokens[start:start + n] == target:
            return start + n

    # small fuzzy rescue
    best_end = None
    best_score = 0.0

    for start in range(
        0,
        len(transcript_tokens) - n + 1,
    ):

        score = difflib.SequenceMatcher(
            None,
            target,
            transcript_tokens[start:start + n],
        ).ratio()

        if score > best_score:
            best_score = score
            best_end = start + n

    if best_score >= 0.85:
        return best_end

    return None


def nearest_preceding_anchor(
    index: int,
    anchors: dict[int, int],
):

    for previous in range(index - 1, -1, -1):

        if previous in anchors:
            return previous, anchors[previous]

    return -1, 0


def estimated_cursor(
    index: int,
    segments: list[str],
    anchors: dict[int, int],
    transcript_length: int,
):

    anchor_index, anchor_cursor = (
        nearest_preceding_anchor(
            index,
            anchors,
        )
    )

    advance = sum(
        segment_word_count(segments[i])
        for i in range(
            anchor_index + 1,
            index,
        )
    )

    return min(
        transcript_length,
        anchor_cursor + advance,
    )


def make_excerpt(
    transcript: str,
    token_spans,
    cursor: int,
) -> str:

    if not token_spans:
        return ""

    start = max(
        0,
        cursor - LOCAL_WINDOW_WORDS // 2,
    )

    end = min(
        len(token_spans),
        cursor + LOCAL_WINDOW_WORDS,
    )

    if start >= len(token_spans):
        return ""

    return transcript[
        token_spans[start][0]:
        token_spans[end - 1][1]
    ]


def align_segments_hybrid(
    client,
    model_name: str,
    transcript: str,
    segments: list[str],
    max_workers: int = 4,
) -> tuple[
    list[str | None],
    list[str],
]:

    transcript_tokens, token_spans = (
        tokenize_with_offsets(transcript)
    )

    spans = [None] * len(segments)

    methods = [
        "unresolved"
        for _ in segments
    ]

    anchors = {}

    cursor = 0

    fallback_needed = []

    # --------------------------------------------------
    # Pass 1
    # Local mechanical match + global exact rescue
    # --------------------------------------------------

    for index, segment in enumerate(segments):

        start, score, end = mechanical_match(
            segment,
            transcript_tokens,
            cursor,
        )

        if (
            start is not None
            and score >= MECHANICAL_THRESHOLD
        ):
            spans[index] = span_text(
                transcript,
                token_spans,
                start,
                end,
            )

            methods[index] = "mechanical"

            anchors[index] = end

            cursor = end

            continue

        exact_match = global_exact_match(
            segment,
            transcript_tokens,
        )

        if exact_match is not None:

            start, end = exact_match

            spans[index] = span_text(
                transcript,
                token_spans,
                start,
                end,
            )

            methods[index] = "global_exact"

            anchors[index] = end

            cursor = end

            continue

        window_start = max(
            0,
            cursor - LOCAL_WINDOW_WORDS // 2,
        )

        window_end = min(
            len(transcript_tokens),
            cursor + LOCAL_WINDOW_WORDS,
        )

        fallback_needed.append(
            (
                index,
                segment,
                window_start,
                window_end,
            )
        )

        cursor = min(
            len(transcript_tokens),
            cursor + segment_word_count(segment),
        )

    # --------------------------------------------------
    # Pass 2
    # LLM fallback
    # --------------------------------------------------

    def fallback_worker(item):

        index, segment, start, end = item

        if (
            not token_spans
            or start >= len(token_spans)
            or end <= 0
        ):
            return None

        excerpt = transcript[
            token_spans[start][0]:
            token_spans[end - 1][1]
        ]

        return call_llm_fallback(
            client,
            model_name,
            segment,
            excerpt,
        )

    if fallback_needed:

        results = [
            fallback_worker(item)
            for item in fallback_needed
        ]

        for item, result in zip(
            fallback_needed,
            results,
        ):
            index, _, _, _ = item

            if result is None:
                continue

            spans[index] = result

            methods[index] = "llm_fallback"

            true_end = locate_returned_span(
                result,
                transcript_tokens,
            )

            if true_end is not None:
                anchors[index] = true_end

    # --------------------------------------------------
    # Pass 3
    # Repair unresolved segments
    # --------------------------------------------------

    unresolved = [
        i
        for i, method in enumerate(methods)
        if method == "unresolved"
    ]

    for index in unresolved:

        retry_cursor = estimated_cursor(
            index,
            segments,
            anchors,
            len(transcript_tokens),
        )

        retry_start = max(
            0,
            retry_cursor
            - max(
                20,
                segment_word_count(
                    segments[index]
                ),
            ),
        )

        start, score, end = mechanical_match(
            segments[index],
            transcript_tokens,
            retry_start,
        )

        if (
            start is not None
            and score >= MECHANICAL_THRESHOLD
        ):
            spans[index] = span_text(
                transcript,
                token_spans,
                start,
                end,
            )

            methods[index] = "mechanical_repaired"

            anchors[index] = end

            continue

        exact_match = global_exact_match(
            segments[index],
            transcript_tokens,
        )

        if exact_match is not None:

            start, end = exact_match

            spans[index] = span_text(
                transcript,
                token_spans,
                start,
                end,
            )

            methods[index] = "global_exact_repaired"

            anchors[index] = end

            continue

        start, _, end = wide_local_fuzzy_match(
            segments[index],
            transcript_tokens,
            retry_cursor,
        )

        if start is not None:

            spans[index] = span_text(
                transcript,
                token_spans,
                start,
                end,
            )

            methods[index] = "wide_fuzzy_repaired"

            anchors[index] = end

    # --------------------------------------------------
    # Pass 4
    # Second LLM fallback using corrected anchors
    # --------------------------------------------------

    still_unresolved = [
        i
        for i, method in enumerate(methods)
        if method == "unresolved"
    ]

    repair_items = []

    for index in still_unresolved:

        retry_cursor = estimated_cursor(
            index,
            segments,
            anchors,
            len(transcript_tokens),
        )

        excerpt = make_excerpt(
            transcript,
            token_spans,
            retry_cursor,
        )

        repair_items.append(
            (
                index,
                segments[index],
                excerpt,
            )
        )

    def repair_worker(item):

        _, segment, excerpt = item

        if not excerpt:
            return None

        return call_llm_fallback(
            client,
            model_name,
            segment,
            excerpt,
        )

    if repair_items:

        results = [
            repair_worker(item)
            for item in repair_items
        ]

        for item, result in zip(
            repair_items,
            results,
        ):

            index, _, _ = item

            if result is None:
                continue

            spans[index] = result

            methods[index] = "llm_repaired"

    return spans, methods