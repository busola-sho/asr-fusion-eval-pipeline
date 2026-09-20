"""
Severity scoring prompts for ASR transcription error evaluation.
Two conditions for comparison: DIRECT_SEVERITY_PROMPT vs STRUCTURED_SEVERITY_PROMPT.
Both score against the same 0-4 rubric; the structured condition forces
error-type identification and consequence reasoning before the final score.
"""

SEVERITY_RUBRIC = """
Severity scale (0-4), describing the meaning-level impact of ASR transcription
errors on a reference (ground-truth) vs hypothesis (ASR output) sentence pair.
Judge only from the pair given - do not assume outside context.

0 - No meaning change: surface form only (casing, punctuation, filler words,
    dialectal/phonetic variation, or an unambiguous spelling variant of the
    same proper noun) - no change to propositional content.
1 - Trivial/cosmetic error: wording differs or is garbled, but the meaning
    is confidently and cleanly recoverable, no material fact changes. Covers
    nonsensical renderings with no real alternative (e.g. "Japer" for
    "Jaipur") - even where the substituted words are real, if the resulting
    phrase doesn't cohere into a believable alternative claim, treat it the
    same way (e.g. "seeing her goggle" for "saying hey Google"). Requires
    CONFIDENT, CLEAN recovery: if reconstruction is needed, or any doubt
    remains, use level 2 instead. For longer, multi-clause utterances,
    prefer 2 over 1 for any genuine (even minor) content-word change - more
    text means more room for a subtle shift to go unnoticed.
2 - Ambiguous meaning shift: genuine ambiguity, real reconstruction needed,
    or residual doubt about intent. Covers connector/relational word changes
    (and/or/in/of/on) that alter how two items relate rather than what they
    individually are (e.g. "activities ON work time" -> "activities AND work
    time" turns one benefit into two vague ones); the same word garbled
    differently more than once in a sentence; and total, unrecoverable loss
    of a detail (content simply gone, not just unclear) - loss alone stays
    at 2, never 3/4, since those levels require a CLEAR meaning change to be
    asserted, and severe garbling can't assert anything clearly. No
    unambiguous different material fact is asserted at this level.
3 - Clear factual error: unambiguously asserts a different material fact -
    entity, number, time, location, quantity, or action - with no real
    ambiguity about the new claim. Includes substitution of a real,
    plausible, different entity (e.g. "Joan" for "John", "Jasper" for
    "Jaipur", "Shannon" for "Schengen"), as opposed to nonsense with no real
    alternative (level 1).
4 - Critical meaning alteration: reverses, fabricates, or fundamentally
    changes the central assertion - flipped negation, a tense/aspect shift
    that changes whether a consequential action happened/will happen, a
    different actor, a reversed alibi/sequence, or fabricated content. Does
    NOT require a literal negation word: any swap (demonstrative, pronoun,
    modal, tense) that flips the claim with nothing in the sentence to
    correct it qualifies (e.g. "This is not a better way" -> "There is not
    a better way" reverses the recommendation). Also covers hallucinated
    content filling a span marked inaudible/unclear (e.g. [inc]) - score
    as 4 by default, since there's no ground truth to check it against.

Not scored at all: a reference that is ONLY a non-lexical tag (e.g.
"<OVERLAP>" alone) has no comparable content - exclude it rather than
scoring 0-4.
"""

DIRECT_SEVERITY_PROMPT = """You are evaluating the severity of a speech-to-text
transcription error for use in a high-stakes policing context, where the
transcript may be relied on as evidence.

{rubric}

Reference (ground truth): "{reference}"
Hypothesis (ASR output): "{hypothesis}"

Compare the reference and hypothesis, and assign a single severity score from
0 to 4 using the scale above. Base your judgement only on the pair given -
do not assume outside context.

Respond in exactly this format:
Severity: <0-4>
Justification: <one to two sentences explaining your score>
""".format(rubric=SEVERITY_RUBRIC, reference="{reference}", hypothesis="{hypothesis}")