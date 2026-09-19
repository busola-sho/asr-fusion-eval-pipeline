#prompt/prompts.py

UNANCHORED_FUSION_PROMPT = """You are given three ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the three transcripts
- Change sentence structure or word order beyond individual word swaps

Return only the final transcript, nothing else."""