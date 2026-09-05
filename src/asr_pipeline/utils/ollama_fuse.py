#utils/ollama_fuse.py

from ollama import client
from asr_pipeline.prompt import UNANCHORED_FUSION_PROMPT

OLLAMA_HOST = "http://localhost:11434"

def initialise_client():
    client = Client(host=OLLAMA_HOST)
    return client

def ollama_select(client, model_name, hyp_by_model, num_predict, retries=2):
    
        hyp_block = "\n".join(
            f"Transcript {i+1} ({model}): {hyp}"
            for i, (model, hyp) in enumerate(hyp_by_model.items())
        )

        for attempt in range(retries + 1):
            try:
                response = client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": UNANCHORED_FUSION_PROMPT},
                        {"role": "user",   "content": hyp_block},
                    ],
                    options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
                    keep_alive="30m",
                    think=False,
                )
                return response.message.content.strip()
            except Exception as e:
                if attempt == retries:
                    print(f"  ERROR (select): {e}")
                    return None
        return None