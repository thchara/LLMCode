import pandas as pd
from llmcode.coding import code_texts   # function already in repo
from llmcode.llms import query_LLM      # ensures backend routing
import os, pathlib

# 1. parameters
DATA_PATH = pathlib.Path("my_data/transcripts.csv")
SEED_PATH = pathlib.Path("my_data/seed_codes.csv")  # may not exist
OUT_PATH  = pathlib.Path("my_data/transcripts_coded.csv")

# 2. load data
df = pd.read_csv(DATA_PATH)

# 3. few-shot examples (if you made them)
if SEED_PATH.exists():
    seeds = pd.read_csv(SEED_PATH)
    few_shot_texts  = list(seeds["text"])
    few_shot_codes  = list(seeds["codes"])
else:
    few_shot_texts = few_shot_codes = []

# 4. coding instruction (explicit, concise)
instruction = (
    "Assign ONE OR MORE hierarchical codes to each text.\n"
    "Possible top-level codes: Observation, Hypothesis, Evaluation, Decision.\n"
    "Use '>' to separate levels (e.g., 'Hypothesis > Colour').\n"
    "Return codes as semicolon-separated list; no extra text."
)

# 5. call the helper (uses LM Studio via LLMCODE_BACKEND=local)
coded = code_texts(
    texts=list(df["text"]),
    coding_instruction=instruction,
    few_shot_texts=few_shot_texts,
    few_shot_codes=few_shot_codes,
    gpt_model="local-llama",        # any model name is OK for local
    backend="local",                # explicit for clarity
    temperature=0.2,
)

df["codes_llm"] = coded
df.to_csv(OUT_PATH, index=False)
print("↳ saved", OUT_PATH)
