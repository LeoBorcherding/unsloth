import os
from unsloth import FastLanguageModel

os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

# Keep model small to reduce setup time and downloads.
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
OUT_DIR = "tmp/issue_5054_gguf"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Repro call from issue #5054.
model.save_pretrained_gguf(
    OUT_DIR,
    tokenizer,
    quantization_method = "q8_0",
)

print("DONE")
