from vllm import LLM, SamplingParams
from PIL import Image
import os
from datasets import load_dataset


# datasets = load_dataset("tau/commonsense_qa", split="train")

llm = LLM("Qwen/Qwen2.5-7B-Instruct", enable_prefix_caching=True)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256, skip_special_tokens=True)

inputs = [
    "Hello!",
    "What is the life of a human?",
]

tokenizer = llm.get_tokenizer()

messages1 = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is the life of a human? Please answer in English."},
    ],
    tokenize=False,
)

m1_tokenized = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is the life of a human? Please answer in English."},
    ],
    tokenize=True,
)

messages2 = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is your name? Please answer in English."},
    ],
    tokenize=False,
)

output = llm.generate([messages1, messages2], sampling_params)
# print(output[0].outputs[0].text)
# print(output[1].outputs[0].text)
print(m1_tokenized)
print(output[0].prompt_token_ids)
import pdb; pdb.set_trace()