from vllm import LLM, SamplingParams
from PIL import Image
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


from vllm import LLM, SamplingParams
from PIL import Image
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import json
import warnings
import torch

model_name = "Qwen/Qwen3-8B"

# Suppress specific warnings if needed (optional)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not find image processor.*")

# --- Your Initial Code ---
model_name = "Qwen/Qwen2.5-7B-Instruct" # Using a smaller, common instruct model for demonstration
# It's usually sufficient to use the tokenizer for text processing with chat templates
# AutoProcessor is more for multimodal models, but often includes the tokenizer.
# Using AutoTokenizer directly is often clearer for text-only tasks.
try:
    # Prefer AutoTokenizer for text tasks unless processor is specifically needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Using AutoModelForCausalLM is correct
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    # Set pad_token if it's not set (common for some models like Qwen)
    if tokenizer.pad_token is None:
        print("Warning: pad_token not set. Using eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

except Exception as e:
    print(f"Error loading model/tokenizer '{model_name}': {e}")
    print("Please ensure the model name is correct and you have access.")
    exit()


prompts = [
    "{\n  \"cards\": [\"9\", \"8\", \"3\", \"8\"],\n  \"number\": [9, 8, 3, 8],\n  \"thoughts\": \"To solve this, I will try different combinations of the four numbers 9, 8, 3, and 8 using the allowed operators. One potential solution is to multiply 8 by 3, then subtract the difference between 9 and 8. This can be represented as (8 * 3) - (9 - 8).\",\n  \"formula\": \"(8 * 3) - (9 - 8) = 24\"\n}",
    "{\n  \"cards\": [\"4\", \"3\", \"2\", \"K\"],\n  \"number\": [4, 3, 2, 10],\n  \"thoughts\": \"step by step reasoning process to build the formula\",\n  \"formula\": \"4*3+2+10=24\"\n}",     # Corrected formula example
]

messages = [
    [
        {"role": "system", "content": "You are a helpful assistant."}, # Often good practice to include a system prompt
        {"role": "user", "content": prompt},
    ]
    for prompt in prompts
]

# --- Process Prompts and Get Token IDs ---
# Use return_tensors='pt' to get PyTorch tensors
# Use padding=True to handle sequences of different lengths
# Keep track of the attention_mask as well, which padding=True provides
tokenized_output = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    # enable_thinking=False, # This parameter might not be standard. Check Qwen specific docs if needed.
    add_generation_prompt=True, # Adds prompt indicating model should start generating
    return_tensors='pt',
    padding=True # Pad sequences to the longest in the batch
).to(model.device) # Ensure tensors are on the same device as the model

input_ids = tokenized_output # This is usually the key for input IDs

# --- Tokenize the Markers ---
# Important: Tokenize them exactly as they appear in the prompt string.
# add_special_tokens=False prevents adding BOS/EOS tokens around just these markers.
thoughts_marker = "thoughts\":"
formula_marker = "formula\":"

# Use encode to get token IDs. We need the list of IDs.
thoughts_marker_ids = tokenizer.encode(thoughts_marker, add_special_tokens=False)
formula_marker_ids = tokenizer.encode(formula_marker, add_special_tokens=False)

import pdb; pdb.set_trace()

print(f"Token IDs for '{thoughts_marker}': {thoughts_marker_ids}")
print(f"Token IDs for '{formula_marker}': {formula_marker_ids}")

# --- Create the Mask ---
batch_size, seq_length = input_ids.shape
# Initialize mask tensor with 1s, matching shape and device of input_ids
# Use float type for the mask values (0.2 and 1.0)
target_masks = torch.ones_like(input_ids, dtype=torch.float, device=input_ids.device)

# Helper function to find subsequence indices
def find_subsequence_indices(sequence_list, subsequence_list):
    """Finds the start index of the first occurrence of subsequence_list in sequence_list."""
    len_sub = len(subsequence_list)
    if len_sub == 0:
        return 0 # Empty subsequence found at start
    for i in range(len(sequence_list) - len_sub + 1):
        if sequence_list[i:i+len_sub] == subsequence_list:
            return i
    return -1 # Not found

# Iterate through each item in the batch
for i in range(batch_size):
    current_ids_list = input_ids[i].tolist() # Convert tensor row to list for easier searching

    # Find the start index of the marker sequences
    thoughts_start_idx = find_subsequence_indices(current_ids_list, thoughts_marker_ids)
    formula_start_idx = find_subsequence_indices(current_ids_list, formula_marker_ids)

    # Check if both markers were found and in the correct order
    if thoughts_start_idx != -1 and formula_start_idx != -1 and thoughts_start_idx < formula_start_idx:
        # Calculate the end index of the thoughts marker sequence
        thoughts_end_idx = thoughts_start_idx + len(thoughts_marker_ids)

        # Apply the 0.2 mask value *between* the end of "thoughts:" and the start of "formula:"
        # Slice format is [start:end], where end is exclusive
        target_masks[i, thoughts_end_idx:formula_start_idx] = 0.2
        print(f"Batch {i}: Found markers. Applying mask between index {thoughts_end_idx} and {formula_start_idx}.")
    else:
        # Handle cases where markers aren't found or are in the wrong order (optional)
        print(f"Batch {i}: Markers not found or not in expected order. Mask remains all 1s.")
        # You might want to raise an error or handle this differently depending on requirements.

# --- Display Results ---
print("\nInput Prompts:")
for p in prompts:
    print(p)

print("\nTokenized Input IDs (Batch):")
print(input_ids)
# print("\nAttention Mask (from padding):") # Provided by apply_chat_template with padding=True
# print(attention_mask) # You might need this attention_mask for model inference later

print("\nGenerated Target Masks (Batch):")
print(target_masks)

# Example of how you might use this mask (e.g., in a custom loss function)
# Assuming you have 'labels' corresponding to input_ids for training
# loss_fct = torch.nn.CrossEntropyLoss(reduction='none') # Calculate loss per token
# logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
# Shift logits and labels for next token prediction
# shift_logits = logits[..., :-1, :].contiguous()
# shift_labels = labels[..., 1:].contiguous()
# shift_masks = target_masks[..., 1:].contiguous() # Shift mask accordingly
#
# loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
# loss = loss.view(batch_size, seq_length - 1)
# weighted_loss = (loss * shift_masks).sum() / shift_masks.sum() # Apply mask and calculate mean
# print(f"\nExample Weighted Loss (conceptual): {weighted_loss.item()}")
import pdb; pdb.set_trace()




llm = LLM(model_name, enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=512, skip_special_tokens=True)

prompts = [
    "What is the life of a human? Please answer in English and within 256 words.",
    "Where is the capital of Japan? Please answer in English.",
]

messages = [
    [
        {"role": "user", "content": prompt},
    ]
    for prompt in prompts
]

chat_prompts = processor.apply_chat_template(messages, tokenize=False, enable_thinking=False, add_generation_prompt=True)

import pdb; pdb.set_trace()

outputs = llm.generate(chat_prompts, sampling_params)
print(outputs[0].outputs[0].text)
print("--------------------------------")
print(outputs[1].outputs[0].text)
import pdb; pdb.set_trace()





with open("data.json", "r") as f:
    data = json.load(f)

import pdb; pdb.set_trace()

response = '\n## Previous your response (2 steps ago):\n```json\n{\n  "cards": [4, 5, 3, 9],\n  "number": [4, 5, 3, 9],\n  "thought": "I need to use the numbers 4, 5, 3, and 9 to form a formula that equals 24. I can use the \'*\' operator to multiply 3 and 9, which gives me 27. Then, I can subtract 27 from 4 and 5 to get 24.",\n  "formula": "(4 * 9) - (3 * 5)"\n}\n```'
example = """{
  "cards": [4, 3, 2, "K"],
  "number": [4, 3, 2, 10],
  "formula": "4*3+2+10=24"
}"""

import re
# ```json\n と \n``` の間の文字列を抽出する正規表現パターン
# (.*?) は非貪欲マッチで、最も内側の```を見つけるのを助ける
# re.DOTALL は '.' が改行文字にもマッチするようにするフラグ
pattern = r"```json\n(.*?)\n```"

# 正規表現でマッチする部分を検索
match = re.search(pattern, response, re.DOTALL)

# マッチが見つかった場合
if match:
    # マッチした部分のうち、カッコ()で囲まれたグループ(JSON本体)を取得
    json_part = match.group(1)

    print("--- 抽出されたJSON部分 ---")
    print(json_part)
    print("------------------------")

    # 抽出したJSON部分を Python の辞書に変換
    try:
        data = json.loads(json_part)
        print("--- JSONパース結果 ---")
        print(data)
        print("----------------------")
        # これで data['cards'] や data['formula'] のようにアクセスできます
        print("Formula:", data.get("formula")) # .get()を使うとキーがなくてもエラーにならない

    except json.JSONDecodeError as e:
        print(f"抽出されたJSON部分のパースに失敗しました: {e}")
        print("抽出された文字列が正しいJSON形式か確認してください。")

else:
    print("文字列内に ```json ... ``` のパターンが見つかりませんでした。")
import pdb; pdb.set_trace()

# JSON文字列をPythonの辞書に変換
try:
  output_dict = json.loads(example)
  print(output_dict)
  # output_dict を使った処理
except json.JSONDecodeError as e:
  print(f"JSONデコードエラーが発生しました: {e}")

import pdb; pdb.set_trace()


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

prompt = ["What is the life of a human? Please answer in English.", "What is the life of a cat?"]
prompt_ids = [tokenizer(p, return_tensors="pt", padding=False, truncation=False)["input_ids"] for p in prompt]

messages1 = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is the life of a human? Please answer in English."},
    ],
    tokenize=True,
)

import pdb; pdb.set_trace()

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