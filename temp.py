from vllm import LLM, SamplingParams
from PIL import Image
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import json


# with open("data.json", "r") as f:
#     data = json.load(f)

# import pdb; pdb.set_trace()

# response = '\n## Previous your response (2 steps ago):\n```json\n{\n  "cards": [4, 5, 3, 9],\n  "number": [4, 5, 3, 9],\n  "thought": "I need to use the numbers 4, 5, 3, and 9 to form a formula that equals 24. I can use the \'*\' operator to multiply 3 and 9, which gives me 27. Then, I can subtract 27 from 4 and 5 to get 24.",\n  "formula": "(4 * 9) - (3 * 5)"\n}\n```'
# example = """{
#   "cards": [4, 3, 2, "K"],
#   "number": [4, 3, 2, 10],
#   "formula": "4*3+2+10=24"
# }"""

# import re
# # ```json\n と \n``` の間の文字列を抽出する正規表現パターン
# # (.*?) は非貪欲マッチで、最も内側の```を見つけるのを助ける
# # re.DOTALL は '.' が改行文字にもマッチするようにするフラグ
# pattern = r"```json\n(.*?)\n```"

# # 正規表現でマッチする部分を検索
# match = re.search(pattern, response, re.DOTALL)

# # マッチが見つかった場合
# if match:
#     # マッチした部分のうち、カッコ()で囲まれたグループ(JSON本体)を取得
#     json_part = match.group(1)

#     print("--- 抽出されたJSON部分 ---")
#     print(json_part)
#     print("------------------------")

#     # 抽出したJSON部分を Python の辞書に変換
#     try:
#         data = json.loads(json_part)
#         print("--- JSONパース結果 ---")
#         print(data)
#         print("----------------------")
#         # これで data['cards'] や data['formula'] のようにアクセスできます
#         print("Formula:", data.get("formula")) # .get()を使うとキーがなくてもエラーにならない

#     except json.JSONDecodeError as e:
#         print(f"抽出されたJSON部分のパースに失敗しました: {e}")
#         print("抽出された文字列が正しいJSON形式か確認してください。")

# else:
#     print("文字列内に ```json ... ``` のパターンが見つかりませんでした。")
# import pdb; pdb.set_trace()

# # JSON文字列をPythonの辞書に変換
# try:
#   output_dict = json.loads(example)
#   print(output_dict)
#   # output_dict を使った処理
# except json.JSONDecodeError as e:
#   print(f"JSONデコードエラーが発生しました: {e}")

# import pdb; pdb.set_trace()


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map="cuda")

# prompt = ["What is the life of a human? Please answer in English.", "What is the life of a cat?"]
# prompt_ids = [tokenizer(p, return_tensors="pt", padding=False, truncation=False)["input_ids"] for p in prompt]

# messages1 = tokenizer.apply_chat_template(
#     [
#         {"role": "user", "content": "What is the life of a human? Please answer in English."},
#     ],
#     tokenize=True,
# )

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "What is the life of a human? Please answer in English."},
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
print(model_inputs["input_ids"])
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.6
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

print("-----------------------------")

llm = LLM("Qwen/Qwen2.5-3B-Instruct", enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.6, max_tokens=512, skip_special_tokens=True)

prompts = [
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "What is the life of a human? Please answer in English.",
]

output = llm.generate(prompts, sampling_params)
print(output[0].outputs[0].text)
print(f"input_ids: {output[0].prompt_token_ids}")

# print("-----------------------------")
# output = llm.generate(prompt, sampling_params)
# print(output[0].outputs[0].text)
# print(f"input_ids: {output[0].prompt_token_ids}")

import pdb; pdb.set_trace()

# datasets = load_dataset("tau/commonsense_qa", split="train")



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