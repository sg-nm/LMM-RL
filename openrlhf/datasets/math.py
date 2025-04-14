"""
Hendryck's Math dataset

{'problem': 'A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.',
 'level': 'Level 1',
 'type': 'Counting & Probability',
 'solution': 'The spinner is guaranteed to land on exactly one of the three regions, so we know that the sum of the probabilities of it landing in each region will be 1. If we let the probability of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, from which we have $x=\\boxed{\\frac{1}{4}}$.'}

"""

import random
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from typing import Iterable, Any, Union
import json
from pathlib import Path
from openrlhf.datasets.math_data.math_utils import parse_question, parse_ground_truth, construct_prompt

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def preprocess_data(examples, data_name='math', prompt_type="qwen25-math-cot", n_sampling=1):
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue

        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, prompt_type)
        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }
        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)
    
    return samples



class MathDataset(Dataset):
    def __init__(self, data_path, prompt_type="qwen25-math-cot", ratio=1.0):
        if os.path.exists(data_path):
            examples = list(load_jsonl(data_path))
        else:
            raise FileNotFoundError(f"File {data_path} not found")

        examples = examples[:int(len(examples) * ratio)]
        
        # add 'idx' in the first column
        if "idx" not in examples[0]:
            examples = [{"idx": i, **example} for i, example in enumerate(examples)]
        # dedepulicate & sort
        examples = sorted(examples, key=lambda x: x["idx"])
        self.data_name = 'math'
        self.prompt_type = prompt_type

        self.data = preprocess_data(examples, self.data_name, self.prompt_type, n_sampling=1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example =  self.data[idx]
        return example["prompt"], example["gt_cot"]



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from vllm import LLM, SamplingParams
    from openrlhf.textgrad.custom_reward_functions import check_answer_math
    from tqdm import tqdm

    model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"


    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=True,
    )

    dataset = MathDataset(data_path="/home/suganuma/src/lmm-r1/openrlhf/datasets/math_data/test.jsonl")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    accs = []
    
    for prompts, gt_cots in tqdm(dataloader):

        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0,
                top_p=1.0,
                max_tokens=1024,
                n=1,
                stop=None,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in model_name_or_path.lower()
                    else None
                ),
            ),
        )
        outputs = sorted(outputs, key=lambda x: int(x.request_id))  # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]

        acc, each_score = check_answer_math(outputs, gt_cots, prompt_type="qwen25-math-cot", data_name="math")
        print(f"Accuracy: {acc}")
        accs.append(acc)

    print(f"Average Accuracy: {sum(accs) / len(accs)}")
