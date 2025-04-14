"""
Custom reward functions for TextGrad.
"""

import re
from openrlhf.datasets.math_data.math_utils import run_execute, extract_answer
from openrlhf.datasets.math_data.python_executor import PythonExecutor
from openrlhf.datasets.math_data.evaluate import evaluate


def check_answer_commonsense_qa(output_text: list[str], labels: list[str]) -> list[int]:
    """
    Check if the answer is correct for the commonsense QA task.
    output_text: list of strings, each string is a response from the model.
        Response format:
        Think: <rationale>
        Answer: <answer> like Answer: C
    labels: list of strings, each string is the correct answer.
        Label format:
        Answer: <answer>

    Returns:
        correct: list of integers, each integer is 1 if the answer is correct, 0 otherwise.
    """

    """
    Flexible answer checker for commonsense QA task.
    Matches: "Answer: X", "Answer: (X)", "Answer: 'X'", etc.
    """
    def extract_answer(text):
        # Matches things like: Answer: X, Answer: (X), Answer: 'X', Answer: "X"
        match = re.search(r"Answer:\s*[\('\"]?([A-Za-z])[\)'\"]?", text)
        return match.group(1).upper() if match else None

    answers = [extract_answer(text) for text in output_text]
    label_answers = [extract_answer(label) for label in labels]

    correct = [1 if a is not None and a == l else 0 for a, l in zip(answers, label_answers)]
    return correct


def check_answer_math(output_text: list[str], labels: list[str], prompt_type: str="qwen25-math-cot", data_name: str='math') -> list[int]:
    """
    Check if the answer is correct for Hendrycks's math benchmark.
    Assume labels are 'gt_cot' from MathDataset.
    """
    executor = PythonExecutor(get_answer_from_stdout=True)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    n_sampling = 1

    codes = []
    for output in output_text:
        for stop_word in stop_words:
            if stop_word in output:
                output = output.split(stop_word)[0].strip()
        codes.append(output)
    
    # extract preds
    results = [
        run_execute(executor, code, prompt_type, data_name) for code in codes
    ]

    all_samples = []

    for i, gt_cot in enumerate(labels):
        code = codes[i * n_sampling : (i + 1) * n_sampling]
        result = results[i * n_sampling : (i + 1) * n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        gt_ans = extract_answer(gt_cot, data_name)
        sample = {
            "idx": i,
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "code": code,
            "report": reports,
            "pred": preds
        }
        all_samples.append(sample)

    # add processed samples
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=prompt_type,
        execute=True,
    )

    # print(f"Accuracy: {result_json['acc']}")
    return result_json['acc'], result_json['each_score']

