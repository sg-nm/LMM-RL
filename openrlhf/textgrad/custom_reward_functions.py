"""
Custom reward functions for TextGrad.
"""

import re


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

    # # extract answer letter from output_text
    # answer_pattern = r"Answer:\s*(\w)"
    # answers = []
    # for text in output_text:
    #     match = re.search(answer_pattern, text)
    #     if match:
    #         answers.append(match.group(1))
    #     else:
    #         answers.append(None)  # No answer found
    
    # # extract answer letter from labels
    # label_answers = []
    # for label in labels:
    #     match = re.search(answer_pattern, label)
    #     if match:
    #         label_answers.append(match.group(1))
    #     else:
    #         label_answers.append(None)  # No answer found
    
    # # check if each answer is equal to the corresponding label
    # correct = [1 if a is not None and a == l else 0 for a, l in zip(answers, label_answers)]
    # return correct

