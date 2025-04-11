"""
CommonSenseQA dataset

{'id': '075e483d21c29a511267ef62bedc0461',
 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
 'question_concept': 'punishing',
 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
  'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
 'answerKey': 'A'}
"""


from torch.utils.data import Dataset
from tqdm import tqdm


PROMPT = """Answer a given question using the following output format.

## Output Format
Thought: provide your thoughts behind the answer
Answer: only provide the choice label from the given choices, e.g. Answer: C

## Question
{question}

## Choices
{choices}
"""


def preprocess_data(data, input_template=None, input_key="question", label_key="answerKey", apply_chat_template=None) -> str:
    if apply_chat_template:
        question = data[input_key]
        choice_list = "\n".join([f"{label}: {choice}" for label, choice in zip(data["choices"]["label"], data["choices"]["text"])])
        prompt = PROMPT.format(question=question, choices=choice_list)
        message = [{"role": "user", "content": prompt}]
        prompt = apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    else:
        question = data[input_key]
        choice_list = "\n".join([f"{label}: {choice}" for label, choice in zip(data["choices"]["label"], data["choices"]["text"])])
        prompt = PROMPT.format(question=question, choices=choice_list)
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else "Answer: " + data[label_key]
    return prompt, label


class CommonSenseQADataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", "question")
        label_key = getattr(self.strategy.args, "label_key", "answerKey")
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx]
