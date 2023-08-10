from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import zero_pad_sequences, exist_and_not_none

def preprocess_data(data):
    # stanfordnlp/SHP
    if exist_and_not_none(data, 'human_ref_A'):
        prompt = "Human: " +  data['history'] + "\nAssistant: "
        preferA = bool(data['labels'])
        chosen = data['human_ref_A'] if preferA else data['human_ref_B']
        reject = data['human_ref_B'] if preferA else data['human_ref_A']
    # Anthropic/hh-rlhf
    # tasksource/oasst1_pairwise_rlhf_reward
    elif exist_and_not_none(data, 'chosen'):
        prompt = data['prompt'] if exist_and_not_none(data, 'prompt') else ""
        if prompt.startswith('prompter:'):
            prompt = prompt.replace('prompter:', 'Human:').replace('assistant:', '\nAssistant:') + '\nAssistant:'

        chosen = data['chosen']
        reject = data['rejected']
    # lvwerra/stack-exchange-paired
    elif exist_and_not_none(data, 'response_j'):
        prompt = "Human: " +  data['question'] + "\nAssistant: "
        chosen = data['response_j']
        reject = data['response_k']
    else:
        raise ValueError("reward_dataset key error")
    return prompt, chosen, reject


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy) -> None:
        super().__init__()
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject = preprocess_data(data)
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)


    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]

        chosen = prompt + chosen + " " + self.tokenizer.eos_token 
        chosen_token = self.tokenizer(chosen,
                                    max_length=self.max_length,
                                    padding=False,
                                    truncation=True,
                                    return_tensors="pt")
        
        reject = prompt + reject + " " + self.tokenizer.eos_token 
        reject_token = self.tokenizer(reject,
                                    max_length=self.max_length,
                                    padding=False,
                                    truncation=True,
                                    return_tensors="pt")
        
        return chosen_token["input_ids"], chosen_token["attention_mask"], reject_token[\
            "input_ids"], reject_token["attention_mask"]

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        for chosen_id, chosen_mask, reject_id, rejects_mask in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks