import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F


from .experience_maker import Experience
from .experience_maker_card_game import Experience_CARDGAME
from openrlhf.models.lmm_kits.base.data_processor import BaseDataProcessor

@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    visual_inputs: Optional[dict]


def split_input_batch(batch: Dict, tokenizer) -> List[Dict]:
    batch_size = len(batch["input_ids"])
    batch_kwargs = [{} for _ in range(batch_size)]
    # first process None values
    keys = []
    for k, v in batch.items():
        if v is not None:
            keys.append(k)
        else:
            for i in range(batch_size):
                batch_kwargs[i][k] = None

    if "pixel_values" in keys and ("input_ids" not in keys or "image_grid_thw" not in keys):
        raise ValueError("Cannot split batch with pixel_values without input_ids and image_grid_thw")
    if "image_grid_thw" in keys and ("input_ids" not in keys):
        raise ValueError("Cannot split batch with image_grid_thw without input_ids")
    for k in ["input_ids", "attention_mask"]:
        if k in keys:
            vals = batch[k]
            if isinstance(vals, torch.Tensor):
                vals = torch.unbind(vals)
            assert batch_size == len(vals)
            for i, v in enumerate(vals):
                batch_kwargs[i][k] = v
    if "pixel_values" in keys:
        thws = batch["image_grid_thw"]  # (total_img_num, (t,h,w))
        pixel_values = batch["pixel_values"]
        vision_start_id = tokenizer("<|vision_start|>")["input_ids"][0]
        vision_end_id = tokenizer("<|vision_end|>")["input_ids"][0]
        for i in range(batch_size):
            input_ids_i = batch_kwargs[i]["input_ids"]
            if not isinstance(input_ids_i, torch.Tensor):
                input_ids_i = torch.tensor(input_ids_i)
            vision_start_num = (input_ids_i == vision_start_id).sum().item()
            vision_end_num = (input_ids_i == vision_end_id).sum().item()
            assert vision_start_num == vision_end_num, f"vision_start_num: {vision_start_num}, vision_end_num: {vision_end_num}"
            img_num = vision_start_num
            if img_num == 0:
                batch_kwargs[i]["pixel_values"] = None
                batch_kwargs[i]["image_grid_thw"] = None
                continue
            thws_i = thws[:img_num]
            assert len(thws_i) == img_num, f"len(thws_i): {len(thws_i)}, img_num: {img_num}"
            thws = thws[img_num:]
            if not isinstance(thws_i, torch.Tensor):
                thws_i = torch.stack(thws_i)
            batch_kwargs[i]["image_grid_thw"] = thws_i
            patchs_num = thws_i.prod(dim=1).sum().item()
            pixel_values_i = pixel_values[:patchs_num]
            assert len(pixel_values_i) == patchs_num, f"len(pixel_values_i): {len(pixel_values_i)}, patchs_num: {patchs_num}"
            pixel_values = pixel_values[patchs_num:]
            batch_kwargs[i]["pixel_values"] = pixel_values_i
        assert len(thws) == 0, f"len(thws): {len(thws)}, pixel_values: {len(pixel_values)}"
        assert len(pixel_values) == 0, f"len(pixel_values): {len(pixel_values)}"
    return batch_kwargs


def split_experience_batch(experience: Experience, data_processor: Optional[BaseDataProcessor], multimodal: bool = False) -> List[BufferItem]:
    batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        
        assert batch_size == len(vals), f"batch_size: {batch_size}, len({key}): {len(vals)}"
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    
    if data_processor is not None:
        visual_inputs_batch = experience.visual_inputs
        visual_inputs_batch['input_ids'] = experience.sequences
        visual_inputs_chunks = split_input_batch(visual_inputs_batch, data_processor.tokenizer)
        for i, visual_inputs in enumerate(visual_inputs_chunks):
            visual_inputs.pop('input_ids')
            batch_kwargs[i]["visual_inputs"] = visual_inputs


    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    if not multimodal:
        for i in range(batch_size):
            batch_kwargs[i]["visual_inputs"] = None
    
    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], data_processor: Optional[BaseDataProcessor], packing_samples=False, multimodal=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    
    if multimodal:
        kwargs["visual_inputs"] = data_processor.make_input_batch([item.visual_inputs for item in items])
    return Experience(**kwargs)

def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, 
        sample_batch_size: int, 
        data_processor: Optional[BaseDataProcessor] = None, 
        limit: int = 0, 
        cpu_offload: bool = True, 
        packing_samples: bool = False,
        drop_maxlen: bool = False,
        maxlen: int = 10**8,
        multimodal: bool = False,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.data_processor = data_processor
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.maxlen = maxlen
        self.drop_maxlen = drop_maxlen
        self.multimodal = multimodal

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience, self.data_processor, self.multimodal)
        # NOTE: No tested
        if self.drop_maxlen:
            original_len = len(items)
            items = list(filter(lambda x: x.sequences.shape[-1] <= self.maxlen, items))
            if original_len - len(items) > 0:
                print(f"drop {original_len - len(items)} samples")
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.data_processor, self.packing_samples, self.multimodal)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.data_processor, self.packing_samples, self.multimodal)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd + 1e-8)


class ReplayBuffer_CARDGAME(ABC):
    """Replay buffer class for card game.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, 
        sample_batch_size: int, 
        data_processor: Optional[BaseDataProcessor] = None, 
        limit: int = 0, 
        cpu_offload: bool = True, 
        packing_samples: bool = False,
        drop_maxlen: bool = False,
        maxlen: int = 10**8,
        multimodal: bool = False,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.data_processor = data_processor
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.maxlen = maxlen
        self.drop_maxlen = drop_maxlen
        self.multimodal = multimodal

    @torch.no_grad()
    def append(self, experience: Experience_CARDGAME) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = self.split_experience_batch(experience, self.data_processor, self.multimodal)

        # NOTE: No tested
        if self.drop_maxlen:
            original_len = len(items)
            items = list(filter(lambda x: x.sequences.shape[-1] <= self.maxlen, items))
            if original_len - len(items) > 0:
                print(f"drop {original_len - len(items)} samples")
        
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = self.remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def split_experience_batch(self, experience: Experience_CARDGAME, data_processor: Optional[BaseDataProcessor], multimodal: bool = False) -> List[BufferItem]:
        batch_size = len(experience.sequences)
        batch_kwargs = [{} for _ in range(batch_size)]
        keys = (
            "sequences",
            "action_log_probs",
            "base_action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
        )
        for key in keys:
            value = getattr(experience, key)
            if value is None:
                for i in range(batch_size):
                    batch_kwargs[i][key] = None
                continue
            vals = value
            if isinstance(vals, torch.Tensor):
                vals = torch.unbind(vals)
            
            assert batch_size == len(vals), f"batch_size: {batch_size}, len({key}): {len(vals)}"
            for i, v in enumerate(vals):
                batch_kwargs[i][key] = v
        
        if data_processor is not None:
            visual_inputs_batch = experience.visual_inputs
            visual_inputs_batch['input_ids'] = experience.sequences
            visual_inputs_chunks = split_input_batch(visual_inputs_batch, data_processor.tokenizer)
            for i, visual_inputs in enumerate(visual_inputs_chunks):
                visual_inputs.pop('input_ids')
                batch_kwargs[i]["visual_inputs"] = visual_inputs


        for i in range(batch_size):
            batch_kwargs[i]["info"] = {}
        for k, v in experience.info.items():
            vals = torch.unbind(v)
            assert batch_size == len(vals)
            for i, vv in enumerate(vals):
                if isinstance(vv, torch.Tensor):
                    assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                    vv = vv.item()
                batch_kwargs[i]["info"][k] = vv

        if not multimodal:
            for i in range(batch_size):
                batch_kwargs[i]["visual_inputs"] = None
        
        items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
        return items
    
    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience_CARDGAME:
        items = random.sample(self.items, self.sample_batch_size)
        experience = self.make_experience_batch(items, self.data_processor, self.packing_samples, self.multimodal)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience_CARDGAME:
        experience = self.make_experience_batch(batch, self.data_processor, self.packing_samples, self.multimodal)
        return experience

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd + 1e-8)
    
    
    def make_experience_batch(self, 
                              items: List[BufferItem], 
                              data_processor: Optional[BaseDataProcessor], 
                              packing_samples=False, 
                              multimodal=False) -> Experience_CARDGAME:
        
        kwargs = {}
        keys = (
            "sequences",
            "action_log_probs",
            "base_action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
        )
        for key in keys:
            vals = [getattr(item, key) for item in items]
            if not packing_samples and not (key == "returns" or key == "advantages"):
                batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
            elif not packing_samples and (key == "returns" or key == "advantages"):
                batch_data = torch.stack(vals, dim=0) if vals[0] is not None else None
            else:
                batch_data = vals if vals[0] is not None else None
            kwargs[key] = batch_data

        kwargs["info"] = {}
        for key in items[0].info.keys():
            vals = torch.tensor([item.info[key] for item in items])
            kwargs["info"][key] = vals
        
        if multimodal:
            kwargs["visual_inputs"] = self.make_input_batch([item.visual_inputs for item in items])
        return Experience_CARDGAME(**kwargs)
    
    def remove_padding_in_sequences(self, items):
        for item in items:
            seq, act_log_prob, base_act_log_prob, att_mask, act_mask = (
                item.sequences,
                item.action_log_probs,
                item.base_action_log_probs,
                item.attention_mask,
                item.action_mask,
            )
            right_pad = (1 - act_mask.long()).sum()
            right_pad = None if right_pad == 0 else -right_pad

            # left_pad for seq and att_mask
            left_pad = att_mask.long().argmax()
            (
                item.sequences,
                item.action_log_probs,
                item.base_action_log_probs,
                item.attention_mask,
                item.action_mask,
            ) = (
                seq[left_pad:right_pad],
                act_log_prob[:right_pad],
                base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
                att_mask[left_pad:right_pad],
                act_mask[:right_pad],
            )
        return items
    
    def make_input_batch(self, inputs: List[Dict]) -> Dict:
        # each element has no batch dimension
        batch = {}
        # collect all keys
        for inp in inputs:
            batch.update({k:None for k,v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                # qwen2vl concat all patches of all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in inputs if k in inp], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Qwen2VLDataProcessor")
        return batch