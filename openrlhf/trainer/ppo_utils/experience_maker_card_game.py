import os
import time
import copy
import math
import gc
import traceback
import re
import json
import random
import numpy as np
import gymnasium as gym
from datetime import timedelta
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Deque
from collections import deque
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from vllm import SamplingParams


from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples, compute_uniform_reward
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray

from qwen_vl_utils import process_vision_info

from openrlhf.textgrad.feedback_vllm import FEEDBACK_PROMPT_CARD_V2, FEEDBACK_PROMPT_CARD, FEEDBACK_PROMPT_BASE_CARD, RESPONSE_PROMPT_CARD, FEEDBACK_PROMPT_SUFFIX, PROMPT_BASE_CARD_REINFORCE, TASK_DESCRIPTION_FOR_FEEDBACKMODEL
from openrlhf.textgrad.custom_reward_functions import check_answer_commonsense_qa, check_answer_math

from openrlhf.trainer.ppo_utils.experience_maker import NaiveExperienceMaker, Samples
from card_env.gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from card_env.gym_cards.config_dataclass import EnvConfig, PromptConfig
from card_env.gym_cards.prompt_lib import PROMPT_FN, example_json_text



@dataclass
class RolloutStorage:
    """
    Each element contains the parallel rollout results for each environment, which is the batch dimension.

    NOTE:
    - visual_inputs is a list of Dicts (len(visual_inputs) = batch_size or num_envs), each Dict has a single image info, which is 'pixel_values' and 'image_grid_thw'.
    """

    output_text: list[str]
    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.Tensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    visual_inputs: Optional[List[Dict]]
    pad_len: Optional[int]
    masks: Optional[list[torch.Tensor]]
    bad_masks: Optional[list[torch.Tensor]]
    obs: Optional[list[Dict]]
    action_log_probs: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    prompt: Optional[list[str]] = None
    sequences_for_KL: Optional[torch.Tensor] = None
    attention_mask_for_KL: Optional[torch.Tensor] = None
    action_mask_for_KL: Optional[torch.Tensor] = None
    reward_diff: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    attention_mask_for_input_ids: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None

    # Optional: Add a method for easy validation
    def __post_init__(self):
        # Example validation: check if rewards and masks have same leading dimension if not None
        if self.rewards is not None and self.masks is not None:
            if self.rewards.shape[0] != self.masks.shape[0]:
                raise ValueError(f"Batch dimension mismatch between rewards ({self.rewards.shape[0]}) and masks ({self.masks.shape[0]})")



def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor

def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience_CARDGAME:
    """A batch of data for each step of parallel envs.
    Left/right padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A), usually None
    returns: (B,)
    advantages: (B,)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)
    visual_inputs: Dict{
        'pixel_values': (B, pixel_values),
        'image_grid_thw': (B, 3)
    }
    info: (B,)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None
    visual_inputs: Optional[dict] = field(default_factory=dict)
    sequences_for_KL: Optional[torch.Tensor] = None
    attention_mask_for_KL: Optional[torch.Tensor] = None
    action_mask_for_KL: Optional[torch.Tensor] = None
    reward_diff: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    attention_mask_for_input_ids: Optional[torch.Tensor] = None


    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        if self.visual_inputs is not None:
            self.visual_inputs = {key: to(value, device) for key, value in self.visual_inputs.items()}
        # if self.visual_inputs is not None:
        #     for i in range(len(self.visual_inputs)):
        #         self.visual_inputs[i] = {key: to(value, device) for key, value in self.visual_inputs[i].items()}
        self.reward_diff = to(self.reward_diff, device)
        self.sequences_for_KL = to(self.sequences_for_KL, device)
        self.attention_mask_for_KL = to(self.attention_mask_for_KL, device)
        self.action_mask_for_KL = to(self.action_mask_for_KL, device)
        self.input_ids = to(self.input_ids, device)
        self.attention_mask_for_input_ids = to(self.attention_mask_for_input_ids, device)
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        if self.visual_inputs is not None:
            self.visual_inputs = {key: pin_memory(value) for key, value in self.visual_inputs.items()}
        # if self.visual_inputs is not None:
        #     for i in range(len(self.visual_inputs)):
        #         self.visual_inputs[i] = {key: pin_memory(value) for key, value in self.visual_inputs[i].items()}
        self.reward_diff = pin_memory(self.reward_diff)
        self.sequences_for_KL = pin_memory(self.sequences_for_KL)
        self.attention_mask_for_KL = pin_memory(self.attention_mask_for_KL)
        self.action_mask_for_KL = pin_memory(self.action_mask_for_KL)
        self.input_ids = pin_memory(self.input_ids)
        self.attention_mask_for_input_ids = pin_memory(self.attention_mask_for_input_ids)
        return self
    



class RemoteExperienceMaker_CardGame(NaiveExperienceMaker):
    def __init__(self, 
                 actor, critic, reward_model, initial_model, tokenizer, data_processor, feedback_data_processor,
                 prompt_max_len, kl_controller, strategy, remote_rm_url, reward_fn,
                 vllm_engines: List = None, 
                 feedback_model = None, 
                 packing_samples=False, 
                 multimodal=True, 
                 feedback_rewards=None,
                 envs: gym.vector.AsyncVectorEnv = None,
                 env_config: EnvConfig = None,
                 prompt_config: PromptConfig = None,
                 **kwargs):
        # super().__init__(*args, **kwargs)
        super().__init__(
            actor=actor,
            critic=critic,
            reward_model=reward_model,
            initial_model=initial_model,
            tokenizer=tokenizer,
            data_processor=data_processor,
            prompt_max_len=prompt_max_len,
            kl_controller=kl_controller,
            strategy=strategy,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn
        )
        self.envs = envs
        self.num_envs = envs.num_envs
        self.env_config = env_config
        self.prompt_config = prompt_config
        self.vllm_engines = vllm_engines
        self.feedback_model = feedback_model
        self.packing_samples = packing_samples
        self.multimodal = multimodal
        self.feedback_rewards = feedback_rewards
        self.feedback_processor = feedback_data_processor
        self.distillation = self.strategy.args.distillation

        assert self.prompt_config.use_vision == self.multimodal, "prompt_config.use_vision and multimodal must be the same."
        if self.prompt_config.use_vision:
            self.prompt_vision = PROMPT_FN[self.prompt_config.prompt_vision]
            self.pattern_vision = self.prompt_config.pattern_vision
        else:
            self.prompt_language = PROMPT_FN[self.prompt_config.prompt_language]
            self.pattern_language = self.prompt_config.pattern_language
        
        self.target_number = self.env_config.target_points
        self.formulate_oracle_arguments()
        self.custom_reward_func = None
        self.processor = data_processor
        self.history_length = 1
        self.json_pattern = r"```json\n(.*?)\n```"


    def formulate_oracle_arguments(self):
        self.oracle_arguments = {}
        self.oracle_arguments['face_card_msg'] = "'J', 'Q', and 'K' count as '10'." if self.env_config.treat_face_cards_as_10 \
                                        else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
        self.oracle_arguments['target_number'] = str(self.target_number)
        self.oracle_arguments['example_json_text'] = example_json_text
        
    def formulate_vision_arguments(self, vision_res_list, info_batch):
        for i in range(len(vision_res_list)):
            if 'cards' not in vision_res_list[i].keys():
                # hard code gt cards into dict
                vision_res_list[i]['cards'] = info_batch['Plain Cards'][i]
    
    def formulate_prompt(self, task_prompt: str, obs_batch: np.ndarray = None, previous_responses = None, previous_feedbacks = None, previous_verify_infos = None) -> Tuple[List[dict], List[dict]]:
        
        messages = [[] for _ in range(self.num_envs)]
        messages_no_feedback = [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            contents = []
            contents_no_feedback = []
            if len(previous_feedbacks[i]) > 0 and len(previous_responses[i]) > 0 and len(previous_verify_infos[i]) > 0:
                assert len(previous_responses[i]) == len(previous_feedbacks[i]) == len(previous_verify_infos[i]), "The number of previous responses, feedbacks, and verify infos must be the same."
                if obs_batch[i] is not None:
                    pil_image = Image.fromarray(obs_batch[i])
                    contents.append({"type": "image", "image": pil_image})
                    contents_no_feedback.append({"type": "image", "image": pil_image})
                base_prompt = FEEDBACK_PROMPT_BASE_CARD.format(task=task_prompt)
                contents.append({"type": "text", "text": base_prompt})
                contents_no_feedback.append({"type": "text", "text": task_prompt})
                for idx, (prev_response, prev_feedback, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_feedbacks[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append({"type": "text", "text": f"\n## Previous your response ({self.history_length - idx} steps ago):\n{prev_response}"})
                    contents.append({"type": "text", "text": f"\n## Verification\nYou failed this trial because {prev_verify_info}"})
                    contents.append({"type": "text", "text": f"\n## Feedback ({self.history_length - idx} steps ago):\n{prev_feedback}"})
                    contents_no_feedback.append({"type": "text", "text": f"\n## Previous your response ({self.history_length - idx} steps ago):\n{prev_response}"})
                    contents_no_feedback.append({"type": "text", "text": f"\nYou failed this trial because {prev_verify_info}"})
            else:
                if obs_batch[i] is not None:
                    pil_image = Image.fromarray(obs_batch[i])
                    contents.append({"type": "image", "image": pil_image})
                    contents.append({"type": "text", "text": task_prompt})
                    contents_no_feedback.append({"type": "image", "image": pil_image})
                    contents_no_feedback.append({"type": "text", "text": task_prompt})
                else:
                    contents.append({"type": "text", "text": task_prompt})
                    contents_no_feedback.append({"type": "text", "text": task_prompt})
            
            messages[i] = [
                {"role": "user",
                 "content": contents,
                },
            ]
            messages_no_feedback[i] = [
                {"role": "user",
                 "content": contents_no_feedback,
                },
            ]
        return messages, messages_no_feedback

    def formulate_prompt_for_LLMStudent(self, task_prompts: List[str], 
                                        previous_responses: List[List[str]], previous_feedbacks: List[List[str]], previous_verify_infos: List[List[str]]) -> List[dict]:
        
        messages = [[] for _ in range(self.num_envs)]
        messages_no_feedback = [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            contents = []
            contents_no_feedback = []
            
            if len(previous_responses[i]) > 0 and len(previous_verify_infos[i]) > 0 and len(previous_feedbacks[i]) > 0:
                assert len(previous_responses[i]) == len(previous_feedbacks[i]) == len(previous_verify_infos[i]), "The number of previous responses, feedbacks, and verify infos must be the same."
                feedback_prompt = FEEDBACK_PROMPT_BASE_CARD.format(task=task_prompts[i])
                contents.append(feedback_prompt)
                contents_no_feedback.append(feedback_prompt)
                # contents_no_feedback.append(task_prompts[i])
                for idx, (prev_response, prev_feedback, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_feedbacks[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append(f"\n## Previous your answer ({self.history_length - idx} steps ago):\n{prev_response}")
                    contents.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
                    contents.append(f"\n## Feedback ({self.history_length - idx} steps ago):\n{prev_feedback}")
                    contents_no_feedback.append(f"\n## Previous your answer ({self.history_length - idx} steps ago):\n{prev_response}")
                    contents_no_feedback.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
            
            else:
                contents.append(task_prompts[i])
                contents_no_feedback.append(task_prompts[i])
            
            messages[i] = [
                {"role": "user",
                 "content": "\n".join(contents),
                },
            ]
            messages_no_feedback[i] = [
                {"role": "user",
                 "content": "\n".join(contents_no_feedback),
                },
            ]
        return messages, messages_no_feedback
    
    def formulate_feedback_prompt(self, task_prompt: str, obs_batch: List[Image.Image], info_batch: dict,
                         previous_responses: List[List[str]], previous_feedbacks: List[List[str]], previous_verify_infos: List[List[str]]) -> List[dict]:
        
        messages = [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            contents = []
            if obs_batch[i] is not None:
                contents.append({"type": "image", "image": obs_batch[i]})
            feedback_prompt = FEEDBACK_PROMPT_CARD.format(task=task_prompt)
            contents.append({"type": "text", "text": feedback_prompt})
            if (len(previous_responses[i]) == len(previous_verify_infos[i]) == len(previous_feedbacks[i])):
            # if previous_responses[i] is not None and previous_verify_infos[i] is not None and previous_feedbacks[i] is not None:
                assert len(previous_responses[i]) == len(previous_feedbacks[i]) == len(previous_verify_infos[i]), "The number of previous responses, feedbacks, and verify infos must be the same."
                for idx, (prev_response, prev_feedback, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_feedbacks[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append({"type": "text", "text": f"\n## Previous model's answer ({self.history_length - idx} steps ago):\n{prev_response}"})
                    contents.append({"type": "text", "text": f"\n## Verification message\nYou failed this trial because {prev_verify_info}"})
                    contents.append({"type": "text", "text": f"\n## Answer examples\ncards: {info_batch['Plain Cards'][i]}, number: {info_batch['Numbers'][i]}, formula: {info_batch['Solution'][i]}"})
                    contents.append({"type": "text", "text": f"\n## Your previous feedback ({self.history_length - idx} steps ago):\n{prev_feedback}"})
            
            elif (len(previous_responses[i]) == len(previous_verify_infos[i])):
            # elif previous_responses[i] is not None and previous_verify_infos[i] is not None:
                for idx, (prev_response, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append({"type": "text", "text": f"\n## Previous model's answer ({self.history_length - idx} steps ago):\n{prev_response}"})
                    contents.append({"type": "text", "text": f"\n## Verification message\nYou failed this trial because {prev_verify_info}"})
                    contents.append({"type": "text", "text": f"\n## Answer examples\ncards: {info_batch['Plain Cards'][i]}, number: {info_batch['Numbers'][i]}, formula: {info_batch['Solution'][i]}"})
            else:
                raise ValueError("The number of previous responses, feedbacks, and verify infos must be the same.")
            
            contents.append({"type": "text", "text": "Feedback:"})
            
            messages[i] = [
                {"role": "user",
                 "content": contents,
                },
            ]
        return messages
    
    # def formulate_feedback_prompt_for_LLMTeacher(self, task_prompts: List[str], info_batch: dict,
    #                      previous_responses: List[List[str]], previous_feedbacks: List[List[str]], previous_verify_infos: List[List[str]]) -> List[dict]:
        
    #     messages = [[] for _ in range(self.num_envs)]

    #     for i in range(self.num_envs):
    #         contents = []
    #         feedback_prompt = FEEDBACK_PROMPT_CARD.format(task=task_prompts[i]) if not self.multimodal else FEEDBACK_PROMPT_CARD.format(task=task_prompts[0])
    #         contents.append(feedback_prompt)
    #         if (len(previous_responses[i]) == len(previous_verify_infos[i]) == len(previous_feedbacks[i])):
    #             assert len(previous_responses[i]) == len(previous_feedbacks[i]) == len(previous_verify_infos[i]), "The number of previous responses, feedbacks, and verify infos must be the same."
    #             for idx, (prev_response, prev_feedback, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_feedbacks[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
    #                 contents.append(f"\n## Previous Model's answer ({self.history_length - idx} steps ago):\n{prev_response}")
    #                 contents.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
    #                 contents.append(f"\n## Answer examples\ncards: {info_batch['Plain Cards'][i]}, number: {info_batch['Numbers'][i]}, formula: {info_batch['Solution'][i]}")
    #                 # contents.append(f"\n## Your previous feedback ({self.history_length - idx} steps ago):\n{prev_feedback}")
            
    #         elif (len(previous_responses[i]) == len(previous_verify_infos[i])):
    #             for idx, (prev_response, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
    #                 contents.append(f"\n## Previous Model's answer ({self.history_length - idx} steps ago):\n{prev_response}")
    #                 contents.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
    #                 contents.append(f"\n## Answer examples\ncards: {info_batch['Plain Cards'][i]}, number: {info_batch['Numbers'][i]}, formula: {info_batch['Solution'][i]}")
    #         else:
    #             raise ValueError("The number of previous responses, feedbacks, and verify infos must be the same.")
            
    #         contents.append("\nFeedback:")
            
    #         messages[i] = [
    #             {"role": "user",
    #              "content": "\n".join(contents),
    #             },
    #         ]
    #         # messages[i] = [
    #         #     {"role": "user",
    #         #      "content": "\n".join(contents) + FEEDBACK_PROMPT_SUFFIX,
    #         #     },
    #         # ]
    #     return messages

    def formulate_feedback_prompt_for_LLMTeacher(self, task_prompts: List[str], info_batch: dict,
                         previous_responses: List[List[str]], previous_feedbacks: List[List[str]], previous_verify_infos: List[List[str]]) -> List[dict]:
        
        messages = [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            contents = []
            contents.append(FEEDBACK_PROMPT_CARD_V2)
            if (len(previous_responses[i]) == len(previous_verify_infos[i]) == len(previous_feedbacks[i])):
                assert len(previous_responses[i]) == len(previous_feedbacks[i]) == len(previous_verify_infos[i]), "The number of previous responses, feedbacks, and verify infos must be the same."
                for idx, (prev_response, prev_feedback, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_feedbacks[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append(f"\n## Previous Model's answer ({self.history_length - idx} steps ago):\n{prev_response}")
                    contents.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
                    if len(info_batch['Solution'][i]) > 3:
                        answer_samples = random.sample(info_batch['Solution'][i], 3)
                    else:
                        answer_samples = info_batch['Solution'][i]
                    contents.append(f"\n## Formula examples\nformula: {answer_samples}")
                    # contents.append(f"\n## Your previous feedback ({self.history_length - idx} steps ago):\n{prev_feedback}")
            
            elif (len(previous_responses[i]) == len(previous_verify_infos[i])):
                for idx, (prev_response, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append(f"\n## Previous Model's answer ({self.history_length - idx} steps ago):\n{prev_response}")
                    contents.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
                    if len(info_batch['Solution'][i]) > 3:
                        answer_samples = random.sample(info_batch['Solution'][i], 3)
                    else:
                        answer_samples = info_batch['Solution'][i]
                    contents.append(f"\n## Formula examples\nformula: {answer_samples}")
            else:
                raise ValueError("The number of previous responses, feedbacks, and verify infos must be the same.")
            
            contents.append("\nYour feedback:")
            
            messages[i] = [
                {"role": "user",
                 "content": "\n".join(contents),
                },
            ]
        return messages
    
    def formulate_base_prompt_using_feedback(self, task_prompt: str, obs_batch: List[Image.Image],
                         previous_responses: List[List[str]], previous_feedbacks: List[List[str]], previous_verify_infos: List[List[str]]) -> List[dict]:
        
        assert len(previous_responses) == len(previous_feedbacks) == len(previous_verify_infos), "The number of previous responses, feedbacks, and verify infos must be the same."
        messages = [[] for _ in range(len(obs_batch))]

        for i in range(len(obs_batch)):
            contents = []
            if obs_batch[i] is not None:
                contents.append({"type": "image", "image": obs_batch[i]})
            base_prompt = FEEDBACK_PROMPT_BASE_CARD.format(task=task_prompt)
            contents.append({"type": "text", "text": base_prompt})
            for idx, (prev_response, prev_feedback, prev_verify_info) in enumerate(zip(previous_responses[i], previous_feedbacks[i], previous_verify_infos[i])):
                contents.append({"type": "text", "text": f"\n## Previous model's answer ({len(previous_responses[i]) - idx} steps ago):\n{prev_response}"})
                contents.append({"type": "text", "text": f"\n## Verification message\nYou failed this trial because {prev_verify_info}"})
                contents.append({"type": "text", "text": f"\n## Your previous feedback ({len(previous_responses[i]) - idx} steps ago):\n{prev_feedback}"})
            
            contents.append({"type": "text", "text": RESPONSE_PROMPT_CARD})
            
            messages[i] = [
                {"role": "user",
                 "content": contents,
                },
            ]
        return messages
    
    @torch.no_grad()
    def collect_trajectories(self, episode_id: int=None, **generate_kwargs) -> deque[RolloutStorage]:
        """
        Rollout the environment and return the experiences.
        """

        ## returns of envs.reset() ##
        ## obs_batch: numpy with shape (num_envs, resolution, resolution, 3).
        ## info_batch.keys() = ['Cards', '_Cards', 'Plain Cards', '_Plain Cards', 'Numbers', '_Numbers', 
        ##                      'Formula', '_Formula', 'Solution', '_Solution', 'Remaining Numbers', '_Remaining Numbers', 'Remaining Step', 
        ##                      '_Remaining Step', 'Verify Info', '_Verify Info']
        obs_batch, info_batch = self.envs.reset()

        previous_responses = [[] for _ in range(self.num_envs)]
        previous_feedbacks = [[] for _ in range(self.num_envs)]
        previous_verify_infos = [[] for _ in range(self.num_envs)]
        previous_rewards = [[] for _ in range(self.num_envs)]

        if self.prompt_config.use_vision:
            prompt, _ = self.prompt_vision, self.pattern_vision
        else:
            prompt, _ = self.prompt_language, self.pattern_language
        
        replay_buffer = deque(maxlen=self.env_config.num_steps + 1)
        episode_start = np.zeros(self.num_envs, dtype=bool)

        
        if self.strategy.is_rank_0() and self.strategy.args.log:
            logs = []

        
        for step in tqdm(range(self.env_config.num_steps), desc="Collecting trajectories", disable=not self.strategy.is_rank_0()):
            vision_res_list = [{} for _ in range(self.num_envs)]
            language_res_list = [{} for _ in range(self.num_envs)]
            self.formulate_vision_arguments(vision_res_list, info_batch)
            task_prompts = [prompt.format(**vision_res_list[i], **language_res_list[i], **self.oracle_arguments) for i in range(self.num_envs)]
            if self.multimodal:
                messages, messages_no_feedback = self.formulate_prompt(task_prompts[0], 
                                    obs_batch=obs_batch,
                                    previous_responses=previous_responses,
                                    previous_feedbacks=previous_feedbacks,
                                    previous_verify_infos=previous_verify_infos)
            else:
                messages, messages_no_feedback = self.formulate_prompt_for_LLMStudent(task_prompts, 
                                    previous_responses=previous_responses,
                                    previous_feedbacks=previous_feedbacks,
                                    previous_verify_infos=previous_verify_infos)
            
            # base model inference w/o feedback
            if all(len(feedback) == 0 for feedback in previous_feedbacks):
                if self.multimodal:
                    mini_batch = self._generate_vllm(obs_batch, messages, self.multimodal, **generate_kwargs)
                else:
                    mini_batch = self._generate_vllm_language(messages, **generate_kwargs)
            else:
                # base model inference w/o feedback
                if self.multimodal:
                    mini_batch = self._generate_vllm_with_feedback(messages, messages_no_feedback, obs_batch, self.multimodal, **generate_kwargs)
                else:
                    mini_batch = self._generate_vllm_with_feedback_language(messages, messages_no_feedback, **generate_kwargs)
            
            # preprocessing the model response to align with json style.
            for i, model_response in enumerate(mini_batch.output_text):
                # remove <|im_end|> because the env requires a pure json format.
                if "<|im_end|>" in model_response:
                    mini_batch.output_text[i] = model_response.replace("<|im_end|>", "")
                # handle the case that the model provides ```json ...``` format as recent models do.
                try:
                    match = re.search(self.json_pattern, model_response, re.DOTALL)
                    if match:
                        mini_batch.output_text[i] = match.group(1)
                except:
                    pass
                
            obs_batch, rewards, terminations, truncations, info_batch = self.envs.step(mini_batch.output_text)
            episode_start = np.logical_or(terminations, truncations)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            terminations_tensor = torch.tensor(terminations, dtype=torch.bool)
            truncations_tensor = torch.tensor(truncations, dtype=torch.bool)
            # Calculate masks (1 if episode continues, 0 if it ended). Use .float() to convert boolean tensor to float (0.0 or 1.0)
            masks_tensor = 1.0 - terminations_tensor.float()
            # Calculate bad_masks (commonly 1 if episode did NOT truncate, 0 if it did)
            bad_masks_tensor = 1.0 - truncations_tensor.float()

            mini_batch.rewards = rewards_tensor     # Shape: (num_envs,)
            mini_batch.returns = rewards_tensor
            mini_batch.masks = masks_tensor         # Shape: (num_envs,)
            mini_batch.bad_masks = bad_masks_tensor # Shape: (num_envs,)
            reward_diff = torch.zeros_like(rewards_tensor, dtype=torch.float32)
            for i in range(self.num_envs):
                if len(previous_rewards[i]) > 0:
                    reward_diff[i] = rewards[i] - previous_rewards[i][-1] # until 0508
                    # if previous_rewards[i][-1] != 0:
                    #     reward_diff[i] = (rewards[i] - previous_rewards[i][-1]) / math.fabs(previous_rewards[i][-1])
                    # else:
                    #     reward_diff[i] = rewards[i] - previous_rewards[i][-1]
                else:
                    reward_diff[i] = rewards[i]
            mini_batch.reward_diff = reward_diff
            replay_buffer.append(copy.deepcopy(mini_batch))
            
            for i, model_response in enumerate(mini_batch.output_text):
                previous_responses[i].append(model_response)
            for i in range(self.num_envs):
                previous_verify_infos[i].append(info_batch["Verify Info"][i])
                previous_rewards[i].append(rewards[i])

            # get feedbacks from teacher model
            # feedbacks = self.get_feedbacks(task_prompt, obs_batch, info_batch, previous_responses, previous_verify_infos, previous_feedbacks, self.multimodal, **generate_kwargs)
            feedbacks, feedback_prompts = self.get_feedbacks_from_LLMTeacher(task_prompts, info_batch, previous_responses, previous_verify_infos, previous_feedbacks, self.multimodal, **generate_kwargs)

            for i, feedback in enumerate(feedbacks):
                previous_feedbacks[i].append(feedback)

            # Clear the history for environment i
            for i in range(self.num_envs):
                # if episode_start[i]:
                if terminations[i] or truncations[i]:
                    previous_responses[i] = []
                    previous_feedbacks[i] = []
                    previous_verify_infos[i] = []
                    previous_rewards[i] = []


            if self.strategy.is_rank_0() and self.strategy.args.log:
                log = {
                    "step": step,
                    "prompt": mini_batch.prompt[0],
                    "model_response": mini_batch.output_text[0],
                    "verify_info": info_batch["Verify Info"][0],
                    "feedback": feedbacks[0],
                    "reward": rewards[0],
                    "gt_cards": info_batch["Plain Cards"][0],
                    "feedback_prompt": feedback_prompts[0][0]["content"],
                }
                logs.append(log)

        
        if self.strategy.is_rank_0() and self.strategy.args.log:
            log_dir = self.strategy.args.output_log_dir
            log_file = os.path.join(log_dir, f"episode_{episode_id}.json" if episode_id is not None else "log.json")
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        
        return replay_buffer

    def compute_and_store_returns_in_buffer(self, replay_buffer: Deque[RolloutStorage], gamma: float = 0.9) -> None:
        """
        Computes discounted cumulative returns and stores them *in-place*
        within the 'returns' field of each RolloutStorage object in the deque.

        Args:
            replay_buffer: A deque containing RolloutStorage objects.
                        Each object must have 'rewards' and 'masks' attributes
                        (assumed to be tensors or convertible to tensors) and
                        an assignable 'returns' attribute.
            gamma: The discount factor.
        """
        if not replay_buffer:
            print("Warning: Replay buffer is empty. No returns computed.")
            return # Nothing to do

        buffer_len = len(replay_buffer)
        # Infer num_envs and device from the first element
        first_rollout = replay_buffer[0]
        if not isinstance(first_rollout, RolloutStorage):
            raise TypeError("replay_buffer must contain RolloutStorage instances.")
        if first_rollout.rewards is None or first_rollout.masks is None:
            raise ValueError("First RolloutStorage in buffer missing 'rewards' or 'masks'. Cannot proceed.")

        try:
            # Assume rewards/masks are tensors or tensor-like (e.g., lists of numbers)
            rewards_tensor = torch.as_tensor(first_rollout.rewards)
            num_envs = rewards_tensor.shape[0]
            device = rewards_tensor.device
        except (TypeError, IndexError, AttributeError) as e:
            raise ValueError(f"Could not determine num_envs or device from first rollout's rewards. Error: {e}")

        # Initialize the return for the step *after* the last one in the buffer
        next_return = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Iterate backwards through the buffer (from T-1 down to 0)
        for t in reversed(range(buffer_len)):
            rollout_storage_t = replay_buffer[t]

            if not isinstance(rollout_storage_t, RolloutStorage):
                print(f"Warning: Element at index {t} is not a RolloutStorage object. Skipping.")
                continue
            if rollout_storage_t.rewards is None or rollout_storage_t.masks is None:
                print(f"Warning: RolloutStorage at index {t} missing 'rewards' or 'masks'. Skipping return calculation for this step.")
                # Set returns to NaN or zeros, or handle as error? Setting to zeros for safety.
                rollout_storage_t.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
                next_return = torch.zeros(num_envs, dtype=torch.float32, device=device) # Reset future return
                continue

            # Ensure rewards and masks are tensors on the correct device
            rewards_t = torch.as_tensor(rollout_storage_t.rewards, dtype=torch.float32, device=device)
            masks_t = torch.as_tensor(rollout_storage_t.masks, dtype=torch.float32, device=device) # mask=1 if not done

            if rewards_t.shape[0] != num_envs or masks_t.shape[0] != num_envs:
                print(f"Warning: Inconsistent num_envs at step {t}. Expected {num_envs}, got rewards={rewards_t.shape[0]}, masks={masks_t.shape[0]}. Skipping.")
                rollout_storage_t.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
                next_return = torch.zeros(num_envs, dtype=torch.float32, device=device) # Reset future return
                continue

            # Calculate returns for the current step t: G_t = r_{t+1} + gamma * G_{t+1} * mask_t
            current_returns = rewards_t + gamma * next_return * masks_t
            # --- Store the calculated returns IN-PLACE ---
            rollout_storage_t.returns = current_returns
            # Update next_return for the previous step (t-1)
            next_return = current_returns
        print(f"Successfully computed and stored returns in {buffer_len} buffer elements.")

    
    @torch.no_grad()
    def make_experience_list(self, episode_id: int=None, **generate_kwargs) -> List[Experience_CARDGAME]:
        
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            batch_vllm_engine_call(self.feedback_model, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        replay_buffer = self.collect_trajectories(episode_id, **generate_kwargs)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
            batch_vllm_engine_call(self.feedback_model, "sleep")
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        # compute cumulative returns
        self.compute_and_store_returns_in_buffer(replay_buffer, generate_kwargs["gamma"])

        all_experiences = []
        for mini_batch in tqdm(replay_buffer, total=len(replay_buffer), desc="make_experience", disable=not self.strategy.is_rank_0()):
            if self.multimodal:
                experience = self.make_experience(mini_batch)
            else:
                experience = self.make_experience_language(mini_batch)
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
            all_experiences.append(experience)

        return all_experiences

    def split_into_chunks(self, data, chunk_size):
        """Split data into chunks of specified size."""
        if isinstance(data, torch.Tensor):
            return torch.split(data, chunk_size)
        elif isinstance(data, dict):
            return [{k: v[i:i+chunk_size] for k, v in data.items()} for i in range(0, len(next(iter(data.values()))), chunk_size)]
        else:
            return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    @torch.no_grad()
    def make_experience(self, mini_batch: RolloutStorage) -> Experience_CARDGAME:
        """
        Turn samples into experience by calculating logprobs, rewards, and kl divergence.
        The size of each element in vllm_outputs corresponds to self.strategy.args.micro_rollout_batch_size.

        This function does the following:
        1. Get log_probs of the initial_model and base model to compute KL distance to refine reward values
        2. Pack the above information into Experience dataclass
        """
        start_time = time.time()
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        chunk_size = self.strategy.args.micro_train_batch_size

        # Extract values from samples
        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        action_mask = mini_batch.action_mask
        packed_seq_lens = mini_batch.packed_seq_lens
        visual_inputs = self.make_input_batch(mini_batch.visual_inputs)

        # Split data into chunks
        sequences_chunks = self.split_into_chunks(sequences, chunk_size)
        attention_mask_chunks = self.split_into_chunks(attention_mask, chunk_size)
        action_mask_chunks = self.split_into_chunks(action_mask, chunk_size)
        packed_seq_lens_chunks = self.split_into_chunks(packed_seq_lens, chunk_size) if packed_seq_lens is not None else [None] * len(sequences_chunks)
        visual_inputs_chunks = [self.make_input_batch(mini_batch.visual_inputs[i:i+chunk_size]) for i in range(0, len(mini_batch.visual_inputs), chunk_size)]

        # Move current chunk to CPU for remote processing
        seq_chunk_cpu_list = [seq_chunk.to("cpu") for seq_chunk in sequences_chunks]
        attn_chunk_cpu_list = [attn_chunk.to("cpu") for attn_chunk in attention_mask_chunks]
        action_mask_chunk_cpu_list = [action_mask_chunk.to("cpu") for action_mask_chunk in action_mask_chunks]
        vis_inputs_chunk_cpu_list = [{k: v.to("cpu") for k, v in vis_inputs_chunk.items() if k != "input_ids" and k != "attention_mask"} for vis_inputs_chunk in visual_inputs_chunks]
        logps_allgather_list = [True] * len(seq_chunk_cpu_list)
        
        
        # Process initial model chunk
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=seq_chunk_cpu_list,
                action_mask=action_mask_chunk_cpu_list,
                attention_mask=attn_chunk_cpu_list,
                logps_allgather=logps_allgather_list,
                visual_inputs=vis_inputs_chunk_cpu_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(seq_chunk_cpu_list))
        
        
        # Initialize lists to store results
        all_action_log_probs = []
        all_base_action_log_probs = []

        # Process each chunk
        for i in range(len(sequences_chunks)):
            # Clear GPU cache before processing new chunk
            torch.cuda.empty_cache()

            # Get current chunk
            seq_chunk = sequences_chunks[i]
            attn_chunk = attention_mask_chunks[i]
            action_mask_chunk = action_mask_chunks[i]
            packed_seq_chunk = packed_seq_lens_chunks[i]
            vis_inputs_chunk = visual_inputs_chunks[i]

            # Process actor model chunk
            actor_vis_inputs = None if vis_inputs_chunk is None else {k: v.to(device) for k, v in vis_inputs_chunk.items() if k != "input_ids" and k != "attention_mask"}
            
            action_log_probs_chunk = self.actor(
                seq_chunk.to(device),
                action_mask_chunk,
                attn_chunk.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=True,
                packed_seq_lens=packed_seq_chunk,
                visual_inputs=actor_vis_inputs,
            )

            # Store results
            all_action_log_probs.append(action_log_probs_chunk)

            # Clear memory
            del seq_chunk, attn_chunk, action_mask_chunk, packed_seq_chunk, vis_inputs_chunk
            torch.cuda.empty_cache()

        
        base_action_log_probs_list = ray.get([base_action_log_probs_ref])[0]
        if base_action_log_probs_list[0] is not None:
            base_action_log_probs_list = [base_action_log_probs_list[i].to(device) for i in range(len(base_action_log_probs_list))]
        base_action_log_probs = torch.cat(base_action_log_probs_list, dim=0)
        
        # # Concatenate results
        action_log_probs = torch.cat(all_action_log_probs, dim=0)

        actor_value_rm_time = time.time() - start_time
        start = time.time()
        wait_time = time.time() - start

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        rewards = mini_batch.rewards
        r = rewards.to(device)

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=mini_batch.action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        if not self.packing_samples:
            kl_mean = masked_mean(kl, mini_batch.action_mask, dim=-1)
        else:
            num_actions = mini_batch.num_actions
            packed_seq_lens = mini_batch.packed_seq_lens
            if self.strategy.ring_attn_group is not None:
                assert mini_batch.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=mini_batch.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=None,
                    kl=kl,
                )
            # Convert tensor into list of tensors for easier manipulation within dataset
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "return": mini_batch.returns,
            "response_length": mini_batch.response_length,
            "total_length": mini_batch.total_length,
            "num_actions": mini_batch.num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience_CARDGAME(
            sequences,
            action_log_probs,
            base_action_log_probs,
            values=None,
            returns=mini_batch.returns,
            advantages=mini_batch.returns,
            attention_mask=attention_mask,
            action_mask=mini_batch.action_mask,
            info=info,
            kl=kl,
            visual_inputs=visual_inputs,
            sequences_for_KL=mini_batch.sequences_for_KL if self.distillation else None,
            attention_mask_for_KL=mini_batch.attention_mask_for_KL if self.distillation else None,
            action_mask_for_KL=mini_batch.action_mask_for_KL if self.distillation else None,
            reward_diff=mini_batch.reward_diff if self.distillation else None,
        )

        self.actor.train()  # Reset model state
        return experience
    
    @torch.no_grad()
    def make_experience_language(self, mini_batch: RolloutStorage) -> Experience_CARDGAME:
        """
        Turn samples into experience by calculating logprobs, rewards, and kl divergence.
        The size of each element in vllm_outputs corresponds to self.strategy.args.micro_rollout_batch_size.

        This function does the following:
        1. Get log_probs of the initial_model and base model to compute KL distance to refine reward values
        2. Pack the above information into Experience dataclass
        """
        start_time = time.time()
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        chunk_size = self.strategy.args.micro_train_batch_size

        # Extract values from samples
        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        action_mask = mini_batch.action_mask
        packed_seq_lens = mini_batch.packed_seq_lens

        # Split data into chunks
        sequences_chunks = self.split_into_chunks(sequences, chunk_size)
        attention_mask_chunks = self.split_into_chunks(attention_mask, chunk_size)
        action_mask_chunks = self.split_into_chunks(action_mask, chunk_size)
        packed_seq_lens_chunks = self.split_into_chunks(packed_seq_lens, chunk_size) if packed_seq_lens is not None else [None] * len(sequences_chunks)

        # Move current chunk to CPU for remote processing
        seq_chunk_cpu_list = [seq_chunk.to("cpu") for seq_chunk in sequences_chunks]
        attn_chunk_cpu_list = [attn_chunk.to("cpu") for attn_chunk in attention_mask_chunks]
        action_mask_chunk_cpu_list = [action_mask_chunk.to("cpu") for action_mask_chunk in action_mask_chunks]
        logps_allgather_list = [True] * len(seq_chunk_cpu_list)
        
        
        # Process initial model chunk
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=seq_chunk_cpu_list,
                action_mask=action_mask_chunk_cpu_list,
                attention_mask=attn_chunk_cpu_list,
                logps_allgather=logps_allgather_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(seq_chunk_cpu_list))
        
        
        # # values
        # if self.critic is not None:
        #     value_ref = self.critic.forward_batch.remote(
        #         sequences=seq_chunk_cpu_list,
        #         action_mask=action_mask_chunk_cpu_list,
        #         attention_mask=attn_chunk_cpu_list,
        #     )
        #     # avoid CUDA OOM when colocate models
        #     if args.colocate_critic_reward or args.colocate_all_models:
        #         ray.get([value_ref])
        #         ray.get([self.critic.empty_cache.remote()])
        # else:
        #     value_ref = ray.put([None] * len(seq_chunk_cpu_list))
        
        
        # Initialize lists to store results
        all_action_log_probs = []

        # Process each chunk
        for i in range(len(sequences_chunks)):
            torch.cuda.empty_cache()

            # Get current chunk
            seq_chunk = sequences_chunks[i]
            attn_chunk = attention_mask_chunks[i]
            action_mask_chunk = action_mask_chunks[i]
            packed_seq_chunk = packed_seq_lens_chunks[i]

            action_log_probs_chunk = self.actor(
                seq_chunk.to(device),
                action_mask_chunk,
                attn_chunk.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=True,
                packed_seq_lens=packed_seq_chunk,
            )

            # Store results
            all_action_log_probs.append(action_log_probs_chunk)

            # Clear memory
            del seq_chunk, attn_chunk, action_mask_chunk, packed_seq_chunk
            torch.cuda.empty_cache()

        
        base_action_log_probs_list = ray.get([base_action_log_probs_ref])[0]
        if base_action_log_probs_list[0] is not None:
            base_action_log_probs_list = [base_action_log_probs_list[i].to(device) for i in range(len(base_action_log_probs_list))]
        base_action_log_probs = torch.cat(base_action_log_probs_list, dim=0)
        
        # # values
        # values_list = ray.get([value_ref])[0]
        # if values_list[0] is not None:
        #     values_list = [values_list[i].to(device) for i in range(len(values_list))]
        # values = torch.cat(values_list, dim=0)
        
        # # Concatenate results
        action_log_probs = torch.cat(all_action_log_probs, dim=0)

        actor_value_rm_time = time.time() - start_time
        start = time.time()
        wait_time = time.time() - start

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        rewards = mini_batch.rewards
        r = rewards.to(device)

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=mini_batch.action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        if not self.packing_samples:
            kl_mean = masked_mean(kl, mini_batch.action_mask, dim=-1)
        else:
            num_actions = mini_batch.num_actions
            packed_seq_lens = mini_batch.packed_seq_lens
            if self.strategy.ring_attn_group is not None:
                assert mini_batch.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=mini_batch.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=None,
                    kl=kl,
                )
            # Convert tensor into list of tensors for easier manipulation within dataset
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "return": mini_batch.returns,
            "response_length": mini_batch.response_length,
            "total_length": mini_batch.total_length,
            "num_actions": mini_batch.num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience_CARDGAME(
            sequences,
            action_log_probs,
            base_action_log_probs,
            values=None,
            returns=mini_batch.returns,
            advantages=mini_batch.returns,
            attention_mask=attention_mask,
            action_mask=mini_batch.action_mask,
            info=info,
            kl=kl,
            visual_inputs=None,
            sequences_for_KL=mini_batch.sequences_for_KL if self.distillation else None,
            attention_mask_for_KL=mini_batch.attention_mask_for_KL if self.distillation else None,
            action_mask_for_KL=mini_batch.action_mask_for_KL if self.distillation else None,
            reward_diff=mini_batch.reward_diff,
        )

        self.actor.train()  # Reset model state
        return experience
    

    def _generate_vllm(self, obs_batch:List[np.ndarray], messages: List[dict], multimodal=True, **kwargs) -> RolloutStorage:
        """
        Create prompts for base model inference and give them to vLLM, and return the model responses.

        all_prompts: List[dict], each dict is like:
        {
            "role": "user",
            "content": contents,
        }
        """
        
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", self.strategy.args.generate_max_len),
            min_tokens=kwargs.get("min_new_tokens", 16),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        prompt_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        visual_inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.prompt_max_len
        )
        visual_inputs_chunks = self.split_input_batch(visual_inputs)
        visual_inputs = []
        for visual_inputs_chunk in visual_inputs_chunks:
            visual_inputs_chunk.pop("input_ids")
            visual_inputs_chunk.pop("attention_mask")
            # visual_inputs_chunk = {k: v.to("cuda") for k, v in visual_inputs_chunk.items()}
            visual_inputs.append(visual_inputs_chunk)
        # visual_inputs.pop("input_ids")
        # visual_inputs.pop("attention_mask")
        # visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()} # 'pixel_values', 'image_grid_thw'

        # # Expand prompt list based on the number of samples per prompt
        # all_prompts = sum([[prompt] for prompt in prompt_texts], [])
        batch_size = (len(messages) + len(llms) - 1) // len(llms)

        # Prepare inputs for vLLM
        refs = []
        vllm_inputs = []
        if multimodal:
            for i, llm in enumerate(llms):
                msg_batch = messages[i * batch_size : (i + 1) * batch_size]
                obs_batch_slice = obs_batch[i * batch_size : (i + 1) * batch_size]
                prompts = self.processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=True)
                vllm_inputs = [
                    {
                        # "prompt": prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>'),
                        "prompt": prompt,
                        "multi_modal_data": {"image": obs},
                        "mm_processor_kwargs": kwargs["processor_kwargs"]
                    }
                    for prompt, obs in zip(prompts, obs_batch_slice)
                ]
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                )
        else:
            for i, llm in enumerate(llms):
                for msg in messages[i * batch_size : (i + 1) * batch_size]:
                    prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    vllm_inputs.append({"prompt": prompt})
                    refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))

        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        model_responses_list = []
        for output in all_outputs:
            model_responses_list.append(output.outputs[0].text)

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in all_outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for j, output in enumerate(all_outputs):
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)
            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        sequences = sequences.to("cuda")
        attention_mask = attention_mask.to("cuda")
        action_mask = action_mask.to("cuda")

        self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())

        # for logging
        prompt_texts_img_pad_removed = [prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>') for prompt in prompt_texts]

        mini_batch = RolloutStorage(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        visual_inputs=visual_inputs,
                        pad_len=None,
                        output_text=model_responses_list,
                        obs=obs_batch,
                        rewards=None,
                        masks=None,
                        bad_masks=None,
                        action_log_probs=None,
                        prompt=prompt_texts_img_pad_removed,
        )

        return mini_batch

    def _generate_vllm_language(self, messages: List[dict], **kwargs) -> RolloutStorage:
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", self.strategy.args.generate_max_len),
            min_tokens=kwargs.get("min_new_tokens", 16),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        # sampling_params = SamplingParams(
        #     temperature=0.7,
        #     top_p=0.8,
        #     top_k=20,
        #     max_tokens=kwargs.get("max_new_tokens", self.strategy.args.generate_max_len),
        #     min_tokens=kwargs.get("min_new_tokens", 16),
        #     skip_special_tokens=kwargs.get("skip_special_tokens", False),
        #     include_stop_str_in_output=True,
        # )

        batch_size = (len(messages) + len(llms) - 1) // len(llms)

        refs = []
        prompts = []
        for i, llm in enumerate(llms):
            msg = messages[i * batch_size : (i + 1) * batch_size]
            if "Qwen3" in self.strategy.args.pretrain:
                prompt = self.processor.apply_chat_template(msg, tokenize=False, enable_thinking=False, add_generation_prompt=True)
            else:
                prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            flattened_prompts = [p for p in prompt]
            prompts.extend(flattened_prompts)
            refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=prompt))

        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        model_responses_list = []
        for output in all_outputs:
            model_responses_list.append(output.outputs[0].text)

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in all_outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for j, output in enumerate(all_outputs):
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)
            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        sequences = sequences.to("cuda")
        attention_mask = attention_mask.to("cuda")
        action_mask = action_mask.to("cuda")

        self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())

        mini_batch = RolloutStorage(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        visual_inputs=None,
                        pad_len=None,
                        output_text=model_responses_list,
                        obs=None,
                        rewards=None,
                        masks=None,
                        bad_masks=None,
                        action_log_probs=None,
                        prompt=prompts,
                        sequences_for_KL=sequences if self.distillation else None,
                        attention_mask_for_KL=attention_mask if self.distillation else None,
                        action_mask_for_KL=action_mask if self.distillation else None,
        )

        return mini_batch
    
    def get_feedbacks(self, task_prompt: str, obs_batch: List[Image.Image], info_batch: dict,
                      previous_responses: List[List[str]], previous_verify_infos: List[List[str]], previous_feedbacks: List[List[str]], 
                      multimodal=True, **kwargs) -> List[str]:
        """
        1. Extract model responses from output_text (i.e, extract "Answer: ..." from "Thought: ..." and "Answer: ...")
        2. Create prompts for feedback model
        3. Expand prompt list based on the number of samples per prompt
        4. Distribute requests to engines and collect responses to outputs
        5. Return feedbacks
        """
        self.response_length_list = []
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size
        if len(self.feedback_model) <= world_size:
            llms = [self.feedback_model[rank % len(self.feedback_model)]]
        else:
            llms = self.feedback_model[rank::world_size]

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            min_tokens=1,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
        )

        feedback_prompts = self.formulate_feedback_prompt(task_prompt, obs_batch, info_batch, previous_responses, previous_feedbacks, previous_verify_infos)
        batch_size = (len(feedback_prompts) + len(llms) - 1) // len(llms)

        # Prepare inputs for vLLM
        refs = []
        vllm_inputs = []
        if multimodal:
            for i, llm in enumerate(llms):
                msg_batch = feedback_prompts[i * batch_size : (i + 1) * batch_size]
                obs_batch_slice = obs_batch[i * batch_size : (i + 1) * batch_size]
                prompts = self.processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=True)
                vllm_inputs = [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": obs},
                        "mm_processor_kwargs": kwargs["processor_kwargs"]
                    }
                    for prompt, obs in zip(prompts, obs_batch_slice)
                ]
                refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))

        else:
            for i, llm in enumerate(llms):
                for msg in feedback_prompts[i * batch_size : (i + 1) * batch_size]:
                    prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    vllm_inputs.append({"prompt": prompt})
                    refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))

        ray.get(refs)

        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        feedbacks = []
        for output in all_outputs:
            feedbacks.append(output.outputs[0].text)
        
        return feedbacks
    
    def get_feedbacks_from_LLMTeacher(self, task_prompts: List[str], info_batch: dict,
                      previous_responses: List[List[str]], previous_verify_infos: List[List[str]], previous_feedbacks: List[List[str]], 
                      multimodal=True, **kwargs) -> List[str]:
        """
        1. Extract model responses from output_text (i.e, extract "Answer: ..." from "Thought: ..." and "Answer: ...")
        2. Create prompts for feedback model
        3. Expand prompt list based on the number of samples per prompt
        4. Distribute requests to engines and collect responses to outputs
        5. Return feedbacks
        """
        self.response_length_list = []
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size
        if len(self.feedback_model) <= world_size:
            llms = [self.feedback_model[rank % len(self.feedback_model)]]
        else:
            llms = self.feedback_model[rank::world_size]

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=30,
            min_p=0.0,
            max_tokens=512,
            min_tokens=1,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
        )

        feedback_prompts = self.formulate_feedback_prompt_for_LLMTeacher(task_prompts, info_batch, previous_responses, previous_feedbacks, previous_verify_infos)
        batch_size = (len(feedback_prompts) + len(llms) - 1) // len(llms)

        refs = []
        for i, llm in enumerate(llms):
            msg = feedback_prompts[i * batch_size : (i + 1) * batch_size]
            # prompt = self.feedback_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            if "Qwen3" in self.strategy.args.feedback_model:
                prompt = self.feedback_processor.apply_chat_template(msg, tokenize=False, enable_thinking=False, add_generation_prompt=True)
            else:
                prompt = self.feedback_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=prompt))

        ray.get(refs)

        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        feedbacks = []
        for output in all_outputs:
            feedbacks.append(output.outputs[0].text)
        
        return feedbacks, feedback_prompts
    
    def _generate_vllm_with_feedback(
        self, 
        messages: List[dict],
        messages_no_feedback: List[dict],
        obs_batch: np.ndarray,
        multimodal=True, 
        **kwargs) -> RolloutStorage:
        """
        Generate new samples with feedbacks from the base model.
        For RL training, concat the original prompt ids and the new output ids.
        """
        
        self.response_length_list = []
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", self.strategy.args.generate_max_len),
            min_tokens=kwargs.get("min_new_tokens", 16),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        feedback_prompts_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        visual_inputs = self.processor(
            text=feedback_prompts_texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.prompt_max_len
        )
        visual_inputs_chunks = self.split_input_batch(visual_inputs)
        visual_inputs = []
        for visual_inputs_chunk in visual_inputs_chunks:
            visual_inputs_chunk.pop("input_ids")
            visual_inputs_chunk.pop("attention_mask")
            visual_inputs_chunk = {k: v.to("cuda") for k, v in visual_inputs_chunk.items()}
            visual_inputs.append(visual_inputs_chunk)
        # visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()} # 'pixel_values', 'image_grid_thw'

        base_prompts_ids = [
            self.processor.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
            for msg in messages_no_feedback
        ]
        # all_feedback_prompts = sum([[prompt] for prompt in feedback_prompts_texts], [])
        base_prompts_ids = [base_prompt_ids[0] for base_prompt_ids in base_prompts_ids]
        batch_size = (len(messages) + len(llms) - 1) // len(llms)

        refs = []
        vllm_inputs = []
        if multimodal:
            for i, llm in enumerate(llms):
                msg_batch = messages[i * batch_size : (i + 1) * batch_size]
                obs_batch_slice = obs_batch[i * batch_size : (i + 1) * batch_size]
                # base_msg_batch = messages_no_feedback[i * batch_size : (i + 1) * batch_size]
                prompts = self.processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=True)
                vllm_inputs = [
                    {
                        # "prompt": prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>'),
                        "prompt": prompt,
                        "multi_modal_data": {"image": obs},
                    }
                    for prompt, obs in zip(prompts, obs_batch_slice)
                ]
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                )
                # for prompt, obs in zip(all_feedback_prompts[i * batch_size : (i + 1) * batch_size], obs_batch[i * batch_size : (i + 1) * batch_size]):
                #     vllm_inputs.append({
                #         "prompt": prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>'),
                #         "multi_modal_data": {"image": obs},
                #         "mm_processor_kwargs": kwargs["processor_kwargs"]
                #     })
                    
        else:
            for i, llm in enumerate(llms):
                for msg in messages[i * batch_size : (i + 1) * batch_size]:
                    prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    vllm_inputs.append({"prompt": prompt})
                    refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))

        ray.get(refs)

        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        output_list = []
        for output in all_outputs:
            output_list.append(output.outputs[0].text)

        if not self.packing_samples or self.multimodal:
            # NOTE: concat all outputs to following format:
            #
            # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
            # | token token token token token | token token [EOS] [PAD] |
            # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
            # |<---------- prompt ----------->|<-------- answer ------->|
            max_input_len, max_output_len = 0, 0
            for output in all_outputs:
                max_input_len = max(max_input_len, len(output.prompt_token_ids))
                max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            sequences = []
            for j, output in enumerate(all_outputs):
                # left padding input
                input_len = len(base_prompts_ids[j])
                input_ids = [pad_token_id] * (max_input_len - input_len) + list(base_prompts_ids[j])
                # right padding output
                output_len = len(output.outputs[0].token_ids)
                output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
                # concat input and output
                sequences.append(input_ids + output_ids)

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = self.actor.process_sequences(
                sequences, max_input_len, eos_token_id, pad_token_id
            )
            sequences = sequences.to("cuda")
            attention_mask = attention_mask.to("cuda")
            action_mask = action_mask.to("cuda")
            
            # visual_inputs = self.processor(feedback_prompts_texts, self.prompt_max_len, device="cuda")
            # visual_inputs.pop("input_ids")
            # visual_inputs.pop("attention_mask")
            # visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()} # 'pixel_values', 'image_grid_thw'
            self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())
            
            # for logging
            prompt_texts_img_pad_removed = [prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>') for prompt in feedback_prompts_texts]
            
            mini_batch = RolloutStorage(
                            sequences=sequences,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                            visual_inputs=visual_inputs,
                            pad_len=None,
                            output_text=output_list,
                            obs=obs_batch,
                            rewards=None,
                            masks=None,
                            bad_masks=None,
                            action_log_probs=None,
                            prompt = prompt_texts_img_pad_removed,
            )
        # else:
        #     # NOTE: concat all outputs to following format:
        #     #
        #     # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
        #     # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
        #     # This will lead to better inference performance in terms of effificency.
        #     pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        #     sequences = []
        #     packed_seq_lens = []
        #     attention_mask = []
        #     num_actions = []
        #     for j, output in enumerate(outputs):
        #         input_len = len(base_prompt_ids[j])
        #         output_len = len(output.outputs[0].token_ids)
        #         packed_seq_lens.append(input_len + output_len)
        #         sequences.extend(base_prompt_ids[j] + list(output.outputs[0].token_ids))
        #         # sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))  # org imple. for standard RL
        #         attention_mask.extend([j + 1] * (input_len + output_len))
        #         num_actions.append(max(1, output_len))

        #     # pad seq makes the sequence a multiple of ring_attention_size.
        #     pad_len = None
        #     if self.strategy.ring_attn_group is not None:
        #         pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
        #             sequences=sequences,
        #             attention_mask=attention_mask,
        #             num_actions=num_actions,
        #             packed_seq_lens=packed_seq_lens,
        #             ring_attn_group=self.strategy.ring_attn_group,
        #             pad_token_id=pad_token_id,
        #         )

        #     sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
        #     attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
        #     action_mask = None
        #     response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
        #     self.response_length_list.extend(num_actions)
        #     total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
        #     vllm_outputs_list.append(
        #         vLLM_outputs(
        #             sequences=sequences,
        #             attention_mask=attention_mask,
        #             action_mask=None,
        #             num_actions=num_actions,
        #             packed_seq_lens=packed_seq_lens,
        #             response_length=response_length,
        #             total_length=total_length,
        #             prompts=prompts,
        #             visual_inputs=None,
        #             labels=labels,
        #             pad_len=pad_len,
        #             output_text=output_list,
        #         )
        #     )
        return mini_batch

    def _generate_vllm_with_feedback_language(
        self, 
        messages: List[dict],
        messages_no_feedback: List[dict],
        **kwargs) -> RolloutStorage:
        """
        Generate new samples with feedbacks from the base model.
        For RL training, concat the original prompt ids and the new output ids.
        """
        
        self.response_length_list = []
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", self.strategy.args.generate_max_len),
            min_tokens=kwargs.get("min_new_tokens", 16),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # feedback_prompts_texts = [
        #     self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        #     for msg in messages
        # ]
        # base_prompts_ids = [
        #     self.processor.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        #     for msg in messages_no_feedback
        # ]
        feedback_prompts_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, enable_thinking=False, add_generation_prompt=True) if "Qwen3" in self.strategy.args.pretrain else self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        base_prompts_ids = [
            self.processor.apply_chat_template(msg, tokenize=True, enable_thinking=False, add_generation_prompt=True) if "Qwen3" in self.strategy.args.pretrain else self.processor.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
            for msg in messages_no_feedback
        ]

        # base_prompts_ids = [base_prompt_ids[0] for base_prompt_ids in base_prompts_ids]
        batch_size = (len(messages) + len(llms) - 1) // len(llms)

        refs = []
        for i, llm in enumerate(llms):
            msg =  messages[i * batch_size : (i + 1) * batch_size]
            if "Qwen3" in self.strategy.args.pretrain:
                prompt = self.processor.apply_chat_template(msg, tokenize=False, enable_thinking=False, add_generation_prompt=True)
            else:
                prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=prompt))

        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        output_list = []
        for output in all_outputs:
            output_list.append(output.outputs[0].text)

        if not self.packing_samples:
            # NOTE: concat all outputs to following format:
            #
            # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
            # | token token token token token | token token [EOS] [PAD] |
            # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
            # |<---------- prompt ----------->|<-------- answer ------->|
            max_input_len, max_output_len = 0, 0
            for j in range(len(base_prompts_ids)):
                max_input_len = max(max_input_len, len(base_prompts_ids[j]))
            for output in all_outputs:
                # max_input_len = max(max_input_len, len(output.prompt_token_ids))
                max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

            max_input_len_for_KL = 0
            for output in all_outputs:
                max_input_len_for_KL = max(max_input_len_for_KL, len(output.prompt_token_ids))
            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            sequences = []
            sequences_for_KL = []
            for j, output in enumerate(all_outputs):
                input_len = len(base_prompts_ids[j])
                input_ids = [pad_token_id] * (max_input_len - input_len) + list(base_prompts_ids[j])
                output_len = len(output.outputs[0].token_ids)
                output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
                sequences.append(input_ids + output_ids)
                
                if self.distillation:
                    input_len = len(output.prompt_token_ids)
                    input_feedback_ids = [pad_token_id] * (max_input_len_for_KL - input_len) + list(output.prompt_token_ids)
                    output_feedback_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
                    sequences_for_KL.append(input_feedback_ids + output_feedback_ids)
                    # assert len(sequences_for_KL[-1]) == len(sequences[-1]), f"len(sequences_for_KL[-1]): {len(sequences_for_KL[-1])}, len(sequences[-1]): {len(sequences[-1])}"

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = self.process_sequences_v2(
                sequences, max_output_len, eos_token_id, pad_token_id
            )
            sequences = sequences.to("cuda")
            attention_mask = attention_mask.to("cuda")
            action_mask = action_mask.to("cuda")
            self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())

            if self.distillation:
                sequences_for_KL = torch.tensor(sequences_for_KL)
                sequences_for_KL, attention_mask_for_KL, action_mask_for_KL = self.process_sequences_v2(
                    sequences_for_KL, max_output_len, eos_token_id, pad_token_id
                )
                sequences_for_KL = sequences_for_KL.to("cpu")
                attention_mask_for_KL = attention_mask_for_KL.to("cpu")
                action_mask_for_KL = action_mask_for_KL.to("cpu")
                assert action_mask_for_KL.size(1) == action_mask.size(1), f"action_mask_for_KL.size(1): {action_mask_for_KL.size(1)}, action_mask.size(1): {action_mask.size(1)}"

            
            mini_batch = RolloutStorage(
                            sequences=sequences,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                            visual_inputs=None,
                            pad_len=None,
                            output_text=output_list,
                            obs=None,
                            rewards=None,
                            masks=None,
                            bad_masks=None,
                            action_log_probs=None,
                            prompt = feedback_prompts_texts,
                            sequences_for_KL=sequences_for_KL if self.distillation else None,
                            attention_mask_for_KL=attention_mask_for_KL if self.distillation else None,
                            action_mask_for_KL=action_mask_for_KL if self.distillation else None
            )
        return mini_batch

    
    def process_sequences_v2(self, sequences: torch.Tensor, ouput_len, eos_token_id, pad_token_id):
        """
        Process generated sequences to create attention masks and action masks.

        Args:
            sequences (torch.Tensor): Generated sequence tensor
            input_len (int): Length of the input sequence
            eos_token_id (int): Token ID for the end-of-sequence token
            pad_token_id (int): Token ID for the padding token

        Returns:
            tuple: A tuple containing three elements:
                - sequences: Original sequence
                - attention_mask: Attention mask indicating valid token positions
                - action_mask: Action mask indicating valid action token positions
        """
        # Create initial attention mask by marking positions that are neither EOS nor padding tokens
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # Find the position of the last valid token in each sequence
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)

        # Handle cases where EOS tokens might appear in the middle of the prompt (for Llama3 and Qwen2 models)
        # Find the position of the first valid token in each sequence
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # Create position mask
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # Generate final attention mask, keeping only positions between first and last valid tokens
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # In reinforcement learning, the state transition is represented as:
        # state_i (current token) + action_i (next token) -> state_i+1 (next token)
        # Generate state sequence from input_len-1 to second-to-last token
        state_seq = sequences[:, -ouput_len: -1]
        # Generate action mask indicating valid action token positions
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask
    
    
    
    def split_input_batch(self, batch: Dict) -> List[Dict]:
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
            vision_start_id = self.processor.tokenizer("<|vision_start|>")["input_ids"][0]
            vision_end_id = self.processor.tokenizer("<|vision_end|>")["input_ids"][0]
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
    
    
    def make_input_batch(self, visual_inputs: List[Dict]) -> Dict:
        """
        - visual_inputs:
            - List of Dicts, each Dict has a single image info, which is 'pixel_values' and 'image_grid_thw'.
        - Output:
            - Dict has the following keys:
                - 'pixel_values': (total_img_num, pixel_values)
                - 'image_grid_thw': (total_img_num, 3)
        """
        # each element has no batch dimension
        batch = {}
        # collect all keys
        for inp in visual_inputs:
            batch.update({k:None for k,v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in visual_inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                # qwen2vl concat all patches of all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in visual_inputs if k in inp], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Qwen2VLDataProcessor")
        return batch
    
    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None




class RemoteExperienceMaker_CardGame_REINFORCE(NaiveExperienceMaker):
    def __init__(self, 
                 actor, critic, reward_model, initial_model, tokenizer, data_processor,
                 prompt_max_len, kl_controller, strategy, remote_rm_url, reward_fn,
                 vllm_engines: List = None, 
                 packing_samples=False, 
                 multimodal=True, 
                 envs: gym.vector.AsyncVectorEnv = None,
                 env_config: EnvConfig = None,
                 prompt_config: PromptConfig = None,
                 **kwargs):
        super().__init__(
            actor=actor,
            critic=critic,
            reward_model=reward_model,
            initial_model=initial_model,
            tokenizer=tokenizer,
            data_processor=data_processor,
            prompt_max_len=prompt_max_len,
            kl_controller=kl_controller,
            strategy=strategy,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn
        )
        self.envs = envs
        self.num_envs = envs.num_envs
        self.env_config = env_config
        self.prompt_config = prompt_config
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.multimodal = multimodal
        self.distillation = self.strategy.args.distillation

        assert self.prompt_config.use_vision == self.multimodal, "prompt_config.use_vision and multimodal must be the same."
        if self.prompt_config.use_vision:
            self.prompt_vision = PROMPT_FN[self.prompt_config.prompt_vision]
            self.pattern_vision = self.prompt_config.pattern_vision
        else:
            self.prompt_language = PROMPT_FN[self.prompt_config.prompt_language]
            self.pattern_language = self.prompt_config.pattern_language
        
        self.target_number = self.env_config.target_points
        self.formulate_oracle_arguments()
        self.custom_reward_func = None
        self.processor = data_processor
        self.history_length = 1
        self.json_pattern = r"```json\n(.*?)\n```"


    def formulate_oracle_arguments(self):
        self.oracle_arguments = {}
        self.oracle_arguments['face_card_msg'] = "'J', 'Q', and 'K' count as '10'." if self.env_config.treat_face_cards_as_10 \
                                        else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
        self.oracle_arguments['target_number'] = str(self.target_number)
        self.oracle_arguments['example_json_text'] = example_json_text
        
    def formulate_vision_arguments(self, vision_res_list, info_batch):
        for i in range(len(vision_res_list)):
            if 'cards' not in vision_res_list[i].keys():
                # hard code gt cards into dict
                vision_res_list[i]['cards'] = info_batch['Plain Cards'][i]
    
    def formulate_prompt(self, task_prompt: str, obs_batch: np.ndarray = None, previous_responses = None, previous_verify_infos = None) -> Tuple[List[dict], List[dict]]:
        
        messages = [[] for _ in range(self.num_envs)]
        messages_no_verification = [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            contents = []
            contents_no_verification = []
            if len(previous_responses[i]) > 0 and len(previous_verify_infos[i]) > 0:
                assert len(previous_responses[i]) == len(previous_verify_infos[i]), "The number of previous responses, feedbacks, and verify infos must be the same."
                if obs_batch[i] is not None:
                    pil_image = Image.fromarray(obs_batch[i])
                    contents.append({"type": "image", "image": pil_image})
                    contents_no_verification.append({"type": "image", "image": pil_image})
                base_prompt = PROMPT_BASE_CARD_REINFORCE.format(task=task_prompt)
                contents.append({"type": "text", "text": base_prompt})
                contents_no_verification.append({"type": "text", "text": base_prompt})
                # contents_no_verification.append({"type": "text", "text": task_prompt})
                for idx, (prev_response, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append({"type": "text", "text": f"\n## Previous your response ({self.history_length - idx} steps ago):\n{prev_response}"})
                    contents.append({"type": "text", "text": f"\n## Verification\nYou failed this trial because {prev_verify_info}"})
                    contents_no_verification.append({"type": "text", "text": f"\n## Previous your response ({self.history_length - idx} steps ago):\n{prev_response}"})
            else:
                if obs_batch[i] is not None:
                    pil_image = Image.fromarray(obs_batch[i])
                    contents.append({"type": "image", "image": pil_image})
                    contents.append({"type": "text", "text": task_prompt})
                    contents_no_verification.append({"type": "image", "image": pil_image})
                    contents_no_verification.append({"type": "text", "text": task_prompt})
                else:
                    contents.append({"type": "text", "text": task_prompt})
                    contents_no_verification.append({"type": "text", "text": task_prompt})
            
            messages[i] = [
                {"role": "user",
                 "content": contents,
                },
            ]
            messages_no_verification[i] = [
                {"role": "user",
                 "content": contents_no_verification,
                },
            ]
        return messages, messages_no_verification
    
    def formulate_prompt_for_LLMStudent(self, task_prompts: List[str], previous_responses: List[List[str]], previous_verify_infos: List[List[str]]) -> List[dict]:
        
        messages = [[] for _ in range(self.num_envs)]
        messages_no_verification = [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            contents = []
            contents_no_verification = []
            
            if len(previous_responses[i]) > 0 and len(previous_verify_infos[i]) > 0:
                assert len(previous_responses[i]) == len(previous_verify_infos[i]), "The number of previous responses, and verify infos must be the same."
                feedback_prompt = PROMPT_BASE_CARD_REINFORCE.format(task=task_prompts[i])
                contents.append(feedback_prompt)
                contents_no_verification.append(feedback_prompt)
                # contents_no_verification.append(task_prompts[i])
                for idx, (prev_response, prev_verify_info) in enumerate(zip(previous_responses[i][-self.history_length:], previous_verify_infos[i][-self.history_length:])):
                    contents.append(f"\n## Previous your answer ({self.history_length - idx} steps ago):\n{prev_response}")
                    contents.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
                    contents_no_verification.append(f"\n## Previous your answer ({self.history_length - idx} steps ago):\n{prev_response}")
            
            else:
                contents.append(task_prompts[i])
                contents_no_verification.append(task_prompts[i])
            
            messages[i] = [
                {"role": "user",
                 "content": "\n".join(contents),
                },
            ]
            messages_no_verification[i] = [
                {"role": "user",
                 "content": "\n".join(contents_no_verification),
                },
            ]
        return messages, messages_no_verification

    
    @torch.no_grad()
    def collect_trajectories(self, episode_id: int=None, **generate_kwargs) -> deque[RolloutStorage]:
        """
        Rollout the environment and return the experiences.
        """

        ## returns of envs.reset() ##
        ## obs_batch: numpy with shape (num_envs, resolution, resolution, 3).
        ## info_batch.keys() = ['Cards', '_Cards', 'Plain Cards', '_Plain Cards', 'Numbers', '_Numbers', 
        ##                      'Formula', '_Formula', 'Solution', '_Solution', 'Remaining Numbers', '_Remaining Numbers', 'Remaining Step', 
        ##                      '_Remaining Step', 'Verify Info', '_Verify Info']
        obs_batch, info_batch = self.envs.reset()

        previous_responses = [[] for _ in range(self.num_envs)]
        previous_verify_infos = [[] for _ in range(self.num_envs)]
        previous_rewards = [[] for _ in range(self.num_envs)]

        if self.prompt_config.use_vision:
            prompt, _ = self.prompt_vision, self.pattern_vision
        else:
            prompt, _ = self.prompt_language, self.pattern_language
        
        replay_buffer = deque(maxlen=self.env_config.num_steps + 1)
        episode_start = np.zeros(self.num_envs, dtype=bool)

        
        if self.strategy.is_rank_0() and self.strategy.args.log:
            logs = []

        
        for step in tqdm(range(self.env_config.num_steps), desc="Collecting trajectories", disable=not self.strategy.is_rank_0()):
            vision_res_list = [{} for _ in range(self.num_envs)]
            language_res_list = [{} for _ in range(self.num_envs)]
            self.formulate_vision_arguments(vision_res_list, info_batch)
            task_prompts = [prompt.format(**vision_res_list[i], **language_res_list[i], **self.oracle_arguments) for i in range(self.num_envs)]
            if self.multimodal:
                messages, messages_no_verification = self.formulate_prompt(task_prompts[0], 
                                    obs_batch=obs_batch,
                                    previous_responses=previous_responses,
                                    previous_verify_infos=previous_verify_infos)
            else:
                messages, messages_no_verification = self.formulate_prompt_for_LLMStudent(task_prompts, 
                                    previous_responses=previous_responses,
                                    previous_verify_infos=previous_verify_infos)
            
            # base model inference w/o verification
            if self.strategy.args.no_verification:
                if self.multimodal:
                    mini_batch = self._generate_vllm(obs_batch, messages_no_verification, self.multimodal, **generate_kwargs)
                else:
                    mini_batch = self._generate_vllm_language(messages_no_verification, **generate_kwargs)
            else:
                # base model inference w/ verification
                if self.multimodal:
                    mini_batch = self._generate_vllm(obs_batch, messages, self.multimodal, **generate_kwargs)
                else:
                    mini_batch = self._generate_vllm_language(messages, **generate_kwargs)
            
            # preprocessing the model response to align with json style.
            for i, model_response in enumerate(mini_batch.output_text):
                # remove <|im_end|> because the env requires a pure json format.
                if "<|im_end|>" in model_response:
                    mini_batch.output_text[i] = model_response.replace("<|im_end|>", "")
                # handle the case that the model provides ```json ...``` format as recent models do.
                try:
                    match = re.search(self.json_pattern, model_response, re.DOTALL)
                    if match:
                        mini_batch.output_text[i] = match.group(1)
                except:
                    pass
                
            obs_batch, rewards, terminations, truncations, info_batch = self.envs.step(mini_batch.output_text)
            episode_start = np.logical_or(terminations, truncations)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            terminations_tensor = torch.tensor(terminations, dtype=torch.bool)
            truncations_tensor = torch.tensor(truncations, dtype=torch.bool)
            # Calculate masks (1 if episode continues, 0 if it ended). Use .float() to convert boolean tensor to float (0.0 or 1.0)
            masks_tensor = 1.0 - terminations_tensor.float()
            # Calculate bad_masks (commonly 1 if episode did NOT truncate, 0 if it did)
            bad_masks_tensor = 1.0 - truncations_tensor.float()

            mini_batch.rewards = rewards_tensor     # Shape: (num_envs,)
            mini_batch.returns = rewards_tensor
            mini_batch.masks = masks_tensor         # Shape: (num_envs,)
            mini_batch.bad_masks = bad_masks_tensor # Shape: (num_envs,)
            reward_diff = torch.zeros_like(rewards_tensor, dtype=torch.float32)
            for i in range(self.num_envs):
                if len(previous_rewards[i]) > 0:
                    reward_diff[i] = rewards[i] - previous_rewards[i][-1]
                else:
                    reward_diff[i] = 0.0
            mini_batch.reward_diff = reward_diff
            replay_buffer.append(copy.deepcopy(mini_batch))
            
            for i, model_response in enumerate(mini_batch.output_text):
                previous_responses[i].append(model_response)
            for i in range(self.num_envs):
                previous_verify_infos[i].append(info_batch["Verify Info"][i])
                previous_rewards[i].append(rewards[i])

            # Clear the history for environment i
            for i in range(self.num_envs):
                # if episode_start[i]:
                if terminations[i] or truncations[i]:
                    previous_responses[i] = []
                    previous_verify_infos[i] = []
                    previous_rewards[i] = []

            if self.strategy.is_rank_0() and self.strategy.args.log:
                log = {
                    "step": step,
                    "prompt": mini_batch.prompt[0],
                    "model_response": mini_batch.output_text[0],
                    "verify_info": info_batch["Verify Info"][0],
                    "reward": rewards[0],
                    "gt_cards": info_batch["Plain Cards"][0],
                }
                logs.append(log)

        
        if self.strategy.is_rank_0() and self.strategy.args.log:
            log_dir = self.strategy.args.output_log_dir
            log_file = os.path.join(log_dir, f"episode_{episode_id}.json" if episode_id is not None else "log.json")
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        
        return replay_buffer

    def compute_and_store_returns_in_buffer(self, replay_buffer: Deque[RolloutStorage], gamma: float = 0.9) -> None:
        """
        Computes discounted cumulative returns and stores them *in-place*
        within the 'returns' field of each RolloutStorage object in the deque.

        Args:
            replay_buffer: A deque containing RolloutStorage objects.
                        Each object must have 'rewards' and 'masks' attributes
                        (assumed to be tensors or convertible to tensors) and
                        an assignable 'returns' attribute.
            gamma: The discount factor.
        """
        if not replay_buffer:
            print("Warning: Replay buffer is empty. No returns computed.")
            return # Nothing to do

        buffer_len = len(replay_buffer)
        # Infer num_envs and device from the first element
        first_rollout = replay_buffer[0]
        if not isinstance(first_rollout, RolloutStorage):
            raise TypeError("replay_buffer must contain RolloutStorage instances.")
        if first_rollout.rewards is None or first_rollout.masks is None:
            raise ValueError("First RolloutStorage in buffer missing 'rewards' or 'masks'. Cannot proceed.")

        try:
            # Assume rewards/masks are tensors or tensor-like (e.g., lists of numbers)
            rewards_tensor = torch.as_tensor(first_rollout.rewards)
            num_envs = rewards_tensor.shape[0]
            device = rewards_tensor.device
        except (TypeError, IndexError, AttributeError) as e:
            raise ValueError(f"Could not determine num_envs or device from first rollout's rewards. Error: {e}")

        # Initialize the return for the step *after* the last one in the buffer
        next_return = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Iterate backwards through the buffer (from T-1 down to 0)
        for t in reversed(range(buffer_len)):
            rollout_storage_t = replay_buffer[t]

            if not isinstance(rollout_storage_t, RolloutStorage):
                print(f"Warning: Element at index {t} is not a RolloutStorage object. Skipping.")
                continue
            if rollout_storage_t.rewards is None or rollout_storage_t.masks is None:
                print(f"Warning: RolloutStorage at index {t} missing 'rewards' or 'masks'. Skipping return calculation for this step.")
                # Set returns to NaN or zeros, or handle as error? Setting to zeros for safety.
                rollout_storage_t.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
                next_return = torch.zeros(num_envs, dtype=torch.float32, device=device) # Reset future return
                continue

            # Ensure rewards and masks are tensors on the correct device
            rewards_t = torch.as_tensor(rollout_storage_t.rewards, dtype=torch.float32, device=device)
            masks_t = torch.as_tensor(rollout_storage_t.masks, dtype=torch.float32, device=device) # mask=1 if not done

            if rewards_t.shape[0] != num_envs or masks_t.shape[0] != num_envs:
                print(f"Warning: Inconsistent num_envs at step {t}. Expected {num_envs}, got rewards={rewards_t.shape[0]}, masks={masks_t.shape[0]}. Skipping.")
                rollout_storage_t.returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
                next_return = torch.zeros(num_envs, dtype=torch.float32, device=device) # Reset future return
                continue

            # Calculate returns for the current step t: G_t = r_{t+1} + gamma * G_{t+1} * mask_t
            current_returns = rewards_t + gamma * next_return * masks_t
            # --- Store the calculated returns IN-PLACE ---
            rollout_storage_t.returns = current_returns
            # Update next_return for the previous step (t-1)
            next_return = current_returns
        print(f"Successfully computed and stored returns in {buffer_len} buffer elements.")

    
    def compute_values(self, replay_buffer: Deque[RolloutStorage]) -> None:
        args = self.strategy.args
        device = torch.cuda.current_device()
        chunk_size = self.strategy.args.micro_train_batch_size
        buffer_len = len(replay_buffer)
        for turn in range(buffer_len):
            mini_batch = replay_buffer[turn]
            input_ids = mini_batch.input_ids
            input_ids_chunks = self.split_into_chunks(input_ids, chunk_size)
            input_ids_chunk_cpu_list = [input_ids_chunk.to("cpu") for input_ids_chunk in input_ids_chunks]
            attention_mask_for_input_ids_chunks = self.split_into_chunks(mini_batch.attention_mask_for_input_ids, chunk_size)
            attention_mask_for_input_ids_chunk_cpu_list = [attention_mask_for_input_ids_chunk.to("cpu") for attention_mask_for_input_ids_chunk in attention_mask_for_input_ids_chunks]

            # compute values
            value_ref = self.critic.forward_batch.remote(
                sequences=input_ids_chunk_cpu_list,
                attention_mask=attention_mask_for_input_ids_chunk_cpu_list,
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
            
            values_list = ray.get([value_ref])[0]
            if values_list[0] is not None:
                values_list = [values_list[i].to(device) for i in range(len(values_list))]
            values = torch.cat(values_list, dim=0)
            # store values
            mini_batch.values = values.squeeze(-1) # Shape: (num_envs, )
            # print(f"mini_batch.values.shape: {mini_batch.values.shape}")

    
    
    @torch.no_grad()
    def compute_advantages_and_returns(
        self,
        replay_buffer: Deque[RolloutStorage],
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns of each turn for multi-turn envs.
        """
        
        T = len(replay_buffer)
        device = replay_buffer[0].rewards.device
        advantages_reversed = []
        lastgaelam = 0.0

        for t in reversed(range(T)):
            # 1. non-terminal flag: 1 if episode is still running AFTER step t
            nonterminal = 1.0 - replay_buffer[t].masks               # masks==1 ➜ done
            if hasattr(replay_buffer[t], "bad_masks"):                # optional time-limit reset
                nonterminal = nonterminal * replay_buffer[t].bad_masks

            nonterminal = nonterminal.to(device)
            next_value = replay_buffer[t + 1].values.to(device) if t < T - 1 else 0.0

            delta = replay_buffer[t].rewards + gamma * next_value * nonterminal - replay_buffer[t].values.to(device)

            lastgaelam = delta + gamma * lambd * nonterminal * lastgaelam
            advantages_reversed.append(lastgaelam)
            replay_buffer[t].returns    = lastgaelam + replay_buffer[t].values.to(device)
            replay_buffer[t].advantages = lastgaelam

        # (optional) Normalize once, in-place
        if self.strategy.args.normalize_advantages:
            advantages_chronological = advantages_reversed[::-1]
            advantages = torch.stack(advantages_chronological, dim=0).to(torch.float32)
            adv_mean, adv_std = advantages.mean(), advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-6)
            for t in range(T):
                replay_buffer[t].advantages = advantages[t]

    
    @torch.no_grad()
    def make_experience_list(self, episode_id: int=None, **generate_kwargs) -> List[Experience_CARDGAME]:
        
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        replay_buffer = self.collect_trajectories(episode_id, **generate_kwargs)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        # compute cumulative returns
        self.compute_and_store_returns_in_buffer(replay_buffer, generate_kwargs["gamma"])

        if self.critic is not None:
            print("compute_values...")
            self.compute_values(replay_buffer)

        if self.strategy.args.advantage_estimator == "gae":
            self.compute_advantages_and_returns(replay_buffer, generate_kwargs["gamma"], generate_kwargs["lambd"])
        
        all_experiences = []
        for mini_batch in tqdm(replay_buffer, total=len(replay_buffer), desc="make_experience", disable=not self.strategy.is_rank_0()):
            if self.multimodal:
                experience = self.make_experience(mini_batch)
            else:
                experience = self.make_experience_language(mini_batch)
            
            # # compute advantages and returns for PPO
            # if self.strategy.args.advantage_estimator == "gae":
            #     experience.advantages, experience.returns = self.get_advantages_and_returns(
            #         values=experience.values,
            #         reward=experience.info["reward"],
            #         action_mask=experience.action_mask,
            #         gamma=generate_kwargs["gamma"],
            #         lambd=generate_kwargs["lambd"],
            #     )

            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
            all_experiences.append(experience)

        
        if self.critic is not None:
            for experience in all_experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        
        return all_experiences

    def split_into_chunks(self, data, chunk_size):
        """Split data into chunks of specified size."""
        if isinstance(data, torch.Tensor):
            return torch.split(data, chunk_size)
        elif isinstance(data, dict):
            return [{k: v[i:i+chunk_size] for k, v in data.items()} for i in range(0, len(next(iter(data.values()))), chunk_size)]
        else:
            return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    @torch.no_grad()
    def make_experience(self, mini_batch: RolloutStorage) -> Experience_CARDGAME:
        """
        Turn samples into experience by calculating logprobs, rewards, and kl divergence.
        The size of each element in vllm_outputs corresponds to self.strategy.args.micro_rollout_batch_size.

        This function does the following:
        1. Get log_probs of the initial_model and base model to compute KL distance to refine reward values
        2. Pack the above information into Experience dataclass
        """
        start_time = time.time()
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        chunk_size = self.strategy.args.micro_train_batch_size

        # Extract values from samples
        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        action_mask = mini_batch.action_mask
        packed_seq_lens = mini_batch.packed_seq_lens
        visual_inputs = self.make_input_batch(mini_batch.visual_inputs)

        # Split data into chunks
        sequences_chunks = self.split_into_chunks(sequences, chunk_size)
        attention_mask_chunks = self.split_into_chunks(attention_mask, chunk_size)
        action_mask_chunks = self.split_into_chunks(action_mask, chunk_size)
        packed_seq_lens_chunks = self.split_into_chunks(packed_seq_lens, chunk_size) if packed_seq_lens is not None else [None] * len(sequences_chunks)
        visual_inputs_chunks = [self.make_input_batch(mini_batch.visual_inputs[i:i+chunk_size]) for i in range(0, len(mini_batch.visual_inputs), chunk_size)]

        # Move current chunk to CPU for remote processing
        seq_chunk_cpu_list = [seq_chunk.to("cpu") for seq_chunk in sequences_chunks]
        attn_chunk_cpu_list = [attn_chunk.to("cpu") for attn_chunk in attention_mask_chunks]
        action_mask_chunk_cpu_list = [action_mask_chunk.to("cpu") for action_mask_chunk in action_mask_chunks]
        vis_inputs_chunk_cpu_list = [{k: v.to("cpu") for k, v in vis_inputs_chunk.items() if k != "input_ids" and k != "attention_mask"} for vis_inputs_chunk in visual_inputs_chunks]
        logps_allgather_list = [True] * len(seq_chunk_cpu_list)
        
        
        # Process initial model chunk
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=seq_chunk_cpu_list,
                action_mask=action_mask_chunk_cpu_list,
                attention_mask=attn_chunk_cpu_list,
                logps_allgather=logps_allgather_list,
                visual_inputs=vis_inputs_chunk_cpu_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(seq_chunk_cpu_list))
        
        
        # Initialize lists to store results
        all_action_log_probs = []
        all_base_action_log_probs = []

        # Process each chunk
        for i in range(len(sequences_chunks)):
            # Clear GPU cache before processing new chunk
            torch.cuda.empty_cache()

            # Get current chunk
            seq_chunk = sequences_chunks[i]
            attn_chunk = attention_mask_chunks[i]
            action_mask_chunk = action_mask_chunks[i]
            packed_seq_chunk = packed_seq_lens_chunks[i]
            vis_inputs_chunk = visual_inputs_chunks[i]

            # Process actor model chunk
            actor_vis_inputs = None if vis_inputs_chunk is None else {k: v.to(device) for k, v in vis_inputs_chunk.items() if k != "input_ids" and k != "attention_mask"}
            
            action_log_probs_chunk = self.actor(
                seq_chunk.to(device),
                action_mask_chunk,
                attn_chunk.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=True,
                packed_seq_lens=packed_seq_chunk,
                visual_inputs=actor_vis_inputs,
            )

            # Store results
            all_action_log_probs.append(action_log_probs_chunk)

            # Clear memory
            del seq_chunk, attn_chunk, action_mask_chunk, packed_seq_chunk, vis_inputs_chunk
            torch.cuda.empty_cache()

        
        base_action_log_probs_list = ray.get([base_action_log_probs_ref])[0]
        if base_action_log_probs_list[0] is not None:
            base_action_log_probs_list = [base_action_log_probs_list[i].to(device) for i in range(len(base_action_log_probs_list))]
        base_action_log_probs = torch.cat(base_action_log_probs_list, dim=0)
        
        # # Concatenate results
        action_log_probs = torch.cat(all_action_log_probs, dim=0)

        actor_value_rm_time = time.time() - start_time
        start = time.time()
        wait_time = time.time() - start

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        rewards = mini_batch.rewards
        r = rewards.to(device)

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=mini_batch.action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        if not self.packing_samples:
            kl_mean = masked_mean(kl, mini_batch.action_mask, dim=-1)
        else:
            num_actions = mini_batch.num_actions
            packed_seq_lens = mini_batch.packed_seq_lens
            if self.strategy.ring_attn_group is not None:
                assert mini_batch.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=mini_batch.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=None,
                    kl=kl,
                )
            # Convert tensor into list of tensors for easier manipulation within dataset
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "return": mini_batch.returns,
            "response_length": mini_batch.response_length,
            "total_length": mini_batch.total_length,
            "num_actions": mini_batch.num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience_CARDGAME(
            sequences,
            action_log_probs,
            base_action_log_probs,
            values=None,
            returns=mini_batch.returns,
            advantages=mini_batch.returns,
            attention_mask=attention_mask,
            action_mask=mini_batch.action_mask,
            info=info,
            kl=kl,
            visual_inputs=visual_inputs,
            sequences_for_KL=mini_batch.sequences_for_KL if self.distillation else None,
            attention_mask_for_KL=mini_batch.attention_mask_for_KL if self.distillation else None,
            action_mask_for_KL=mini_batch.action_mask_for_KL if self.distillation else None,
            reward_diff=mini_batch.reward_diff if self.distillation else None,
        )

        self.actor.train()  # Reset model state
        return experience
    
    @torch.no_grad()
    def make_experience_language(self, mini_batch: RolloutStorage) -> Experience_CARDGAME:
        """
        Turn samples into experience by calculating logprobs, rewards, and kl divergence.
        The size of each element in vllm_outputs corresponds to self.strategy.args.micro_rollout_batch_size.

        This function does the following:
        1. Get log_probs of the initial_model and base model to compute KL distance to refine reward values
        2. Pack the above information into Experience dataclass
        """
        start_time = time.time()
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        chunk_size = self.strategy.args.micro_train_batch_size

        # Extract values from samples
        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        action_mask = mini_batch.action_mask
        packed_seq_lens = mini_batch.packed_seq_lens
        # values = mini_batch.values
        # if self.critic is not None:
        #     input_ids = mini_batch.input_ids
        #     input_ids_chunks = self.split_into_chunks(input_ids, chunk_size)
        #     input_ids_chunk_cpu_list = [input_ids_chunk.to("cpu") for input_ids_chunk in input_ids_chunks]
        #     attention_mask_for_input_ids_chunks = self.split_into_chunks(mini_batch.attention_mask_for_input_ids, chunk_size)
        #     attention_mask_for_input_ids_chunk_cpu_list = [attention_mask_for_input_ids_chunk.to("cpu") for attention_mask_for_input_ids_chunk in attention_mask_for_input_ids_chunks]

        # Split data into chunks
        sequences_chunks = self.split_into_chunks(sequences, chunk_size)
        attention_mask_chunks = self.split_into_chunks(attention_mask, chunk_size)
        action_mask_chunks = self.split_into_chunks(action_mask, chunk_size)
        packed_seq_lens_chunks = self.split_into_chunks(packed_seq_lens, chunk_size) if packed_seq_lens is not None else [None] * len(sequences_chunks)

        # Move current chunk to CPU for remote processing
        seq_chunk_cpu_list = [seq_chunk.to("cpu") for seq_chunk in sequences_chunks]
        attn_chunk_cpu_list = [attn_chunk.to("cpu") for attn_chunk in attention_mask_chunks]
        action_mask_chunk_cpu_list = [action_mask_chunk.to("cpu") for action_mask_chunk in action_mask_chunks]
        logps_allgather_list = [True] * len(seq_chunk_cpu_list)
        
        
        # Process initial model chunk
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=seq_chunk_cpu_list,
                action_mask=action_mask_chunk_cpu_list,
                attention_mask=attn_chunk_cpu_list,
                logps_allgather=logps_allgather_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(seq_chunk_cpu_list))
        
        
        # # values
        # if self.critic is not None:
        #     value_ref = self.critic.forward_batch.remote(
        #         sequences=input_ids_chunk_cpu_list,
        #         attention_mask=attention_mask_for_input_ids_chunk_cpu_list,
        #     )
        #     # avoid CUDA OOM when colocate models
        #     if args.colocate_critic_reward or args.colocate_all_models:
        #         ray.get([value_ref])
        #         ray.get([self.critic.empty_cache.remote()])
        # else:
        #     value_ref = ray.put([None] * len(seq_chunk_cpu_list))
        
        
        # Initialize lists to store results
        all_action_log_probs = []

        # Process each chunk
        for i in range(len(sequences_chunks)):
            # Clear GPU cache before processing new chunk
            torch.cuda.empty_cache()

            # Get current chunk
            seq_chunk = sequences_chunks[i]
            attn_chunk = attention_mask_chunks[i]
            action_mask_chunk = action_mask_chunks[i]
            packed_seq_chunk = packed_seq_lens_chunks[i]

            action_log_probs_chunk = self.actor(
                seq_chunk.to(device),
                action_mask_chunk,
                attn_chunk.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=True,
                packed_seq_lens=packed_seq_chunk,
            )

            # Store results
            all_action_log_probs.append(action_log_probs_chunk)

            # Clear memory
            del seq_chunk, attn_chunk, action_mask_chunk, packed_seq_chunk
            torch.cuda.empty_cache()

        
        base_action_log_probs_list = ray.get([base_action_log_probs_ref])[0]
        if base_action_log_probs_list[0] is not None:
            base_action_log_probs_list = [base_action_log_probs_list[i].to(device) for i in range(len(base_action_log_probs_list))]
        base_action_log_probs = torch.cat(base_action_log_probs_list, dim=0)
        
        # # values
        # values_list = ray.get([value_ref])[0]
        # if values_list[0] is not None:
        #     values_list = [values_list[i].to(device) for i in range(len(values_list))]
        # values = torch.cat(values_list, dim=0)
        
        # # Concatenate results
        action_log_probs = torch.cat(all_action_log_probs, dim=0)

        actor_value_rm_time = time.time() - start_time
        start = time.time()
        wait_time = time.time() - start

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        rewards = mini_batch.rewards
        r = rewards.to(device)

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=mini_batch.action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        sequences = mini_batch.sequences
        attention_mask = mini_batch.attention_mask
        if not self.packing_samples:
            kl_mean = masked_mean(kl, mini_batch.action_mask, dim=-1)
        else:
            num_actions = mini_batch.num_actions
            packed_seq_lens = mini_batch.packed_seq_lens
            if self.strategy.ring_attn_group is not None:
                assert mini_batch.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=mini_batch.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=None, # if we consider turn-level values, we need to pass None here.
                    kl=kl,
                )
            # Convert tensor into list of tensors for easier manipulation within dataset
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "return": mini_batch.returns,
            "response_length": mini_batch.response_length,
            "total_length": mini_batch.total_length,
            "num_actions": mini_batch.num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience_CARDGAME(
            sequences,
            action_log_probs,
            base_action_log_probs,
            values=mini_batch.values if self.critic is not None else None,
            returns=mini_batch.returns,
            # advantages=mini_batch.returns,
            advantages=mini_batch.advantages if self.critic is not None else mini_batch.returns,
            attention_mask=attention_mask,
            action_mask=mini_batch.action_mask,
            info=info,
            kl=kl,
            visual_inputs=None,
            input_ids=mini_batch.input_ids if self.critic is not None else None,
            attention_mask_for_input_ids=mini_batch.attention_mask_for_input_ids if self.critic is not None else None,
        )

        self.actor.train()  # Reset model state
        return experience
    

    def _generate_vllm(self, obs_batch:List[np.ndarray], messages: List[dict], multimodal=True, **kwargs) -> RolloutStorage:
        """
        Create prompts for base model inference and give them to vLLM, and return the model responses.

        all_prompts: List[dict], each dict is like:
        {
            "role": "user",
            "content": contents,
        }
        """
        
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", self.strategy.args.generate_max_len),
            min_tokens=kwargs.get("min_new_tokens", 16),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        prompt_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        visual_inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.prompt_max_len
        )
        visual_inputs_chunks = self.split_input_batch(visual_inputs)
        visual_inputs = []
        for visual_inputs_chunk in visual_inputs_chunks:
            visual_inputs_chunk.pop("input_ids")
            visual_inputs_chunk.pop("attention_mask")
            # visual_inputs_chunk = {k: v.to("cuda") for k, v in visual_inputs_chunk.items()}
            visual_inputs.append(visual_inputs_chunk)
        # visual_inputs.pop("input_ids")
        # visual_inputs.pop("attention_mask")
        # visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()} # 'pixel_values', 'image_grid_thw'

        # # Expand prompt list based on the number of samples per prompt
        # all_prompts = sum([[prompt] for prompt in prompt_texts], [])
        batch_size = (len(messages) + len(llms) - 1) // len(llms)

        # Prepare inputs for vLLM
        refs = []
        vllm_inputs = []
        if multimodal:
            for i, llm in enumerate(llms):
                msg_batch = messages[i * batch_size : (i + 1) * batch_size]
                obs_batch_slice = obs_batch[i * batch_size : (i + 1) * batch_size]
                prompts = self.processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=True)
                vllm_inputs = [
                    {
                        # "prompt": prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>'),
                        "prompt": prompt,
                        "multi_modal_data": {"image": obs},
                        "mm_processor_kwargs": kwargs["processor_kwargs"]
                    }
                    for prompt, obs in zip(prompts, obs_batch_slice)
                ]
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                )
        else:
            for i, llm in enumerate(llms):
                for msg in messages[i * batch_size : (i + 1) * batch_size]:
                    prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    vllm_inputs.append({"prompt": prompt})
                    refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))

        ray.get(refs)

        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(3)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        model_responses_list = []
        for output in all_outputs:
            model_responses_list.append(output.outputs[0].text)

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in all_outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for j, output in enumerate(all_outputs):
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)
            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        sequences = sequences.to("cuda")
        attention_mask = attention_mask.to("cuda")
        action_mask = action_mask.to("cuda")

        self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())

        # for logging
        prompt_texts_img_pad_removed = [prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>') for prompt in prompt_texts]

        mini_batch = RolloutStorage(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        visual_inputs=visual_inputs,
                        pad_len=None,
                        output_text=model_responses_list,
                        obs=obs_batch,
                        rewards=None,
                        masks=None,
                        bad_masks=None,
                        action_log_probs=None,
                        prompt=prompt_texts_img_pad_removed,
        )

        return mini_batch

    def _generate_vllm_language(self, messages: List[dict], **kwargs) -> RolloutStorage:
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", self.strategy.args.generate_max_len),
            min_tokens=kwargs.get("min_new_tokens", 16),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        batch_size = (len(messages) + len(llms) - 1) // len(llms)

        
        try:
            refs = []
            prompts = []
            for i, llm in enumerate(llms):
                msg = messages[i * batch_size : (i + 1) * batch_size]
                prompt = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                flattened_prompts = [p for p in prompt]
                prompts.extend(flattened_prompts)
                refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=prompt))

            ray.get(refs)

            # Retrieve and combine results from all outputs
            all_output_refs = []
            for i, llm in enumerate(llms):
                all_output_refs.append(llm.get_responses.remote(rank))
            all_outputs = sum(ray.get(all_output_refs), [])

            model_responses_list = []
            for output in all_outputs:
                model_responses_list.append(output.outputs[0].text)

        # # Retrieve and combine results from all outputs
        # all_output_refs = []
        # for i, llm in enumerate(llms):
        #     all_output_refs.append(llm.get_responses.remote(rank))
        # all_outputs = sum(ray.get(all_output_refs), [])

        # model_responses_list = []
        # for output in all_outputs:
        #     model_responses_list.append(output.outputs[0].text)

        except Exception as e:
            print(f"Error: {e}")
            print(f"Output: {prompts}")
        
        
        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in all_outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        input_ids_for_PPO = []
        for j, output in enumerate(all_outputs):
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)
            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
            # concat input and output
            sequences.append(input_ids + output_ids)
            if self.critic is not None:
                input_ids_for_PPO.append(input_ids)
        
        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        sequences = sequences.to("cuda")
        attention_mask = attention_mask.to("cuda")
        action_mask = action_mask.to("cuda")

        self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())

        if self.critic is not None:
            input_ids_for_PPO = torch.tensor(input_ids_for_PPO)
            # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
            attention_mask_for_input_ids = (input_ids_for_PPO.ne(eos_token_id) & input_ids_for_PPO.ne(pad_token_id)).to(dtype=torch.long)
            seq_length = attention_mask_for_input_ids.size(1)
            eos_indices = seq_length - attention_mask_for_input_ids.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
            input_ids_for_PPO.scatter_(dim=1, index=eos_indices, value=eos_token_id)
            first_token_indices = attention_mask_for_input_ids.long().argmax(dim=1, keepdim=True)
            mask = torch.arange(seq_length).unsqueeze(0).expand(input_ids_for_PPO.size(0), -1).to(device=input_ids_for_PPO.device)
            attention_mask_for_input_ids = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)
            attention_mask_for_input_ids = attention_mask_for_input_ids.to("cuda")
            input_ids_for_PPO = input_ids_for_PPO.to("cuda")


        mini_batch = RolloutStorage(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        visual_inputs=None,
                        pad_len=None,
                        output_text=model_responses_list,
                        obs=None,
                        rewards=None,
                        masks=None,
                        bad_masks=None,
                        action_log_probs=None,
                        prompt=prompts,
                        input_ids=input_ids_for_PPO if self.strategy.args.advantage_estimator == "gae" else None,
                        attention_mask_for_input_ids=attention_mask_for_input_ids if self.strategy.args.advantage_estimator == "gae" else None
        )

        return mini_batch

    
    def process_sequences_v2(self, sequences: torch.Tensor, ouput_len, eos_token_id, pad_token_id):
        """
        Process generated sequences to create attention masks and action masks.

        Args:
            sequences (torch.Tensor): Generated sequence tensor
            input_len (int): Length of the input sequence
            eos_token_id (int): Token ID for the end-of-sequence token
            pad_token_id (int): Token ID for the padding token

        Returns:
            tuple: A tuple containing three elements:
                - sequences: Original sequence
                - attention_mask: Attention mask indicating valid token positions
                - action_mask: Action mask indicating valid action token positions
        """
        # Create initial attention mask by marking positions that are neither EOS nor padding tokens
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # Find the position of the last valid token in each sequence
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)

        # Handle cases where EOS tokens might appear in the middle of the prompt (for Llama3 and Qwen2 models)
        # Find the position of the first valid token in each sequence
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        # Create position mask
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        # Generate final attention mask, keeping only positions between first and last valid tokens
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # In reinforcement learning, the state transition is represented as:
        # state_i (current token) + action_i (next token) -> state_i+1 (next token)
        # Generate state sequence from input_len-1 to second-to-last token
        state_seq = sequences[:, -ouput_len: -1]
        # Generate action mask indicating valid action token positions
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask
    
    
    
    def split_input_batch(self, batch: Dict) -> List[Dict]:
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
            vision_start_id = self.processor.tokenizer("<|vision_start|>")["input_ids"][0]
            vision_end_id = self.processor.tokenizer("<|vision_end|>")["input_ids"][0]
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
    
    
    def make_input_batch(self, visual_inputs: List[Dict]) -> Dict:
        """
        - visual_inputs:
            - List of Dicts, each Dict has a single image info, which is 'pixel_values' and 'image_grid_thw'.
        - Output:
            - Dict has the following keys:
                - 'pixel_values': (total_img_num, pixel_values)
                - 'image_grid_thw': (total_img_num, 3)
        """
        # each element has no batch dimension
        batch = {}
        # collect all keys
        for inp in visual_inputs:
            batch.update({k:None for k,v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in visual_inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                # qwen2vl concat all patches of all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in visual_inputs if k in inp], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Qwen2VLDataProcessor")
        return batch
    
    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None