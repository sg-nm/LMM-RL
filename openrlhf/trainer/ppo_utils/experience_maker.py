import os
import time
import copy
import math
import gc
import traceback
from datetime import timedelta
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from io import BytesIO


from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray

from qwen_vl_utils import process_vision_info

from gui_env.robust_parallel_desktop_env import ParallelDesktopEnv
from agents.ui_tars import build_singleturn_prompt, parse_action_qwen2vl, parsing_response_to_pyautogui_code
from openrlhf.trainer.ppo_utils.rewards import gui_agent_format_reward, english_format_reward
from openrlhf.textgrad.feedback_vllm import FEEDBACK_PROMPT, FEEDBACK_PROMPT_BASE
from openrlhf.textgrad.custom_reward_functions import check_answer_commonsense_qa

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

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
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    visual_inputs: the visual input for vlm training
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    visual_inputs: Optional[Dict]
    labels: list[str]
    pad_len: Optional[int]

@dataclass
class Samples_GUI:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    visual_inputs: the visual input for vlm training
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    visual_inputs: Optional[Dict]
    labels: list[str]
    pad_len: Optional[int]
    reward: Optional[float]
    completion: Optional[str]


@dataclass
class vLLM_outputs:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    visual_inputs: the visual input for vlm training
    """

    prompts: list[str]
    output_text: list[str]
    labels: list[str]
    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    visual_inputs: Optional[Dict]
    pad_len: Optional[int]



class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        data_processor,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        self.response_length_list = []
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func


    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        self.response_length_list = []
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
    
            inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
            visual_inputs = {}
            for k,v in inputs.items():
                if k not in ["input_ids", "attention_mask"]:
                    visual_inputs[k] = v


            labels = all_labels[i : i + args.micro_rollout_batch_size]
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                visual_inputs=visual_inputs,
                labels=labels,
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        raise NotImplementedError("This method should be implemented by the subclass.")

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    ) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_labels, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        visual_inputs = samples.visual_inputs

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )
        visual_inputs_cpu = None
        if visual_inputs is not None:
            visual_inputs_cpu = {k: v.to("cpu") for k, v in visual_inputs.items()}        
        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(
                    rm.forward.remote(
                        sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens, pad_sequence=True, visual_inputs=visual_inputs_cpu
                    )
                )
        else:
            # remote RM
            if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

                if self.custom_reward_func:
                    r = self.custom_reward_func.remote(queries, samples.prompts, samples.labels)
                    r_refs.append(r)
                else:
                    for rm in self.remote_rm_url:
                        r = remote_rm_fn_ray.remote(
                            rm, queries=queries, prompts=samples.prompts, labels=samples.labels
                        )
                        r_refs.append(r)
            else:
                r_refs.append(ray.put(None))

        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        # log probs
        action_log_probs = self.actor(
            sequences, 
            num_actions, 
            attention_mask, 
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
            visual_inputs=visual_inputs
        )
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)

        # broadcast rewards to all ring attention ranks when using remote RM
        if self.remote_rm_url and self.strategy.ring_attn_group is not None:
            if self.strategy.ring_attn_rank == 0:
                dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                dist.broadcast_object_list(
                    rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )

        total_rewards = [r.pop('rewards').to(device) if isinstance(r,dict) else r.to(device) for r in rewards]
        specific_rewards = {}
        for r in rewards:
            if isinstance(r,dict):
                for k in r.keys():
                    r[k] = r[k].to(device)
                specific_rewards.update(r)

        r = self.reward_fn(total_rewards) if len(total_rewards) > 0 else total_rewards[0]

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            if self.strategy.ring_attn_group is not None:
                assert samples.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=samples.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=value,
                    kl=kl,
                )
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            **specific_rewards
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
            visual_inputs=visual_inputs
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams
        self.response_length_list = []
        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])

        # Distribute requests to engines and collect responses to outputs
        refs = []
        # For VLM
        for i, llm in enumerate(llms):
            messages = all_prompts[i * batch_size : (i + 1) * batch_size]
            if messages:
                prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images = [self.data_processor.get_images_from_messages(m) for m in messages]
                vllm_inputs = [{
                        "prompt": p,
                        "multi_modal_data":{"image": imgs} if imgs else None,
                        "mm_processor_kwargs": kwargs["processor_kwargs"]
                    } for p, imgs in zip(prompts,images)]
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                )


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

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
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
                # Collect for visual input
                
                visual_inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
                visual_inputs.pop("input_ids")
                visual_inputs.pop("attention_mask")
                visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()}
                self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        visual_inputs=visual_inputs,
                        labels=labels,
                        pad_len=None,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                self.response_length_list.extend(num_actions)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                # Collect for visual input
                visual_inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
                visual_inputs.pop("input_ids")
                visual_inputs.pop("attention_mask")
                visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()}
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        visual_inputs=visual_inputs,
                        labels=labels,
                        pad_len=pad_len,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None


class RemoteExperienceMaker_TG(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, feedback_model = None, packing_samples=False, multimodal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.feedback_model = feedback_model
        self.packing_samples = packing_samples
        self.multimodal = multimodal
        self.apply_chat_template = self.tokenizer.apply_chat_template
        self.custom_reward_func = check_answer_commonsense_qa

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        
        ## == NaiveExperienceMaker.make_experience_list() == ##
        args = self.strategy.args
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            vllm_outputs_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        all_experiences = []
        for vllm_outputs in tqdm(vllm_outputs_list, total=len(vllm_outputs_list), desc="make_experience", disable=not self.strategy.is_rank_0()):
        # for vllm_outputs in vllm_outputs_list:
            experiences = self.make_experience(vllm_outputs)
            # Process experiences (reward shaping, etc.)
            experiences, rewards = self.process_experiences(experiences)

            # calculate return and advantages
            for experience, reward in zip(experiences, rewards):
                experience = experience.to_device("cuda")
                reward = reward.to(device="cuda")
                num_actions = experience.info["num_actions"]
                reward = compute_reward(
                    reward,
                    self.kl_ctl.value,
                    experience.kl,
                    action_mask=experience.action_mask,
                    num_actions=num_actions,
                    reward_clip_range=args.reward_clip_range,
                )

                if self.advantage_estimator == "gae":
                    experience.advantages, experience.returns = self.get_advantages_and_returns(
                        experience.values,
                        reward,
                        experience.action_mask,
                        generate_kwargs["gamma"],
                        generate_kwargs["lambd"],
                    )
                elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                    experience.returns = self.get_cumulative_returns(
                        reward,
                        experience.action_mask,
                        generate_kwargs["gamma"],
                    )
                    experience.advantages = deepcopy(experience.returns)
                else:
                    raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

                # calculate the return info.
                if not getattr(self, "packing_samples", False):
                    return_sums = reward.sum(dim=-1)
                else:
                    return_sums = torch.tensor(
                        [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                    )
                experience.info["return"] = return_sums
                # remove unnecessary info
                experience.kl = None
                del experience.info["num_actions"]
                experience.to_device("cpu")

                all_experiences.append(experience)
        ## == NaiveExperienceMaker.make_experience_list() == ##
        
        if self.critic is not None:
            for experience in all_experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)

        return all_experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM generation
        vllm_outputs_list = self._generate_vllm(all_prompts, all_labels, multimodal=self.multimodal, 
                                                n_samples_per_prompt=self.strategy.args.n_samples_per_prompt, 
                                                pack_size=self.strategy.args.micro_rollout_batch_size, **generate_kwargs)
        return vllm_outputs_list

    @torch.no_grad()
    def make_experience(self, vllm_outputs: vLLM_outputs) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, rewards, and kl divergence.
        The size of each element in vllm_outputs corresponds to self.strategy.args.micro_rollout_batch_size.

        This function does the following:
        1. Get feedbacks from a teacher model
        2. Add the feedbacks to the original prompts
        3. Generate new responses from the base model
        4. Get log_probs of the initial_model and base model to compute KL distance to refine reward values
        5. Pack the above information into Experience dataclass
        """
        start_time = time.time()
        # if dist.get_rank() == 0:
        #     logger.info(f"🚀 Starting experience making...")
        args = self.strategy.args
        assert args.n_feedback_samples_per_prompt == args.micro_train_batch_size, "n_feedback_samples_per_prompt must be equal to micro_train_batch_size"
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        prompts = vllm_outputs.prompts
        output_text = vllm_outputs.output_text
        labels = vllm_outputs.labels
        sequences = vllm_outputs.sequences
        attention_mask = vllm_outputs.attention_mask
        action_mask = vllm_outputs.action_mask
        num_actions = vllm_outputs.num_actions
        packed_seq_lens = vllm_outputs.packed_seq_lens

        # compute rewards from the original responses: base_rewards = [r1, r2, ...] with the size of self.strategy.args.micro_rollout_batch_size.
        base_rewards = self.custom_reward_func(output_text, labels)

        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.feedback_model, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()
        # get all_feedbacks: [[feedback1 for prompt1, feedback2 for prompt1, ...], [feedback1 for prompt2, feedback2 for prompt2, ...], ...].  len(all_feedbacks) == len(prompts), len(all_feedbacks[i]) == args.n_feedback_samples_per_prompt.
        all_feedbacks, question_prompts, model_responses, all_labels = self.get_feedbacks(prompts, output_text, labels, multimodal=self.multimodal)
        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.feedback_model, "sleep")
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        
        # add the feedbacks to the original prompts
        output_text_feedback = [
            FEEDBACK_PROMPT_BASE.format(question=prompt, model_answer=model_response, feedback=feedback)
            for prompt, model_response, feedbacks in zip(question_prompts, model_responses, all_feedbacks)
            for feedback in feedbacks # len(feedbacks) == args.n_feedback_samples_per_prompt
        ]

        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()
        # get new responses from the model with feedbacks.
        # new_samples_list = [vllm_outputs1, vllm_outputs2, ...], each vllm_outputsX has self.strategy.args.n_feedback_samples_per_prompt samples.
        new_samples_list = self._generate_vllm(output_text_feedback, all_labels, multimodal=self.multimodal, 
                                                    n_samples_per_prompt=1, pack_size=self.strategy.args.n_feedback_samples_per_prompt)
        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        # get new rewards from the new responses
        new_rewards = []
        for new_samples in new_samples_list:
            new_rewards.append(self.custom_reward_func(new_samples.output_text, new_samples.labels))
        
        if dist.get_rank() == 0:
            # only show the items where base_rewards == 0
            for i, (base, new) in enumerate(zip(base_rewards, new_rewards)):
                if base == 0:
                    print(f"score after feedback: {new}")

        # calculate the difference of rewards from the original responses
        # new_rewards = [[r1, r2, ...], [r1, r2, ...], ...], each list is the size of self.strategy.args.n_feedback_samples_per_prompt.
        assert len(base_rewards) == len(new_rewards), "The number of base rewards and new rewards must be the same. Current number of base rewards: {}, current number of new rewards: {}".format(len(base_rewards), len(new_rewards))
        diff_rewards = torch.zeros((len(base_rewards), len(new_rewards[0])))
        for sample_id, new_r in enumerate(new_rewards):
            for i in range(len(new_r)):
                if new_r[i] == 0 and base_rewards[sample_id] == 0:
                    diff_rewards[sample_id, i] = -1.0
                elif new_r[i] == 1 and base_rewards[sample_id] == 1:
                    diff_rewards[sample_id, i] = 0.5
                else:
                    diff_rewards[sample_id, i] = new_r[i] - base_rewards[sample_id]
        
        # make Experience objects from new_samples_list
        experiences = []
        # Extract all information from samples in one pass, convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in new_samples_list]
        attention_mask_list = [s.attention_mask for s in new_samples_list]
        num_actions_list = [s.num_actions for s in new_samples_list]
        packed_seq_lens_list = [s.packed_seq_lens for s in new_samples_list]

        # Move data to CPU for remote processing
        sequences_cpu_list = [seq.to("cpu") for seq in sequences_list]
        attention_mask_cpu_list = [mask.to("cpu") for mask in attention_mask_list]
        
        # Batch call initial model
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward_batch.remote(
                sequences=sequences_cpu_list,
                num_actions=num_actions_list,
                attention_mask=attention_mask_cpu_list,
                logps_allgather=[True] * len(new_samples_list),
                packed_seq_lens=packed_seq_lens_list,
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put([None] * len(new_samples_list))

        # Batch call actor model
        action_log_probs_list = []
        for seq, num_acts, attn_mask, packed_lens in zip(
            sequences_cpu_list, num_actions_list, attention_mask_cpu_list, packed_seq_lens_list
        ):
            action_log_probs = self.actor(
                seq.to(device),
                num_acts,
                attn_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=True,
                packed_seq_lens=packed_lens,
            )
            action_log_probs_list.append(action_log_probs)

        actor_value_rm_time = time.time() - start_time
        # Wait for all remote calls to complete
        start = time.time()
        base_action_log_probs_set = ray.get([base_action_log_probs_ref])
        base_action_log_probs_list = base_action_log_probs_set[0]
        wait_time = time.time() - start

        # Avoid CUDA OOM when colocate models
        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, rewards) in enumerate(
            zip(new_samples_list, action_log_probs_list, base_action_log_probs_list, diff_rewards)
        ):
            if base_action_log_probs is not None:
                base_action_log_probs = base_action_log_probs.to(device)

            # Broadcast rewards to all ring attention ranks when using remote RM
            rewards = [rewards]
            if self.remote_rm_url and self.strategy.ring_attn_group is not None:
                if self.strategy.ring_attn_rank == 0:
                    dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
                else:
                    dist.broadcast_object_list(rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group)
            r = rewards[0].to(device)

            if (self.initial_model is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    action_mask=samples.action_mask,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

            sequences = samples.sequences
            attention_mask = samples.attention_mask
            if not self.packing_samples:
                kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            else:
                num_actions = samples.num_actions
                packed_seq_lens = samples.packed_seq_lens
                if self.strategy.ring_attn_group is not None:
                    assert samples.pad_len is not None
                    sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                        pad_len=samples.pad_len,
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
                "response_length": samples.response_length,
                "total_length": samples.total_length,
                "num_actions": samples.num_actions,
            }

            if self.strategy.args.perf:
                self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
                self.perf_stats["wait_time"] += wait_time

            experience = Experience(
                sequences,
                action_log_probs,
                base_action_log_probs,
                None,
                None,
                None,
                attention_mask,
                samples.action_mask,
                info,
                kl,
                visual_inputs=samples.visual_inputs
            )
            experiences.append(experience)

        self.actor.train()  # Reset model state
        end_time = time.time()
        duration = end_time - start_time
        # if dist.get_rank() == 0:
        #     time_str = str(timedelta(seconds=duration)).split(".")[0]
        #     logger.info(f"✨ Experience making completed in {time_str}")
        return experiences


        # start = time.time()
        # sequences_cpu, attention_mask_cpu = (
        #     sequences.to("cpu"),
        #     attention_mask.to("cpu"),
        # )

        # visual_inputs_cpu = None
        # # init log probs for KL loss
        # if self.initial_model is not None:
        #     base_action_log_probs_ref = self.initial_model.forward.remote(
        #         sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu
        #     )

        #     if args.colocate_actor_ref or args.colocate_all_models:
        #         ray.get([base_action_log_probs_ref])
        #         ray.get([self.initial_model.empty_cache.remote()])
        # else:
        #     base_action_log_probs_ref = ray.put(None)

        # # rewards for baseline model without feedbacks
        # r_refs = []
        # # support remote RM API with ray
        # if not self.remote_rm_url:
        #     # for rm in self.reward_model:
        #     #     r_refs.append(
        #     #         rm.forward.remote(
        #     #             sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens, pad_sequence=True, visual_inputs=visual_inputs_cpu
        #     #         )
        #     #     )
        #     assert self.custom_reward_func is not None, "custom_reward_func must be provided when remote_rm_url is not provided"
        #     r_refs = self.custom_reward_func(output_text, labels)
        # else:
        #     # remote RM
        #     if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
        #         if not self.packing_samples:
        #             queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
        #         else:
        #             sequences_list = []
        #             offset = 0
        #             tokens_list = sequences_cpu.tolist()[0]
        #             for length in packed_seq_lens:
        #                 sequences_list.append(tokens_list[offset : offset + length])
        #                 offset += length
        #             queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

        #         if self.custom_reward_func:
        #             r = self.custom_reward_func.remote(queries, vllm_outputs.prompts, vllm_outputs.labels)
        #             r_refs.append(r)
        #         else:
        #             for rm in self.remote_rm_url:
        #                 r = remote_rm_fn_ray.remote(
        #                     rm, queries=queries, prompts=vllm_outputs.prompts, labels=vllm_outputs.labels
        #                 )
        #                 r_refs.append(r)
        #     else:
        #         r_refs.append(ray.put(None))

        # if args.colocate_all_models and not self.remote_rm_url:
        #     ray.get(r_refs)
        #     ray.get([self.reward_model[0].empty_cache.remote()])

        # # get all_feedbacks: [[feedback1 for prompt1, feedback2 for prompt1, ...], [feedback1 for prompt2, feedback2 for prompt2, ...], ...].  len(all_feedbacks) == len(prompts).
        # all_feedbacks, all_labels = self.get_feedbacks(prompts, output_text, labels, multimodal=self.multimodal)
        # # add the feedbacks to the original prompts
        # output_text_feedback = [
        #     FEEDBACK_PROMPT_BASE.format(question=prompt, model_answer=model_response, feedback=feedback)
        #     for prompt, model_response, feedbacks in zip(prompts, output_text, all_feedbacks)
        #     for feedback in feedbacks
        # ]
        # # get new responses from the model with feedbacks
        # new_vllm_outputs_list = self._generate_vllm(output_text_feedback, all_labels, multimodal=self.multimodal, n_samples_per_prompt=1)
        # # get new rewards from the new responses
        # new_r_refs = self.custom_reward_func(new_vllm_outputs_list.output_text, new_vllm_outputs_list.labels)
        # # calculate the difference of rewards from the original responses
        # feedback_r_refs = [new_r - r for new_r, r in zip(new_r_refs, r_refs)]
        # total_rewards = torch.tensor(feedback_r_refs, dim=0)

        # # log probs to compute KL to refine reward values
        # action_log_probs = self.actor(
        #     sequences, 
        #     num_actions, 
        #     attention_mask, 
        #     ring_attn_group=self.strategy.ring_attn_group,
        #     logps_allgather=True,
        #     packed_seq_lens=packed_seq_lens,
        #     visual_inputs=visual_inputs
        # )

        # # wait initial/critic/reward model done
        # ref_values = ray.get([base_action_log_probs_ref])
        # base_action_log_probs = ref_values[0]
        # if base_action_log_probs is not None:
        #     base_action_log_probs = base_action_log_probs.to(device)

        # if (self.initial_model is not None) and (not args.use_kl_loss):
        #     kl = compute_approx_kl(
        #         action_log_probs,
        #         base_action_log_probs,
        #         action_mask=action_mask,
        #         kl_estimator=self.strategy.args.kl_estimator,
        #     )
        # else:
        #     kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        # if not self.packing_samples:
        #     kl_mean = masked_mean(kl, action_mask, dim=-1)
        # else:
        #     if self.strategy.ring_attn_group is not None:
        #         assert vllm_outputs.pad_len is not None
        #         sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
        #             pad_len=vllm_outputs.pad_len,
        #             sequences=sequences,
        #             attention_mask=attention_mask,
        #             num_actions=num_actions,
        #             packed_seq_lens=packed_seq_lens,
        #             ring_attn_group=self.strategy.ring_attn_group,
        #             action_log_probs=action_log_probs,
        #             values=value,
        #             kl=kl,
        #         )
        #     # convert tensor into list of tensors so that it's easier to manipulate
        #     # within dataset.
        #     sequences = unpacking_samples(sequences, packed_seq_lens)
        #     attention_mask = None
        #     action_log_probs = unpacking_samples(action_log_probs, num_actions)
        #     if value is not None:
        #         value = unpacking_samples(value, num_actions)
        #     if base_action_log_probs is not None:
        #         base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

        #     kl = unpacking_samples(kl, num_actions)
        #     kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        # if not args.use_kl_loss:
        #     base_action_log_probs = None

        # info = {
        #     "kl": kl_mean,
        #     "reward": r,
        #     "response_length": vllm_outputs.response_length,
        #     "total_length": vllm_outputs.total_length,
        #     "num_actions": num_actions,
        # }

        # if args.colocate_actor_ref or args.colocate_all_models:
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()
        

        # for i in range(0, len(new_vllm_outputs_list), self.strategy.args.micro_rollout_batch_size):
        #     new_vllm_outputs = new_vllm_outputs_list[i : i + self.strategy.args.micro_rollout_batch_size]
        #     sequences = new_vllm_outputs.sequences
        #     attention_mask = new_vllm_outputs.attention_mask
        #     action_mask = new_vllm_outputs.action_mask
        #     num_actions = new_vllm_outputs.num_actions
        #     visual_inputs = new_vllm_outputs.visual_inputs
        #     packed_seq_lens = new_vllm_outputs.packed_seq_lens

        #     # log probs
        #     action_log_probs = self.actor(
        #         sequences, 
        #         num_actions, 
        #         attention_mask, 
        #         ring_attn_group=self.strategy.ring_attn_group,
        #         logps_allgather=True,
        #         packed_seq_lens=packed_seq_lens,
        #         visual_inputs=visual_inputs
        #     )
        
        

        # # # broadcast rewards to all ring attention ranks when using remote RM
        # # if self.remote_rm_url and self.strategy.ring_attn_group is not None:
        # #     if self.strategy.ring_attn_rank == 0:
        # #         dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
        # #     else:
        # #         dist.broadcast_object_list(
        # #             rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
        # #         )

        # # r = self.reward_fn(total_rewards) if len(total_rewards) > 0 else total_rewards[0]

        # # avoid CUDA OOM when colocate models
        # if args.colocate_critic_reward and not self.remote_rm_url:
        #     ray.get([self.reward_model[0].empty_cache.remote()])

        # if self.strategy.args.perf:
        #     self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
        #     self.perf_stats["wait_time"] += wait_time

        # experience = Experience(
        #     sequences,
        #     action_log_probs,
        #     base_action_log_probs,
        #     None,
        #     None,
        #     None,
        #     attention_mask,
        #     action_mask,
        #     info,
        #     kl,
        #     visual_inputs=visual_inputs
        # )

        # self.actor.train()  # reset model state
        # return experience

    def _generate_vllm(self, all_prompts: List[str], all_labels, multimodal=False, n_samples_per_prompt=1, pack_size=32, **kwargs) -> List[vLLM_outputs]:
        from vllm import SamplingParams
        self.response_length_list = []
        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # convert the prompts to chat-style
        all_prompts_chat = [self.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) 
                            for prompt in all_prompts]

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts_chat], [])
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])

        # Distribute requests to engines and collect responses to outputs
        refs = []
        if multimodal:
            # For VLM
            for i, llm in enumerate(llms):
                messages = all_prompts[i * batch_size : (i + 1) * batch_size]
                if messages:
                    prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images = [self.data_processor.get_images_from_messages(m) for m in messages]
                    vllm_inputs = [{
                            "prompt": p,
                            "multi_modal_data":{"image": imgs} if imgs else None,
                            "mm_processor_kwargs": kwargs["processor_kwargs"]
                        } for p, imgs in zip(prompts,images)]
                    refs.append(
                        llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                    )
        else:
            for i, llm in enumerate(llms):
                messages = all_prompts[i * batch_size : (i + 1) * batch_size]
                if messages:
                    # prompts already are chat-style due to PromptDataset in ActorModelRayActor_TG class
                    refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=messages))

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


        vllm_outputs_list = []
        for i in range(0, len(all_outputs), pack_size):
            outputs = all_outputs[i : i + pack_size]
            prompts = all_prompts[i : i + pack_size]
            labels = all_labels[i : i + pack_size]

            output_list = []
            for output in outputs:
                output_list.append(output.outputs[0].text)

            if not self.packing_samples or self.multimodal:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
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
                # Collect for visual input
                
                visual_inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
                visual_inputs.pop("input_ids")
                visual_inputs.pop("attention_mask")
                visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()}
                self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())
                vllm_outputs_list.append(
                    vLLM_outputs(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        visual_inputs=visual_inputs,
                        labels=labels,
                        pad_len=None,
                        output_text=output_list
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                # This will lead to better inference performance in terms of effificency.
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                self.response_length_list.extend(num_actions)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                vllm_outputs_list.append(
                    vLLM_outputs(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        visual_inputs=None,
                        labels=labels,
                        pad_len=pad_len,
                        output_text=output_list
                    )
                )
        return vllm_outputs_list

    
    def get_feedbacks(self, all_prompts: list[str], output_text: list[str], labels: list[str], multimodal=False) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        1. Extract model responses from output_text (i.e, extract "Answer: ..." from "Thought: ..." and "Answer: ...")
        2. Create prompts for feedback model
        3. Expand prompt list based on the number of samples per prompt
        4. Distribute requests to engines and collect responses to outputs
        5. Return feedbacks


        all_prompts: list[str]: all_prompts[0] is like this:
        '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer a given question using the following output format.\n\n## Output Format\nThought: provide your thoughts behind the answer\nAnswer: only provide the choice label from the given choices, e.g. Answer: C\n\n## Question\nThe king needed to feel safe, where did he go?\n\n## Choices\nA: castle\nB: throne room\nC: deck of cards\nD: fort\nE: court\n<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n<|im_start|>assistant\n'
        
        output_text: list[str]: output_text[0] is like this:
        'Thought: The king, as a figurehead, typically needs a secure location to feel safe. Castles have long been recognized as strong and secure structures, often seen in historical depictions or stories of kings and warriors. Throne rooms are associated with royal power but are not identified as primary safety havens for a king in general public imagination. The deck of cards is not a secure place for any individual. Forts are strategic locations meant for defensive purposes, not necessarily for general safety. Courts are locations for legal and political decisions but are not typically seen as safe havens for a king. Given these considerations, the castle is the most logical and historically accurate choice.\n\nAnswer: A<|im_end|>'
        """
        from vllm import SamplingParams
        self.response_length_list = []
        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.feedback_model) <= world_size:
            llms = [self.feedback_model[rank % len(self.feedback_model)]]
        else:
            llms = self.feedback_model[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            min_tokens=1,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
        )

        # create prompts for feedback model
        question_prompts = []
        for prompt in all_prompts:
            question_choice_special_chars = prompt.split("## Question")[1].strip()
            question_choice = question_choice_special_chars.split("<|im_end|>")[0].strip()
            question_prompts.append(question_choice)

        model_responses = []
        for output in output_text:
            thought = ""
            answer = ""
            # "Thought:" が含まれているかチェック
            if "Thought:" in output:
                try:
                    thought_part = output.split("Thought:", 1)[1]
                    # "Answer:" でさらに分けられるかチェック
                    if "Answer:" in thought_part:
                        thought = thought_part.split("Answer:", 1)[0].strip()
                        answer_part = thought_part.split("Answer:", 1)[1]
                    else:
                        thought = thought_part.strip()
                        answer_part = ""
                except IndexError:
                    thought = "No thought is provided."
                    answer_part = ""
            else:
                answer_part = output  # Thoughtがない場合は全文をanswerとして扱ってもOK（用途による）

            # "Answer:" を含んでいる or fallback で answer_part が存在する場合
            if "Answer:" in output or answer_part:
                try:
                    answer = answer_part.split("<|im_end|>")[0].strip()
                except IndexError:
                    answer = answer_part.strip()
            model_responses.append(thought + "\n" + answer)

        all_prompts_feedback = [FEEDBACK_PROMPT.format(question=prompt, answer=label, model_answer=model_response) for prompt, label, model_response in zip(question_prompts, labels, model_responses)]
        all_prompts_chat = [self.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in all_prompts_feedback]

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_feedback_samples_per_prompt for prompt in all_prompts_chat], [])
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        all_labels = sum([[label] * args.n_feedback_samples_per_prompt for label in labels], [])

        # Distribute requests to engines and collect responses to outputs
        refs = []
        if multimodal:
            # For VLM
            for i, llm in enumerate(llms):
                messages = all_prompts[i * batch_size : (i + 1) * batch_size]
                if messages:
                    prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images = [self.data_processor.get_images_from_messages(m) for m in messages]
                    vllm_inputs = [{
                            "prompt": p,
                            "multi_modal_data":{"image": imgs} if imgs else None,
                            "mm_processor_kwargs": kwargs["processor_kwargs"]
                        } for p, imgs in zip(prompts,images)]
                    refs.append(
                        llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                    )
        else:
            for i, llm in enumerate(llms):
                messages = all_prompts[i * batch_size : (i + 1) * batch_size]
                if messages:
                    # prompts already are chat-style due to PromptDataset in ActorModelRayActor_TG class
                    refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=messages))

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

        feedbacks = []
        for output in all_outputs:
            feedbacks.append(output.outputs[0].text)
        # divide the feedbacks by the number of samples per prompt
        feedbacks = [feedbacks[i:i+args.n_feedback_samples_per_prompt] for i in range(0, len(feedbacks), args.n_feedback_samples_per_prompt)]
        
        return feedbacks, question_prompts, model_responses, all_labels
    
    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None



class RemoteExperienceMaker_GUI(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, parallel_env: ParallelDesktopEnv = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert parallel_env is not None
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.parallel_env = parallel_env
        self.gamma = 0.9
        self.gae_lambda = 0.95
        self.mini_batch_size = 4

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(self, task_config: Dict, task_id: int, **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """


        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        
        args = self.strategy.args

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # Reset environment and rollout episode
        obs_list = self.parallel_env.reset(task_config)
        
        # Explore GUI environment and collect trajectories, samples_list: [Samples_GUI, ...]. len(samples_list) = num_parallel_envs
        samples_list = self.rollout_episode(obs_list, task_id, task_config, args, action_reference=args.action_reference)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        
        # flatten samples_list
        flattened_samples_list = []
        for env_id in range(len(samples_list)):
            flattened_samples_list.extend(samples_list[env_id])

        
        
        ## TODO: implement make_experience for GUI agent
        experiences = []
        mini_batch = []
        for sample in tqdm(flattened_samples_list, desc="make_experience", disable=not self.strategy.is_rank_0()):
            if len(mini_batch) < self.mini_batch_size:
                mini_batch.append(sample)
                continue
            experiences.append(self.make_experience(mini_batch).to_device("cpu"))
            mini_batch = []
        if mini_batch:
            experiences.append(self.make_experience(mini_batch).to_device("cpu"))

        
        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        
        
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences
    

    
    @torch.no_grad()
    def rollout_episode(self, obs_list, task_idx, task_config, args, action_reference: bool = False):
        """
        obs_list is init states of the environment
            : List[Dict] = [{"screenshot": screenshot, "instruction": instruction}, ...]. 
            : len(obs_list) = num_parallel_envs
        
        This will:
        - rollout k trajectories for each environment
        - using internal reward model(s) to get rewards
        - return samples_list: [Samples_GUI, ...], len(samples_list) = num_parallel_envs
            - Samples_GUI has information of each step in the trajectory
        """
        num_envs = len(obs_list)
        previous_images_per_env = [[] for _ in range(num_envs)]
        previous_actions_per_env = [[] for _ in range(num_envs)]
        message = build_singleturn_prompt(
            obs_list[0][0]["obs"], 
            previous_images=None,
            previous_actions=None,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            action_reference=action_reference
        )
        messages = message * args.n_samples_per_prompt

        done_flags = [False] * num_envs
        step_indices = [0] * num_envs
        all_examples = [[] for _ in range(num_envs)]

        # Track resources for cleanup
        previous_all_prompts_text = [None] * num_envs
        samples_list = [[] for _ in range(num_envs)]

        # episode starts
        # Continue until all environments are done or reach max steps
        # with self.rollout_sharding_manager:
        while not all(done_flags) and any(idx < args.env_max_steps for idx in step_indices):
            # For environments that are still active
            active_indices = [i for i, done in enumerate(done_flags) if not done and step_indices[i] < args.env_max_steps]
            # create the correspondances between active_indices and orders such as {0: <active_index>, 1: <active_index>, ...}
            active_indices_orders = {active_indices[i]: i for i in range(len(active_indices))}

            if not active_indices:
                break   # All environments are done or reached max steps
            
            # Prepare prompts only for active environments
            active_messages = [messages[i] for i in active_indices]

            prompt_texts = [
                self.data_processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
                for msg in active_messages
            ]

            ## when we want to handle manually.
            ## Process image information for active environments
            # image_inputs, video_inputs = process_vision_info(active_messages)
            # inputs = self.data_processor(
            #     text=prompt_texts,
            #     images=image_inputs,
            #     videos=video_inputs,
            #     padding=True,
            #     return_tensors="pt",
            # )
            # prompt_ids = inputs.input_ids
            # prompt_mask = inputs.attention_mask
            image_inputs, video_inputs = process_vision_info(active_messages)
            visual_inputs = self.data_processor(
                text=prompt_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            visual_inputs.pop("input_ids")
            visual_inputs.pop("attention_mask")

            ## need to resize the image ([self.min_pixels, self.max_pixels])
            active_images = []
            for i in range(num_envs):
                if i in active_indices:
                    try:
                        image = Image.open(BytesIO(obs_list[i][-1]["obs"]["screenshot"]))
                        if image.width * image.height > args.max_pixels:
                            resize_factor = math.sqrt(args.max_pixels / (image.width * image.height))
                            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                            image = image.resize((width, height))
                        if image.width * image.height < args.min_pixels:
                            resize_factor = math.sqrt(args.min_pixels / (image.width * image.height))
                            width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
                            image = image.resize((width, height))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        active_images.append(image)
                    except Exception as e:
                        logger.error(f"Error processing image for environment {i}: {e}")
                        # Provide a placeholder image
                        placeholder = Image.new('RGB', (224, 224), color='gray')
                        active_images.append(placeholder)
                else:
                    active_images.append(None)

            # Prepare inputs for vLLM
            all_prompts_text = []
            active_multi_modal_data = [
                {
                    "image": previous_images_per_env[i][-args.history_n:] + [active_images[i]],
                }
                for i in active_indices
            ]
            for prompt, multi_modal_data in zip(prompt_texts, active_multi_modal_data):
                all_prompts_text.append({
                    "prompt": prompt.replace('<|image_pad|>', '').replace('<|vision_start|>', '<|vision_start|><|image_pad|>'),
                    "multi_modal_data": multi_modal_data,  # Will be handled by vLLM's preprocessor
                })


            try:
                all_outputs = self._generate_vllm_text(all_prompts_text)
                max_input_len, max_output_len = 0, 0
                completion_ids = []
                for output in all_outputs:
                    completion_ids.append(output.outputs[0].token_ids)
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))
            except Exception as e:
                logger.error(f"Error in vLLM generation: {e}")
                logger.error(traceback.format_exc())
                logger.info(f"Use previous_all_prompts_text: {previous_all_prompts_text} for rollout")
                all_outputs = self._generate_vllm_text(previous_all_prompts_text)
                completion_ids = []
                for output in all_outputs:
                    completion_ids.append(output.outputs[0].token_ids)

            completions_text = self.data_processor.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            print(f"completions_text: {completions_text}")
            
            # parse the completions_text to get the action commands represented by pyautogui
            parsed_actions = []
            for completion in completions_text:
                try:
                    parsed_action = parse_action_qwen2vl(completion, args.bin_nums)
                    parsed_actions.append(parsed_action)
                except Exception as e:
                    # print(f"Error when parsing action: {completion}, with error:\n{e}")
                    parsed_actions.append([None])
            
            active_actions = []
            active_actions_parse_rewards = []
            for parsed_action in parsed_actions:
                each_env_actions = []
                action_parse_reward = 0.0
                for action in parsed_action:
                    if action is not None:
                        try:
                            pyautogui_code = parsing_response_to_pyautogui_code(action, args.screen_height, args.screen_width)
                            action_parse_reward += 0.2
                        except Exception as e:
                            pyautogui_code = f"import pyautogui\nimport time\n"
                            # action_parse_reward -= 0.2
                    else:
                        pyautogui_code = f"import pyautogui\nimport time\n"
                        # action_parse_reward -= 0.2
                    each_env_actions.append(pyautogui_code)
                active_actions.append(each_env_actions)
                active_actions_parse_rewards.append(action_parse_reward)
            
            # Prepare full action list with placeholders for inactive environments
            all_actions = [None] * num_envs
            all_actions_parse_rewards = [0.0] * num_envs
            for i, idx in enumerate(active_indices):
                all_actions[idx] = active_actions[i]
                all_actions_parse_rewards[idx] = active_actions_parse_rewards[i]
            
            # get the next obs from the environment via extracted actions: len(obs_list) == num_parallel_envs
            try:
                obs_list = self.parallel_env.step_all(all_actions, step_indices, active_indices, task_idx)  # [[obs, reward, done, info], ...].
            except Exception as e:
                logger.error(f"Error in environment step: {e}")
                logger.error(traceback.format_exc())
                # Continue with existing obs_list

            # Update state for each environment
            pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            prompts = []
            for rank in range(num_envs):
                # skip environments that are already done
                if done_flags[rank]:
                    continue
                
                obs = obs_list[rank]
                if rank in active_indices:
                    # Update step index
                    step_indices[rank] += 1
                    # parse reward and done state
                    reward = obs[-1].get("reward", 0)
                    is_done = obs[-1].get("done", False)

                    # store the screenshot and action
                    previous_images_per_env[rank].append(active_images[rank])
                    previous_actions_per_env[rank].append(all_actions[rank])

                    # Update done flag and examples
                    if is_done or step_indices[rank] >= args.env_max_steps - 1:
                        done_flags[rank] = True
                    
                    # Preprocess each active sample and storefor RL training
                    output = all_outputs[active_indices_orders[rank]]
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)
                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
                    # concat input and output
                    sequence = []
                    sequence.append(input_ids + output_ids)
                    current_prompt = active_messages[rank][-1]["content"][-1]["text"] if "text" in active_messages[rank][-1]["content"][-1]["type"] else active_messages[rank][-1]["content"][-2]["text"]
                    prompts.append(current_prompt)
                    # prepare the prompts for the next step
                    if not done_flags[rank]:
                        message = build_singleturn_prompt(
                            obs[-1]["obs"],
                            previous_images=previous_images_per_env[rank][-args.history_n:],
                            previous_actions=previous_actions_per_env[rank][-args.history_n:],
                            max_pixels=args.max_pixels,
                            min_pixels=args.min_pixels,
                            action_reference=action_reference
                        )
                        messages[rank] = message[0]

                    sequence = torch.tensor(sequence)
                    sequence, attention_mask, action_mask = self.actor.process_sequence(
                            sequence, max_input_len, eos_token_id, pad_token_id
                        )
                    format_reward = gui_agent_format_reward(completions_text[active_indices_orders[rank]])
                    language_reward = english_format_reward(completions_text[active_indices_orders[rank]])
                    reward = all_actions_parse_rewards[rank] + format_reward + language_reward
                    samples_list[rank].append(
                        Samples_GUI(
                            sequences=sequence,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            num_actions=action_mask.size(1),
                            packed_seq_lens=None,
                            response_length=action_mask.float().sum(dim=-1),
                            total_length=attention_mask.float().sum(dim=-1),
                            prompts=prompts,
                            visual_inputs=visual_inputs,
                            labels=None,
                            pad_len=None,
                            reward=reward,
                            completion=completions_text[active_indices_orders[rank]],
                        )
                    )
                    
                else:  # This environment reached max steps or is done
                    done_flags[rank] = True

            # deep copy the prompts to previous_all_prompts_text
            previous_all_prompts_text = copy.deepcopy(all_prompts_text)
            
        
        # evaluate the final state of the all environments
        try:
            goal_rewards = self.parallel_env.evaluate_all(task_config)
        except Exception as e:
            logger.error(f"Error in environment evaluation: {e}")
            logger.error(traceback.format_exc())
            goal_rewards = [0.0] * num_envs

        for rank in range(len(samples_list)):
            samples_list[rank][-1]['reward'] += goal_rewards[rank]

        # compute final reward by accumulated rewards
        for rank in range(len(samples_list)):
            for step in reversed(range(len(samples_list[rank]))):
                if step == len(samples_list[rank]) - 1:
                    continue
                else:
                    samples_list[rank][step]['reward'] = samples_list[rank][step]['reward'] + self.gamma * samples_list[rank][step + 1]['reward']
        return samples_list
        
    
    def _generate_vllm_text(self, all_prompts_text: List[Dict], **kwargs) -> List[Samples]:
        """
        Generate samples using vLLM.
        """
        from vllm import SamplingParams
        self.response_length_list = []
        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("generate_max_len", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        refs = []
        for i, llm in enumerate(llms):
            refs.append(llm.add_request.remote(rank, sampling_params=sampling_params, vllm_vision_input=all_prompts_text))

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
        return all_outputs

    
    
    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_labels, **generate_kwargs)

        # vLLM generation
        samples = self._generate_vllm(all_prompts, all_labels, **generate_kwargs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: list[Samples_GUI]) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = []
        attention_mask = []
        action_mask = []
        num_actions = []
        packed_seq_lens = []
        visual_inputs = []
        for sample in samples:
            sequences.append(sample.sequences)
            attention_mask.append(sample.attention_mask)
            action_mask.append(sample.action_mask)
            num_actions.append(sample.num_actions)
            packed_seq_lens.append(sample.packed_seq_lens)
            visual_inputs.append(sample.visual_inputs)

        # convert to tensor
        sequences = torch.stack(sequences)
        attention_mask = torch.stack(attention_mask)
        action_mask = torch.stack(action_mask)
        num_actions = torch.stack(num_actions)
        packed_seq_lens = torch.stack(packed_seq_lens)
        visual_inputs = torch.stack(visual_inputs)
        
        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )
        visual_inputs_cpu = None
        if visual_inputs is not None:
            visual_inputs_cpu = {k: v.to("cpu") for k, v in visual_inputs.items()}        
        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens, visual_inputs=visual_inputs_cpu
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(
                    rm.forward.remote(
                        sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens, pad_sequence=True, visual_inputs=visual_inputs_cpu
                    )
                )
        else:
            # remote RM
            if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                if not self.packing_samples:
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens:
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

                if self.custom_reward_func:
                    r = self.custom_reward_func.remote(queries, samples.prompts, samples.labels)
                    r_refs.append(r)
                else:
                    for rm in self.remote_rm_url:
                        r = remote_rm_fn_ray.remote(
                            rm, queries=queries, prompts=samples.prompts, labels=samples.labels
                        )
                        r_refs.append(r)
            else:
                r_refs.append(ray.put(None))

        if args.colocate_all_models and not self.remote_rm_url:
            ray.get(r_refs)
            ray.get([self.reward_model[0].empty_cache.remote()])

        # log probs
        action_log_probs = self.actor(
            sequences, 
            num_actions, 
            attention_mask, 
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
            visual_inputs=visual_inputs
        )
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)

        # broadcast rewards to all ring attention ranks when using remote RM
        if self.remote_rm_url and self.strategy.ring_attn_group is not None:
            if self.strategy.ring_attn_rank == 0:
                dist.broadcast_object_list(rewards, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                dist.broadcast_object_list(
                    rewards, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )

        total_rewards = [r.pop('rewards').to(device) if isinstance(r,dict) else r.to(device) for r in rewards]
        specific_rewards = {}
        for r in rewards:
            if isinstance(r,dict):
                for k in r.keys():
                    r[k] = r[k].to(device)
                specific_rewards.update(r)

        r = self.reward_fn(total_rewards) if len(total_rewards) > 0 else total_rewards[0]

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            if self.strategy.ring_attn_group is not None:
                assert samples.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=samples.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=value,
                    kl=kl,
                )
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            **specific_rewards
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
            visual_inputs=visual_inputs
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams
        self.response_length_list = []
        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])

        # Distribute requests to engines and collect responses to outputs
        refs = []
        # For VLM
        for i, llm in enumerate(llms):
            messages = all_prompts[i * batch_size : (i + 1) * batch_size]
            if messages:
                prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images = [self.data_processor.get_images_from_messages(m) for m in messages]
                vllm_inputs = [{
                        "prompt": p,
                        "multi_modal_data":{"image": imgs} if imgs else None,
                        "mm_processor_kwargs": kwargs["processor_kwargs"]
                    } for p, imgs in zip(prompts,images)]
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs)
                )


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

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
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
                # Collect for visual input
                
                visual_inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
                visual_inputs.pop("input_ids")
                visual_inputs.pop("attention_mask")
                visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()}
                self.response_length_list.extend(attention_mask.float().sum(dim=-1).tolist())
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        visual_inputs=visual_inputs,
                        labels=labels,
                        pad_len=None,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                self.response_length_list.extend(num_actions)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                # Collect for visual input
                visual_inputs = self.data_processor(prompts, self.prompt_max_len, device="cuda")
                visual_inputs.pop("input_ids")
                visual_inputs.pop("attention_mask")
                visual_inputs = {k: v.to("cuda") for k, v in visual_inputs.items()}
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        visual_inputs=visual_inputs,
                        labels=labels,
                        pad_len=pad_len,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None