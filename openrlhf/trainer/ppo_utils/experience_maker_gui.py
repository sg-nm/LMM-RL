import os
import time
import copy
import math
import gc
import traceback
import logging
from datetime import timedelta
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
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

from openrlhf.trainer.ppo_utils.experience_maker import NaiveExperienceMaker, Samples
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples, compute_uniform_reward
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn_ray

from qwen_vl_utils import process_vision_info

from openrlhf.trainer.ppo_utils.rewards import gui_agent_format_reward, english_format_reward
from openrlhf.textgrad.feedback_vllm import FEEDBACK_PROMPT, FEEDBACK_PROMPT_BASE, PROMPT_BASE

from gui_env.robust_parallel_desktop_env import ParallelDesktopEnv
from gui_env.agent_utils import build_prompt_for_actor, parse_action_qwen2vl, parsing_response_to_pyautogui_code


logger = logging.getLogger(__name__)

def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor

def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor

@dataclass
class Experience_GUI:
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



class RemoteExperienceMaker_GUI(NaiveExperienceMaker):
    def __init__(self, 
                 actor, critic, reward_model, initial_model, tokenizer, data_processor, feedback_data_processor,
                 prompt_max_len, kl_controller, strategy, remote_rm_url, reward_fn,
                 vllm_engines: List = None, 
                 feedback_model = None, 
                 packing_samples=False, 
                 parallel_env: ParallelDesktopEnv = None,
                 grounding_model = None,
                 grounding_data_processor = None,
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
        self.parallel_env = parallel_env
        self.vllm_engines = vllm_engines
        self.feedback_model = feedback_model
        self.feedback_data_processor = feedback_data_processor
        self.grounding_model = grounding_model
        self.grounding_data_processor = grounding_data_processor
        self.packing_samples = packing_samples
        self.gamma = 0.9
        self.gae_lambda = 0.95
        self.mini_batch_size = 4

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(self, task_config: Dict, task_id: int, **generate_kwargs) -> List[Experience_GUI]:
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
        obs_list = self.parallel_env.reset()
        
        # Explore GUI environment and collect trajectories, samples_list: [Samples_GUI, ...]. len(samples_list) = num_parallel_envs
        samples_list = self.rollout_episode(obs_list, args, action_reference=args.action_reference)

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
    def rollout_episode(self, obs_list, args, action_reference: bool = False):
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

        messages = []
        for obs in obs_list:
            message = build_prompt_for_actor(
                obs[0]["obs"], 
                previous_images=None,
                previous_actions=None,
                max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                action_reference=action_reference
            )
            messages.append(message)

        # message = build_prompt_for_actor(
        #     obs_list[0][0]["obs"], 
        #     previous_images=None,
        #     previous_actions=None,
        #     max_pixels=args.max_pixels,
        #     min_pixels=args.min_pixels,
        #     action_reference=action_reference
        # )
        # messages = message * args.n_samples_per_prompt # for GRPO

        done_flags = [False] * num_envs
        step_indices = [0] * num_envs
        all_examples = [[] for _ in range(num_envs)]

        # Track resources for cleanup
        previous_all_prompts_text = [None] * num_envs
        samples_list = [[] for _ in range(num_envs)]

        # episode starts. Continue until all environments are done or reach max steps
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

            try:
                all_outputs = self._generate_vllm(active_messages, active_multi_modal_data)


            except Exception as e:
                logger.error(f"Error in vLLM generation: {e}")
            
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

    def _generate_vllm(self, all_messages: List[dict], active_multi_modal_data: List[dict], **kwargs) -> RolloutStorage:
        from vllm import SamplingParams
        self.response_length_list = []
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 512),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        batch_size = (len(all_messages) + len(llms) - 1) // len(llms)

        # Distribute requests to engines and collect responses to outputs
        refs = []
        for i, llm in enumerate(llms):
            messages = all_messages[i * batch_size : (i + 1) * batch_size]
            multi_modal_data = active_multi_modal_data[i * batch_size : (i + 1) * batch_size]
            prompts = self.data_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            vllm_inputs = [
                {
                    "prompt": p,
                    "multi_modal_data": m,
                    # "mm_processor_kwargs": kwargs["processor_kwargs"]
                }
                for p, m in zip(prompts, multi_modal_data)
            ]
            refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))

        ray.get(refs)
        # Make sure all requests are sent.
        if self.strategy.ring_attn_group is None:
            torch.distributed.barrier()
        else:
            time.sleep(2)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        model_responses_list = []
        for output in all_outputs:
            model_responses_list.append(output.outputs[0].text)

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

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None