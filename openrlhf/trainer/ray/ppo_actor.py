from contextlib import ExitStack
import itertools
import math
import os
import re
import socket
import json
import random
import time
import copy
import yaml
from typing import Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from torch.optim import Optimizer
from PIL import Image
import numpy as np

import deepspeed
import ray
import torch
import torch.distributed
from transformers.trainer import get_scheduler
from transformers import AutoProcessor
from datasets import load_dataset

import gymnasium as gym


from openrlhf.datasets import PromptDataset, SFTDataset, CommonSenseQADataset, MathDataset
from openrlhf.models import Actor, MultiModalActor
from openrlhf.models.lmm_kits.utils import get_data_processor
from openrlhf.trainer import PPOTrainer
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker, RemoteExperienceMaker_GUI, RemoteExperienceMaker_TG
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils import blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import init_process_group
from peft.peft_model import PeftModel
from .launcher import BasePPORole
from .utils import get_physical_gpu_id

from openrlhf.trainer.ppo_utils.experience_maker_card_game import RemoteExperienceMaker_CardGame, RemoteExperienceMaker_CardGame_REINFORCE
# from gui_env.robust_parallel_desktop_env import ParallelDesktopEnv
from openrlhf.textgrad.custom_reward_functions import check_answer_commonsense_qa, check_answer_math
from openrlhf.trainer.ppo_utils.replay_buffer import ReplayBuffer_CARDGAME

from card_env.gym_cards.prompt_lib import PROMPT_FN, example_json_text
from card_env.gym_cards.config_dataclass import EnvConfig, PromptConfig, load_config_from_yaml, EvalEnvConfig
from card_env.gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline


def make_env(env_config: Union[EnvConfig, EvalEnvConfig], language_only=False, seed=42, ood=False):
    def _init():
        config_dict = {k: v for k, v in vars(env_config).items() if k != "id" and k != "num_steps" and k != "num_evaluations"}
        config_dict["language_only"] = language_only
        config_dict["seed"] = seed
        config_dict["ood"] = ood
        return GeneralPointEnv_oneline(**config_dict)
        # env = GeneralPointEnv_oneline(**vars(env_config), language_only=language_only)
        # return env
    return _init


class ActorPPOTrainer(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote

        self.experience_maker = RemoteExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.data_processor,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and (self.strategy.args.colocate_all_models or self.strategy.args.colocate_actor_vllm):
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch.distributed.barrier()

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}

        # 2. triger remote critic model training
        if self.critic_train_remote:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.reload_states.remote())

            critic_status_ref = self.critic.fit.remote()

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref))
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.offload_states.remote())

        if self.strategy.args.colocate_all_models:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.reload_states()

            status.update(super().ppo_train(global_steps))

            if self.strategy.args.deepspeed_enable_sleep:
                self.offload_states()

            torch.cuda.empty_cache()

            # # 4. broadcast weights to vllm engines
            # if self.vllm_engines is not None:
            #     if self.strategy.args.vllm_enable_sleep:
            #         batch_vllm_engine_call(self.vllm_engines, "wake_up")

            #     torch.distributed.barrier()
            #     torch.cuda.synchronize()
            #     self._broadcast_to_vllm()

            #     if self.strategy.args.vllm_enable_sleep:
            #         batch_vllm_engine_call(self.vllm_engines, "sleep")
            #         torch.distributed.barrier()
            #         torch.cuda.synchronize()

        # 5. wait remote critic model training done
        if self.critic_train_remote and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def _get_leaf_modules(self,model,use_lora):
        leaf_modules = []
        lora_module_keyword = ["lora_","base_layer"]
        class IsoParamWrapper:
            """
            Some modules may have isolated parameters that are not in submodules.
            This class wraps such parameters in a module so that they can be treated uniformly.
            """
            def __init__(self, name, parameter):
                self.name = name
                self.parameter = parameter

            def named_parameters(self,prefix=None):
                # self.name is already the full name. No need to add prefix
                return [(self.name,self.parameter)]

        for name,module in model.named_modules():
            if len(list(module.children())) == 0 or (use_lora and hasattr(module,"base_layer")):
                leaf_modules.append((name,module))
            else:
                #find isolated parameter
                for pname, p in module.named_parameters(recurse=False,prefix=name):
                    leaf_modules.append((pname,IsoParamWrapper(pname,p)))
        if use_lora:
            leaf_modules = [(n,m) for n,m in leaf_modules if not any([keyword in n for keyword in lora_module_keyword])]
        return leaf_modules

    def _broadcast_module(self,module,prefix=None,empty_cache=False,need_gather=False):
        count, num_params = 0, len(list(module.named_parameters()))
        for name, param in module.named_parameters(prefix=prefix):
            # broadcast
            count += 1
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=empty_cache and count==num_params,
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=empty_cache and count==num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        use_lora = False
        if isinstance(model, PeftModel):
            lora_model = model.base_model
            model = lora_model.model
            use_lora = True

        leaf_modules = self._get_leaf_modules(model,use_lora) # parameters of leaf_modules should not overlap
        count, num_modules = 0, len(leaf_modules)
        for key,module in leaf_modules:
            count += 1
            with ExitStack() as stack:
                need_gather = self.strategy.args.zero_stage == 3
                module_name = key.split(".")[-1]
                raw_module = module
                if use_lora and hasattr(raw_module, "base_layer"):
                    #This is a lora module
                    stack.enter_context(deepspeed.zero.GatheredParameters(raw_module.parameters(), enabled=need_gather))
                    raw_module.merge(safe_merge=True)
                    # we don't really replace the module, but we utilize _replace_module to get the merged module
                    fake_parent = type('FakeParent',(),{})()
                    lora_model._replace_module(fake_parent, module_name, raw_module.get_base_layer(), raw_module)
                    module = getattr(fake_parent, module_name)
                    need_gather = False
                    stack.callback(raw_module.unmerge)

                self._broadcast_module(module, prefix=key, empty_cache=count==num_modules,need_gather=need_gather)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.processor,
                save_path,
            )
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
        torch.distributed.barrier()

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)


class ActorPPOTrainer_TG(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        feedback_model = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray and TextFeedback.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            feedback_model (FeedbackModel_vllm, optional): feedback model for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.feedback_model = feedback_model
        self.critic_train_remote = critic_train_remote

        self.experience_maker = RemoteExperienceMaker_TG(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.data_processor,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            feedback_model=self.feedback_model,
            packing_samples=self.strategy.args.packing_samples,
            multimodal=self.strategy.args.multimodal,
            feedback_rewards=self.strategy.args.feedback_rewards,
        )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.strategy.args.colocate_all_models:
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch.distributed.barrier()

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        if global_steps == 0 or global_steps == 1:
            if self.strategy.args.dataset_name == "math":
                eval_accuracy = self.evaluate_math()
            else:
                eval_accuracy = self.evaluate()
            print(f"Init eval accuracy: {eval_accuracy}")
            status["eval_accuracy"] = eval_accuracy
            if self.strategy.is_rank_0() and self._wandb is not None:
                logs = {
                    "train/%s" % k: v
                    for k, v in {**status, "global_step": global_steps}.items()
                }
                self._wandb.log(logs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        # 2. triger remote critic model training
        if self.critic_train_remote:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.reload_states.remote())

            critic_status_ref = self.critic.fit.remote()

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref))
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.offload_states.remote())

        if self.strategy.args.colocate_all_models:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.reload_states()

            status.update(super().ppo_train(global_steps))

            if self.strategy.args.deepspeed_enable_sleep:
                self.offload_states()

            torch.cuda.empty_cache()

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                torch.distributed.barrier()
                torch.cuda.synchronize()
                self._broadcast_to_vllm()

                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        # 5. wait remote critic model training done
        if self.critic_train_remote and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        # 6. evaluate
        if self.strategy.args.eval:
            if self.strategy.args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
                batch_vllm_engine_call(self.vllm_engines, "wake_up")
                torch.distributed.barrier()
                torch.cuda.synchronize()
            if self.strategy.args.dataset_name == "math":
                eval_accuracy = self.evaluate_math()
            else:
                eval_accuracy = self.evaluate()
            status["eval_accuracy"] = eval_accuracy

            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            torch.cuda.synchronize()

        return status

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def _get_leaf_modules(self,model,use_lora):
        leaf_modules = []
        lora_module_keyword = ["lora_","base_layer"]
        class IsoParamWrapper:
            """
            Some modules may have isolated parameters that are not in submodules.
            This class wraps such parameters in a module so that they can be treated uniformly.
            """
            def __init__(self, name, parameter):
                self.name = name
                self.parameter = parameter

            def named_parameters(self,prefix=None):
                # self.name is already the full name. No need to add prefix
                return [(self.name,self.parameter)]

        for name,module in model.named_modules():
            if len(list(module.children())) == 0 or (use_lora and hasattr(module,"base_layer")):
                leaf_modules.append((name,module))
            else:
                #find isolated parameter
                for pname, p in module.named_parameters(recurse=False,prefix=name):
                    leaf_modules.append((pname,IsoParamWrapper(pname,p)))
        if use_lora:
            leaf_modules = [(n,m) for n,m in leaf_modules if not any([keyword in n for keyword in lora_module_keyword])]
        return leaf_modules

    def _broadcast_module(self,module,prefix=None,empty_cache=False,need_gather=False):
        count, num_params = 0, len(list(module.named_parameters()))
        for name, param in module.named_parameters(prefix=prefix):
            # broadcast
            count += 1
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=empty_cache and count==num_params,
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=empty_cache and count==num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        use_lora = False
        if isinstance(model, PeftModel):
            lora_model = model.base_model
            model = lora_model.model
            use_lora = True

        leaf_modules = self._get_leaf_modules(model,use_lora) # parameters of leaf_modules should not overlap
        count, num_modules = 0, len(leaf_modules)
        for key,module in leaf_modules:
            count += 1
            with ExitStack() as stack:
                need_gather = self.strategy.args.zero_stage == 3
                module_name = key.split(".")[-1]
                raw_module = module
                if use_lora and hasattr(raw_module, "base_layer"):
                    #This is a lora module
                    stack.enter_context(deepspeed.zero.GatheredParameters(raw_module.parameters(), enabled=need_gather))
                    raw_module.merge(safe_merge=True)
                    # we don't really replace the module, but we utilize _replace_module to get the merged module
                    fake_parent = type('FakeParent',(),{})()
                    lora_model._replace_module(fake_parent, module_name, raw_module.get_base_layer(), raw_module)
                    module = getattr(fake_parent, module_name)
                    need_gather = False
                    stack.callback(raw_module.unmerge)

                self._broadcast_module(module, prefix=key, empty_cache=count==num_modules,need_gather=need_gather)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.processor,
                save_path,
            )
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
        torch.distributed.barrier()

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def evaluate(self):
        self.actor.eval()
        from vllm import SamplingParams
        self.response_length_list = []
        # llms = self.vllm_engines
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            top_k=-1,
            max_tokens=768,
            min_tokens=16,
            skip_special_tokens=False,
            include_stop_str_in_output=False,
        )

        eval_batch_size = 512
        correct = 0
        total = 0

        for prompts, labels in tqdm(self.prompts_dataloader_eval, total=len(self.prompts_dataloader_eval), desc="Evaluate", disable=not self.strategy.is_rank_0()):
            all_prompts_chat = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

            refs = []
            with torch.no_grad():
                for i, llm in enumerate(llms):
                    messages = all_prompts_chat[i * eval_batch_size : (i + 1) * eval_batch_size]
                    if messages:
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

            model_responses = []
            for output in all_outputs:
                model_responses.append(output.outputs[0].text)
            
            # calculate the accuracy
            judges = check_answer_commonsense_qa(model_responses, labels)
            correct += sum(judges)
            total += len(judges)

        # Convert to tensors and move to CUDA
        correct_tensor = torch.tensor(correct, device="cuda", dtype=torch.float)
        total_tensor = torch.tensor(total, device="cuda", dtype=torch.float)

        # All-reduce across all processes
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

        if total_tensor.item() > 0:
            accuracy = correct_tensor.item() / total_tensor.item()
        else:
            accuracy = 0.0

        if self.strategy.is_rank_0():
            print(f"Evaluate Accuracy (avg over all ranks): {accuracy:.4f}")

        return accuracy
    
    def evaluate_math(self):
        self.actor.eval()
        from vllm import SamplingParams
        self.response_length_list = []
        # llms = self.vllm_engines
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            top_k=-1,
            max_tokens=768,
            min_tokens=16,
            skip_special_tokens=False,
            include_stop_str_in_output=False,
        )

        eval_batch_size = 512
        correct = 0
        total = 0

        for prompts, labels in tqdm(self.prompts_dataloader_eval, total=len(self.prompts_dataloader_eval), desc="Evaluate", disable=not self.strategy.is_rank_0()):
            refs = []
            with torch.no_grad():
                for i, llm in enumerate(llms):
                    messages = prompts[i * eval_batch_size : (i + 1) * eval_batch_size]
                    if messages:
                        refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=messages))
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

            model_responses = []
            for output in all_outputs:
                model_responses.append(output.outputs[0].text)
            
            # calculate the accuracy
            acc, each_score = check_answer_math(model_responses, labels, prompt_type="qwen25-math-cot", data_name="math")
            # judges = check_answer_commonsense_qa(model_responses, labels)
            correct += acc
            total += 1

        # Convert to tensors and move to CUDA
        correct_tensor = torch.tensor(correct, device="cuda", dtype=torch.float)
        total_tensor = torch.tensor(total, device="cuda", dtype=torch.float)

        # All-reduce across all processes
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)

        if total_tensor.item() > 0:
            accuracy = correct_tensor.item() / total_tensor.item()
        else:
            accuracy = 0.0

        if self.strategy.is_rank_0():
            print(f"Evaluate Accuracy (avg over all ranks): {accuracy:.4f}")

        return accuracy


        


@ray.remote(num_gpus=1)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            exclude_modules=strategy.args.exclude_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)
        # Support freeze some parameter
        if hasattr(strategy.args, "freeze_prefix") and strategy.args.freeze_prefix:
            frozen_count = 0
            total_params = 0
            for name, param in actor.model.named_parameters():
                total_params += 1
                if any(name.startswith(prefix) for prefix in strategy.args.freeze_prefix):
                    param.requires_grad = False
                    frozen_count += 1
            strategy.print(f"Froze {frozen_count}/{total_params} parameters based on prefixes: {strategy.args.freeze_prefix}")

        # configure tokenizer
        
        self.data_processor = get_data_processor(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )
        self.tokenizer = self.data_processor.tokenizer

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_datasets()

        # configure scheduler
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        self.prompts_dataset = PromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
            True,
            True,
        )

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(
                            len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            data_processor=self.data_processor,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # for GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            processor_kwargs=args.processor_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

            trainer._broadcast_to_vllm()

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "sleep")
                torch.distributed.barrier()
                torch.cuda.synchronize()

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.data_processor.processor,
            args.save_path,
        )


@ray.remote(num_gpus=1)
class ActorModelRayActor_TG(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            exclude_modules=strategy.args.exclude_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)
        # Support freeze some parameter
        if hasattr(strategy.args, "freeze_prefix") and strategy.args.freeze_prefix:
            frozen_count = 0
            total_params = 0
            for name, param in actor.model.named_parameters():
                total_params += 1
                if any(name.startswith(prefix) for prefix in strategy.args.freeze_prefix):
                    param.requires_grad = False
                    frozen_count += 1
            strategy.print(f"Froze {frozen_count}/{total_params} parameters based on prefixes: {strategy.args.freeze_prefix}")

        if args.multimodal:
            self.data_processor = get_data_processor(pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        else:
            self.data_processor = None
        self.tokenizer = get_tokenizer(pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_datasets(args.dataset_name)

        # configure scheduler
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

    def prepare_datasets(self, dataset_name='commonsense_qa'):
        strategy = self.strategy
        args = self.strategy.args
        if dataset_name == 'commonsense_qa':
            prompts_data = load_dataset(args.prompt_data, split='train')
            self.prompts_dataset = CommonSenseQADataset(prompts_data, self.tokenizer, strategy, input_template=args.input_template)
            self.prompts_dataloader = strategy.setup_dataloader(
                self.prompts_dataset,
                args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
                True,
                True,
            )
            if args.eval:
                prompts_data = load_dataset(args.prompt_data, split='validation')
                self.prompts_dataset_eval = CommonSenseQADataset(prompts_data, self.tokenizer, strategy, input_template=args.input_template)
                self.prompts_dataloader_eval = strategy.setup_dataloader(
                    self.prompts_dataset_eval,
                    args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
                    True,
                    True,
                )
            else:
                self.prompts_dataloader_eval = None

        elif dataset_name == 'math':
            self.prompts_dataset = MathDataset(args.prompt_data)
            self.prompts_dataloader = strategy.setup_dataloader(
                self.prompts_dataset,
                args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
                True,
                True,
            )
            if args.eval:
                self.prompts_dataset_eval = MathDataset(args.prompt_eval_data, ratio=0.3)
                self.prompts_dataloader_eval = strategy.setup_dataloader(
                    self.prompts_dataset_eval,
                    args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
                    True,
                    True,
                )
            else:
                self.prompts_dataloader_eval = None

        self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        feedback_model = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer_TG(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            feedback_model=feedback_model,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            data_processor=self.data_processor,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # for GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            processor_kwargs=args.processor_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

            trainer._broadcast_to_vllm()

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "sleep")
                torch.distributed.barrier()
                torch.cuda.synchronize()

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.prompts_dataloader_eval,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.data_processor.processor,
            args.save_path,
        )

    def evaluate(self, vllm_engines):
        self.actor.eval()
        from vllm import SamplingParams
        self.response_length_list = []
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        if len(vllm_engines) <= world_size:
            llms = [vllm_engines[rank % len(vllm_engines)]]
        else:
            llms = vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            min_tokens=1,
            skip_special_tokens=False,
            include_stop_str_in_output=False,
        )

        eval_batch_size = 512
        correct = 0
        total = 0

        for prompts, labels in self.prompts_dataloader_eval:
            all_prompts_chat = [self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

            refs = []
            with torch.no_grad():
                for i, llm in enumerate(llms):
                    messages = all_prompts_chat[i * eval_batch_size : (i + 1) * eval_batch_size]
                    if messages:
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

            model_responses = []
            for output in all_outputs:
                model_responses.append(output.outputs[0].text)
            
            # calculate the accuracy
            judges = check_answer_commonsense_qa(model_responses, labels)
            correct += sum(judges)
            total += len(judges)

        print(f"Evaluate Accuracy: {correct / total}")
        return correct / total



@ray.remote(num_gpus=1)
class ActorModelRayActor_Card(BasePPORole):
    """
    Class for Card Game environment
    """
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        if self.strategy.args.multimodal:
            actor = MultiModalActor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                lora_rank=strategy.args.lora_rank,
                lora_alpha=strategy.args.lora_alpha,
                target_modules=strategy.args.target_modules,
                exclude_modules=strategy.args.exclude_modules,
                lora_dropout=strategy.args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
                packing_samples=strategy.args.packing_samples,
                temperature=strategy.args.temperature,
                use_liger_kernel=strategy.args.use_liger_kernel,
                freeze_vision_encoder=strategy.args.freeze_vision_encoder,
            )
        else:
            actor = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                lora_rank=strategy.args.lora_rank,
                lora_alpha=strategy.args.lora_alpha,
                target_modules=strategy.args.target_modules,
                exclude_modules=strategy.args.exclude_modules,
                lora_dropout=strategy.args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
                packing_samples=strategy.args.packing_samples,
                temperature=strategy.args.temperature,
                use_liger_kernel=strategy.args.use_liger_kernel,
            )

        # strategy.print(actor)

        # Support freeze some parameter
        if hasattr(strategy.args, "freeze_prefix") and strategy.args.freeze_prefix:
            frozen_count = 0
            total_params = 0
            for name, param in actor.model.named_parameters():
                total_params += 1
                if any(name.startswith(prefix) for prefix in strategy.args.freeze_prefix):
                    param.requires_grad = False
                    frozen_count += 1
            strategy.print(f"Froze {frozen_count}/{total_params} parameters based on prefixes: {strategy.args.freeze_prefix}")

        if args.multimodal:
            min_pixels = 256*28*28
            max_pixels = 1280*28*28
            self.data_processor = AutoProcessor.from_pretrained(pretrain, padding_side="left", min_pixels=min_pixels, max_pixels=max_pixels)
            if args.feedback_model:
                self.feedback_data_processor = AutoProcessor.from_pretrained(args.feedback_model, padding_side="left")
            else:
                self.feedback_data_processor = None
        else:
            self.data_processor = AutoProcessor.from_pretrained(pretrain, padding_side="left")
            if args.feedback_model:
                self.feedback_data_processor = AutoProcessor.from_pretrained(args.feedback_model, padding_side="left")
            else:
                self.feedback_data_processor = None
        self.tokenizer = get_tokenizer(pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_env()

        # configure scheduler
        self.num_update_steps_per_episodes = (self.num_steps * self.num_envs * args.max_epochs // args.train_batch_size)

        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

    
    
    def prepare_env(self):
        args = self.strategy.args
        self.prompts_dataloader = None
        self.prompts_dataloader_eval = None
        self.pretrain_dataloader = None

        self.configs = load_config_from_yaml(args.env_config)
        if args.multimodal:
            self.configs.prompt_config.use_vision = True
        else:
            self.configs.prompt_config.use_vision = False
            self.configs.prompt_config.use_language = True
        self.action_space = []
        self.num_envs = self.configs.num_envs
        self.num_eval_envs = self.configs.num_eval_envs
        self.num_steps = self.configs.env_config.num_steps
        self.num_updates = self.configs.num_updates
        self.compute_return_kwargs = self.configs.compute_return_kwargs
        env_fns = [make_env(self.configs.env_config, language_only=self.configs.prompt_config.use_language, seed=args.seed + idx) for idx in range(self.num_envs)]
        self.envs = None
        self.eval_envs = None
        try:
            print(f"Creating {self.num_envs} parallel environments...")
            # context="spawn" is often more robust, especially on Windows/macOS
            self.envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
            # self.envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            print("Environments created.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()

        if args.eval:
            try:
                # in-distribution
                env_fns = [make_env(self.configs.eval_env_config, language_only=self.configs.prompt_config.use_language, seed=args.seed + 10*(idx+1), ood=False) for idx in range(self.num_eval_envs)]
                self.iid_eval_envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
                self.iid_eval_config = copy.deepcopy(self.configs.eval_env_config)
                # ood (change the rule of the face cards)
                self.configs.eval_env_config.treat_face_cards_as_10 = False
                env_fns = [make_env(self.configs.eval_env_config, language_only=self.configs.prompt_config.use_language, seed=args.seed + 100*(idx+1), ood=True) for idx in range(self.num_eval_envs)]
                self.ood_rule_eval_envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
                self.ood_rule_eval_config = copy.deepcopy(self.configs.eval_env_config)
                if args.multimodal:
                    # ood (change cards' visual appearance)
                    self.configs.eval_env_config.treat_face_cards_as_10 = True
                    self.configs.eval_env_config.face_cards_color = "red"
                    env_fns = [make_env(self.configs.eval_env_config, language_only=self.configs.prompt_config.use_language, seed=args.seed + 1000*(idx+1), ood=True) for idx in range(self.num_eval_envs)]
                    self.ood_visual_eval_envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
                    self.ood_visual_eval_config = copy.deepcopy(self.configs.eval_env_config)
                    self.eval_envs = [self.iid_eval_envs, self.ood_rule_eval_envs, self.ood_visual_eval_envs]
                    self.eval_env_configs = [self.iid_eval_config, self.ood_rule_eval_config, self.ood_visual_eval_config]
                else:
                    self.configs.eval_env_config.treat_face_cards_as_10 = True
                    self.configs.eval_env_config.target_points = 36
                    env_fns = [make_env(self.configs.eval_env_config, language_only=self.configs.prompt_config.use_language, seed=args.seed + 1000*(idx+1), ood=True) for idx in range(self.num_eval_envs)]
                    self.ood_num_eval_envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
                    self.ood_num_eval_config = copy.deepcopy(self.configs.eval_env_config)
                    self.eval_envs = [self.iid_eval_envs, self.ood_rule_eval_envs, self.ood_num_eval_envs]
                    self.eval_env_configs = [self.iid_eval_config, self.ood_rule_eval_config, self.ood_num_eval_config]
                print("Eval environments created.")
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                import traceback
                traceback.print_exc()

        else:
            self.eval_envs = None
            self.eval_env_configs = None


    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        feedback_model = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer_CardGame(
            strategy=strategy,
            actor=self.actor,
            critic=critic_model,
            reward_model=reward_model,
            initial_model=initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            feedback_model=feedback_model,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            data_processor=self.data_processor,
            feedback_data_processor=self.feedback_data_processor,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # for GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            processor_kwargs=args.processor_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
            # for card game env
            envs=self.envs,
            eval_envs=self.eval_envs,
            env_config=self.configs.env_config,
            eval_env_configs=self.eval_env_configs,
            prompt_config=self.configs.prompt_config,
        )

        # # broadcast checkpoint
        # ckpt_path = os.path.join(args.ckpt_path, "_actor")
        # if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
        #     # vLLM wakeup when vllm_enable_sleep
        #     if self.strategy.args.vllm_enable_sleep:
        #         batch_vllm_engine_call(vllm_engines, "wake_up")
        #     torch.distributed.barrier()
        #     torch.cuda.synchronize()

        #     trainer._broadcast_to_vllm()

        #     # vLLM offload when vllm_enable_sleep
        #     if self.strategy.args.vllm_enable_sleep:
        #         batch_vllm_engine_call(vllm_engines, "sleep")
        #         torch.distributed.barrier()
        #         torch.cuda.synchronize()

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.prompts_dataloader_eval,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

class ActorPPOTrainer_CardGame(PPOTrainer):
    def __init__(
        self,
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        ema_model,
        actor_optim: Optional[Optimizer], # fit から None が渡される可能性を考慮
        critic_optim: Optional[Optimizer], # fit から None が渡される可能性を考慮
        actor_scheduler, # fit から None が渡される可能性を考慮
        critic_scheduler, # fit から None が渡される可能性を考慮
        vllm_engines: List = None,
        feedback_model = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        envs: gym.vector.AsyncVectorEnv = None,
        eval_envs: Union[gym.vector.AsyncVectorEnv, List[gym.vector.AsyncVectorEnv]] = None,
        env_config: EnvConfig = None,
        eval_env_configs: Union[EvalEnvConfig, List[EvalEnvConfig]] = None,
        prompt_config: PromptConfig = None,
        **kwargs,
    ):
        """PPOTrainer for ray and TextFeedback.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            feedback_model (FeedbackModel_vllm, optional): feedback model for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        # kwargs から 'data_processor' を取得し、同時に kwargs から削除する
        ppo_data_processor = kwargs.pop('data_processor', None)
        ppo_feedback_data_processor = kwargs.pop('feedback_data_processor', None)
        # remote_rm_url も同様に kwargs から削除 (必要であれば)
        # super には渡さない想定であれば pop するだけでよい
        kwargs.pop('remote_rm_url', None) # super に渡さないので値は不要
        super().__init__(
            strategy,
            actor,
            critic,
            reward_model,
            initial_model, # 5番目
            ema_model,     # 6番目
            actor_optim,   # 7番目
            critic_optim,  # 8番目
            actor_scheduler,# 9番目
            critic_scheduler,# 10番目
            data_processor=ppo_data_processor, # PPOTrainer用の data_processor を渡す
            # remote_rm_url は PPOTrainer が str を期待するため、ここでは渡さないか、
            # self_remote_rm_url_list の最初の要素などを渡す (仕様による)
            # remote_rm_url=self_remote_rm_url_list[0] if self_remote_rm_url_list else None, # 例
            **kwargs,      # 残りのキーワード引数 (ema_beta, init_kl_coef など)
        )
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.feedback_model = feedback_model
        self.critic_train_remote = critic_train_remote
        self.envs = envs
        self.env_config = env_config
        self.eval_envs = eval_envs
        self.eval_env_configs = eval_env_configs
        self.prompt_config = prompt_config
        self.feedback_data_processor = ppo_feedback_data_processor

        # override replay buffer
        packing_samples = getattr(self.strategy.args, "packing_samples", False)
        self.replay_buffer = ReplayBuffer_CARDGAME(
            sample_batch_size=self.strategy.args.micro_train_batch_size,
            data_processor=self.data_processor,
            packing_samples=packing_samples,
            drop_maxlen=self.strategy.args.drop_maxlen, 
            maxlen=self.strategy.args.generate_max_len + self.strategy.args.prompt_max_len,
            multimodal=self.strategy.args.multimodal,
        )

        if self.strategy.args.advantage_estimator == "reinforce":
                self.experience_maker = RemoteExperienceMaker_CardGame_REINFORCE(
                actor=self.actor,
                critic=self.critic,
                reward_model=self.reward_model,
                initial_model=self.initial_model,
                tokenizer=self.tokenizer,
                data_processor=self.data_processor,
                prompt_max_len=self.prompt_max_len,
                kl_controller=self.kl_ctl,
                strategy=self.strategy,
                remote_rm_url=self.remote_rm_url,
                reward_fn=self.reward_fn,
                vllm_engines=self.vllm_engines,
                packing_samples=self.strategy.args.packing_samples,
                multimodal=self.strategy.args.multimodal,
                envs=self.envs,
                env_config=self.env_config,
                prompt_config=self.prompt_config,
            )
        else:
            self.experience_maker = RemoteExperienceMaker_CardGame(
                actor=self.actor,
                critic=self.critic,
                reward_model=self.reward_model,
                initial_model=self.initial_model,
                tokenizer=self.tokenizer,
                data_processor=self.data_processor,
                feedback_data_processor=self.feedback_data_processor,
                prompt_max_len=self.prompt_max_len,
                kl_controller=self.kl_ctl,
                strategy=self.strategy,
                remote_rm_url=self.remote_rm_url,
                reward_fn=self.reward_fn,
                vllm_engines=self.vllm_engines,
                feedback_model=self.feedback_model,
                packing_samples=self.strategy.args.packing_samples,
                multimodal=self.strategy.args.multimodal,
                feedback_rewards=self.strategy.args.feedback_rewards,
                envs=self.envs,
                env_config=self.env_config,
                prompt_config=self.prompt_config,
            )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.strategy.args.colocate_all_models:
            self.use_cuda_ipc = True

        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)

        torch.distributed.barrier()

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

        if global_steps == 0 and self.strategy.args.eval:
            success_rate_dict = self.evaluate()
            for k, v in success_rate_dict.items():
                status[k] = v
            if self.strategy.is_rank_0():
                print(f"Init eval success rate: {success_rate_dict}")
            # status["eval_success_rate"] = success_rate
            if self.strategy.is_rank_0() and self._wandb is not None:
                logs = {
                    "train/%s" % k: v
                    for k, v in {**status, "global_step": global_steps}.items()
                }
                self._wandb.log(logs)

        # vLLM offload when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        torch.cuda.synchronize()

        # 2. triger remote critic model training
        if self.critic_train_remote:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.reload_states.remote())

            critic_status_ref = self.critic.fit.remote()

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref))
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.offload_states.remote())

        if self.strategy.args.colocate_all_models or self.strategy.args.colocate_actor_vllm:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.reload_states()

            status.update(super().ppo_train(global_steps))

            if self.strategy.args.deepspeed_enable_sleep:
                self.offload_states()

            torch.cuda.empty_cache()

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                torch.distributed.barrier()
                torch.cuda.synchronize()
                self._broadcast_to_vllm()

                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        # 5. wait remote critic model training done
        if self.critic_train_remote and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        # 6. evaluate
        if self.strategy.args.eval and global_steps % 2 == 0:
            if self.strategy.args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
                batch_vllm_engine_call(self.vllm_engines, "wake_up")
                torch.distributed.barrier()
                torch.cuda.synchronize()
            success_rate_dict = self.evaluate()
            for k, v in success_rate_dict.items():
                status[k] = v

            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            torch.cuda.synchronize()

        return status

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def _get_leaf_modules_LLM(self,model,use_lora):
        leaf_modules = []
        lora_module_keyword = ["lora_","base_layer"]
        class IsoParamWrapper:
            """
            Some modules may have isolated parameters that are not in submodules.
            This class wraps such parameters in a module so that they can be treated uniformly.
            """
            def __init__(self, name, parameter):
                self.name = name
                self.parameter = parameter

            def named_parameters(self,prefix=None):
                # self.name is already the full name. No need to add prefix
                return [(self.name,self.parameter)]

        for name,module in model.named_modules():
            if len(list(module.children())) == 0 or (use_lora and hasattr(module,"base_layer")):
                leaf_modules.append((name,module))
            else:
                #find isolated parameter
                for pname, p in module.named_parameters(recurse=False,prefix=name):
                    leaf_modules.append((pname,IsoParamWrapper(pname,p)))
        if use_lora:
            leaf_modules = [(n,m) for n,m in leaf_modules if not any([keyword in n for keyword in lora_module_keyword])]
        return leaf_modules

    
    ## NOTE: For Qwen2.5-VL
    def _get_leaf_modules(self, model, use_lora):
        leaf_modules = []
        lora_module_keyword = ["lora_", "base_layer"]
        
        class IsoParamWrapper:
            """
            Some modules may have isolated parameters that are not in submodules.
            This class wraps such parameters in a module so that they can be treated uniformly.
            """
            def __init__(self, name, parameter):
                self.name = name
                self.parameter = parameter

            def named_parameters(self, prefix=None):
                # self.name is already the full name. No need to add prefix
                return [(self.name, self.parameter)]
        
        # 1. まず基本モジュール（layers以外）を処理
        for name, module in model.named_modules():
            # model.layersは別途処理するのでスキップ
            if name.startswith("layers.") or name == "layers":
                continue
                
            if len(list(module.children())) == 0 or (use_lora and hasattr(module, "base_layer")):
                leaf_modules.append((name, module))
            else:
                # 独立したパラメータを処理
                for pname, p in module.named_parameters(recurse=False, prefix=name):
                    leaf_modules.append((pname, IsoParamWrapper(pname, p)))
        
        # 2. 次にmodel.layersを明示的に処理
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            if self.strategy.is_rank_0():
                print(f"Processing model.layers: {len(layers)} layers found")
            
            # 各レイヤーを処理
            for i, layer in enumerate(layers):
                layer_prefix = f"model.layers.{i}"
                
                # 各レイヤー内の子モジュールを処理
                for child_name, child in layer.named_children():
                    full_name = f"{layer_prefix}.{child_name}"
                    
                    if len(list(child.children())) == 0 or (use_lora and hasattr(child, "base_layer")):
                        leaf_modules.append((full_name, child))
                    else:
                        # 子モジュール内のリーフモジュールを処理
                        for sub_name, sub_module in child.named_modules():
                            if sub_name == "":  # 自分自身はスキップ
                                continue
                            
                            sub_full_name = f"{full_name}.{sub_name}"
                            if len(list(sub_module.children())) == 0 or (use_lora and hasattr(sub_module, "base_layer")):
                                leaf_modules.append((sub_full_name, sub_module))
                        
                        # モジュール内の独立したパラメータを処理
                        for param_name, param in child.named_parameters(recurse=False):
                            param_full_name = f"{full_name}.{param_name}"
                            leaf_modules.append((param_full_name, IsoParamWrapper(param_full_name, param)))
        
        # LoRAモジュールをフィルタリング
        if use_lora:
            orig_count = len(leaf_modules)
            leaf_modules = [(n, m) for n, m in leaf_modules if not any([keyword in n for keyword in lora_module_keyword])]
            if self.strategy.is_rank_0():
                print(f"Filtered LoRA modules: {orig_count} -> {len(leaf_modules)}")
        
        # デバッグ情報
        if self.strategy.is_rank_0():
            module_types = {}
            for name, _ in leaf_modules:
                top_level = name.split('.')[0]
                module_types[top_level] = module_types.get(top_level, 0) + 1
            
            # 種類別のモジュール数を表示
            print(f"Leaf modules by type: {module_types}")
            
            # トランスフォーマーレイヤーの数を確認
            layer_count = 0
            for name, _ in leaf_modules:
                if "model.layers" in name:
                    layer_count += 1
            print(f"Transformer layer modules: {layer_count}")
        
        return leaf_modules


    def _broadcast_module(self, module, prefix=None, empty_cache=False, need_gather=False):
        count, num_params = 0, len(list(module.named_parameters()))
        for name, param in module.named_parameters(prefix=prefix):
            # broadcast
            count += 1
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=empty_cache and count==num_params,
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective
                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
                            
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=empty_cache and count==num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()
    
    # def _broadcast_to_vllm(self):
    #     use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
    #     cache_reset_refs = []
    #     if use_prefix_cache and torch.distributed.get_rank() == 0:
    #         # clear prefix cache
    #         for engine in self.vllm_engines:
    #             cache_reset_refs.append(engine.reset_prefix_cache.remote())

    #     torch.cuda.empty_cache()
    #     model = self.actor.model.module
    #     use_lora = False
    #     if isinstance(model, PeftModel):
    #         lora_model = model.base_model
    #         model = lora_model.model
    #         use_lora = True

    #     leaf_modules = self._get_leaf_modules(model,use_lora) # parameters of leaf_modules should not overlap
    #     count, num_modules = 0, len(leaf_modules)
    #     for key, module in leaf_modules:
    #         print(f"Broadcasting module: {key}")
    #         count += 1
    #         with ExitStack() as stack:
    #             need_gather = self.strategy.args.zero_stage == 3
    #             module_name = key.split(".")[-1]
    #             raw_module = module
    #             if use_lora and hasattr(raw_module, "base_layer"):
    #                 #This is a lora module
    #                 stack.enter_context(deepspeed.zero.GatheredParameters(raw_module.parameters(), enabled=need_gather))
    #                 raw_module.merge(safe_merge=True)
    #                 # we don't really replace the module, but we utilize _replace_module to get the merged module
    #                 fake_parent = type('FakeParent',(),{})()
    #                 lora_model._replace_module(fake_parent, module_name, raw_module.get_base_layer(), raw_module)
    #                 module = getattr(fake_parent, module_name)
    #                 need_gather = False
    #                 stack.callback(raw_module.unmerge)

    #             self._broadcast_module(module, prefix=key, empty_cache=count==num_modules,need_gather=need_gather)

    #     if cache_reset_refs:
    #         ray.get(cache_reset_refs)
    #     torch.cuda.empty_cache()
    #     torch.distributed.barrier()

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        
        # モデルを取得
        model = self.actor.model.module
        use_lora = False
        if isinstance(model, PeftModel):
            lora_model = model.base_model
            model = lora_model.model
            use_lora = True

        # デバッグ情報の追加
        if self.strategy.is_rank_0():
            print("=== Model Structure Diagnosis ===")
            print(f"Model Type: {type(model).__name__}")
            
            # トップレベルの属性を確認
            print("Model Attributes:", list(vars(model).keys()))
            
            # モデル内にlayersが存在するか確認
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                print(f"layers found: {len(model.model.layers)} layers")
            elif hasattr(model, 'layers'):
                print(f"layers found directly: {len(model.layers)} layers")
            else:
                print("Warning: No layers found in model structure!")

        # リーフモジュールの収集（修正版）
        if self.args.multimodal:
            leaf_modules = self._get_leaf_modules(model, use_lora)
        else:
            leaf_modules = self._get_leaf_modules_LLM(model, use_lora)
        
        if self.strategy.is_rank_0():
            print(f"Total leaf modules to process: {len(leaf_modules)}")
        
        # モジュールをvLLMに同期
        count, num_modules = 0, len(leaf_modules)
        for key, module in leaf_modules:
            # if self.strategy.is_rank_0():
            #     print(f"Broadcasting module: {key}")
            count += 1
            with ExitStack() as stack:
                need_gather = self.strategy.args.zero_stage == 3
                module_name = key.split(".")[-1]
                raw_module = module
                if use_lora and hasattr(raw_module, "base_layer"):
                    #This is a lora module
                    stack.enter_context(deepspeed.zero.GatheredParameters(raw_module.parameters(), enabled=need_gather))
                    raw_module.merge(safe_merge=True)
                    # we don't really replace the module, but we utilize _replace_module to get the merged module
                    fake_parent = type('FakeParent',(),{})()
                    lora_model._replace_module(fake_parent, module_name, raw_module.get_base_layer(), raw_module)
                    module = getattr(fake_parent, module_name)
                    need_gather = False
                    stack.callback(raw_module.unmerge)

                self._broadcast_module(module, prefix=key, empty_cache=count==num_modules, need_gather=need_gather)
                
                # 進捗状況を表示（10%ごと）
                if self.strategy.is_rank_0() and count % max(1, num_modules // 10) == 0:
                    print(f"Progress: {count}/{num_modules} modules processed ({count/num_modules*100:.1f}%)")

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        
        # 検証情報
        if self.strategy.is_rank_0():
            print("=== Weight update complete ===")

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.processor,
                save_path,
            )
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
        torch.distributed.barrier()

    
    def verify_weight_updates(self):
        """更新された重みを検証する関数"""
        if not self.strategy.is_rank_0() or self.vllm_engines is None:
            return
        
        # サンプリングするパラメータのリスト
        key_params = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.10.self_attn.k_proj.weight",
            "model.layers.20.mlp.gate_proj.weight",
            "model.layers.35.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight"
        ]
        
        print("=== 重み更新検証 ===")
        # vLLMエンジンから重みを取得
        weights_ref = self.vllm_engines[0].get_model_weights.remote(key_params)
        weights = ray.get(weights_ref)
        
        # 結果を表示
        for name, weight in zip(key_params, weights):
            if weight is not None:
                print(f"{name}: Shape={weight.shape}, Mean={weight.mean().item():.6f}")
            else:
                print(f"{name}: Not found")
    
    
    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def evaluate(self):
        if self.strategy.is_rank_0():
            print("Evaluating...")
        self.actor.eval()
        from vllm import SamplingParams
        self.response_length_list = []
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        ## original
        # sampling_params = SamplingParams(
        #     temperature=0,
        #     top_p=1.0,
        #     top_k=-1,
        #     max_tokens=512,
        #     min_tokens=1,
        #     skip_special_tokens=False,
        #     include_stop_str_in_output=False,
        # )
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=512,
            min_tokens=1,
            skip_special_tokens=False,
            include_stop_str_in_output=False,
        )

        success_rates = []
        success_nums = []
        total_nums = []

        for eval_id, (eval_env, eval_env_config) in enumerate(zip(self.eval_envs, self.eval_env_configs)):

            success  = 0
            total = 0
            num_envs = eval_env.num_envs
            history_length = 2
            responses = [[] for _ in range(num_envs)]
            oracle_arguments = self.formulate_oracle_arguments(eval_env_config)
            json_pattern = r"```json\n(.*?)\n```"

            if self.prompt_config.use_vision:
                prompt_vision = PROMPT_FN[self.prompt_config.prompt_vision]
                pattern_vision = self.prompt_config.pattern_vision
            else:
                prompt_language = PROMPT_FN[self.prompt_config.prompt_language]
                pattern_language = self.prompt_config.pattern_language

            for t in range(eval_env_config.num_evaluations):
                obs_batch, info_batch = eval_env.reset()
                if t == 0 and self.args.multimodal:
                    image = Image.fromarray(obs_batch[0])
                    
                previous_responses = [[] for _ in range(num_envs)]
                previous_verify_infos = [[] for _ in range(num_envs)]
                active_envs = np.ones(num_envs, dtype=bool)  # Track which environments are still active
                episode_rewards = np.zeros(num_envs)  # Track rewards for each environment

                while active_envs.any():
                    vision_res_list = [{} for _ in range(num_envs)]
                    language_res_list = [{} for _ in range(num_envs)]
                    
                    if self.prompt_config.use_vision:
                        prompt, pattern = prompt_vision, pattern_vision
                    else:
                        prompt, pattern = prompt_language, pattern_language

                    # Only process active environments
                    active_indices = np.where(active_envs)[0]
                    active_obs_batch = obs_batch[active_envs]
                    active_previous_responses = [previous_responses[i] for i in active_indices]
                    active_previous_verify_infos = [previous_verify_infos[i] for i in active_indices]

                    for i in active_indices:
                        if 'cards' not in vision_res_list[i].keys():
                            vision_res_list[i]['cards'] = info_batch['Plain Cards'][i]
                    
                    task_prompts = [prompt.format(**vision_res_list[i], **language_res_list[i], **oracle_arguments) for i in active_indices]
                    if self.args.multimodal:
                        messages = self.formulate_prompt(len(active_indices), 
                                                    task_prompts[0], 
                                                    obs_batch=active_obs_batch,
                                                    previous_responses=active_previous_responses,
                                                    previous_verify_infos=active_previous_verify_infos,
                                                    history_length=history_length)
                    else:
                        messages = self.formulate_prompt_for_LLMStudent(len(active_indices), task_prompts, 
                                                                        active_previous_responses,
                                                                        active_previous_verify_infos,
                                                                        history_length=history_length)

                    batch_size = (len(messages) + len(llms) - 1) // len(llms)
                    refs = []
                    if self.args.multimodal:
                        for i, llm in enumerate(llms):
                            msg_batch = messages[i * batch_size : (i + 1) * batch_size]
                            obs_batch_slice = active_obs_batch[i * batch_size : (i + 1) * batch_size]
                            prompts = self.data_processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=True)
                            vllm_inputs = [
                                {
                                    "prompt": prompt,
                                    "multi_modal_data": {"image": obs},
                                }
                                for prompt, obs in zip(prompts, obs_batch_slice)
                            ]
                            refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))
                    else:
                        for i, llm in enumerate(llms):
                            msg_batch = messages[i * batch_size : (i + 1) * batch_size]
                            if "Qwen3" in self.strategy.args.pretrain:
                                prompt = self.data_processor.apply_chat_template(msg_batch, tokenize=False, enable_thinking=False, add_generation_prompt=True)
                            else:
                                prompt = self.data_processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=True)
                            refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=prompt))
                    
                    ray.get(refs)

                    if self.strategy.ring_attn_group is None:
                        torch.distributed.barrier()
                    else:
                        time.sleep(2)

                    all_output_refs = []
                    for i, llm in enumerate(llms):
                        all_output_refs.append(llm.get_responses.remote(rank))
                    all_outputs = sum(ray.get(all_output_refs), [])

                    model_responses_list = []
                    for output in all_outputs:
                        model_responses_list.append(output.outputs[0].text)

                    # Create a full-sized responses list with None for inactive environments
                    full_responses = ["None"] * num_envs
                    for response, idx in zip(model_responses_list, active_indices):
                        full_responses[idx] = response

                    # preprocessing the model response to align with json style.
                    for i, model_response in enumerate(full_responses):
                        if model_response is None or model_response == "None":
                            continue
                        if "<|im_end|>" in model_response:
                            model_response = model_response.replace("<|im_end|>", "")
                        try:
                            match = re.search(json_pattern, model_response, re.DOTALL)
                        except TypeError as e:
                            pass
                        if match:
                            full_responses[i] = match.group(1)
                        else:
                            full_responses[i] = model_response
                    obs_batch, rewards, terminations, truncations, info_batch = eval_env.step(full_responses)
                    done = np.logical_or(terminations, truncations)

                    # 報酬更新
                    for idx in active_indices:
                        episode_rewards[idx] = rewards[idx]

                    # 履歴更新
                    for idx, txt in zip(active_indices, model_responses_list):
                        previous_responses[idx].append(txt)
                        previous_verify_infos[idx].append(info_batch["Verify Info"][idx])

                    # エピソード終了の処理
                    for idx in active_indices:
                        if done[idx]:
                            active_envs[idx] = False
                            total += 1
                            if episode_rewards[idx] >= 5 or "Correct solution" in info_batch["Verify Info"][idx]:
                                success += 1

                    # log
                    for j in range(len(active_indices)):
                        env_id = active_indices[j]
                        log = {
                            "env_idx": str(env_id),
                            "prompt": task_prompts[j],
                            "responses": full_responses[env_id],
                            "verify_info": info_batch["Verify Info"][env_id],
                            "reward": rewards[env_id],
                        }
                        responses[env_id].append(log)
                    # # 最初のTrialでEnv0のログを保存
                    # if t == 0 and active_envs[0]:
                    #     log = {
                    #         "prompt": task_prompts[0],
                    #         "responses": full_responses[0],
                    #         "verify_info": info_batch["Verify Info"][0],
                    #         "reward": rewards[0],
                    #     }
                    #     responses.append(log)

            # 最初のTrialの結果をファイル出力
            if self.strategy.is_rank_0():
                if not os.path.exists(self.strategy.args.output_log_dir + f"/eval_results_{eval_id}"):
                    os.makedirs(self.strategy.args.output_log_dir + f"/eval_results_{eval_id}")
                for env_id in range(num_envs):
                    if env_id < 20:
                        with open(self.strategy.args.output_log_dir + f"/eval_results_{eval_id}/log_{env_id}.json", "w") as f:
                            json.dump(responses[env_id], f, indent=4)
                if self.args.multimodal:
                    image.save(self.strategy.args.output_log_dir + f"/eval_results_{eval_id}/image.png")


            # Convert to tensors and move to CUDA
            correct_tensor = torch.tensor(success, device="cuda", dtype=torch.float)
            total_tensor = torch.tensor(total, device="cuda", dtype=torch.float)
            # All-reduce across all processes
            torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
            success_rate = correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0
            success_rates.append(success_rate)
            success_nums.append(correct_tensor.item())
            total_nums.append(total_tensor.item())
            # if self.strategy.is_rank_0():
            #     print(f"Evaluate Accuracy (avg over all ranks): {success_rate:.4f}")
            
        
        if self.args.multimodal:
            results = {
                f"In-distribution SR": success_rates[0],
                f"OOD (rule change) SR": success_rates[1],
                f"OOD (visual change) SR": success_rates[2]
            }
        else:
            results = {
                f"In-distribution SR": success_rates[0],
                f"OOD (rule change) SR": success_rates[1],
                f"OOD (number change) SR": success_rates[2]
            }

        if self.strategy.is_rank_0():
            print(f"In-distribution SR: {success_rates[0]:.4f} ({success_nums[0]}/{total_nums[0]})")
            print(f"OOD (rule change) SR: {success_rates[1]:.4f} ({success_nums[1]}/{total_nums[1]})")
            if self.args.multimodal:
                print(f"OOD (visual change) SR: {success_rates[2]:.4f} ({success_nums[2]}/{total_nums[2]})")
            else:
                print(f"OOD (number change) SR: {success_rates[2]:.4f} ({success_nums[2]}/{total_nums[2]})")

        return results

    
    def formulate_oracle_arguments(self, eval_env_config: EvalEnvConfig):
        oracle_arguments = {}
        oracle_arguments['face_card_msg'] = "'J', 'Q', and 'K' count as '10'." if eval_env_config.treat_face_cards_as_10 \
                                        else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
        oracle_arguments['target_number'] = str(eval_env_config.target_points)
        oracle_arguments['example_json_text'] = example_json_text
        return oracle_arguments
    
    def formulate_prompt(self, num_envs: int, task_prompt: str, obs_batch: np.ndarray = None, previous_responses = None, previous_verify_infos = None, history_length: int = 2) -> Tuple[List[dict], List[dict]]:
        messages = [[] for _ in range(num_envs)]
        for i in range(num_envs):
            contents = []
            if len(previous_responses[i]) > 0 and len(previous_verify_infos[i]) > 0:
                assert len(previous_responses[i]) == len(previous_verify_infos[i]), "The number of previous responses and verify infos must be the same."
                if obs_batch[i] is not None:
                    pil_image = Image.fromarray(obs_batch[i])
                    contents.append({"type": "image", "image": pil_image})
                contents.append({"type": "text", "text": task_prompt})
                for idx, (prev_response, prev_verify_info) in enumerate(zip(previous_responses[i][-history_length:], previous_verify_infos[i][-history_length:])):
                    contents.append({"type": "text", "text": f"\n## Previous your response ({history_length - idx} steps ago):\n{prev_response}"})
                    contents.append({"type": "text", "text": f"\n## Verification\nYou failed this trial because {prev_verify_info}"})
            else:
                if obs_batch[i] is not None:
                    pil_image = Image.fromarray(obs_batch[i])
                    contents.append({"type": "image", "image": pil_image})
                    contents.append({"type": "text", "text": task_prompt})
                else:
                    contents.append({"type": "text", "text": task_prompt})
            
            messages[i] = [
                {"role": "user",
                 "content": contents,
                },
            ]
        return messages

    def formulate_prompt_for_LLMStudent(self, num_envs: int, task_prompts: List[str], previous_responses: List[List[str]], previous_verify_infos: List[List[str]], history_length: int = 2) -> List[dict]:
        messages = [[] for _ in range(num_envs)]

        for i in range(num_envs):
            contents = []
            contents.append(task_prompts[i])
            if len(previous_responses[i]) > 0 and len(previous_verify_infos[i]) > 0:
                assert len(previous_responses[i]) == len(previous_verify_infos[i]), "The number of previous responses and verify infos must be the same."
                for idx, (prev_response, prev_verify_info) in enumerate(zip(previous_responses[i][-history_length:], previous_verify_infos[i][-history_length:])):
                    contents.append(f"\n## Previous your response ({history_length - idx} steps ago):\n{prev_response}")
                    contents.append(f"\n## Verification message\nYou failed this trial because {prev_verify_info}")
            
            messages[i] = [
                {"role": "user",
                 "content": "\n".join(contents),
                },
            ]
        return messages




class ActorPPOTrainer_GUI(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        remote_rm_url: List[str] = None,
        critic_train_remote: bool = False,
        parallel_env = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote

        self.experience_maker = RemoteExperienceMaker_GUI(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.data_processor,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
            parallel_env=parallel_env,
        )

        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.strategy.args.colocate_all_models:
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = init_process_group(
                    backend=backend,
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )

            ray.get(refs)
        torch.distributed.barrier()

    
    def fit(
        self,
        args,
        task_configs: List[Dict],
        consumed_samples: int = 0,
        num_update_steps_per_episodes: int = 1,
    ) -> None:
        
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        ## Episode loop
        for episode in range(start_episode, args.num_episodes):
            pbar = tqdm(
                range(len(task_configs)),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for task_id, task_config in enumerate(task_configs):
                experiences = self.experience_maker.make_experience_list(task_config, task_id, **self.generate_kwargs)
                for i, experience in enumerate(experiences):
                    if i == 0:
                        output = self.tokenizer.batch_decode(experience.sequences[0].unsqueeze(0), skip_special_tokens=True)
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)

                if self.args.advantage_estimator != "group_norm":
                    self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                self.replay_buffer.clear()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    
    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()
        status = {}

        # 2. triger remote critic model training
        if self.critic_train_remote:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.reload_states.remote())

            critic_status_ref = self.critic.fit.remote()

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref))
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic.offload_states.remote())

        if self.strategy.args.colocate_all_models:
            torch.distributed.barrier()

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.reload_states()

            status.update(super().ppo_train(global_steps))

            if self.strategy.args.deepspeed_enable_sleep:
                self.offload_states()

            torch.cuda.empty_cache()

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "wake_up")

                torch.distributed.barrier()
                torch.cuda.synchronize()
                self._broadcast_to_vllm()

                if self.strategy.args.vllm_enable_sleep:
                    batch_vllm_engine_call(self.vllm_engines, "sleep")
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

        # 5. wait remote critic model training done
        if self.critic_train_remote and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def _get_leaf_modules(self,model,use_lora):
        leaf_modules = []
        lora_module_keyword = ["lora_","base_layer"]
        class IsoParamWrapper:
            """
            Some modules may have isolated parameters that are not in submodules.
            This class wraps such parameters in a module so that they can be treated uniformly.
            """
            def __init__(self, name, parameter):
                self.name = name
                self.parameter = parameter

            def named_parameters(self,prefix=None):
                # self.name is already the full name. No need to add prefix
                return [(self.name,self.parameter)]

        for name,module in model.named_modules():
            if len(list(module.children())) == 0 or (use_lora and hasattr(module,"base_layer")):
                leaf_modules.append((name,module))
            else:
                #find isolated parameter
                for pname, p in module.named_parameters(recurse=False,prefix=name):
                    leaf_modules.append((pname,IsoParamWrapper(pname,p)))
        if use_lora:
            leaf_modules = [(n,m) for n,m in leaf_modules if not any([keyword in n for keyword in lora_module_keyword])]
        return leaf_modules

    def _broadcast_module(self,module,prefix=None,empty_cache=False,need_gather=False):
        count, num_params = 0, len(list(module.named_parameters()))
        for name, param in module.named_parameters(prefix=prefix):
            # broadcast
            count += 1
            if not self.use_cuda_ipc:
                use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=empty_cache and count==num_params,
                        )
                        for engine in self.vllm_engines
                    ]

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    if torch.distributed.get_rank() == 0:
                        if use_ray:
                            import ray.util.collective as collective

                            collective.broadcast(param.data, 0, group_name=self._model_update_group)
                        else:
                            torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                        ray.get(refs)
            # CUDA IPC
            else:
                from torch.multiprocessing.reductions import reduce_tensor

                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
                    weight = param.data.clone()
                    ipc_handle = reduce_tensor(weight)

                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight_cuda_ipc.remote(
                                name,
                                dtype=param.dtype,
                                shape=shape,
                                ipc_handles=ipc_handles,
                                empty_cache=empty_cache and count==num_params,
                            )
                            for engine in self.vllm_engines
                        ]
                        ray.get(refs)
                    torch.distributed.barrier()
                    torch.cuda.synchronize()

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        use_lora = False
        if isinstance(model, PeftModel):
            lora_model = model.base_model
            model = lora_model.model
            use_lora = True

        leaf_modules = self._get_leaf_modules(model,use_lora) # parameters of leaf_modules should not overlap
        count, num_modules = 0, len(leaf_modules)
        for key,module in leaf_modules:
            count += 1
            with ExitStack() as stack:
                need_gather = self.strategy.args.zero_stage == 3
                module_name = key.split(".")[-1]
                raw_module = module
                if use_lora and hasattr(raw_module, "base_layer"):
                    #This is a lora module
                    stack.enter_context(deepspeed.zero.GatheredParameters(raw_module.parameters(), enabled=need_gather))
                    raw_module.merge(safe_merge=True)
                    # we don't really replace the module, but we utilize _replace_module to get the merged module
                    fake_parent = type('FakeParent',(),{})()
                    lora_model._replace_module(fake_parent, module_name, raw_module.get_base_layer(), raw_module)
                    module = getattr(fake_parent, module_name)
                    need_gather = False
                    stack.callback(raw_module.unmerge)

                self._broadcast_module(module, prefix=key, empty_cache=count==num_modules,need_gather=need_gather)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _save_checkpoint(self, args, tag, client_states):
        # call remote critic
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ref = self.critic.save_checkpoint.remote(tag)
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.processor,
                save_path,
            )
        # wait
        if not self.disable_ds_ckpt:
            if self.critic_train_remote:
                ray.get(ref)
        torch.distributed.barrier()

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)


@ray.remote(num_gpus=1)
class ActorModelRayActor_GUI(BasePPORole):
    """
    This is a modified version of ActorModelRayActor for GUI training.
    It is used to train the actor model with GUI envs.
    This class has
        - Parallel GUI envs
        - Task configuration for each GUI env

    In the fit function, the above information will be given to collect episodes through ActorPPOTrainer_GUI.RemoteExperienceMaker_GUI.make_experience_list.
    
    NOTE:
        - We can refer to world_size and rank as self._world_size and self._rank.
    """
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            exclude_modules=strategy.args.exclude_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)
        # Support freeze some parameter
        if hasattr(strategy.args, "freeze_prefix") and strategy.args.freeze_prefix:
            frozen_count = 0
            total_params = 0
            for name, param in actor.model.named_parameters():
                total_params += 1
                if any(name.startswith(prefix) for prefix in strategy.args.freeze_prefix):
                    param.requires_grad = False
                    frozen_count += 1
            strategy.print(f"Froze {frozen_count}/{total_params} parameters based on prefixes: {strategy.args.freeze_prefix}")

        # configure tokenizer
        
        self.data_processor = get_data_processor(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )
        self.tokenizer = self.data_processor.tokenizer

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2)

        # prepare parallel envs
        self.prepare_envs()

        # configure scheduler
        self.num_update_steps_per_episodes = (
            len(self.task_configs) * args.n_samples_per_prompt * args.max_epochs // args.train_batch_size
        )
        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)


    def prepare_envs(self):
        args = self.strategy.args

        with open(args.task_configs_file, "r") as f:
            self.task_configs = json.load(f)
        
        random.seed(args.seed + self._rank)
        random.shuffle(self.task_configs)

        print(f"Generating {args.n_samples_per_prompt} environments...")
        self.parallel_env = ParallelDesktopEnv(
            num_containers=args.n_samples_per_prompt,
            docker_image_name=args.docker_image_name,
            host_task_dir=args.host_task_dir,
            action_space=args.action_space,
            task_config_list=self.task_configs,
            process_id=self._rank,
        )
        self.parallel_env.launch_containers()

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer_GUI(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            data_processor=self.data_processor,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # for GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            processor_kwargs=args.processor_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
            parallel_env=self.parallel_env,
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

            trainer._broadcast_to_vllm()

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(vllm_engines, "sleep")
                torch.distributed.barrier()
                torch.cuda.synchronize()

        trainer.fit(
            args,
            self.task_configs,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.data_processor.processor,
            args.save_path,
        )