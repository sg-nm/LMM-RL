from contextlib import ExitStack
import itertools
import math
import os
import socket
import json
import random
import time
import yaml
from typing import Callable, Dict, List, Optional
from tqdm import tqdm
from torch.optim import Optimizer

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

from openrlhf.trainer.ppo_utils.experience_maker_card_game import RemoteExperienceMaker_CardGame
# from gui_env.robust_parallel_desktop_env import ParallelDesktopEnv
from openrlhf.textgrad.custom_reward_functions import check_answer_commonsense_qa, check_answer_math
from openrlhf.trainer.ppo_utils.replay_buffer import ReplayBuffer_CARDGAME

from card_env.gym_cards.prompt_lib import PROMPT_FN
from card_env.gym_cards.config_dataclass import EnvConfig, PromptConfig, load_config_from_yaml
from card_env.gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline


def make_env(env_config: EnvConfig, language_only=False):
    def _init():
        config_dict = {k: v for k, v in vars(env_config).items() if k != "id" and k != "num_steps"}
        config_dict["language_only"] = language_only
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
            # self.data_processor = get_data_processor(pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
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
        self.prepare_env()

        # configure scheduler
        self.num_update_steps_per_episodes = (self.num_steps * self.num_envs * args.max_epochs // args.train_batch_size)
        # num_rollouts_per_episodes = (
        #     self.num_update_steps_per_episodes
        #     * args.train_batch_size
        #     // args.max_epochs
        #     // args.rollout_batch_size
        #     // args.n_samples_per_prompt
        # )
        # assert num_rollouts_per_episodes != 0, f"num_rollouts_per_episodes is 0. {args.train_batch_size}, {args.max_epochs}, {args.rollout_batch_size}, {args.n_samples_per_prompt}"

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
        self.action_space = []
        self.num_envs = self.configs.num_envs
        self.num_steps = self.configs.env_config.num_steps
        self.num_updates = self.configs.num_updates
        self.compute_return_kwargs = self.configs.compute_return_kwargs
        env_fns = [make_env(self.configs.env_config, language_only=self.configs.prompt_config.use_language) for _ in range(self.num_envs)]
        self.envs = None
        try:
            print(f"Creating {self.num_envs} parallel environments...")
            # context="spawn" is often more robust, especially on Windows/macOS
            # self.envs = gym.vector.AsyncVectorEnv(env_fns, context="spawn", autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            self.envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            print("Environments created.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()

        # self.env = gym.make(**self.configs.env_config, language_only=self.configs.prompt_config.use_language)
        # self.id = self.configs.env_config.id
        # if 'gym_cards' in self.configs.env_config.id:
        #     self.target_number = self.configs.env_config.target_points
        #     self.treat_face_cards_as_10 = self.configs.env_config.treat_face_cards_as_10

        # # prompt related settings
        # self.enable_verification = self.configs.prompt_config.enable_verification
        # self.use_vision = self.configs.prompt_config.use_vision
        # self.use_language = self.configs.prompt_config.use_language
        # if self.configs.prompt_config.use_vision:
        #     self.prompt_vision = [PROMPT_FN[i] for i in self.configs.prompt_config.prompt_vision]
        #     self.pattern_vision = self.configs.prompt_config.pattern_vision
        #     assert len(self.prompt_vision) == len(self.prompt_vision), "Young folk, check # of prompts & # of patterns in vision."
            
        # if self.configs.prompt_config.use_language:
        #     self.prompt_language = [PROMPT_FN[i] for i in self.configs.prompt_config.prompt_language]
        #     self.pattern_language = self.configs.prompt_config.pattern_language
        #     assert len(self.prompt_language) == len(self.prompt_language), "Young folk, check # of prompts & # of patterns in language."

        # self.generation_config = self.configs.generation_config

        # self.oracle_arguments = {}
        # if 'gym_cards' in self.id:
        #     self.oracle_arguments['valid_ops'] = self.action_space
        #     self.oracle_arguments['face_card_msg'] = "'J', 'Q', and 'K' count as '10'." if self.treat_face_cards_as_10 \
        #                                     else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
        #     self.oracle_arguments['target_number'] = str(self.target_number)
        # elif 'gym_virl' in self.id:
        #     self.oracle_arguments['action_space'] = self.action_space



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
            env_config=self.configs.env_config,
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
        env_config: EnvConfig = None,
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
        self.prompt_config = prompt_config

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

        self.experience_maker = RemoteExperienceMaker_CardGame(
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

        # # vLLM wakeup when vllm_enable_sleep
        # if self.strategy.args.vllm_enable_sleep:
        #     from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
        #     batch_vllm_engine_call(self.vllm_engines, "wake_up")
        #     torch.distributed.barrier()
        #     torch.cuda.synchronize()

        # if global_steps == 0 or global_steps == 1:
        #     if self.strategy.args.dataset_name == "math":
        #         eval_accuracy = self.evaluate_math()
        #     else:
        #         eval_accuracy = self.evaluate()
        #     print(f"Init eval accuracy: {eval_accuracy}")
        #     status["eval_accuracy"] = eval_accuracy
        #     if self.strategy.is_rank_0() and self._wandb is not None:
        #         logs = {
        #             "train/%s" % k: v
        #             for k, v in {**status, "global_step": global_steps}.items()
        #         }
        #         self._wandb.log(logs)

        # # vLLM offload when vllm_enable_sleep
        # if self.strategy.args.vllm_enable_sleep:
        #     batch_vllm_engine_call(self.vllm_engines, "sleep")
        # torch.cuda.empty_cache()
        # torch.distributed.barrier()
        # torch.cuda.synchronize()

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

        # # 6. evaluate
        # if self.strategy.args.eval:
        #     if self.strategy.args.vllm_enable_sleep:
        #         from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
        #         batch_vllm_engine_call(self.vllm_engines, "wake_up")
        #         torch.distributed.barrier()
        #         torch.cuda.synchronize()
        #     if self.strategy.args.dataset_name == "math":
        #         eval_accuracy = self.evaluate_math()
        #     else:
        #         eval_accuracy = self.evaluate()
        #     status["eval_accuracy"] = eval_accuracy

        #     if self.strategy.args.vllm_enable_sleep:
        #         batch_vllm_engine_call(self.vllm_engines, "sleep")
        #     torch.cuda.empty_cache()
        #     torch.distributed.barrier()
        #     torch.cuda.synchronize()

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

    # def _broadcast_to_vllm(self):
    #     """
    #     Consolidate every ZeRO‑3 shard into full FP16 tensors,
    #     build a CPU state_dict, and push it to all vLLM actors.
    #     """
    #     ds_engine: deepspeed.DeepSpeedEngine = self.actor.model

    #     # 1) Manually build a full fp16 state dict on rank 0
    #     full_sd = {}
    #     if torch.distributed.get_rank() == 0:
    #         # `ds_engine.module` is your original nn.Module
    #         for name, param in ds_engine.module.named_parameters():
    #             if hasattr(param, "ds_id"):
    #                 # gather this shard → get full tensor → re‑shard on exit
    #                 with deepspeed.zero.GatheredParameters([param], enabled=True):
    #                     full_sd[name] = param.data.clone().cpu()
    #             else:
    #                 full_sd[name] = param.data.clone().cpu()

    #     # 2) Broadcast the dict to every vLLM actor
    #     if torch.distributed.get_rank() == 0:
    #         # one RPC per actor, loading all weights in one shot
    #         refs = [
    #             vllm_engine.load_state_dict.remote(full_sd, strict=False)
    #             for vllm_engine in self.vllm_engines
    #         ]
    #         ray.get(refs)

    #     # 3) Barrier so that every rank waits until the load is done
    #     torch.distributed.barrier()
    #     torch.cuda.synchronize()


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
                    # Clear active sub-modules before partitioning
                    if hasattr(param, 'ds_active_sub_modules'):
                        # Store the active sub-modules to restore later
                        active_sub_modules = param.ds_active_sub_modules.copy()
                        param.ds_active_sub_modules.clear()
                    
                    try:
                        if torch.distributed.get_rank() == 0:
                            if use_ray:
                                import ray.util.collective as collective
                                collective.broadcast(param.data, 0, group_name=self._model_update_group)
                            else:
                                torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                            ray.get(refs)
                    finally:
                        # Restore active sub-modules
                        if hasattr(param, 'ds_active_sub_modules') and 'active_sub_modules' in locals():
                            param.ds_active_sub_modules.update(active_sub_modules)
                            
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
    
    # def _broadcast_module(self,module,prefix=None,empty_cache=False,need_gather=False):
    #     count, num_params = 0, len(list(module.named_parameters()))
    #     for name, param in module.named_parameters(prefix=prefix):
    #         # broadcast
    #         count += 1
    #         if not self.use_cuda_ipc:
    #             use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
    #             # Fire all vllm engines for broadcast
    #             if torch.distributed.get_rank() == 0:
    #                 shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
    #                 refs = [
    #                     engine.update_weight.remote(
    #                         name, dtype=param.dtype, shape=shape, empty_cache=empty_cache and count==num_params,
    #                     )
    #                     for engine in self.vllm_engines
    #                 ]

    #             # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
    #             with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
    #                 if torch.distributed.get_rank() == 0:
    #                     if use_ray:
    #                         import ray.util.collective as collective

    #                         collective.broadcast(param.data, 0, group_name=self._model_update_group)
    #                     else:
    #                         torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
    #                     ray.get(refs)
    #         # CUDA IPC
    #         else:
    #             from torch.multiprocessing.reductions import reduce_tensor

    #             # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
    #             with deepspeed.zero.GatheredParameters([param], enabled=need_gather):
    #                 weight = param.data.clone()
    #                 ipc_handle = reduce_tensor(weight)

    #                 ipc_handle = {get_physical_gpu_id(): ipc_handle}
    #                 ipc_handle_list = [None] * torch.distributed.get_world_size()
    #                 torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

    #                 if torch.distributed.get_rank() == 0:
    #                     ipc_handles = {}
    #                     for d in ipc_handle_list:
    #                         ipc_handles.update(d)

    #                     shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
    #                     refs = [
    #                         engine.update_weight_cuda_ipc.remote(
    #                             name,
    #                             dtype=param.dtype,
    #                             shape=shape,
    #                             ipc_handles=ipc_handles,
    #                             empty_cache=empty_cache and count==num_params,
    #                         )
    #                         for engine in self.vllm_engines
    #                     ]
    #                     ray.get(refs)
    #                 torch.distributed.barrier()
    #                 torch.cuda.synchronize()

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