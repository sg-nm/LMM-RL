"""
Parallel Desktop Environment for efficient multi-GPU training
"""

import os
import sys
import time
import json
from typing import List, Dict
import datetime
from gui_env.desktop_env import DesktopEnv


class ParallelDesktopEnv:
    """
    Manages multiple Docker-based desktop environments in parallel.
    """

    def __init__(
        self,
        num_containers: int = 4,
        docker_image_name: str = "myrepo/ubuntu2004-pptx-150drills:custom",
        host_task_dir: str = "/home/user/src/gui_agent/drills_vm/",
        action_space: str = "pyautogui",
        task_config_list: List[Dict] = None,
        example_result_dir: str = "results/parallel_env",
        process_id: int = 0
    ):
        self.num_containers = num_containers
        self.envs: List[DesktopEnv] = []
        self.docker_image_name = docker_image_name
        self.host_task_dir = host_task_dir
        self.action_space = action_space
        self.task_config_list = task_config_list
        assert self.task_config_list is not None, "task_config_list is not provided"
        self.num_samples = len(self.task_config_list)
        self.time_stamps = [[] for _ in range(self.num_containers)]
        self.example_result_dir = example_result_dir + f"/process_{process_id}"
        if not os.path.exists(self.example_result_dir):
            os.makedirs(self.example_result_dir, exist_ok=True)
        
    def launch_containers(self):
        """
        Launch multiple containers in parallel.
        """
        for _ in range(self.num_containers):
            env = DesktopEnv(
                docker_image_name=self.docker_image_name,
                host_task_dir=self.host_task_dir,
                action_space=self.action_space
            )
            self.envs.append(env)
        time.sleep(1)

    def reset_all(self, task_config_list: List[Dict]):
        """
        Reset each environment with a potentially different task_config.
        """
        # Optionally each container can have a different config or the same config.
        for env, config in zip(self.envs, task_config_list):
            env.reset(config)

    def reset(self, task_config: Dict, rank: int = None):
        """
        Reset the environment with a specific task_config.
        """
        self.time_stamps = [[] for _ in range(self.num_containers)]
        obs_list = [[] for _ in range(self.num_containers)]
        for container_id, env in enumerate(self.envs):
            idx = f"{rank}_{container_id}" if rank is not None else container_id
            obs = env.reset(task_config, idx)
            obs_list[container_id].append({"obs": obs, "reward": 0, "done": False, "info": {"status": "initial"}})
        return obs_list
    
    def step_all(self, all_env_actions: List[List[str]], step_indices: List[int], active_indices: List[int], task_idx: int) -> List[Dict]:
        """
        Send one action per container, gather the next observations.
        """
        assert len(all_env_actions) == len(self.envs), "The number of actions must be equal to the number of containers."
        # Create the directory for the task if it doesn't exist
        task_dir = os.path.join(self.example_result_dir, f"task_{task_idx}")
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)

        obs = [[] for _ in range(self.num_containers)]

        for rank, (env, actions) in enumerate(zip(self.envs, all_env_actions)):
            if rank not in active_indices or actions is None:
                # Add a placeholder for the observation
                obs[rank].append({
                    "obs": {"screenshot": None, "instruction": None},
                    "reward": 0,
                    "done": True,
                    "info": {"status": "inactive"}
                })
                continue

            if actions is not None:
                o = None
                for action in actions:
                    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                    o, r, done, info = env.step(action, pause=2)

                    # Save screenshot and trajectory information
                    if rank == 0:
                        with open(os.path.join(task_dir, f"step_{step_indices[rank] + 1}_{action_timestamp}.jpg"), "wb") as _f:
                            _f.write(o['screenshot'])
                    self.time_stamps[rank].append({
                        "step_num": step_indices[rank] + 1,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "reward": r,
                        "done": done,
                        "screenshot_file": f"step_{step_indices[rank] + 1}_{action_timestamp}.jpg"
                    })
                    with open(os.path.join(task_dir, f"trajectory_{rank}.json"), "w") as f:
                        json.dump(self.time_stamps[rank], f, indent=4)

                    if done:
                        print(f"The episode is done in env {rank}.")
                        break

                if o is not None:
                    obs[rank].append({"obs": o, "reward": r, "done": done, "info": info})
                else:
                    o = env._get_obs()
                    obs[rank].append({"obs": o, "reward": 0, "done": False, "info": {"status": "error of action parsing at step " + str(step_indices[rank] + 1)}})

            else:
                print(f"action is None in env {rank}.")
                o, r, done, info = env.step('WAIT', pause=1)
                obs[rank].append({"obs": o, "reward": r, "done": done, "info": info})

        return obs

    def get_observations(self):
        """
        Get current screenshots & instructions from each container (if needed).
        """
        obs_list = []
        for env in self.envs:
            obs_list.append(env._get_obs())
        return obs_list

    def evaluate_all(self, task_config: Dict, rank: int = None):
        """
        Evaluate the task in all containers.
        """
        rewards = []
        for container_id, env in enumerate(self.envs):
            success, reward = env.evaluate(task_config, container_id=f"{rank}_{container_id}")
            rewards.append(reward)
        return rewards
    
    
    def close_all(self):
        """
        Stop all Docker containers.
        """
        for env in self.envs:
            env.stop_emulator()
        self.envs = []