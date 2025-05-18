"""
Parallel Desktop Environment for efficient multi-GPU training with robust error handling
"""

import os
import sys
import time
import json
import gc
import logging
import datetime
import traceback
from typing import List, Dict, Optional, Union, Any
from filelock import FileLock
from gui_env.desktop_env import DesktopEnv

# Configure logging
logger = logging.getLogger("gui_env.parallel_desktop_env")
logger.setLevel(logging.INFO)

class ContainerStatus:
    """Track status and health of a single container"""
    def __init__(self, container_id: str):
        self.id = container_id
        self.healthy = True
        self.restart_count = 0
        self.last_restart_time = None
        self.last_health_check = time.time()
        self.consecutive_failures = 0

class ParallelDesktopEnv:
    """
    Manages multiple Docker-based desktop environments in parallel with robust error handling.
    """

    def __init__(
        self,
        num_containers: int = 4,
        docker_image_name: str = "myrepo/ubuntu2004-pptx-150drills:custom",
        host_task_dir: str = "/home/user/src/gui_agent/drills_vm/",
        action_space: str = "pyautogui",
        task_config_list: List[Dict] = None,
        example_result_dir: str = "results/parallel_env",
        cache_dir: str = "cache",
        process_id: int = 0,
        max_restart_attempts: int = 10,
        health_check_interval: int = 300  # 5 minutes
    ):
        self.num_containers = num_containers
        self.envs: List[DesktopEnv] = [None] * num_containers
        self.docker_image_name = docker_image_name
        self.host_task_dir = host_task_dir
        self.action_space = action_space
        self.task_config_list = task_config_list
        self.max_restart_attempts = max_restart_attempts
        self.health_check_interval = health_check_interval
        self.cache_dir = cache_dir
        self.rank = process_id
        
        assert self.task_config_list is not None, "task_config_list is not provided"
        self.num_samples = len(self.task_config_list)
        self.time_stamps = [[] for _ in range(self.num_containers)]
        self.example_result_dir = example_result_dir + f"/process_{process_id}"
        if not os.path.exists(self.example_result_dir):
            os.makedirs(self.example_result_dir, exist_ok=True)
            
        # Track container statuses
        self.container_status = [None] * num_containers
        
        # Lock for synchronizing container operations
        temp_dir = "/tmp"
        self.lock_file = os.path.join(temp_dir, 'parallel_env_operations.lck')

        # task_id counter
        self.task_id_counter = 0
        
    def launch_containers(self):
        """
        Launch multiple containers in parallel with better error handling.
        """
        for i in range(self.num_containers):
            container_id = f"container_{i}"
            self.container_status[i] = ContainerStatus(container_id)
            
            try:
                env = DesktopEnv(
                    docker_image_name=self.docker_image_name,
                    host_task_dir=self.host_task_dir,
                    action_space=self.action_space,
                    cache_dir=self.cache_dir
                )
                self.envs[i] = env
                # self.envs.append(env)
                logger.info(f"Successfully launched container {i} with ID {container_id}")
            except Exception as e:
                logger.error(f"Failed to launch container {i}: {e}")
                # Add placeholder to maintain indexing
                self.envs[i] = None
                self.container_status[i].healthy = False
                
        # Force garbage collection after container launch
        gc.collect()
        time.sleep(1)
        
        # Verify all containers are responsive
        self._check_all_container_health()

    def _check_container_health(self, env_index: int) -> bool:
        """
        Check if a specific container is healthy by trying to get a screenshot.
        """
        if env_index >= len(self.envs) or self.envs[env_index] is None:
            return False
            
        try:
            env = self.envs[env_index]
            # Try to perform a simple operation that would fail if container is unhealthy
            screenshot = env._get_screenshot()
            
            # Update health status
            if screenshot is not None:
                self.container_status[env_index].healthy = True
                self.container_status[env_index].consecutive_failures = 0
                self.container_status[env_index].last_health_check = time.time()
                return True
            else:
                self.container_status[env_index].healthy = False
                self.container_status[env_index].consecutive_failures += 1
                logger.warning(f"Container {env_index} health check failed: Screenshot is None")
                return False
        except Exception as e:
            logger.error(f"Container {env_index} health check failed with error: {e}")
            self.container_status[env_index].healthy = False
            self.container_status[env_index].consecutive_failures += 1
            return False
            
    def _check_all_container_health(self) -> List[int]:
        """
        Check health of all containers and return list of unhealthy container indices.
        """
        unhealthy_containers = []
        
        for i in range(len(self.envs)):
            # Only check containers that haven't been checked recently
            current_time = time.time()
            if (current_time - self.container_status[i].last_health_check > self.health_check_interval):
                is_healthy = self._check_container_health(i)
                if not is_healthy:
                    unhealthy_containers.append(i)
                    
        return unhealthy_containers
        
    def restart_container(self, rank: int, env_index: int) -> bool:
        """
        Restart a specific container with proper error handling and resource cleanup.
        """
        if env_index >= len(self.envs):
            logger.error(f"Invalid container index: {env_index}")
            return False
            
        status = self.container_status[env_index]
        
        # Check if we've exceeded the max restart attempts
        if status.restart_count >= self.max_restart_attempts:
            logger.error(f"Container {env_index} has been restarted {status.restart_count} times, exceeding the limit of {self.max_restart_attempts}")
            return False
            
        # Increment restart count and update container ID
        status.restart_count += 1
        old_id = status.id
        status.id = f"{rank}_{env_index}_restart_{status.restart_count}"
        status.last_restart_time = time.time()
        
        logger.info(f"Restarting container {rank}_{env_index} (ID: {old_id}) as {status.id}")
        
        # Use file lock to prevent port allocation conflicts
        lock = FileLock(self.lock_file, timeout=60)
        
        try:
            with lock:
                # Stop the old container if it exists
                if self.envs[env_index] is not None:
                    try:
                        self.envs[env_index].stop_emulator()
                    except Exception as e:
                        logger.error(f"Error stopping container {rank}_{env_index}: {e}")
                    
                    # Explicitly delete the container reference to release resources
                    # del self.envs[env_index]
                    # gc.collect()
                    
                # Wait to ensure resources are released properly
                time.sleep(5)
                
                # Create a new container
                try:
                    new_env = DesktopEnv(
                        docker_image_name=self.docker_image_name,
                        host_task_dir=self.host_task_dir,
                        action_space=self.action_space
                    )
                    
                    # Replace in our list
                    self.envs[env_index] = new_env
                    self.envs[env_index]._start_emulator(status.id)
                    status.healthy = True
                    logger.info(f"Container {env_index} successfully restarted as {status.id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to create new container for index {env_index}: {e}")
                    logger.error(traceback.format_exc())
                    self.envs[env_index] = None
                    status.healthy = False
                    return False
        except Exception as e:
            logger.error(f"Error during container restart process: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # Force garbage collection after restart operations
            gc.collect()

    def reset(self, task_config: Dict = None) -> List[List[Dict]]:
        """
        Reset environments with health checks and automatic container recovery.
        """
        # Reset timestamps and observation lists
        self.time_stamps = [[] for _ in range(self.num_containers)]
        obs_list = [[] for _ in range(self.num_containers)]
        
        # Check for and restart unhealthy containers
        unhealthy_containers = self._check_all_container_health()
        for container_idx in unhealthy_containers:
            logger.warning(f"Container {container_idx} is unhealthy, attempting restart before reset")
            self.restart_container(self.rank, container_idx)
        
        # Reset each container
        for container_id, env in enumerate(self.envs):
            idx = f"{self.rank}_{container_id}" if self.rank is not None else container_id
            
            # Skip if container is None (failed to initialize or restart)
            if env is None:
                obs_list[container_id].append({
                    "obs": {"screenshot": None, "instruction": task_config.get("instruction", "")},
                    "reward": 0, 
                    "done": False, 
                    "info": {"status": "container_unavailable"}
                })
                continue
                
            try:
                if self.task_id_counter >= len(self.task_config_list):
                    self.task_id_counter = 0
                task_config = self.task_config_list[self.task_id_counter] if task_config is None else task_config
                self.task_id_counter += 1
                obs = env.reset(task_config, idx)
                obs_list[container_id].append({"obs": obs, "reward": 0, "done": False, "info": {"status": "initial"}})
                # Update health status on successful reset
                self.container_status[container_id].healthy = True
                self.container_status[container_id].consecutive_failures = 0
                
            except Exception as e:
                logger.error(f"Error resetting container {idx}: {e}")
                # Try to restart the container
                if self.restart_container(self.rank, container_id):
                    try:
                        # Try reset again with new container
                        if self.task_id_counter >= len(self.task_config_list):
                            self.task_id_counter = 0
                        task_config = self.task_config_list[self.task_id_counter] if task_config is None else task_config
                        self.task_id_counter += 1
                        obs = self.envs[container_id].reset(task_config, idx)
                        obs_list[container_id].append({"obs": obs, "reward": 0, "done": False, "info": {"status": "restarted"}})
                        
                    except Exception as e2:
                        logger.error(f"Reset still failing after restart for container {container_id}: {e2}")
                        obs_list[container_id].append({
                            "obs": {"screenshot": None, "instruction": task_config.get("instruction", "")},
                            "reward": 0, 
                            "done": False, 
                            "info": {"status": "reset_failed_after_restart"}
                        })
                else:
                    # If restart failed, return empty observation
                    if self.task_id_counter >= len(self.task_config_list):
                        self.task_id_counter = 0
                    task_config = self.task_config_list[self.task_id_counter] if task_config is None else task_config
                    self.task_id_counter += 1
                    obs_list[container_id].append({
                        "obs": {"screenshot": None, "instruction": task_config.get("instruction", "")},
                        "reward": 0, 
                        "done": False, 
                        "info": {"status": "reset_failed_restart_failed"}
                    })
                    
        
        # Force garbage collection after reset operations
        gc.collect()
        return obs_list
    
    def step_all(self, all_env_actions: List[List[str]], step_indices: List[int], active_indices: List[int]) -> List[List[Dict]]:
        """
        Send actions to containers with better error handling and recovery.
        """
        assert len(all_env_actions) == len(self.envs), "The number of actions must be equal to the number of containers."
        
        # Create task directory
        task_dir = os.path.join(self.example_result_dir, f"task_{self.task_id_counter}")
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)

        obs = [[] for _ in range(self.num_containers)]

        # Process each environment
        for container_id, env in enumerate(self.envs):
            actions = all_env_actions[container_id] if container_id < len(all_env_actions) else None
            
            # Skip if container is inactive or actions are None
            if container_id not in active_indices or actions is None or env is None:
                obs[container_id].append({
                    "obs": {"screenshot": None, "instruction": None, "action_reference": None},
                    "reward": 0,
                    "done": True,
                    "info": {"status": "inactive_or_no_actions"}
                })
                continue

            # Check container health if needed
            current_time = time.time()
            if (current_time - self.container_status[container_id].last_health_check > self.health_check_interval):
                if not self._check_container_health(container_id):
                    logger.warning(f"Container {container_id} is unhealthy, restarting before step")
                    restart_success = self.restart_container(self.rank, container_id)
                    if not restart_success:
                        obs[container_id].append({
                            "obs": {"screenshot": None, "instruction": None, "action_reference": None},
                            "reward": 0,
                            "done": True,
                            "info": {"status": "container_restart_failed"}
                        })
                        continue
                    # Since restart changes container state, we need to bail on this step
                    obs[container_id].append({
                        "obs": {"screenshot": None, "instruction": None, "action_reference": None},
                        "reward": 0,
                        "done": False,
                        "info": {"status": "container_restarted_skipping_step"}
                    })
                    continue
            
            # Execute actions
            try:
                o = None
                r = 0
                done = False
                info = {}
                
                for action in actions:
                    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                    
                    try:
                        o, r, done, info = env.step(action, pause=2)
                    except Exception as e:
                        logger.error(f"Error executing action in container {container_id}: {e}")
                        # Try one more time after a short pause
                        time.sleep(2)
                        try:
                            o, r, done, info = env.step("WAIT", pause=1)
                        except Exception as e2:
                            logger.error(f"Second attempt also failed for container {container_id}: {e2}")
                            self.container_status[container_id].consecutive_failures += 1
                            
                            # If we have too many failures, try restarting
                            if self.container_status[container_id].consecutive_failures >= 3:
                                self.restart_container(self.rank, container_id)
                            
                            # Skip the rest of the actions
                            break

                    # Save screenshot and trajectory information for the first container (for debugging)
                    if container_id == 0:
                        try:
                            if o and o['screenshot']:
                                with open(os.path.join(task_dir, f"step_{step_indices[container_id] + 1}_{action_timestamp}.jpg"), "wb") as _f:
                                    _f.write(o['screenshot'])
                        except Exception as e:
                            logger.error(f"Error saving screenshot: {e}")
                    
                    # Record trajectory
                    self.time_stamps[container_id].append({
                        "step_num": step_indices[container_id] + 1,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "reward": r,
                        "done": done,
                        "screenshot_file": f"step_{step_indices[container_id] + 1}_{action_timestamp}.jpg"
                    })
                    
                    # Save trajectory to file
                    try:
                        with open(os.path.join(task_dir, f"trajectory_{container_id}.json"), "w") as f:
                            json.dump(self.time_stamps[container_id], f, indent=4)
                    except Exception as e:
                        logger.error(f"Error saving trajectory: {e}")

                    if done:
                        logger.info(f"The episode is done in env {container_id}.")
                        break

                # Record observation
                if o is not None:
                    obs[container_id].append({"obs": o, "reward": r, "done": done, "info": info})
                    
                    # Reset failure count on success
                    self.container_status[container_id].consecutive_failures = 0
                else:
                    # Get fresh observation if we couldn't get one from the step
                    try:
                        o = env._get_obs()
                        obs[container_id].append({
                            "obs": o, 
                            "reward": 0, 
                            "done": False, 
                            "info": {"status": "error_of_action_parsing_at_step_" + str(step_indices[container_id] + 1)}
                        })
                    except Exception as e:
                        logger.error(f"Error getting observation after action failure: {e}")
                        obs[container_id].append({
                            "obs": {"screenshot": None, "instruction": None, "action_reference": None},
                            "reward": 0,
                            "done": False,
                            "info": {"status": "observation_error_after_action_failure"}
                        })

            except Exception as e:
                logger.error(f"Unexpected error in container {container_id}: {e}")
                logger.error(traceback.format_exc())
                
                obs[container_id].append({
                    "obs": {"screenshot": None, "instruction": None, "action_reference": None},
                    "reward": 0,
                    "done": True,
                    "info": {"status": "unexpected_error", "error": str(e)}
                })
                
                # Increment failure count and try restart if needed
                self.container_status[container_id].consecutive_failures += 1
                if self.container_status[container_id].consecutive_failures >= 3:
                    self.restart_container(self.rank, container_id)

        # Force garbage collection after stepping all environments
        gc.collect()
        return obs

    def evaluate_all(self, task_config: Dict, rank: int = None) -> List[float]:
        """
        Evaluate the task in all containers with improved error handling.
        """
        rewards = []
        containers_to_restart = []
        
        for container_id, env in enumerate(self.envs):
            # Skip if container is None
            if env is None:
                rewards.append(0.0)
                continue
                
            try:
                container_tag = f"{rank}_{container_id}" if rank is not None else str(container_id)
                success, reward, needs_restart = env.evaluate(task_config, container_id=container_tag)
                rewards.append(reward)
                if success:
                    logger.info(f"!!!!! Container {container_id} evaluation SUCCESS !!!!!!")

                # If container needs restart, add to list
                if needs_restart:
                    containers_to_restart.append(container_id)
                    logger.warning(f"Container {container_id} flagged for restart after evaluation")
                
                # Reset failure count on success
                self.container_status[container_id].consecutive_failures = 0
                
            except Exception as e:
                logger.error(f"Error evaluating container {container_id}: {e}")
                rewards.append(0.0)
                containers_to_restart.append(container_id)
                self.container_status[container_id].consecutive_failures += 1
                
                # # Try to restart and evaluate again
                # if self.container_status[container_id].consecutive_failures < 3:
                #     self.container_status[container_id].consecutive_failures += 1
                #     try:
                #         success, reward = env.evaluate(task_config, container_id=f"{rank}_{container_id}_retry")
                #         rewards.append(reward)
                #     except Exception as e2:
                #         logger.error(f"Retry evaluation failed for container {container_id}: {e2}")
                #         rewards.append(0.0)
                        
                #         # If multiple failures, try restarting the container
                #         if self.container_status[container_id].consecutive_failures >= 3:
                #             self.restart_container(container_id)
                # else:
                #     # Too many failures, restart and give up on evaluation
                #     self.restart_container(container_id)
                #     rewards.append(0.0)
        
        
        # Restart containers as needed
        for container_id in containers_to_restart:
            logger.info(f"Restarting container {container_id} after evaluation")
            self.restart_container(rank, container_id)
        
        # Force garbage collection
        gc.collect()
        return rewards
    
    def close_all(self):
        """
        Stop all Docker containers with proper resource cleanup.
        """
        for i, env in enumerate(self.envs):
            if env is not None:
                try:
                    logger.info(f"Stopping container {i}")
                    env.stop_emulator()
                except Exception as e:
                    logger.error(f"Error stopping container {i}: {e}")
                
                # Clear reference 
                self.envs[i] = None
                
                # Force garbage collection after each container closure
                gc.collect()
                time.sleep(1)
                
        # Final cleanup
        self.envs = [None] * self.num_containers
        self.container_status = [None] * self.num_containers
        gc.collect()

    def restart_container_by_id(self, container_id: str, rank: int) -> bool:
        """
        Find the container by ID and restart it
        """
        for i, env in enumerate(self.envs):
            if env.container and container_id in env.container.name:
                logger.info(f"Found container {container_id} at index {i}, restarting")
                return self.restart_container(rank, i)
        
        logger.error(f"Container ID {container_id} not found in environment list")
        return False