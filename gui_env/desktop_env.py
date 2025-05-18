"""
GUI environment for VLM agents
"""

import logging
import os
import time
import random
import platform
import psutil
import requests
import subprocess
from pathlib import Path
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union
from filelock import FileLock

import gymnasium as gym
import docker

from gui_env.docker_utils import (
    send_tar_archive, make_dir,
    copy_file_inside_container,
    copy_files_inside_container, 
    open_libreoffice_file_via_system, 
    close_all_windows,
    save_current_file,
    execute_python_command,
    execute_action,
    change_directory_group_permissions,
    activate_libreoffice_window,
    get_vm_file
)



logger = logging.getLogger("gui_env.desktop_env")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 120

class PortAllocationError(Exception):
    pass


class DesktopEnv(gym.Env):
    """
    A desktop environment (GUI) with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """
    def __init__(
        self,
        docker_image_name: str = "myrepo/ubuntu2004-pptx-150drills_v3:latest",
        host_task_dir: str = "/home/suganuma/src/gui_agent/drills_vm/",
        action_space: str = "pyautogui",
        cache_dir: str = "/home/suganuma/src/gui_agent/cache",
    ):
        self.docker_image_name = docker_image_name
        self.client = docker.from_env()
        self.environment = {"DISK_SIZE": "16G", "RAM_SIZE": "8G", "CPU_CORES": "4"}  # Modify if needed
        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / 'docker_port_allocation.lck'
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        self.default_user = "default"
        self.default_vnc_port = 15901
        self.default_http_port = 20901
        self.default_automation_port = 30901
        self.docker_vnc_port = 5901
        self.docker_http_port = 10901
        self.docker_automation_port = 10902
        self.container = None
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None

        self._traj_no: int = -1
        self._step_no: int = 0
        self.action_history: List[Dict[str, any]] = []
        self.instruction: str = None
        self.action_reference: str = None
        self.action_space: str = action_space
        self.host_task_dir = host_task_dir
        self.docker_download_dir = f"/home/{self.default_user}/Downloads"
        self.docker_workspace_dir = f"/home/{self.default_user}/Desktop"
        
        # self.cache_dir = '/home/suganuma/src/gui_agent/cache'
        self.cache_dir = cache_dir
        
        self.retry_times = 5
        
    
    def _get_used_ports(self):
        """Get all currently used ports (both system and Docker)."""
        # Get system ports
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        
        # Get Docker container ports
        docker_ports = set()
        for container in self.client.containers.list():
            ports = container.attrs['NetworkSettings']['Ports']
            if ports:
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p['HostPort']) for p in port_mappings)
        
        return system_ports | docker_ports
    
    def _get_available_port(self, start_port: int) -> int:
        """Find next available port starting from start_port."""
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")
    
    def _wait_for_vm_ready(self, timeout: int = 60):
        """Wait for VM to be ready by checking screenshot endpoint."""
        time.sleep(5)
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.automation_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception as e:
                print(f"Error checking screenshot: {e}")
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

        
    def _start_emulator(self, container_id: str = None):
        lock = FileLock(str(self.lock_file), timeout=LOCK_TIMEOUT)

        try:
            with lock:
                # Allocate all required ports
                self.vnc_port = self._get_available_port(self.default_vnc_port)
                self.server_port = self._get_available_port(self.default_http_port)
                self.automation_port = self._get_available_port(self.default_automation_port)

                ports = {
                    self.docker_vnc_port: self.vnc_port,
                    self.docker_http_port: self.server_port,
                    self.docker_automation_port: self.automation_port
                }

                logger.info(f"Ports: {ports}")

                # self.container = self.client.containers.run(
                #     self.docker_image_name,
                #     name="ubuntu2004",
                #     environment=self.environment,
                #     detach=True,                       # Run container in background
                #     auto_remove=True,                  # Automatically remove container on exit
                #     privileged=True,                   # Run in privileged mode
                #     cgroupns="host",                   # Use host's cgroup namespace
                #     cap_add=["SYS_BOOT", "SYS_ADMIN"],  # Additional capabilities
                #     devices=["/dev/kvm"],
                #     tmpfs={"/run": "", "/run/lock": "", "/tmp": ""},  # Tmpfs mounts
                #     volumes={"/sys/fs/cgroup": {"bind": "/sys/fs/cgroup", "mode": "rw"}},  # Bind mount
                #     ports=ports,                       # Dynamic port mapping
                #     stdin_open=True,                   # Keep STDIN open for interactive sessions
                #     tty=True                           # Allocate a pseudo-TTY
                # )
                random_id = random.randint(0, 1000000)
                self.container = self.client.containers.run(
                    self.docker_image_name,
                    name=f"ubuntu2004_{container_id}" if container_id is not None else f"ubuntu2004_{random_id}",
                    environment=self.environment,
                    detach=True,                       
                    auto_remove=True,
                    privileged=True,                   
                    cgroupns="host",                   
                    cap_add=["SYS_BOOT", "SYS_ADMIN"],  
                    devices=[],
                    tmpfs={"/run": "", "/run/lock": "", "/tmp": "rw"},  
                    volumes={
                        "/sys/fs/cgroup": {"bind": "/sys/fs/cgroup", "mode": "rw"},
                        "/dev/shm": {"bind": "/dev/shm", "mode": "rw"},
                    },
                    ports=ports,                       
                    stdin_open=True,                   
                    tty=True                           
                )

                # send the task files to the container
                # logger.info(f"Setting up environment from {self.host_task_dir}...")
                task_dir = os.path.abspath(self.host_task_dir)
                task_files = os.listdir(task_dir)
                task_paths = [os.path.join(task_dir, file) for file in task_files]
                # # make_dir(self.container, self.docker_download_dir)
                # # make_dir(self.container, self.docker_workspace_dir)
                # send_tar_archive(self.container, task_paths, self.docker_download_dir)
                time.sleep(2)
                # copy_files_inside_container(self.container, self.docker_download_dir, self.docker_workspace_dir)
                # time.sleep(2)
                # change_directory_group_permissions(self.docker_workspace_dir, self.automation_port)
            
            logger.info(f"Started container with ports - VNC: {self.vnc_port}, "
                       f"Server: {self.server_port}, Automation: {self.automation_port}")
            self._wait_for_vm_ready()

        except Exception as e:
            # Clean up if anything goes wrong
            if self.container:
                try:
                    self.container.stop()
                    self.container.remove()
                except:
                    pass
            raise e
        
    def stop_emulator(self):
        if self.container:
            logger.info("Stopping VM...")
            try:
                self.container.stop()
                # self.container.remove()
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None

    
    def _close_all_windows(self):
        """
        Uses wmctrl to list and close all open windows in the container.
        Assumes wmctrl is installed and that closing via wmctrl works in your environment.
        """
        # List all windows
        list_result = self.container.exec_run("wmctrl -l", tty=True)
        output = list_result.output.decode("utf-8").strip()
        if not output:
            logger.info("No windows found.")
            return
        logger.info("Open windows:")
        logger.info(output)
        
        # Iterate over each window and close it
        for line in output.splitlines():
            parts = line.split()
            if parts:
                window_id = parts[0]
                # Command to close a window by its ID
                close_cmd = f"wmctrl -ic {window_id}"
                logger.info(f"Closing window {window_id}...")
                self.container.exec_run(close_cmd, tty=True)
                time.sleep(0.5)  # Allow a brief pause for the window to close
        logger.info("All windows closed.")
    
    def _close_confirmation_windows(self, keywords=("Save", "Confirm", "Unsaved")):
        """
        Searches for windows with titles containing any of the specified keywords inside the container,
        and sends a close command using wmctrl.
        
        Args:
            container: Docker container object.
            keywords: Tuple of strings to search for in window titles.
        """
        # List all windows inside the container using wmctrl
        exec_result = self.container.exec_run("wmctrl -l", tty=True)
        output = exec_result.output.decode("utf-8").strip()
        
        if not output:
            logger.info("No windows found inside container.")
            return

        logger.info("List of windows inside container:")
        logger.info(output)

        # Loop through each window and close those that match our keywords
        for line in output.splitlines():
            if any(keyword.lower() in line.lower() for keyword in keywords):
                parts = line.split()
                if parts:
                    window_id = parts[0]
                    logger.info(f"Found confirmation window (ID: {window_id}): {line}")
                    close_cmd = f"wmctrl -ic {window_id}"
                    result = self.container.exec_run(close_cmd, tty=True)
                    if result.exit_code == 0:
                        logger.info(f"Closed confirmation window {window_id}")
                    else:
                        logger.info(f"Failed to close window {window_id}. Exit code: {result.exit_code}")
                    time.sleep(0.5)
    
    def reset(self, task_config: Dict, container_id: str = None):
        """
        Reset the environment.
        Close all windows, start the emulator, and send the task files to the container.
        """
        logger.info("Resetting environment...")
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()

        # If the container is already running, close all windows and copy the task files to the workplace in the container.
        if self.container:
            print(f"Container of {container_id} is already running so open the next file." if container_id is not None else "Container is already running so open the next file.")
            # save_current_file(self.base_file, self.automation_port)
            # logger.info(f"Closing the all windows of {container_id}...")
            # time.sleep(3)
            # flag = close_all_windows(self.automation_port)
            # time.sleep(3)
            # if not flag:
            #     logger.error(f"Restart the container of {container_id} due to the failure of closing all windows.")
            #     self.stop_emulator()
            #     self._start_emulator(container_id+"_restarted")
            #     logger.info(f"Emulator of {container_id} restarted.")
        else:
            logger.info(f"Starting emulator of {container_id}...")
            self._start_emulator(container_id)
            logger.info(f"Emulator of {container_id} started.")

        if task_config is not None:
            self.instruction = task_config["instruction"]
            self.action_reference = task_config["action_reference"] if "action_reference" in task_config else None
            self.task_file = task_config["task_file"]
            self.base_file = task_config["base_file"]
            file_path = os.path.join(self.docker_workspace_dir, "Downloads", self.base_file)
            open_libreoffice_file_via_system(file_path, self.automation_port)
            logger.info(f"Opened file: {self.base_file} at {container_id}")
            time.sleep(7)
        
        observation = self._get_obs()
        return observation
    
    def _get_screenshot(self):
        for _ in range(self.retry_times):
            try:
                screenshot_url = f"http://localhost:{self.automation_port}/screenshot"
                response = requests.get(screenshot_url)
                if response.status_code == 200:
                    return response.content
            except Exception as e:
                logger.error(f"Error getting screenshot: {e}")
                logger.info("Retrying...")
                time.sleep(1)

        logger.error(f"Failed to get screenshot after {self.retry_times} retries")
        return None
    
    def _get_obs(self):
        return {
            "screenshot": self._get_screenshot(),
            "instruction": self.instruction,
            "action_reference": self.action_reference if self.action_reference is not None else ""
        }
    
    def step(self, action, pause=2):
        self._step_no += 1
        self.action_history.append(action)

        reward = 0
        done = False
        info = {}

        if action in ['WAIT', 'FAIL', 'DONE'] or (type(action) == dict and action['action_type'] in ['WAIT', 'FAIL', 'DONE']):
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}
        
        if self.action_space == "computer_13":
            raise NotImplementedError("Computer 13 action space is not implemented yet.")
        
        elif self.action_space == "pyautogui":
            if action in ['WAIT', 'FAIL', 'DONE']:
                execute_action(action, f"http://localhost:{self.automation_port}/execute")
            else:
                # the set of all possible python commands insides `pyautogui`
                execute_python_command(action, f"http://localhost:{self.automation_port}/execute")

        time.sleep(pause)
        observation = self._get_obs()

        return observation, reward, done, info
    

    def evaluate(self, task_config, container_id: str = None, eval_agent=None):
        """
        Evaluate the task performed by an agent - focus only on evaluation.
        """
        success = False
        total_reward = 0
        needs_restart = False  # Flag to indicate if container needs restart
        
        try:
            # 1. Save the current file
            save_success = save_current_file(self.base_file, self.automation_port)
            time.sleep(2)
            
            # 2. Close all windows
            close_success = close_all_windows(self.automation_port, max_attempts=2)
            time.sleep(2)
            
            # 3. Get the file from the container
            file_path = os.path.join(self.docker_workspace_dir, "Downloads", self.base_file)
            file = get_vm_file(file_path, self.automation_port)
            
            if file is not None:
                # Process file and calculate rewards
                task_file_name = self.task_file.split("/")[-1]
                cache_dir = os.path.join(self.cache_dir, container_id)
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                _path = os.path.join(cache_dir, task_file_name)
                with open(_path, "wb") as f:
                    f.write(file)
                
                # Compute rewards...
                eval_script = os.path.join(self.host_task_dir, task_config["eval_script"])
                result = subprocess.run(
                    ["python", eval_script, "--pptx_file", _path], 
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    success = True
                    hard_reward = 10
                else:
                    hard_reward = 0

                if eval_agent is not None:
                    soft_reward = eval_agent.evaluate(file)
                else:
                    soft_reward = 0
                    
                total_reward = hard_reward + soft_reward

                # 4. Restore original file if needed
                if close_success and file is not None:
                    original_file_path = os.path.join(self.docker_download_dir, self.base_file)
                    workspace_path = os.path.join(self.docker_workspace_dir, "Downloads", self.base_file)
                    copy_file_inside_container(self.container, src=original_file_path, dst=workspace_path)
                    time.sleep(1)
            else:
                logger.info(f"Failed to get the file from the container of {container_id}.")
                success = False
                total_reward = 0
                
            # Set restart flag if needed
            if not close_success or file is None:
                needs_restart = True
        
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            success = False
            total_reward = 0
            needs_restart = True
        
        return success, total_reward, needs_restart
    
    # def evaluate(self, task_config, container_id: str = None, eval_agent=None):
    #     """
    #     Evaluate the task peformed by an agent.
    #     """

    #     # activate the window of the target task and save the file
    #     save_current_file(self.base_file, self.automation_port)
    #     time.sleep(2)
    #     # close all windows
    #     flag = close_all_windows(self.automation_port, max_attempts=2)
    #     time.sleep(2)
    #     # get the file from the container
    #     file_path = os.path.join(self.docker_workspace_dir, "Downloads", self.base_file)
    #     file = get_vm_file(file_path, self.automation_port)
    #     if file is not None:
    #         task_file_name = self.task_file.split("/")[-1]
    #         _path = os.path.join(self.cache_dir, task_file_name)
    #         with open(_path, "wb") as f:
    #             f.write(file)
            
    #         ## compute the reward
    #         total_reward = 0
    #         success = False
    #         # predefined reward (running an evaluation python script)
    #         eval_script = os.path.join(self.host_task_dir, task_config["eval_script"])
    #         result = subprocess.run(
    #             ["python", eval_script, "--pptx_file", _path], 
    #             capture_output=True,
    #             text=True
    #         )
    #         if result.returncode == 0:
    #             success = True
    #             hard_reward = 1
    #         else:
    #             hard_reward = 0

    #         # soft reward by Qwen2.5-VL-7B-Instruct
    #         if eval_agent is not None:
    #             soft_reward = eval_agent.evaluate(file)
    #         else:
    #             soft_reward = 0
    #         total_reward = hard_reward + soft_reward

    #     else:
    #         logger.info(f"Failed to get the file from the container of {container_id}.")
    #         success = False
    #         total_reward = 0

    #     # recover the file in the container to an original file
    #     if flag and file is not None:
    #         original_file_path = os.path.join(self.docker_download_dir, self.base_file)
    #         workspace_path = os.path.join(self.docker_workspace_dir, "Downloads", self.base_file)
    #         copy_file_inside_container(self.container, src=original_file_path, dst=workspace_path)
    #         time.sleep(1)
    #     else:
    #         logger.info(f"Restart the container due to the failure of closing all windows or getting the file from the container of {container_id}.")
    #         self.stop_emulator()
    #         self._start_emulator(container_id+"_restarted")
    #         logger.info(f"Emulator of {container_id} restarted.")

    #     return success, total_reward