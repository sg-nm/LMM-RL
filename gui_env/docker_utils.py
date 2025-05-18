import os
import io
import tarfile
import docker
import logging
import requests
import json
import time
import random
from typing import Dict, Any

logger = logging.getLogger("gui_env.docker_utils")

KEYBOARD_KEYS = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace', 'browserback', 'browserfavorites', 'browserforward', 'browserhome', 'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear', 'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete', 'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20', 'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja', 'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail', 'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack', 'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn', 'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn', 'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator', 'shift', 'shiftleft', 'shiftright', 'sleep', 'stop', 'subtract', 'tab', 'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen', 'command', 'option', 'optionleft', 'optionright']


# def create_tar_archive(source_path, arcname=None):
#     """
#     Creates a tar archive in memory of the given source_path.
    
#     :param source_path: Path to file or directory on the host.
#     :param arcname: Name to use inside the archive (defaults to basename of source_path).
#     :return: BytesIO object containing the tar archive.
#     """
#     tar_stream = io.BytesIO()
#     with tarfile.open(fileobj=tar_stream, mode='w') as tar:
#         # arcname lets you control the name/path in the archive
#         tar.add(source_path, arcname=arcname or os.path.basename(source_path))
#     tar_stream.seek(0)
#     return tar_stream

def create_tar_archive(file_paths, arcname_dir=None):
    """
    Creates an in-memory tar archive from a list of file paths.

    Args:
        file_paths (list): List of full paths of files on the host.
        arcname_dir (str): Optional directory name inside the archive for the files.
                           If provided, files will be stored under this directory.
    
    Returns:
        bytes: The tar archive as a bytes object.
    """
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        for file_path in file_paths:
            # Use arcname to define the file name (or directory structure) inside the tar
            if arcname_dir:
                arcname = os.path.join(arcname_dir, os.path.basename(file_path))
            else:
                arcname = os.path.basename(file_path)
            tar.add(file_path, arcname=arcname)
    tar_stream.seek(0)
    return tar_stream.getvalue()


def send_tar_archive(container, host_file_paths, container_dest_path):
    """
    Sends a tar archive to a container.
    """
    tar_stream = create_tar_archive(host_file_paths)
    success = container.put_archive(container_dest_path, tar_stream)
    if success:
        logger.info("File(s) transferred successfully.")
    else:
        logger.info("File transfer failed.")
    return success

def get_vm_file(file_path: str, automation_port: int):
    """
    Gets a file from file_path on a VM server, and returns the file path.
    """
    url = f"http://localhost:{automation_port}/file"
    for _ in range(3):
        try:
            response = requests.post(url, data={"file_path": file_path})
            if response.status_code == 200:
                logger.info("File downloaded successfully")
                return response.content
            else:
                logger.error("Failed to get file. Status code: %d", response.status_code)
                logger.info("Retrying to get file.")
        except Exception as e:
            logger.error("An error occurred while trying to get the file: %s", e)
            logger.info("Retrying to get file.")
        time.sleep(1)

    logger.error("Failed to get file.")
    return None


def copy_file_inside_container(container, src, dst):
    """
    Copies a file from src to dst inside the container.
    
    Args:
        container: Docker container object.
        src (str): Source path inside the container.
        dst (str): Destination path inside the container.
    """

    cmd = f"cp {src} {dst}"
    
    result = container.exec_run(cmd, tty=True)
    if result.exit_code == 0:
        logger.info(f"Successfully copied {src} to {dst}")
    else:
        error_message = result.output.decode('utf-8')
        logger.error(f"Failed to copy files. Exit code: {result.exit_code}. Error: {error_message}")

def copy_files_inside_container(container, src, dst):
    """
    Copies files from src to dst inside the container.
    
    Args:
        container: Docker container object.
        src (str): Source path inside the container.
        dst (str): Destination path inside the container.
    """
    # Build the command; the -r flag allows copying directories recursively.
    cmd = f"cp -r {src} {dst}"
    # logger.info(f"Executing inside container: {cmd}")
    
    result = container.exec_run(cmd, tty=True)
    if result.exit_code == 0:
        logger.info(f"Successfully copied {src} to {dst}")
    else:
        error_message = result.output.decode('utf-8')
        logger.error(f"Failed to copy files. Exit code: {result.exit_code}. Error: {error_message}")



def make_dir(container, dir_path):
    """
    Creates a directory inside the container.
    """
    cmd = f"mkdir -p {dir_path}"
    logger.info(f"Executing inside container: {cmd}")
    result = container.exec_run(cmd, tty=True)
    if result.exit_code == 0:
        logger.info(f"Successfully created directory: {dir_path}")
    else:
        error_message = result.output.decode('utf-8')
        logger.error(f"Failed to create directory. Exit code: {result.exit_code}. Error: {error_message}")
    

def open_libreoffice_impress(container, file_path):
    """
    Opens a LibreOffice Impress file inside the container by executing
    the appropriate command.
    
    Args:
        container: The Docker container object.
        file_path (str): The absolute path to the Impress file inside the container.
    """
    # Build the command to open the file with LibreOffice Impress
    command = f"libreoffice --impress {file_path}"
    logger.info(f"Executing command inside container: {command}")
    
    # Run the command in detached mode so it doesn't block further operations.
    # tty=True ensures we allocate a pseudo-TTY, which is sometimes necessary
    # for GUI applications.
    exec_result = container.exec_run(command, detach=True, tty=True)
    if exec_result.exit_code == 0:
        logger.info("Command executed successfully.")
    else:
        logger.error(f"Failed to execute command. Exit code: {exec_result.exit_code}")

def change_libreoffice_permissions(container, USER="default"):
    """
    Changes the permissions of the LibreOffice file to allow execution.
    """
    cmd = f"sudo chown -R {USER}:{USER} /home/{USER}/.config/libreoffice/4"
    result = container.exec_run(cmd, tty=True)
    if result.exit_code == 0:
        logger.info("Command executed successfully.")
    else:
        logger.error(f"Failed to execute command {cmd}. Exit code: {result.exit_code}")
    cmd = f"chmod -R u+rwX /home/{USER}/.config/libreoffice/4"
    result = container.exec_run(cmd, tty=True)
    if result.exit_code == 0:
        logger.info("Command executed successfully.")
    else:
        logger.error(f"Failed to execute command {cmd}. Exit code: {result.exit_code}")

def change_directory_group_permissions(directory_path, automation_port=30901, USER="default"):
    """
    Changes the group permissions of the directory to allow execution.
    """
    command = ["sudo", "chown", "-R", f"{USER}:{USER}", f"{directory_path}"]
    payload = json.dumps({"command": command, "shell": False})
    response = requests.post(f"http://localhost:{automation_port}/execute", headers={'Content-Type': 'application/json'}, data=payload)
    if response.status_code != 200:
        logger.error(f"Failed to execute command {command}. Exit code: {response.status_code}")
    time.sleep(1)
    command = ["sudo", "chmod", "777", "-R", f"{directory_path}"]
    payload = json.dumps({"command": command, "shell": False})
    response = requests.post(f"http://localhost:{automation_port}/execute", headers={'Content-Type': 'application/json'}, data=payload)
    if response.status_code != 200:
        logger.error(f"Failed to execute command {command}. Exit code: {response.status_code}")



def open_libreoffice_file(container, file_path):
    """
    Opens a LibreOffice file inside the container by executing
    the appropriate command.
    """
    # check the file extension
    file_extension = os.path.splitext(file_path)[1]
    if file_extension == ".odp":
        command = f"libreoffice --impress {file_path}"
    elif file_extension == ".odt":
        command = f"libreoffice --writer {file_path}"
    elif file_extension == ".ods":
        command = f"libreoffice --calc {file_path}"
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    print(f"Opening LibreOffice file: {file_path}")
    print(f"Command: {command}")

    # Run the command in detached mode so it doesn't block further operations.
    # tty=True ensures we allocate a pseudo-TTY, which is sometimes necessary
    # for GUI applications.
    exec_result = container.exec_run(command, detach=True, tty=True)
    if exec_result.exit_code == 0:
        logger.info("Command executed successfully.")
    else:
        logger.error(f"Failed to open LibreOffice file. Exit code: {exec_result.exit_code}")


def open_libreoffice_file_via_pyautogui(file_path, automation_port):
    """
    Opens a LibreOffice file inside the container by sending pyautogui commands.
    """
    # check the file extension
    file_extension = os.path.splitext(file_path)[1]
    if file_extension == ".odp":
        command = f"libreoffice --impress {file_path}"
    elif file_extension == ".odt":
        command = f"libreoffice --writer {file_path}"
    elif file_extension == ".ods":
        command = f"libreoffice --calc {file_path}"
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    url = f"http://localhost:{automation_port}/run_commands"
    headers = {"Content-Type": "application/json"}
    payload = {
        "commands": [
            {"action": "hotkey", "keys": ["ctrl", "alt", "t"]},  # Open terminal
            {"action": "sleep", "seconds": 1},
            {"action": "typewrite", "message": f"{command}"},
            {"action": "press", "key": "enter"}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response == 200:
        logger.info(f"Successfully opened {file_path}.")
    else:
        logger.error(f"Failed to open LibreOffice file. Exit code: {response}")

def open_libreoffice_file_via_system(file_path, automation_port):
    """
    Opens a LibreOffice file inside the container by executing
    the appropriate command.
    """
    data = {"file_path": file_path}
    response = requests.post(f"http://localhost:{automation_port}/open_libreoffice", json=data)
    if response.status_code != 200:
        logger.error(f"Failed to open LibreOffice file. Exit code: {response}")

def activate_libreoffice_window(file_name, automation_port):
    """
    Activates the LibreOffice window inside the container.
    """
    url = f"http://localhost:{automation_port}/execute"
    command_list = ["wmctrl", "-a", f"{file_name} - LibreOffice Impress"]
    headers = {"Content-Type": "application/json"}
    payload = {"command": command_list, "shell": False}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            logger.error(f"Failed to activate LibreOffice window. Exit code: {response}")
    except Exception as e:
        logger.error(f"Failed to activate LibreOffice window. Error: {e}")


def close_active_window(automation_port):
    """
    Closes the active window inside the container.
    """
    url = f"http://localhost:{automation_port}/run_commands"
    headers = {"Content-Type": "application/json"}
    payload = {"commands": [{"action": "hotkey", "keys": ["alt", "f4"]}]}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        logger.error(f"Failed to close active window. Exit code: {response.status_code}")

# def close_all_windows(automation_port, num_iter=5):
#     """
#     Closes all windows inside the container.
#     """
#     url = f"http://localhost:{automation_port}/run_commands"
#     headers = {"Content-Type": "application/json"}
#     payload = {"commands": [{"action": "hotkey", "keys": ["alt", "f4"]}]}
#     for _ in range(num_iter):
#         response = requests.post(url, headers=headers, data=json.dumps(payload))
#         if response.status_code != 200:
#             logger.error(f"Failed to close active window. Exit code: {response}")
#         time.sleep(1)


def close_all_windows(automation_port, max_attempts=10):
    """
    Closes all windows inside the container, handling save dialogs.
    """
    url = f"http://localhost:{automation_port}/run_commands"
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_attempts):
        # First try Alt+F4 to close window
        close_payload = {"commands": [
            {"action": "hotkey", "keys": ["alt", "f4"]},
            {"action": "sleep", "seconds": 2}  # Wait for dialog if it appears
        ]}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(close_payload))
            if response.status_code != 200:
                logger.error(f"Failed close attempt {attempt+1}. Exit code: {response.status_code}")
                continue
                
            # Check for dialog and handle it - press "Save" (usually the left button)
            # First check if there's a dialog by looking for typical dialog text
            if is_libreoffice_running(url=f"http://localhost:{automation_port}/execute", shell=False):
                save_dialog_payload = {"commands": [{"action": "press", "key": "enter"}]}
                response = requests.post(url, headers=headers, data=json.dumps(save_dialog_payload))
                time.sleep(2)

            # for the additional dialog, press Enter again (e.g., selecting "Saving anymore").
            if is_libreoffice_running(url=f"http://localhost:{automation_port}/execute", shell=False):
                enter_payload = {"commands": [{"action": "press", "key": "enter"}]}
                response = requests.post(url, headers=headers, data=json.dumps(enter_payload))
                time.sleep(1)
            
            # Check if LibreOffice is still running (return True if LibreOffice is running)
            if not is_libreoffice_running(url=f"http://localhost:{automation_port}/execute", shell=False):
                logger.info("Successfully closed all LibreOffice windows")
                return True
            else:
                continue
                
        except Exception as e:
            logger.error(f"Error during close attempt {attempt+1}: {e}")
    
    if is_libreoffice_running(url=f"http://localhost:{automation_port}/execute", shell=False):
        # click somewhere
        url = f"http://localhost:{automation_port}/execute"
        command = f"pyautogui.click(x=500, y=500)"
        pkgs_prefix: str = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
        command_list = ["python3", "-c", pkgs_prefix.format(command=command)]
        payload = json.dumps({"command": command_list, "shell": False})
        response = requests.post(url, headers=headers, data=payload)
        time.sleep(1)
        # type ESC to close the window
        command = f"pyautogui.press('esc')"
        pkgs_prefix: str = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
        command_list = ["python3", "-c", pkgs_prefix.format(command=command)]
        payload = json.dumps({"command": command_list, "shell": False})
        response = requests.post(url, headers=headers, data=payload)
        time.sleep(1)
        # close the window
        url = f"http://localhost:{automation_port}/run_commands"
        headers = {"Content-Type": "application/json"}
        payload = {"commands": [{"action": "hotkey", "keys": ["alt", "f4"]}]}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        time.sleep(1)

    if not is_libreoffice_running(url=f"http://localhost:{automation_port}/execute", shell=False):
        logger.info("Successfully closed all LibreOffice windows")
        return True
    
    logger.error("Failed to close LibreOffice after maximum attempts")
    return False


# ## not recommended to use this function, as it causes the crash of the LibreOffice application. Use close_active_window instead.
# def close_libreoffice_windows(automation_port):
#     """
#     Closes all LibreOffice windows inside the container.
#     """
#     url = f"http://localhost:{automation_port}/close_libreoffice_window"
#     response = requests.post(url)
#     if response.status_code != 200:
#         logger.error(f"Failed to close LibreOffice window. Exit code: {response}")


# def save_current_file(file_name, automation_port):
#     """
#     Saves the current file inside the container.
#     """
#     # activate the window
#     activate_libreoffice_window(file_name, automation_port)
#     time.sleep(0.5)
#     # save the file
#     url = f"http://localhost:{automation_port}/run_commands"
#     headers = {"Content-Type": "application/json"}
#     payload = {"commands": [{"action": "hotkey", "keys": ["ctrl", "s"]}]}
#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(payload))
#         if response.status_code != 200:
#             logger.error(f"Failed to save current file. Exit code: {response.status_code}")
#     except Exception as e:
#         logger.error(f"Failed to save current file. Error: {e}")

def save_current_file(file_name, automation_port):
    """
    Saves the current file inside the container with improved robustness.
    """
    # Activate the window
    activate_libreoffice_window(file_name, automation_port)
    time.sleep(1)  # Give more time for window to activate
    
    url = f"http://localhost:{automation_port}/run_commands"
    headers = {"Content-Type": "application/json"}
    
    # click the window to close unneeded dialogs such as right-clicking a plot
    url = f"http://localhost:{automation_port}/execute"
    command = f"pyautogui.click(x=960, y=50)"
    pkgs_prefix: str = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
    command_list = ["python3", "-c", pkgs_prefix.format(command=command)]
    payload = json.dumps({"command": command_list, "shell": False})
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        logger.error(f"Failed to close unneeded dialogs. Exit code: {response.status_code}")
    
    time.sleep(1)
    url = f"http://localhost:{automation_port}/run_commands"
    headers = {"Content-Type": "application/json"}
    # First attempt Ctrl+S to save
    save_payload = {"commands": [
        {"action": "hotkey", "keys": ["ctrl", "s"]},
        {"action": "sleep", "seconds": 2}  # Wait for save dialog if it appears
    ]}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(save_payload))
        if response.status_code != 200:
            logger.error(f"Failed to initiate save. Exit code: {response.status_code}")
            return False
        
        # # Check for "Save As" dialog and handle it if needed. Press Enter to save the file because the default save location/name is already set.
        # enter_payload = {"commands": [{"action": "press", "key": "enter"}]}
        # response = requests.post(url, headers=headers, data=json.dumps(enter_payload))
        # time.sleep(1)
        # # for the additional dialog, press Enter again (e.g., selecting "Saving anymore").
        # enter_payload = {"commands": [{"action": "press", "key": "enter"}]}
        # response = requests.post(url, headers=headers, data=json.dumps(enter_payload))
        # time.sleep(1)
        return True
        
    except Exception as e:
        logger.error(f"Failed to save current file. Error: {e}")
        return False

def is_libreoffice_running(url="http://localhost:30901/execute", shell=False):
    """
    Check if LibreOffice is running in the Ubuntu Docker container.
    
    Args:
        container_id: The Docker container ID or name
        
    Returns:
        bool: True if LibreOffice is running, False otherwise
    """
    try:
        # Execute the ps command in the container to check for soffice.bin process
        command_list = ["pgrep", "-f", "soffice.bin"]
        payload = json.dumps({"command": command_list, "shell": shell})
        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=payload, timeout=5)
        result = response.json()
        
        # If the command returns a pid, LibreOffice is running
        return result['error'] == None and result['output'] != ""
    except Exception as e:
        logger.error(f"Error checking if LibreOffice is running: {e}")
        return False

def execute_python_command(command, url="http://localhost:30901/execute", shell=False):
    """
    Executes a python command inside the container.
    """
    pkgs_prefix: str = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
    command_list = ["python3", "-c", pkgs_prefix.format(command=command)]
    payload = json.dumps({"command": command_list, "shell": shell})

    # if the command includes "pyperclip.copy", then we need to ignore the timeout error because it runs correctly but cannot receive the response from the server.
    if "pyperclip" in command:
        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=payload, timeout=5)
    
    else:
        for _ in range(3):
            try:
                response = requests.post(url, headers={'Content-Type': 'application/json'}, data=payload, timeout=90)
                if response.status_code == 200:
                    # logger.info("Command executed successfully: %s", response.text)
                    return response.json()
                else:
                    logger.error("Failed to execute command. Status code: %d", response.status_code)
                    logger.info("Retrying to execute command.")
            except requests.exceptions.ReadTimeout:
                break
            except Exception as e:
                logger.error("An error occurred while trying to execute the command: %s", e)
                logger.info("Retrying to execute command.")
            time.sleep(1)

        logger.error("Failed to execute command at execute_python_command function.")
    return None

# def execute_python_command(command, url, shell=False):
#     """
#     Executes a python command inside the container.
#     """
#     if shell:
#         # For shell=True, send a single string command
#         python_command = f"python3 -c 'import pyautogui; import time; pyautogui.FAILSAFE = False; {command}'"
#         payload = json.dumps({"command": python_command, "shell": True})
#     else:
#         # For shell=False, send a list of arguments
#         python_command = ["python3", "-c", f"import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"]
#         payload = json.dumps({"command": python_command, "shell": False})

#     for _ in range(3):
#         try:
#             response = requests.post(url+"/execute", headers={'Content-Type': 'application/json'}, data=payload, timeout=90)
#             if response.status_code == 200:
#                 logger.info("Command executed successfully: %s", response.text)
#                 return response.json()
#             else:
#                 logger.error("Failed to execute command. Status code: %d", response.status_code)
#                 logger.info("Retrying to execute command.")
#         except requests.exceptions.ReadTimeout:
#             break
#         except Exception as e:
#             logger.error("An error occurred while trying to execute the command: %s", e)
#             logger.info("Retrying to execute command.")
#         time.sleep(1)

#     logger.error("Failed to execute command.")
#     return None


def execute_action(action: Dict[str, Any], url: str = "http://localhost:30901/execute"):
    """
    Executes an action on the server computer.
    """
    
    if action in ['WAIT', 'FAIL', 'DONE']:
        return

    action_type = action["action_type"]
    parameters = action["parameters"] if "parameters" in action else {param: action[param] for param in action if param != 'action_type'}
    move_mode = random.choice(
        ["pyautogui.easeInQuad", "pyautogui.easeOutQuad", "pyautogui.easeInOutQuad", "pyautogui.easeInBounce",
            "pyautogui.easeInElastic"])
    duration = random.uniform(0.5, 1)

    if action_type == "MOVE_TO":
        if parameters == {} or None:
            execute_python_command("pyautogui.moveTo()", url)
        elif "x" in parameters and "y" in parameters:
            x = parameters["x"]
            y = parameters["y"]
            execute_python_command(f"pyautogui.moveTo({x}, {y}, {duration}, {move_mode})", url)
        else:
            raise Exception(f"Unknown parameters: {parameters}")

    elif action_type == "CLICK":
        if parameters == {} or None:
            execute_python_command("pyautogui.click()", url)
        elif "button" in parameters and "x" in parameters and "y" in parameters:
            button = parameters["button"]
            x = parameters["x"]
            y = parameters["y"]
            if "num_clicks" in parameters:
                num_clicks = parameters["num_clicks"]
                execute_python_command(
                    f"pyautogui.click(button='{button}', x={x}, y={y}, clicks={num_clicks})", url)
            else:
                execute_python_command(f"pyautogui.click(button='{button}', x={x}, y={y})", url)
        elif "button" in parameters and "x" not in parameters and "y" not in parameters:
            button = parameters["button"]
            if "num_clicks" in parameters:
                num_clicks = parameters["num_clicks"]
                execute_python_command(f"pyautogui.click(button='{button}', clicks={num_clicks})", url)
            else:
                execute_python_command(f"pyautogui.click(button='{button}')", url)
        elif "button" not in parameters and "x" in parameters and "y" in parameters:
            x = parameters["x"]
            y = parameters["y"]
            if "num_clicks" in parameters:
                num_clicks = parameters["num_clicks"]
                execute_python_command(f"pyautogui.click(x={x}, y={y}, clicks={num_clicks})", url)
            else:
                execute_python_command(f"pyautogui.click(x={x}, y={y})", url)
        else:
            raise Exception(f"Unknown parameters: {parameters}")

    elif action_type == "MOUSE_DOWN":
        if parameters == {} or None:
            execute_python_command("pyautogui.mouseDown()", url)
        elif "button" in parameters:
            button = parameters["button"]
            execute_python_command(f"pyautogui.mouseDown(button='{button}')", url)
        else:
            raise Exception(f"Unknown parameters: {parameters}")

    elif action_type == "MOUSE_UP":
        if parameters == {} or None:
            execute_python_command("pyautogui.mouseUp()", url)
        elif "button" in parameters:
            button = parameters["button"]
            execute_python_command(f"pyautogui.mouseUp(button='{button}')", url)
        else:
            raise Exception(f"Unknown parameters: {parameters}")

    elif action_type == "RIGHT_CLICK":
        if parameters == {} or None:
            execute_python_command("pyautogui.rightClick()", url)
        elif "x" in parameters and "y" in parameters:
            x = parameters["x"]
            y = parameters["y"]
            execute_python_command(f"pyautogui.rightClick(x={x}, y={y})", url)
        else:
            raise Exception(f"Unknown parameters: {parameters}")

    elif action_type == "DOUBLE_CLICK":
        if parameters == {} or None:
            execute_python_command("pyautogui.doubleClick()", url)
        elif "x" in parameters and "y" in parameters:
            x = parameters["x"]
            y = parameters["y"]
            execute_python_command(f"pyautogui.doubleClick(x={x}, y={y})", url)
        else:
            raise Exception(f"Unknown parameters: {parameters}")

    elif action_type == "DRAG_TO":
        if "x" in parameters and "y" in parameters:
            x = parameters["x"]
            y = parameters["y"]
            execute_python_command(
                f"pyautogui.dragTo({x}, {y}, duration=1.0, button='left', mouseDownUp=True)", url)

    elif action_type == "SCROLL":
        # todo: check if it is related to the operating system, as https://github.com/TheDuckAI/DuckTrack/blob/main/ducktrack/playback.py pointed out
        if "dx" in parameters and "dy" in parameters:
            dx = parameters["dx"]
            dy = parameters["dy"]
            execute_python_command(f"pyautogui.hscroll({dx})", url)
            execute_python_command(f"pyautogui.vscroll({dy})", url)
        elif "dx" in parameters and "dy" not in parameters:
            dx = parameters["dx"]
            execute_python_command(f"pyautogui.hscroll({dx})", url)
        elif "dx" not in parameters and "dy" in parameters:
            dy = parameters["dy"]
            execute_python_command(f"pyautogui.vscroll({dy})", url)
        else:
            raise Exception(f"Unknown parameters: {parameters}")

    elif action_type == "TYPING":
        if "text" not in parameters:
            raise Exception(f"Unknown parameters: {parameters}")
        # deal with special ' and \ characters
        # text = parameters["text"].replace("\\", "\\\\").replace("'", "\\'")
        # self.execute_python_command(f"pyautogui.typewrite('{text}')")
        text = parameters["text"]
        execute_python_command("pyautogui.typewrite({:})".format(repr(text)), url)

    elif action_type == "PRESS":
        if "key" not in parameters:
            raise Exception(f"Unknown parameters: {parameters}")
        key = parameters["key"]
        if key.lower() not in KEYBOARD_KEYS:
            raise Exception(f"Key must be one of {KEYBOARD_KEYS}")
        execute_python_command(f"pyautogui.press('{key}')", url)

    elif action_type == "KEY_DOWN":
        if "key" not in parameters:
            raise Exception(f"Unknown parameters: {parameters}")
        key = parameters["key"]
        if key.lower() not in KEYBOARD_KEYS:
            raise Exception(f"Key must be one of {KEYBOARD_KEYS}")
        execute_python_command(f"pyautogui.keyDown('{key}')", url)

    elif action_type == "KEY_UP":
        if "key" not in parameters:
            raise Exception(f"Unknown parameters: {parameters}")
        key = parameters["key"]
        if key.lower() not in KEYBOARD_KEYS:
            raise Exception(f"Key must be one of {KEYBOARD_KEYS}")
        execute_python_command(f"pyautogui.keyUp('{key}')", url)

    elif action_type == "HOTKEY":
        if "keys" not in parameters:
            raise Exception(f"Unknown parameters: {parameters}")
        keys = parameters["keys"]
        if not isinstance(keys, list):
            raise Exception("Keys must be a list of keys")
        for key in keys:
            if key.lower() not in KEYBOARD_KEYS:
                raise Exception(f"Key must be one of {KEYBOARD_KEYS}")

        keys_para_rep = "', '".join(keys)
        execute_python_command(f"pyautogui.hotkey('{keys_para_rep}')", url)

    elif action_type in ['WAIT', 'FAIL', 'DONE']:
        pass

    else:
        raise Exception(f"Unknown action type: {action_type}")


if __name__ == "__main__":
    # Create a Docker client
    client = docker.from_env()
    # Retrieve your running container (replace 'your_container_id_or_name' with your container's id or name)
    container = client.containers.get("ubuntu2004-gnome-server")
    # Specify the file or directory on the host that you want to send
    host_path = "path/to/your/file_or_directory"
    # Specify the destination path inside the container where the files will be placed.
    # Make sure this destination path exists or the archive will be extracted relative to container's root.
    container_dest_path = "/destination/path/in/container"
    # Create the tar archive in memory
    tar_stream = create_tar_archive(host_path)
    # Transfer the tar archive to the container
    success = container.put_archive(container_dest_path, tar_stream.getvalue())
    if success:
        print("File(s) transferred successfully.")
    else:
        print("File transfer failed.")
