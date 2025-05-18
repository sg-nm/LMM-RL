"""
UI-TARS agent

"""
import ast
import numpy as np
from PIL import Image
from io import BytesIO
import math
import re
from typing import Dict, List
import base64
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from openai import OpenAI
import backoff
import openai
from google.api_core.exceptions import (
    BadRequest,
    InternalServerError,
    InvalidArgument,
    ResourceExhausted,
)
from requests.exceptions import SSLError

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

## original action space
# UITARS_ACTION_SPACE = """
# click(start_box='[x1, y1, x2, y2]')
# left_double(start_box='[x1, y1, x2, y2]')
# right_single(start_box='[x1, y1, x2, y2]')
# drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')
# hotkey(key='')
# type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
# scroll(start_box='[x1, y1, x2, y2]', direction='down or up or right or left')
# wait() #Sleep for 5s and take a screenshot to check for any changes.
# finished()
# call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
# """

UITARS_ACTION_SPACE = """
click(start_box='[x1, y1, x2, y2]')
left_double(start_box='[x1, y1, x2, y2]')
right_single(start_box='[x1, y1, x2, y2]')
drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='[x1, y1, x2, y2]', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
"""


UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

UITARS_USR_PROMPT_PLAN = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Plan: ...
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Must use {language} in `Plan`, `Thought`, and `Action` part.
- Write a small plan for actions to complete the task in `Plan` part.
- Then summarize your next action (with its target element) based on the current state in one sentence in `Thought` part.
- Finally, output your next action in `Action` part.

## User Instruction
{instruction}
"""

UITARS_USR_PROMPT_ACTION_REFERENCE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Refer to the following `Action Guide` section to generate your next action.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part based on the `Action Guide`.

## Action Guide
{action_reference}

## User Instruction
Based on the above `Action Guide`, perform the following task: {instruction}
"""

ACTOR_PROMPT = """Your task is to complete ## User Instruction below.
Before predicting the next action, you should first think about an overall plan for actions to complete the task and itemize the plan in `Plan` part. You can change the plan if you think your previous plan is not suitable.
Then, think about what to do next based on the plan and the current state in `Thought` part.
Finally, output next action in `Action` part.
If ## Action Guides are provided below, you should refer to them when generating your plan, thoughts, and next action.

## Output Format
```
Plan: <itemized action plans to complete the task>
Thought: <thought about what to do next based on the plan and the current state>
Action: <next action>
```

## Output Example
```
Plan: 1: go to **Slide** on Menu bar. 2: Click **New Slide**.
Thought: I need to select the "Proposed method" slide from the slide thumbnail panel on the left. This is the first step in inserting a new slide.
Action: click(start_box='(105,518)')
```

## Action Guides
{action_reference}

## User Instruction (task)
{instruction}
"""


class UITARSAgent:
    def __init__(
        self,
        max_tokens=1500,
        top_p=0.9,
        top_k=1.0,
        temperature=1.0,
        action_space="pyautogui",
        observation_type="screenshot",
        max_trajectory_length=5,
        a11y_tree_max_tokens=10000,
        infer_mode="qwen2vl_user",
        prompt_style="qwen2vl_user",
        input_swap=False,
        language="English",
        max_steps=15,
        history_n=5,
        screen_height=1080,
        screen_width=1920
    ):
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.vlm = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="empty",
        )
        self.infer_mode = infer_mode
        self.prompt_style = prompt_style
        self.input_swap = input_swap
        self.language = language
        self.max_steps = max_steps
        self.history_n = history_n
        self.screen_height = screen_height
        self.screen_width = screen_width

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []

        self.prompt_action_space = UITARS_ACTION_SPACE
        self.customize_action_parser = parse_action_qwen2vl
        self.action_parse_res_factor = 1000

        self.prompt_template = UITARS_USR_PROMPT_THOUGHT



    def predict(self, obs: Dict, last_action_after_obs: Dict = None) -> List:
        """
        Predict the next action based on the observation.

        args:
            obs: Dict = {"screenshot": self._get_screenshot(), "instruction": self.instruction}
        """

        # Append trajectory
        # print(len(self.observations), len(self.actions), len(self.actions))
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(
            self.thoughts
        ), "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length :]
                _actions = self.actions[-self.max_trajectory_length :]
                _thoughts = self.thoughts[-self.max_trajectory_length :]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts


        if last_action_after_obs is not None and self.infer_mode == "double_image":
            self.history_images.append(last_action_after_obs["screenshot"])

        self.history_images.append(obs["screenshot"])
        base64_image = obs["screenshot"]
        self.observations.append({"screenshot": base64_image, "accessibility_tree": None})
        
        
        if self.infer_mode == "qwen2vl_user":
            user_prompt = self.prompt_template.format(
                instruction=obs["instruction"],
                action_space=self.prompt_action_space,
                language=self.language
            )
        elif self.infer_mode == "qwen2vl_no_thought":
            user_prompt = self.prompt_template.format(
                instruction=obs["instruction"]
            )

        # print(f"user_prompt: {user_prompt}")
        
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]
        if len(self.history_responses) >= self.max_trajectory_length:
            self.history_responses = self.history_responses[-self.max_trajectory_length:]

        max_pixels = 1350 * 28 * 28
        min_pixels = 100 * 28 * 28
        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")
        max_image_nums_under_32k = int(32768*0.75/max_pixels*28*28)
        if len(self.history_images) > max_image_nums_under_32k:
            num_of_images = min(5, len(self.history_images))
            max_pixels = int(32768*0.75) // num_of_images

        for turn, image in enumerate(self.history_images):
            if len(images) >= 5:
                break
            try:
                image = Image.open(BytesIO(image))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            if image.width * image.height > max_pixels:
                resize_factor = math.sqrt(max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                image = image.resize((width, height))
            if image.width * image.height < min_pixels:
                resize_factor = math.sqrt(min_pixels / (image.width * image.height))
                width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
                image = image.resize((width, height))

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):
                    cur_image = images[image_num]
                    encoded_string = pil_to_base64(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                    })
                    image_num += 1
                    
                messages.append({
                    "role": "assistant",
                    "content": history_response
                    # "content": [history_response]
                })

            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1
        
        else:
            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
            })
            image_num += 1

        try_times = 3
        while True:
            if try_times <= 0:
                print(f"Reach max retry times to fetch response from client, as error flag.")
                return "client error", ["DONE"]
            try:
                response = self.vlm.chat.completions.create(
                    model="ui-tars",
                    messages=messages,
                    frequency_penalty=1,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                prediction = response.choices[0].message.content.strip()
                parsed_responses = self.customize_action_parser(
                    prediction,
                    self.action_parse_res_factor,
                    self.screen_height,
                    self.screen_width
                )
                break
            except Exception as e:
                print(f"Error when fetching response from client, with response.")
                prediction = None
                try_times -= 1
                
        if prediction is None:
            return "client error", ["DONE"]

        
        self.history_responses.append(prediction)
        self.thoughts.append(prediction)

        try:
            parsed_responses = self.customize_action_parser(
                prediction,
                self.action_parse_res_factor,
                self.screen_height,
                self.screen_width
            )
        except Exception as e:
            print(f"Parsing action error: {prediction}, with error:\n{e}")
            return f"Parsing action error: {prediction}, with error:\n{e}", ["DONE"]

        actions = []
        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:
                if parsed_response["action_type"] == FINISH_WORD:
                    self.actions.append(actions)
                    return prediction, ["DONE"]
                
                elif parsed_response["action_type"] == WAIT_WORD:
                    self.actions.append(actions)
                    return prediction, ["WAIT"]
                
                elif parsed_response["action_type"] == ENV_FAIL_WORD:
                    self.actions.append(actions)
                    return prediction, ["FAIL"]

                elif parsed_response["action_type"] == CALL_USER:
                    self.actions.append(actions)
                    return prediction, ["FAIL"]

            
            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_response,
                self.screen_height,
                self.screen_width,
                self.input_swap
            )
            actions.append(pyautogui_code)

        self.actions.append(actions)

        if len(self.history_responses) >= self.max_steps:
            # Default to FAIL if exceed max steps
            actions = ["FAIL"]

        return prediction, actions
    
    
    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
            # General exceptions
            SSLError,
            # OpenAI exceptions
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            # Google exceptions
            InvalidArgument,
            ResourceExhausted,
            InternalServerError,
            BadRequest,
            # Groq exceptions
            # todo: check
        ),
        interval=30,
        max_tries=10,
    )
    
    def reset(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []



# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None
    
def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def parse_action_qwen2vl(text, factor, image_height=1080, image_width=1920):
    text = text.strip()
    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            continue
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                float_numbers = [float(num) / factor for num in numbers]
                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions



def normalize_action_str(action_str):
    """
    Normalize action string to handle both regular and escaped quotes.
    """
    # Replace escaped quotes with regular quotes for Python AST parsing
    action_str = action_str.replace("\\'", "'")
    # Replace double quotes with single quotes for consistency
    action_str = action_str.replace('"', "'")
    return action_str


def extract_actions(text):
    """
    Extract actions from text containing "Action:" marker.
    """
    # Check if text contains the Action marker
    if "Action:" not in text:
        return []
        
    # Extract action text after "Action:" marker
    action_str = text.split("Action:")[-1]
    
    # Split into separate actions (by double newlines)
    tmp_all_action = action_str.split("\n\n")
    all_action = []
    
    for action_str in tmp_all_action:
        # Clean up the action string
        action_str = action_str.strip()
        if not action_str:
            continue
            
        # Special handling for type(content='...') actions
        if "type(content" in action_str:
            # Use regex to extract content regardless of quote style
            content_match = re.search(r"type\(content=(?:'|\\'|\")(.*?)(?:'|\\'|\")\)", action_str, re.DOTALL)
            if content_match:
                content = content_match.group(1)
                # Create a properly formatted action string
                action_str = f"type(content='{content}')"
        
        all_action.append(action_str)
    
    return all_action

def parse_action_qwen2vl_v2(text, factor, image_height=1080, image_width=1920):
    text = text.strip()
    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text

    # Extract action strings
    action_strings = extract_actions(text)
    
    # Process each action
    actions = []
    for action_str in action_strings:
        # Normalize the action string for parsing
        normalized_action = normalize_action_str(action_str)
        
        # Replace newlines with escaped newlines
        normalized_action = normalized_action.replace("\n", "\\n")
        
        # Parse the action
        action_instance = parse_action(normalized_action)
        
        # Skip if parsing failed
        if action_instance is None:
            print(f"Action can't parse: {action_str}")
            continue
        
        action_type = action_instance["function"]
        params = action_instance["args"]
        
        # Process action inputs
        action_inputs = {}
        for param_name, param in params.items():
            if param is None or param == "":
                continue
                
            # Convert param to string if it's not already
            param = str(param).strip()
            action_inputs[param_name.strip()] = param
            
            # Handle box coordinates
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # If already in list format, leave as is
                if ori_box.startswith('[') and ori_box.endswith(']'):
                    action_inputs[param_name.strip()] = ori_box
                else:
                    # Handle parentheses format
                    numbers = ori_box.replace("(", "").replace(")", "").split(",")
                    factor = 1000  # Scaling factor from original code
                    
                    try:
                        float_numbers = [float(num.strip()) / factor for num in numbers]
                        
                        # If only two coordinates provided, duplicate them
                        if len(float_numbers) == 2:
                            float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                            
                        action_inputs[param_name.strip()] = str(float_numbers)
                    except ValueError:
                        # If conversion fails, keep original value
                        print(f"Warning: Could not convert coordinates in {ori_box}")
        
        # Create action entry
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": action_str
        })
    return actions


# def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=False) -> str:
#     '''
#     将M模型的输出解析为OSWorld中的action, 生成pyautogui代码字符串
#     参数:
#         response: 包含模型输出的字典，结构类似于：
#         {
#             "action_type": "hotkey",
#             "action_inputs": {
#                 "hotkey": "v ctrl",
#                 "start_box": None,
#                 "end_box": None
#             }
#         }
#     返回:
#         生成的pyautogui代码字符串
#     '''

#     pyautogui_code = f"import pyautogui\nimport time\n"
#     if isinstance(responses, dict):
#         responses = [responses]
#     for response_id, response in enumerate(responses):
#         if "observation" in response:
#             observation = response["observation"]
#         else:
#             observation = ""

#         if "thought" in response:
#             thought = response["thought"]
#         else:
#             thought = ""
        
#         if response_id == 0:
#             pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
#         else:
#             pyautogui_code += f"\ntime.sleep(3)\n"

#         action_dict = response
#         action_type = action_dict.get("action_type")
#         action_inputs = action_dict.get("action_inputs", {})
        
#         if action_type == "hotkey":
#             # Parsing hotkey action
#             if "key" in action_inputs:
#                 hotkey = action_inputs.get("key", "")
#             else:
#                 hotkey = action_inputs.get("hotkey", "")

#             if hotkey:
#                 # Handle other hotkeys
#                 keys = hotkey.split()  # Split the keys by space
#                 pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in keys])})"
        
#         elif action_type == "type":
#             # Parsing typing action using clipboard
#             content = action_inputs.get("content", "")
#             content = escape_single_quotes(content)
#             if content:
#                 if input_swap:
#                     pyautogui_code += f"\nimport pyperclip"
#                     pyautogui_code += f"\npyperclip.copy('{content.strip()}')"
#                     pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
#                     pyautogui_code += f"\ntime.sleep(0.5)\n"
#                     if content.endswith("\n") or content.endswith("\\n"):
#                         pyautogui_code += f"\npyautogui.press('enter')"
#                 else:
#                     pyautogui_code += f"\npyautogui.write('{content.strip()}', interval=0.1)"
#                     pyautogui_code += f"\ntime.sleep(0.5)\n"
#                     if content.endswith("\n") or content.endswith("\\n"):
#                         pyautogui_code += f"\npyautogui.press('enter')"

        
#         elif action_type in ["drag", "select"]:
#             # Parsing drag or select action based on start and end_boxes
#             start_box = action_inputs.get("start_box")
#             end_box = action_inputs.get("end_box")
#             if start_box and end_box:
#                 x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
#                 sx = round(float((x1 + x2) / 2) * image_width, 3)
#                 sy = round(float((y1 + y2) / 2) * image_height, 3)
#                 x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
#                 ex = round(float((x1 + x2) / 2) * image_width, 3)
#                 ey = round(float((y1 + y2) / 2) * image_height, 3)
#                 pyautogui_code += (
#                     f"\npyautogui.moveTo({sx}, {sy})\n"
#                     f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
#                 )

#         elif action_type == "scroll":
#             # Parsing scroll action
#             start_box = action_inputs.get("start_box")
#             if start_box:
#                 x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
#                 x = round(float((x1 + x2) / 2) * image_width, 3)
#                 y = round(float((y1 + y2) / 2) * image_height, 3)
                
#                 # # 先点对应区域，再滚动
#                 # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
#             else:
#                 x = None
#                 y = None
#             direction = action_inputs.get("direction", "")
            
#             if x == None:
#                 if "up" in direction.lower():
#                     pyautogui_code += f"\npyautogui.scroll(5)"
#                 elif "down" in direction.lower():
#                     pyautogui_code += f"\npyautogui.scroll(-5)"
#             else:
#                 if "up" in direction.lower():
#                     pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
#                 elif "down" in direction.lower():
#                     pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

#         elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
#             # Parsing mouse click actions
#             start_box = action_inputs.get("start_box")
#             start_box = str(start_box)
#             if start_box:
#                 start_box = eval(start_box)
#                 if len(start_box) == 4:
#                     x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
#                 elif len(start_box) == 2:
#                     x1, y1 = start_box
#                     x2 = x1
#                     y2 = y1
#                 x = round(float((x1 + x2) / 2) * image_width, 3)
#                 y = round(float((y1 + y2) / 2) * image_height, 3)
#                 if action_type == "left_single" or action_type == "click":
#                     pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
#                 elif action_type == "left_double":
#                     pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
#                 elif action_type == "right_single":
#                     pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
#                 elif action_type == "hover":
#                     pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
#         elif action_type in ["finished"]:
#             pyautogui_code = f"DONE"
        
#         else:
#             pyautogui_code += f"\n# Unrecognized action type: {action_type}"

#     return pyautogui_code






def parse_action_qwen2vl_plan(text, factor, image_height=1080, image_width=1920):
    text = text.strip()
    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    
    if text.startswith("Plan:"):
        plan_pattern = r"Plan: (.+?)(?=\s*Action:|$)"
        plan_hint = "Plan: "
    elif text.startswith("Action_Plan:"):
        plan_pattern = r"Action_Plan: (.+?)(?=\s*Action:|$)"
        plan_hint = "Action_Plan: "
    else:
        plan_pattern = r"Plan: (.+?)(?=\s*Action:|$)"
        plan_hint = "Plan: "
    
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    
    plan = None
    plan_match = re.search(plan_pattern, text, re.DOTALL)
    if plan_match:
        if len(plan_match.groups()) == 1:
            plan = plan_match.group(1).strip()
        elif len(plan_match.groups()) == 2:
            plan = plan_match.group(2).strip()

    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            continue
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                float_numbers = [float(num) / factor for num in numbers]
                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "plan": plan,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions


def parsing_response_to_pyautogui_code_plan(responses, image_height: int, image_width:int, input_swap:bool=False) -> str:
    '''
    将M模型的输出解析为OSWorld中的action, 生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""

        if "plan" in response:
            plan = response["plan"]
        else:
            plan = ""

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nPlan:\n{plan}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(3)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in keys])})"
        
        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{content.strip()}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{content.strip()}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        
        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


def parse_action_to_structure_output(text, factor, origin_resized_height, origin_resized_width, model_type="qwen25vl", max_pixels=16384*28*28, min_pixels=100*28*28):
    text = text.strip()
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(origin_resized_height, origin_resized_width, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)

    # # 正则表达式匹配 Action 字符串
    # if text.startswith("Thought:"):
    #     thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    #     thought_hint = "Thought: "
    # elif text.startswith("Reflection:"):
    #     thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
    #     thought_hint = "Reflection: "
    # elif text.startswith("Action_Summary:"):
    #     thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
    #     thought_hint = "Action_Summary: "
    # else:
    #     thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    #     thought_hint = "Thought: "
    
    reflection, thought = None, None
    thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    
    plan_pattern = r"Plan: (.+?)(?=\s*Thought:|$)"
    plan_match = re.search(plan_pattern, text, re.DOTALL)

    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    
    if plan_match:
        if len(plan_match.groups()) == 1:
            plan = plan_match.group(1).strip()
        elif len(plan_match.groups()) == 2:
            plan = plan_match.group(2).strip()
            thought = plan_match.group(1).strip()
    
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}") 
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num/smart_resize_height))
                        else:
                            float_numbers.append(float(num/smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)
                # print(f"float_numbers: {float_numbers}")

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions


def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=False) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""
        
        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
        
        elif action_type == "press":
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"
            
            elif hotkey == "space":
                hotkey = " "
                
            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
            
        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
        
        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        
        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code



## for Grounding
def build_prompt_for_grounding(current_obs: Dict, previous_images = None, previous_actions: List[str] = None, action_reference: bool = False, max_pixels: int = 1350 * 28 * 28, min_pixels: int = 100 * 28 * 28) -> str:
    """
    Build a prompt with single-turn format, containing previous observations and actions.
    This is effective for Qwen-based models to refer to previous observations and actions.

    args:
        instruction: str, user instruction
        previous_images: List[PIL.Image.Image], previous images
        previous_actions: List[str], previous actions
    """
    # assert len(previous_images) == len(previous_actions)
    contents = []
    if previous_images is not None and previous_actions is not None:
        for idx, (previous_image, previous_action) in enumerate(zip(previous_images, previous_actions)):
            prefix_image_tokens = {"type": "text", "text": f"Previous screenshot ({len(previous_images) - idx} steps ago):"}
            contents.append(prefix_image_tokens)
            image_tokens = {"type": "image", "image": previous_image}
            contents.append(image_tokens)
            prefix_action_tokens = {"type": "text", "text": f"Previous action ({len(previous_images) - idx} steps ago): {previous_action}"}
            contents.append(prefix_action_tokens)
    
    if action_reference:
        if isinstance(current_obs["action_reference"], str):
            action_reference = current_obs["action_reference"]
        else:
            action_reference = "\n".join(current_obs["action_reference"])
        prompt = UITARS_USR_PROMPT_ACTION_REFERENCE.format(instruction=current_obs["instruction"], action_space=UITARS_ACTION_SPACE, action_reference=action_reference, language='English')
    else:
        prompt = UITARS_USR_PROMPT_PLAN.format(instruction=current_obs["instruction"], action_space=UITARS_ACTION_SPACE, language='English')
    
    if current_obs["screenshot"] is not None:
        current_prompt = {"type": "text", "text": prompt + "\n\n" + "Current screenshot:"}
        contents.append(current_prompt)
        image = Image.open(BytesIO(current_obs["screenshot"]))
        if image.width * image.height > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        if image.width * image.height < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
            image = image.resize((width, height))
        if image.mode != "RGB":
            image = image.convert("RGB")
        contents.append({"type": "image", "image": image})
        # contents.append({"type": "image", "image": Image.open(BytesIO(current_obs["screenshot"]))})
    else:
        current_prompt = {"type": "text", "text": prompt + "\n\n" + "Current screenshot is not available."}
        contents.append(current_prompt)
    
    message = [
        {"role": "user",
         "content": contents,
        },
    ]
    return message


def build_prompt_for_actor(
        current_obs: Dict, 
        previous_images = None, 
        previous_actions: List[str] = None, 
        action_reference: bool = False, 
        max_pixels: int = 1350 * 28 * 28, 
        min_pixels: int = 100 * 28 * 28) -> str:
    """
    Build a prompt with single-turn format, containing previous observations and actions.
    This is effective for Qwen-based models to refer to previous observations and actions.

    args:
        instruction: str, user instruction
        previous_images: List[PIL.Image.Image], previous images
        previous_actions: List[str], previous actions
    """
    

    contents = []
    if previous_images is not None and previous_actions is not None:
        for idx, (previous_image, previous_action) in enumerate(zip(previous_images, previous_actions)):
            prefix_image_tokens = {"type": "text", "text": f"Previous screenshot ({len(previous_images) - idx} steps ago):"}
            contents.append(prefix_image_tokens)
            image_tokens = {"type": "image", "image": previous_image}
            contents.append(image_tokens)
            prefix_action_tokens = {"type": "text", "text": f"Previous action ({len(previous_images) - idx} steps ago): {previous_action}"}
            contents.append(prefix_action_tokens)
    
    if action_reference:
        if isinstance(current_obs["action_reference"], str):
            action_reference = current_obs["action_reference"]
        else:
            action_reference = "\n".join(current_obs["action_reference"])
        prompt = ACTOR_PROMPT.format(instruction=current_obs["instruction"], action_reference=action_reference)
    else:
        prompt = ACTOR_PROMPT.format(instruction=current_obs["instruction"], action_reference="None")
    
    if current_obs["screenshot"] is not None:
        current_prompt = {"type": "text", "text": prompt + "\n\n" + "Current screenshot:"}
        contents.append(current_prompt)
        image = Image.open(BytesIO(current_obs["screenshot"]))
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize((resized_width, resized_height))
        # if image.width * image.height > max_pixels:
        #     resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        #     width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        #     image = image.resize((width, height))
        # if image.width * image.height < min_pixels:
        #     resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        #     width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
        #     image = image.resize((width, height))
        if image.mode != "RGB":
            image = image.convert("RGB")
        contents.append({"type": "image", "image": image})
        # contents.append({"type": "image", "image": Image.open(BytesIO(current_obs["screenshot"]))})
    else:
        current_prompt = {"type": "text", "text": prompt + "\n\n" + "Current screenshot is not available."}
        contents.append(current_prompt)
    
    message = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ],
        },
        {
            "role": "user",
            "content": contents,
        },
    ]
    return message


# messages1 = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "screenshot.png"},
#             {"type": "text", "text": instruction},
#             {"type": "image", "image": "screenshot.png"},
#         ],
#     }
# ]

# messages2 = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who are you?"},
# ]

# messages = [messages1, messages2]

# texts = [
#     processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
#     for msg in messages
# ]

# # # Preparation for a single turn inference
# # text = processor.apply_chat_template(
# #     messages, tokenize=False, add_generation_prompt=True
# # )

# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     # text=[text], # for single turn inference
#     text=texts,    # for multi-turn inference
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )