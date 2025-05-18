from vllm import LLM, EngineArgs, SamplingParams
from PIL import Image
import os
import random
from dataclasses import asdict
from typing import NamedTuple, Optional
from vllm.lora.request import LoRARequest
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from typing import Union, Tuple, List
from qwen_agent.tools.base import BaseTool, register_tool
from gui_env.agent_utils import parse_action_to_structure_output, parsing_response_to_pyautogui_code

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

ACTOR_FEEDBACK_PROMPT = """Your task is to help another agent to complete the following task.

## Task
{instruction}

## Model's Action Plan
First, we need to select the "Proposed method" slide from the slide preview pane on the left. Once the correct slide is selected, we can then proceed with the action.

## Model's Action History
click(start_box='(105,523)')

Please provide some hints to the model to help it complete the task.
"""


@register_tool("computer_use")
class ComputerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "scroll",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
                "type": "number",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action in ["left_click", "right_click", "middle_click", "double_click"]:
            return self._mouse_click(action)
        elif action == "key":
            return self._key(params["keys"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "mouse_move":
            return self._mouse_move(params["coordinate"])
        elif action == "left_click_drag":
            return self._left_click_drag(params["coordinate"])
        elif action == "scroll":
            return self._scroll(params["pixels"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _mouse_click(self, button: str):
        raise NotImplementedError()

    def _key(self, keys: List[str]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _mouse_move(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _left_click_drag(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _scroll(self, pixels: int):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()


# model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model_name = "ByteDance-Seed/UI-TARS-1.5-7B"
# llm = LLM("bytedance-research/UI-TARS-7B-DPO", limit_mm_per_prompt={"image": 5}, enable_prefix_caching=True)
llm = LLM(model_name, limit_mm_per_prompt={"image": 5}, enable_prefix_caching=True)

processor = AutoProcessor.from_pretrained(model_name)

# Initialize computer use function
computer_use = ComputerUse(
    cfg={"display_width_px": 1920, "display_height_px": 1080}
)

# Build messages
system_message = NousFnCallPrompt().preprocess_fncall_messages(
    messages=[
        Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
    ],
    functions=[computer_use.function],
    lang=None,
)
system_message = system_message[0].model_dump()

prompt = """Output next action to complete the following task.

## Task
Insert a new slide after "Proposed method" slide.
"""

prompt2 = """Output next action to complete the following task.

## Task
Insert a new table to the current slide.
"""

ref = "We can insert a new slide by pressing `Enter` after selecting the slide where you want to insert a new slide."

prompt = ACTOR_PROMPT.format(action_reference=ref, instruction="Insert a new slide after \"Proposed method\" slide.")
prompt2 = ACTOR_PROMPT.format(action_reference="None", instruction="Insert a new table to the current slide.")

image = Image.open("/home/suganuma/src/lmm-r1_L/LMM-RL-GUI/step_1.jpg").convert("RGB")
image2 = Image.open("/home/suganuma/src/lmm-r1_L/LMM-RL-GUI/step_10.jpg").convert("RGB")
width, height = image.size
r_height, r_width = smart_resize(height, width, min_pixels=100 * 28 * 28, max_pixels=16384*28*28)
image = image.resize((r_width, r_height))
image2 = image2.resize((r_width, r_height))
messages1=[
    # {
    #     "role": "system",
    #     "content": [
    #         {"type": "text", "text": msg["text"]} for msg in system_message["content"]
    #     ],
    # },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

messages2=[
    # {
    #     "role": "system",
    #     "content": [
    #         {"type": "text", "text": msg["text"]} for msg in system_message["content"]
    #     ],
    # },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image2,
            },
            {"type": "text", "text": prompt2},
        ],
    }
]

# prompt_texts = [
#     processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
#     for msg in messages
# ]

messages = [messages1, messages2]

prompt_texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]

print(prompt_texts)

images = [image, image2]
vllm_inputs = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
        #"mm_processor_kwargs": {"min_pixels": 100 * 28 * 28, "max_pixels": 1350 * 28 * 28}
    }
    for prompt, image in zip(prompt_texts, images)
]

sampling_params = SamplingParams(temperature=1.0, top_p=0.7, max_tokens=512)
output = llm.generate(vllm_inputs, sampling_params)
print(output[0].outputs[0].text)
print(output[1].outputs[0].text)

outputs = [output[i].outputs[0].text for i in range(len(output))]
# parsed_outputs = []
# for output, image in zip(outputs, images):
#     image_height, image_width = image.size
#     output = parse_action_to_structure_output(output, factor=1000, origin_resized_height=image_height, origin_resized_width=image_width)
#     output = parsing_response_to_pyautogui_code(output, image_height=image_height, image_width=image_width)
#     parsed_outputs.append(output)

import pdb; pdb.set_trace()
# x = int(output[0].outputs[0].text.split("start_box=")[1].split("(")[0].strip("("))
# y = int(output[0].outputs[0].text.split("start_box=")[1].split(",")[1].strip(")"))
# import cv2
# image = cv2.imread("Screenshot.png")
# img_height, img_width = image.shape[:2]
# x = int(x * img_width / 1000)
# y = int(y * img_height / 1000)
# cv2.circle(image, (x, y), 20, (255, 255, 0), 2)
# cv2.imwrite("Screenshot_with_circle.png", image)
import pdb; pdb.set_trace()

# ## Below is the prompt for computer
# prompt = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

# ## Output Format
# ```\nThought: ...
# Action: ...\n```

# ## Action Space

# click(start_box='<|box_start|>(x1,y1)<|box_end|>')
# left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
# right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
# drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
# hotkey(key='')
# type(content='') #If you want to submit your input, use \"\
# \" at the end of `content`.
# scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
# wait() #Sleep for 5s and take a screenshot to check for any changes.
# finished()
# call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


# ## Note
# - Use English in `Thought` part.
# - Summarize your next action (with its target element) in one sentence in `Thought` part.

# ## User Instruction

# """

# prompt = [r"""Output only the coordinate of one point in your response. What element matches the following task: """]


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

# Qwen2-VL
def run_qwen2_vl(questions: list[str], gpt_answers: list[str], modality: str="image"):
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = ""
    for question, gpt_answer in zip(questions[:-1], gpt_answers):
        prompts += (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{gpt_answer}<|im_end|>\n"
        )
    prompts += (
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{questions[-1]}<|im_end|>\n<|im_start|>assistant\n"
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

# Qwen2-VL
def run_qwen2_vl_seq(questions: list[str], gpt_answers: list[str], modality: str="image"):
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nFirst image:<|vision_start|>{placeholder}<|vision_end|>Second image:<|vision_start|>{placeholder}<|vision_end|>Third image:<|vision_start|>{placeholder}<|vision_end|>"
    for question, gpt_answer in zip(questions[:-1], gpt_answers):
        prompts += (
            f"{question} You answer is {gpt_answer}\n"
        )
    prompts += (
        f"\n{questions[-1]}<|im_end|>\n<|im_start|>assistant\n"
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


image1= Image.open("/home/suganuma/src/gui_agent/active_image_0.png").convert("RGB")
image2= Image.open("/home/suganuma/src/gui_agent/active_image_0.png").convert("RGB")
image3= Image.open("/home/suganuma/src/gui_agent/active_image_0.png").convert("RGB")

image = Image.open("/home/suganuma/src/gui_agent/active_image_0.png").convert("RGB")
image_= Image.open("/home/suganuma/src/gui_agent/active_image_0.png").convert("RGB")

multi_modal_data_1 = {
    "image": [image1, image_, image3],
}

multi_modal_data_2 = {
    "image": [image_, image, image_],
}

prompt = ["Which page is selected?", "Which page is selected?", "Search for the GUI element to increase font size and output only the coordinate of one point in your response."]
gpt_answers = [
    "1",
    "2",
]
req_data = run_qwen2_vl(prompt, gpt_answers, "image")
input_prompt_2 = req_data.prompts

prompt = ["Which page is selected?", "Which page is selected?", "What is the difference between the second and third images?"]
gpt_answers = [
    "1",
    "2",
]
req_data = run_qwen2_vl_seq(prompt, gpt_answers, "image")
input_prompt_1 = req_data.prompts


sampling_params = SamplingParams(temperature=0.6,
                                 max_tokens=128,
                                 stop_token_ids=req_data.stop_token_ids)

inputs = [
    {
        "prompt": input_prompt_1,
        "multi_modal_data": multi_modal_data_1,
    },
    {
        "prompt": input_prompt_2,
        "multi_modal_data": multi_modal_data_2,
    },
]
output = llm.generate(inputs, sampling_params)
print(output[0].outputs[0].text)
print(output[1].outputs[0].text)
import pdb; pdb.set_trace()
# extract x and y from the response
x = int(output[1].outputs[0].text.split(",")[0].strip("("))
y = int(output[1].outputs[0].text.split(",")[1].strip(")"))
# draw a circle on the image at the point
import cv2
image = cv2.imread("Screenshot.png")
img_height, img_width = image.shape[:2]
x = int(x * img_width / 1000)
y = int(y * img_height / 1000)
cv2.circle(image, (x, y), 20, (255, 255, 0), 2)
# x = int(gpt_answers[0].split(",")[0].strip("("))
# y = int(gpt_answers[0].split(",")[1].strip(")"))
# x = int(x * img_width / 1000)
# y = int(y * img_height / 1000)
# cv2.circle(image, (x, y), 20, (255, 0, 255), 3)
cv2.imwrite("Screenshot_with_circle.png", image)


