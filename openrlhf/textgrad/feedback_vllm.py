import torch
import os
from typing import List
import time
import ray


## Prompt for base model without feedbacks
PROMPT_BASE = """Answer a given question using the following output format.

## Output Format
Thought: provide your thoughts behind the answer
Answer: only provide the choice label from the given choices, e.g. Answer: C

## Question
{question}
"""

## Prompt for base model without feedbacks
PROMPT_BASE_MATH = """Please reason step by step, and put your final answer within \\boxed{{}}.

## Question
{question}
"""

## Prompt for feedback models
FEEDBACK_PROMPT = """Your task is to provide a helpful feedback for the following response from another model so that the model can answer the question better. 
You should refer to the `Correct Answer` and `Model's Answer` below and provide the feedback that can improve the model's thought and answer.
Note that do not provide or include the answer (i.e., content of `Correct Answer`) directly in your feedback.
Please also point out the output format of the model's response if it is not correct.

## Question
{question}

## Correct Answer
{answer}

## Model's Answer
{model_answer}

Feedback:
"""

## Prompt for feedback models
FEEDBACK_PROMPT_MATH = """Your task is to provide a helpful feedback for the following response from another model so that the model can answer the math question better. 
You should refer to the `Solution` and `Model's Answer` below and provide the feedback that can improve the model's thought and answer.
Note that do not provide or include the answer (i.e., content of `Solution`) directly in your feedback.

## Question
{question}

## Solution
{answer}

## Model's Answer
{model_answer}

Feedback:
"""

## Prompt for feedback models on card game
FEEDBACK_PROMPT_CARD = """Your task is to provide a helpful feedback for another model so that the model can achieve the goal better. 
You should refer to the `Previous Model's Answer`, `Verification message`, and `Answer examples` below and provide the step by step feedback that can improve the model's thought and answer.
Note that DO NOT provide or include the answer of the task directly in your feedback even if you know it. Your goal is to help the model to improve its thought and answer.
Please also point out the output format of the model's response if it is not correct.

{task}
"""


## Prompt for base model to generate responses with feedbacks
FEEDBACK_PROMPT_BASE = """The followings are a question and the previous answer from you. Unfortunately, your previous answer is incorrect or can be improved. 
Please generate a new response to the question based on the previous answer and the feedback from another model.
Please do not include any explicit references to the feedback in your response (e.g., “As another model pointed out XX ...”, “The feedback is ...”). You should use the feedback to generate a new answer, but act as if you have not seen the feedback.

## Question
{question}

## Previous answer
{model_answer}

## Feedback
{feedback}

Generate your new answer using the following format.

Thought:
Answer:
"""

## Prompt for base model to generate responses with feedbacks
FEEDBACK_PROMPT_BASE_MATH = """The followings are a math question and the previous answer from you. Unfortunately, your previous answer is incorrect or can be improved. 
Please generate a new response to the question based on the previous answer and the feedback from another model.
Please do not include any explicit references to the feedback in your response (e.g., “As another model pointed out XX ...”, “The feedback is ...”). You should use the feedback to generate a new answer, but act as if you have not seen the feedback.

## Question
{question}

## Previous answer
{model_answer}

## Feedback
{feedback}

Please reason step by step, and put your final answer within \\boxed{{}}.
"""

## Prompt for base model to generate responses with feedbacks
FEEDBACK_PROMPT_BASE_CARD = """Your task is to solve the following problem. Unfortunately, your previous answers are incorrect or can be improved. 
Please generate a new answer by considering the feedbacks from another model below.
Please DO NOT include any explicit references to the feedback in your response (e.g., “As another model pointed out XX ...”, “The feedback says ...”).
You should utilize the feedback to provide a correct answer, but act as if you have not seen the feedback.

{task}
"""


## Prompt for base model to generate responses for REINFORCE
PROMPT_BASE_CARD_REINFORCE = """Your task is to solve the following problem. Unfortunately, your previous answers are incorrect or can be improved. 
Please generate a new answer by considering your previous answers below.

{task}
"""



RESPONSE_PROMPT_CARD = """Your response should be a valid json file in the following format:
{{
  "cards": [x, y, z, w], where {face_card_msg},
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "thought": 'a thought process to build the formula',
  "formula": 'an equation that equals {target_number}',
}}
"""


class FeedbackModel_vllm:
    def __init__(self, feedback_vllm_engines, args):
        self.vllm_engines = feedback_vllm_engines
        self.args = args

        if getattr(self.args, "feedback_vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(self.args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"


    def forward(self, all_prompts: List[str], all_labels, **kwargs):
        from vllm import SamplingParams
        self.response_length_list = []
        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

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
        all_prompts = sum([[prompt] * self.args.n_samples_per_prompt for prompt in all_prompts], [])
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        all_labels = sum([[label] * self.args.n_samples_per_prompt for label in all_labels], [])

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
                refs.append(llm.add_requests.remote(rank, sampling_params=sampling_params, vllm_vision_input=vllm_inputs))

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

    def empty_cache(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()