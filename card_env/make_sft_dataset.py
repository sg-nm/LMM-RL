"""
Make SFT dataset from card game data
"""

import os
import json
import random
from PIL import Image
from tqdm import tqdm
from card_env.gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
from card_env.gym_cards.config_dataclass import load_config_from_yaml
import gymnasium as gym


def make_env(env_config, language_only=False, seed=42):
    def _init():
        config_dict = {k: v for k, v in vars(env_config).items() if k != "id" and k != "num_steps" and k != "num_evaluations"}
        config_dict["language_only"] = language_only
        config_dict["seed"] = seed
        return GeneralPointEnv_oneline(**config_dict)
    return _init

num_envs = 128
configs = load_config_from_yaml("/home/suganuma/src/lmm-r1/card_env/gym_cards/configs/card_24.yaml")
env_fns = [make_env(configs.env_config, language_only=False, seed=idx) for idx in range(num_envs)]
# config_dict = {k: v for k, v in vars(configs.env_config).items() if k != "id" and k != "num_steps" and k != "num_evaluations"}
# config_dict["language_only"] = False
# env = GeneralPointEnv_oneline(**config_dict)
envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)

sample_num = 5000
target_number = configs.env_config.target_points

example = {
  "cards": ['4', '3', '2', 'K'],
  "number": [4, 3, 2, 10],
  "formula": "4*3+2+10=24",
}
example_json_text = json.dumps(example, ensure_ascii=False)

prompt = """
## Task Description
You are an expert {target_number} points card game player. You are observing these four cards in the image.
Your goal is to find a formula that evaluates to {target_number} using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.
Note that {face_card_msg}, and each card must be used exactly once.

## Output format
Your response should be a valid JSON file in the following format:
{{
  "cards": [x, y, z, w], where {face_card_msg} Also, omit the suit of the cards.,
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "formula": 'an equation that equals {target_number}. Note that the formula must include "=".',
}}

## Output Example
{example_json_text}

Provide only the JSON.

"""

samples = []
img_out_dir = "/home/suganuma/datasets/card_24/sft/images_v2"
os.makedirs(img_out_dir, exist_ok=True)
os.makedirs("/home/suganuma/datasets/card_24/sft/train_v2", exist_ok=True)

for i in tqdm(range(sample_num), desc="Generating SFT dataset"):
    obs, info = envs.reset()
    card_info = info['Plain Cards']
    face_card_msg = "'J', 'Q', and 'K' count as '10'." if configs.env_config.treat_face_cards_as_10 \
                                        else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
    task_prompt = prompt.format(target_number=target_number, face_card_msg=face_card_msg, example_json_text=example_json_text)
    
    for env_idx in range(num_envs):
        answer_formulas = info['Solution'][env_idx]
        answer_cards = info['Plain Cards'][env_idx]
        answer_numbers = info['Numbers'][env_idx]
        answer_formula = random.choice(answer_formulas)

        answer_data = {
            "cards": answer_cards,
            "number": answer_numbers,
            "formula": str(answer_formula)+"=24"
        }

        answer = json.dumps(answer_data, ensure_ascii=False)

        img_pil = Image.fromarray(obs[env_idx])
        img = img_pil.resize((512, 512))
        img_path = os.path.join(img_out_dir, f"{i:05d}.jpg")
        img.save(img_path)

        sample = {
            "instruction": task_prompt,
            "answer": answer,
            "image_path": img_path,
        }
        samples.append(sample)

        if len(samples) % 10000 == 0:
            with open(os.path.join("/home/suganuma/datasets/card_24/sft/train_v2", "sft_data.json"), "w") as f:
                json.dump(samples, f, indent=4)

with open(os.path.join("/home/suganuma/datasets/card_24/sft/train_v2", "sft_data.json"), "w") as f:
    json.dump(samples, f, indent=4)
