"""
Finalized prompts
Updated: Jan-28-2025
By Tianzhe
"""


import json
example = {
  "cards": ['4', '3', '2', 'K'],
  "number": [4, 3, 2, 10],
  "thought": "<step by step reasoning process to build the formula>",
  "formula": "4*3+2+10=24",
}
example_json_text = json.dumps(example, ensure_ascii=False)

Q_GeneralPoint_EQN_VL = """
## Task Description
You are an expert {target_number} points card game player. You are observing these four cards in the image.
Note that {face_card_msg}, and each card must be used once.
Your goal is to find a formula that evaluates to {target_number} using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

## Output format
Your response should be a valid JSON file in the following format:
{{
  "cards": [x, y, z, w], where {face_card_msg} Also, omit the suit of the cards.,
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "thought": "your step by step thought process to build the formula",
  "formula": "an equation that equals {target_number}",
}}

## Output Example
{example_json_text}

Please only output the json.
"""

Q_GeneralPoint_EQN_VL_REASONING = """
## Task Description
You are an expert {target_number} points card game player. You are observing four cards in the image.
Your goal is to find a formula that evaluates to {target_number} using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.
Note that {face_card_msg}, and each card must be used exactly once.

Please reason step by step to build the formula and provide the reasoning process in the "thought" field below.

## Output format
Your response should be a valid JSON file in the following format:
{{
  "cards": [x, y, z, w], where {face_card_msg} Also, omit the suit of the cards.,
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "thought": "step by step reasoning process to build the formula",
  "formula": "the formula that equals {target_number}",
}}

## Output Example
{example_json_text}

"""

Q_GeneralPoint_EQN_L = """
## Task Description
You are an expert {target_number} points card game player. You will receive a set of 4 cards.
Note that {face_card_msg}, and each card must be used once.
Your goal is to output a formula that evaluates to {target_number} using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

## Input
Cards: {cards}

## Output format
{{
  "cards": [x, y, z, w], where {face_card_msg},
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "formula": 'an equation that equals {target_number}',
}}

"""

"""
    *** Responses templates ***
"""

ResponseEqn = """
{{
  "cards": {cards},
  "number": {numbers},
  "formula": "{formula}",
}}"""



Q_VIRL_L = """
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to follow. You
are also provided with your observation and action history in text. Your goal is to first analyze the instruction and identify the next sentence to be executed. 
Then, you need to provide the action to be taken based on the current observation and instruction.

[Instruction]
{instruction}

[Action space]
{action_space}

[Observations and actions sequence]
{obs_act_seq}

[Output]
{{
  "current observation": latest observation from the observation sequence,
  "current instruction": analyze the full instruction and identify the sentence to be executed,
  "action": the action to be taken chosen from the action space,
}}
"""

ResponseVIRL = """
{{
  "current observation": "{current_observation}",
  "current instruction": "{current_instruction}",
  "action": "{action}",
}}
"""

Q_VIRL_VL = """
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to follow while observing your surrounding stree tviews. You
are also provided with your observation and action history in text. Your goal is to first analyze the instruction and identify the next sentence to be executed. 
Then, you need to provide the action to be taken based on the current observation and instruction.

[Instruction]
{instruction}

[Observation format]
You observe a 2x2 grid of streetview images with the following headings:
[front, right
 back, left]
You need to identify if any of the landmarks in the instruction are visible in the street view grid.

[Action space]
{action_space}

[Observations and actions sequence]
{obs_act_seq}

[Output]
{{
  "current observation": latest observation from the street view grid,
  "current instruction": analyze the full instruction and identify the sentence to be executed,
  "action": the action to be taken chosen from the action space,
}}
"""