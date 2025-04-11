import re

def gui_agent_format_reward(output: str):
    """
    Reward function that checks if the output is formatted correctly.
    Specifically,
        (i)  it checks if the content of Thought: is not empty.
        (ii) it checks if the content of Action: can be parsed into a valid action.
    """
    thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    reward = 0.0
    thought_match = re.search(thought_pattern, output, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
        if thought != "":
            reward += 0.1
    else:
        reward -= 0.1

    return reward


def english_format_reward(output: str):
    """
    Reward function that checks if the output does not include any Chinese or Japanese characters.
    """
    pattern = r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]'
    reward = 0.0
    if re.search(pattern, output):
        reward -= 0.1
    else:
        reward += 0.1
    return reward