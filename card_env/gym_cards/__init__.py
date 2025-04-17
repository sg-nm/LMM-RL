from gymnasium.envs.registration import register
from gym_cards.prompt_lib import PROMPT_FN
from gym_cards.config_dataclass import CardEnvConfig, load_config_from_yaml

register(
    id='gym_cards/GeneralPoint-oneline-v0',
    entry_point='gym_cards.envs:GeneralPointEnv_oneline',
    max_episode_steps=300,
)