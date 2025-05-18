from dataclasses import dataclass
import yaml


@dataclass
class VMConfig:
    num_envs: int
    docker_image_name: str
    host_task_dir: str
    task_configs_file: str
    cache_dir: str

    max_tokens: int
    top_p: float
    top_k: float
    temperature: float

    action_space: str
    observation_type: str
    max_trajectory_length: int
    max_steps: int
    history_n: int
    screen_height: int
    screen_width: int
    bin_nums: int

    max_pixels: int
    min_pixels: int
    result_dir: str




# YAMLファイルの読み込みとConfigクラスへの変換
def load_config_from_yaml(path: str) -> VMConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return VMConfig(
        num_envs=config_dict["num_envs"],
        docker_image_name=config_dict["docker_image_name"],
        host_task_dir=config_dict["host_task_dir"],
        task_configs_file=config_dict["task_configs_file"],
        cache_dir=config_dict["cache_dir"],
        max_tokens=config_dict["max_tokens"],
        top_p=config_dict["top_p"],
        top_k=config_dict["top_k"],
        temperature=config_dict["temperature"],
        action_space=config_dict["action_space"],
        observation_type=config_dict["observation_type"],
        max_trajectory_length=config_dict["max_trajectory_length"],
        max_steps=config_dict["max_steps"],
        history_n=config_dict["history_n"],
        screen_height=config_dict["screen_height"],
        screen_width=config_dict["screen_width"],
        bin_nums=config_dict["bin_nums"],
        max_pixels=config_dict["max_pixels"],
        min_pixels=config_dict["min_pixels"],
        result_dir=config_dict["result_dir"],
    )