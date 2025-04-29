from dataclasses import dataclass
from typing import Optional, List

import yaml


@dataclass
class OptimizerConfig:
    init_lr: float
    eps: float
    weight_decay: float
    lr_max_steps: int
    end_lr: float

@dataclass
class PPOConfig:
    clip_param: float
    ppo_epoch: int
    mini_batch_size: int
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float

@dataclass
class ComputeReturnKwargs:
    use_gae: bool
    gamma: float
    gae_lambda: float
    use_proper_time_limits: bool

@dataclass
class EnvConfig:
    id: str
    target_points: Optional[int] = None
    treat_face_cards_as_10: Optional[bool] = None
    resolution: Optional[int] = None
    face_cards_color: Optional[str] = None
    verify_iter: Optional[int] = None
    num_steps: Optional[int] = None
    
@dataclass
class EvalEnvConfig:
    id: str
    target_points: Optional[int] = None
    treat_face_cards_as_10: Optional[bool] = None
    resolution: Optional[int] = None
    face_cards_color: Optional[str] = None
    verify_iter: Optional[int] = None
    num_evaluations: Optional[int] = None

@dataclass
class PromptConfig:
    use_vision: bool
    use_language: bool
    enable_verification: bool
    prompt_vision: Optional[List[str]] = None
    pattern_vision: Optional[List[str]] = None
    prompt_language: Optional[List[str]] = None
    pattern_language: Optional[List[str]] = None

@dataclass
class GenerationConfig:
    temperature: float
    max_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    thought_prob_coef: Optional[float] = None
    num_beams: Optional[int] = None

@dataclass
class CardEnvConfig:
    grad_accum_steps: int
    optimizer_config: OptimizerConfig
    ppo_config: PPOConfig
    compute_return_kwargs: ComputeReturnKwargs
    report_to: str
    run_name: str
    num_processes: int
    num_updates: int
    env_config: EnvConfig
    eval_env_config: EvalEnvConfig
    model: str
    model_path: str
    prompt_config: PromptConfig
    generation_config: GenerationConfig
    output_dir: str
    seed: int
    save_ckpt: bool
    save_every: int
    num_envs: int
    num_eval_envs: int

# YAMLファイルの読み込みとConfigクラスへの変換
def load_config_from_yaml(path: str) -> CardEnvConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return CardEnvConfig(
        grad_accum_steps=config_dict["grad_accum_steps"],
        optimizer_config=OptimizerConfig(**config_dict["optimizer_config"]),
        ppo_config=PPOConfig(**config_dict["ppo_config"]),
        compute_return_kwargs=ComputeReturnKwargs(**config_dict["compute_return_kwargs"]),
        report_to=config_dict["report_to"],
        run_name=config_dict["run_name"],
        num_processes=config_dict["num_processes"],
        num_updates=config_dict["num_updates"],
        env_config=EnvConfig(**config_dict["env_config"]),
        eval_env_config=EvalEnvConfig(**config_dict["eval_env_config"]),
        model=config_dict["model"],
        model_path=config_dict["model_path"],
        prompt_config=PromptConfig(**config_dict["prompt_config"]),
        generation_config=GenerationConfig(**config_dict["generation_config"]),
        output_dir=config_dict["output_dir"],
        seed=config_dict["seed"],
        save_ckpt=config_dict["save_ckpt"],
        save_every=config_dict["save_every"],
        num_envs=config_dict["num_envs"],
        num_eval_envs=config_dict["num_eval_envs"],
    )