import torch
from chatgpt.models.utils import compute_reward

from .base import Experience, ExperienceMaker
from tqdm import tqdm
from typing import Tuple
from chatgpt.models.utils import masked_mean
import torch
import torch.nn as nn
from chatgpt.models.actor import Actor

SHOW_TIME_DETAILS = False
SHOW_SAMPLE_DETAILS = False

class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    
    def __init__(self,
                 actor: Actor,
                 critic: nn.Module,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 kl_controller,
                 strategy=None) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.kl_ctl = kl_controller
        self.strategy = strategy

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        for i in tqdm(range(1), desc=f'Generate sequence', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            sequences, attention_mask, action_mask = self.actor.generate(input_ids, **generate_kwargs)
            
        num_actions = action_mask.size(1)
        for i in tqdm(range(1), desc=f'Actor forward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            action_log_probs = self.actor(sequences, num_actions, attention_mask)
        
        for i in tqdm(range(1), desc=f'Init model forward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        for i in tqdm(range(1), desc=f'Value model forward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            value = self.critic(sequences, action_mask, attention_mask)
        
        for i in tqdm(range(1), desc=f'Reward model forward', disable=not SHOW_TIME_DETAILS or not self.strategy.is_rank_0()):
            r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(r, self.kl_ctl.value, action_log_probs, base_action_log_probs, action_mask=action_mask)
        advantage, returns = self.get_advantages_and_returns(
            value, reward, action_mask, generate_kwargs['gamma'], generate_kwargs['lambd'])

        if SHOW_SAMPLE_DETAILS:
            self.strategy.print('return', reward.sum(1))
            self.strategy.print('reward', reward)
            self.strategy.print('advantage', advantage)
            self.strategy.print('value', value)
            self.strategy.print('action_log_probs', action_log_probs)
            self.strategy.print('action_mask', action_mask)

        info = {
            'kl': masked_mean(kl, action_mask, dim=-1),
            'rm': r,
            'ret': reward.sum(dim=-1),
            'glen': action_mask.float().sum(dim=-1),
            'tlen': attention_mask.float().sum(dim=-1)
        }
        
        return Experience(sequences, action_log_probs, value, returns, advantage, attention_mask, action_mask, info)

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
