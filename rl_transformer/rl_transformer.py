import math
from typing import Optional, Tuple

import gym
import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from torch import Tensor, nn

from rl_transformer.utils import get_space_dim


class PositionalEncoder(nn.Module):
    """
    Positional encoder.

    Adds PE to the input where

    PE(pos, 2i  ) = sin(pos/1e4^(2i/d))
    PE(pos, 2i+1) = cos(pos/1e4^(2i/d))

    Args:
        d_model (int): Input size which is the size of the embedding vector (H_in)
        max_len (int): Maximum length of the sequence that the positional encoder can process
        dropout (float): Dropout probability
    """

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add the positional encoding to the input.

        Args:
            x (Tensor): Input of shape (N, L, H_in)

        Returns:
            Tensor: Output of shape (N, L, H_in). It corresponds to the
                input, to which we have added the positional encoding
        """
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len]
        return self.dropout(x)


def generate_square_subsequent_mask(size: int) -> Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)


class ActorCriticTransformer(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        obs_size = get_space_dim(env.observation_space)
        action_size = get_space_dim(env.action_space)
        max_len = env.spec.max_episode_steps + 1
        self.input_encoder = nn.Linear(obs_size + action_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mask = generate_square_subsequent_mask(max_len)
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.action_net = nn.Linear(d_model, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size), requires_grad=True)
        self.action_dist = DiagGaussianDistribution(action_size)

    def forward(
        self, observations: Tensor, actions: Tensor, src_key_padding_mask: Optional[Tensor] = None, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the  predicted actions and values given the episodes.

        Args:
            observations (Tensor): Observation of shape (N, L+1, obs_size). Ending with s_t.
            actions (Tensor): Actions of shape (N, L, action_size). Ending with a_{t-1}
            deterministic (bool): If actor acts deterinistially

        Returns:
            Predicted actions, values and log_prob
        """
        actions = torch.hstack((torch.zeros(actions.shape[0], 1, actions.shape[2]), actions))  # (N, L, *) to (N, L+1, *)
        observations_actions = torch.dstack((observations, actions))  # (N, L+1, observation_size + action_size)
        x = self.input_encoder(observations_actions)
        x = self.positional_encoder(x)
        mask = self.mask[: x.size(1), : x.size(1)]
        x = self.transformer_encoder(x, mask, src_key_padding_mask)
        # Critic
        values = self.value_net(x)
        # Actor
        mean_actions = self.action_net(x)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def act(self, observations: Tensor, actions: Tensor, deterministic: bool = False) -> Tensor:
        """
        Compute the  predicted action given the episode.

        Args:
            observations (Tensor): Observation of shape (L+1, obs_size). Ending with s_t.
            actions (Tensor): Actions of shape (L, action_size). Ending with a_{t-1}
            deterministic (bool): If actor acts deterinistially

        Returns:
            Predicted next action
        """
        observations = observations.unsqueeze(0)  # (L, obs_size) -> (1, L, obs_size)
        actions = actions.unsqueeze(0)  # (L, action_size) -> (1, L, action_size)
        with torch.no_grad():
            actions, _, _ = self.forward(observations, actions, deterministic=deterministic)
        return actions[0][-1]  # get only the last action


if __name__ == "__main__":
    import gym

    env = gym.make("Pendulum-v1")
    ac = ActorCriticTransformer(env, d_model=16)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    observations = torch.rand((13, obs_size))
    actions = torch.rand((12, action_size))

    print(ac.act(observations, actions))
