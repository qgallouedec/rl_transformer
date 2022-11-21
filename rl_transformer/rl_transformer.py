import math
from typing import Optional, Tuple

import gym
import torch
from gym import spaces
from torch import Tensor, nn
from torch.distributions import Categorical, Normal

from rl_transformer.utils import get_space_shape, get_space_size, preprocess


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
            x (Tensor): Input of shape (..., L, H_in)

        Returns:
            Tensor: Output of shape (N, L, H_in). It corresponds to the
                input, to which we have added the positional encoding
        """
        seq_len = x.shape[-2]
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
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        obs_size = get_space_size(self.observation_space)
        action_size = get_space_size(self.action_space)
        max_len = env.spec.max_episode_steps + 1
        self.input_encoder = nn.Linear(obs_size + action_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        mask = generate_square_subsequent_mask(max_len)
        self.register_buffer("mask", mask)
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.action_net = nn.Linear(d_model, action_size)
        if isinstance(self.action_space, spaces.Box):
            self.log_std = nn.Parameter(torch.zeros(action_size), requires_grad=True)

    def forward(
        self, observations: Tensor, actions: Tensor, src_key_padding_mask: Optional[Tensor] = None, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the  predicted actions and values given the episodes.

        Args:
            observations (Tensor): Observation of shape (..., L+1, obs_shape). Ending with s_t.
            actions (Tensor): Actions of shape (..., L, action_shape). Ending with a_{t-1}
            deterministic (bool): If actor acts deterinistially

        Returns:
            values, actions, and log_prob
        """
        observations = preprocess(observations, self.observation_space)  # (..., L+1, *obs_shape) -> (..., L+1, obs_size)
        actions = preprocess(actions, self.action_space)  # (..., L, *action_shape) -> (..., L, action_size)
        input_shape = observations.shape[:-1]  # = (..., L)
        pad = torch.zeros(*actions.shape[:-2], 1, actions.shape[-1]).to(actions.device)
        actions = torch.concatenate((pad, actions), dim=-2)  # (..., L, action_size) to (..., L+1, action_size)
        observations_actions = torch.concatenate((observations, actions), dim=-1)  # (..., L+1, observation_size + action_size)
        x = self.input_encoder(observations_actions)
        x = self.positional_encoder(x)
        mask = self.mask[: x.size(-2), : x.size(-2)]
        x = self.transformer_encoder(x, mask, src_key_padding_mask)
        # Critic
        values = self.value_net(x)
        # Actor
        if isinstance(self.action_space, spaces.Box):
            mean_actions = self.action_net(x)
            action_std = torch.ones_like(mean_actions) * self.log_std.exp()
            distribution = Normal(mean_actions, action_std)
            if deterministic:
                actions = distribution.mean
            else:
                actions = distribution.rsample()
        elif isinstance(self.action_space, spaces.Discrete):
            action_logits = self.action_net(x)
            distribution = Categorical(logits=action_logits)
            if deterministic:
                actions = distribution.sample()
            else:
                actions = torch.argmax(distribution.probs, dim=-1)
            log_prob = distribution.log_prob(actions)
        if isinstance(self.action_space, spaces.Box):
            actions = torch.clip(
                actions,
                torch.tensor(self.action_space.low).to(actions.device),
                torch.tensor(self.action_space.high).to(actions.device),
            )
            log_prob = torch.sum(distribution.log_prob(actions), dim=-1)

        return (
            values.reshape(input_shape),
            actions.reshape((*input_shape, *get_space_shape(self.action_space))),
            log_prob.reshape(input_shape),
        )

    def act(self, observations: Tensor, actions: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the predicted action given the episode.

        Args:
            observations (Tensor): Past observation of shape (L+1, obs_shape). Ending with s_t.
            actions (Tensor): Past actions of shape (L, action_shape). Ending with a_{t-1}
            deterministic (bool): If actor acts deterinistially

        Returns:
            value, action, log_prob
        """
        with torch.no_grad():
            values, actions, log_prob = self.forward(observations, actions, deterministic=deterministic)
        return values[-1], actions[-1], log_prob[-1]  # get only the last action

    def get_value(self, observations: Tensor) -> Tensor:
        """
        Compute the value of the current observation.

        Args:
            observations (Tensor): Past observation of shape (L+1, obs_shape). Ending with s_t.
            actions (Tensor): Past actions of shape (L, action_shape). Ending with a_{t-1}

        Returns:
            Value
        """
        obs_shape = get_space_shape(self.observation_space)
        action_space = get_space_shape(self.action_space)
        observations = observations.reshape((1, *obs_shape))  # (obs_shape,) -> (1, obs_shape)
        actions = torch.zeros(0, *action_space).to(observations.device)  # (0, action_shape)
        with torch.no_grad():
            values, _, _ = self.forward(observations, actions)
        return values[0]  # get only the last value
