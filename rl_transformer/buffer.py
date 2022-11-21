from typing import Dict, Optional, Tuple, Union

import gym
import torch
from torch import Tensor

from rl_transformer.utils import NUMPY_TO_TORCH_DTYPE, get_space_shape


class EpisodeBuffer:
    """
    Episode buffer.

    Logic:
    The step increment occurs when the action is applied to the environment.

    Args:
        buffer_size (int): As the number of episodes that can be stored
        env (gym.Env): The envrionment
    """

    def __init__(self, buffer_size: int, env: gym.Env, device: Optional[Union[torch.device, str]] = None) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.max_ep_len = env.spec.max_episode_steps + 1
        observation_shape = get_space_shape(env.observation_space)
        observation_dtype = NUMPY_TO_TORCH_DTYPE[env.observation_space.dtype.type]
        action_shape = get_space_shape(env.action_space)
        action_dtype = NUMPY_TO_TORCH_DTYPE[env.action_space.dtype.type]
        self.observations = torch.zeros(
            (buffer_size, self.max_ep_len, *observation_shape), dtype=observation_dtype, device=self.device
        )
        self.values = torch.zeros((buffer_size, self.max_ep_len), dtype=float, device=self.device)
        self.actions = torch.zeros((buffer_size, self.max_ep_len, *action_shape), dtype=action_dtype, device=self.device)
        self.log_probs = torch.zeros((buffer_size, self.max_ep_len), dtype=float, device=self.device)
        self.rewards = torch.zeros((buffer_size, self.max_ep_len), dtype=float, device=self.device)
        self.dones = torch.zeros((buffer_size, self.max_ep_len), dtype=bool, device=self.device)
        # self.infos = torch.zeros((buffer_size, self.max_ep_len), dtype=dict)
        self.ep_length = torch.zeros((buffer_size,), dtype=int, device=self.device)

        self.ep_idx = -1
        self.t = 0

    def new_episode(self, observation: Tensor, value: float) -> None:
        """
        Start a new episode.

        Args:
            observation (Tensor): Observation
            value (float): Value of the observation
        """
        self.ep_idx += 1
        self.t = 0
        self.observations[self.ep_idx][self.t] = observation
        self.values[self.ep_idx][self.t] = value
        self.t += 1

    def add(
        self, action: Tensor, log_prob: float, observation: Tensor, value: float, reward: float, done: bool, info: Dict
    ) -> None:
        """
        Store a transition.

        Args:
            action (Tensor): Action
            log_prob (float): Log-probability of the action
            observation (Tensor): Observation
            value (float): Value of the observation
            reward (float): Reward
            done (bool): Whether the episode is done
            info (Dict): Info dict
        """
        self.actions[self.ep_idx][self.t - 1] = action
        self.log_probs[self.ep_idx][self.t - 1] = log_prob
        self.observations[self.ep_idx][self.t] = observation
        self.values[self.ep_idx][self.t] = value
        self.rewards[self.ep_idx][self.t] = reward
        self.dones[self.ep_idx][self.t] = done
        # self.infos[self.ep_idx][self.t] = info
        self.ep_length[self.ep_idx] += 1
        self.t += 1

    def get_current_episode(self) -> Tuple[Tensor, Tensor]:
        """
        Return the current observation and action sequence.

        Returns:
            Tuple[Tensor, Tensor]: observations (L+1, obs_shape) and actions (L, action_shape)
        """
        return self.observations[self.ep_idx][: self.t], self.actions[self.ep_idx][: self.t - 1]

    def sample(self, batch_size: int) -> Tuple[Tensor]:
        """
        _summary_

        Args:
            batch_size (int): Batch size

        Returns:
            Tuple[Tensor]: observations, values, actions, log_prob, rewards, dones, infos, src_key_padding_mask
            indexed in the standard way.
        """
        ep_idxs = torch.multinomial(self.ep_length.float(), batch_size, replacement=True)
        ep_length = self.ep_length[ep_idxs]

        # Create the mask
        # True for sampled_start < timesteps < ep_length, and False otherwise
        episode_mask = torch.arange(self.max_ep_len, device=self.device)[None, :] < ep_length[:, None]
        ep_starts = torch.concatenate([torch.randint(high, (1,), device=self.device) for high in ep_length - 1])
        # torch.random.randint(0, self.ep_length[ep_idxs] - 1)

        sampled_start_mask = ep_starts[:, None] <= torch.arange(self.max_ep_len, device=self.device)[None, :]
        src_key_padding_mask = torch.logical_and(episode_mask, sampled_start_mask)

        # Shift right (so that the last observation is right)
        idx = (torch.arange(self.max_ep_len, device=self.device) + self.ep_length[ep_idxs, None]) % self.max_ep_len
        observations = self.observations[ep_idxs[:, None], idx]
        values = self.values[ep_idxs[:, None], idx]
        actions = self.actions[ep_idxs[:, None], idx]
        log_probs = self.log_probs[ep_idxs[:, None], idx]
        rewards = self.rewards[ep_idxs[:, None], idx]
        dones = self.dones[ep_idxs[:, None], idx]
        # infos = self.infos[ep_idxs[:, None], idx]
        infos = [{} for _ in range(batch_size)]
        src_key_padding_mask = src_key_padding_mask[torch.arange(batch_size, device=self.device)[:, None], idx]

        return observations, values, actions, log_probs, rewards, dones, infos, src_key_padding_mask

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.ep_idx = -1
        self.t = 0
