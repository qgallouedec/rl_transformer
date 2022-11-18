from typing import Dict, Tuple

import gym
import numpy as np
import torch
from torch import Tensor

numpy_to_torch_dtype = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


class EpisodeBuffer:
    """
    Episode buffer.

    Logic:
    The step increment occurs when the action is applied to the environment.

    Args:
        buffer_size (int): As the number of episodes that can be stored
        env (gym.Env): The envrionment
    """

    def __init__(self, buffer_size: int, env: gym.Env) -> None:
        self.buffer_size = buffer_size
        self.max_ep_len = env.spec.max_episode_steps + 1
        observation_shape = env.observation_space.shape
        observation_dtype = env.observation_space.dtype.type
        action_shape = env.action_space.shape
        action_dtype = env.action_space.dtype.type
        self.observations = torch.zeros(
            (buffer_size, self.max_ep_len, *observation_shape), dtype=numpy_to_torch_dtype[observation_dtype]
        )
        self.values = torch.zeros((buffer_size, self.max_ep_len), dtype=float)
        self.actions = torch.zeros((buffer_size, self.max_ep_len, *action_shape), dtype=numpy_to_torch_dtype[action_dtype])
        self.log_probs = torch.zeros((buffer_size, self.max_ep_len), dtype=float)
        self.rewards = torch.zeros((buffer_size, self.max_ep_len), dtype=float)
        self.dones = torch.zeros((buffer_size, self.max_ep_len), dtype=bool)
        # self.infos = torch.zeros((buffer_size, self.max_ep_len), dtype=dict)
        self.ep_length = torch.zeros((buffer_size,), dtype=int)

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
        ep_idxs = np.random.choice(np.arange(self.buffer_size), batch_size, p=self.ep_length / self.ep_length.sum())
        ep_length = self.ep_length[ep_idxs]

        # Create the mask
        # True for sampled_start < timesteps < ep_length, and False otherwise
        episode_mask = np.arange(self.max_ep_len)[None, :] < ep_length[:, None]
        ep_starts = np.random.randint(0, self.ep_length[ep_idxs] - 1)
        sampled_start_mask = ep_starts[:, None] <= np.arange(self.max_ep_len)[None, :]
        src_key_padding_mask = np.logical_and(episode_mask, sampled_start_mask)

        # Shift right (so that the last observation is right)
        idx = (np.arange(self.max_ep_len) + self.ep_length[ep_idxs, None]) % self.max_ep_len
        observations = self.observations[ep_idxs[:, None], idx]
        actions = self.actions[ep_idxs[:, None], idx]
        rewards = self.rewards[ep_idxs[:, None], idx]
        dones = self.dones[ep_idxs[:, None], idx]
        infos = self.infos[ep_idxs[:, None], idx]
        src_key_padding_mask = src_key_padding_mask[np.arange(batch_size)[:, None], idx]

        return observations, actions, rewards, dones, infos, src_key_padding_mask

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.ep_idx = -1
        self.t = 0
