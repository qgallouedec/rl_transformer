from typing import Dict, Tuple

import gym
import numpy as np


class EpisodeBuffer:
    """
    Episode buffer.

    Args:
        buffer_size (int): As the number of episodes that can be stored
        env (gym.Env): The envrionment
    """

    def __init__(self, buffer_size: int, env: gym.Env) -> None:
        self.buffer_size = buffer_size
        self.max_ep_len = env.spec.max_episode_steps + 1
        observation_shape = env.observation_space.shape
        observation_dtype = env.observation_space.dtype
        action_shape = env.action_space.shape
        action_dtype = env.action_space.dtype
        self.observations = np.zeros((buffer_size, self.max_ep_len, *observation_shape), dtype=observation_dtype)
        self.actions = np.zeros((buffer_size, self.max_ep_len, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros((buffer_size, self.max_ep_len), dtype=np.float32)
        self.dones = np.zeros((buffer_size, self.max_ep_len), dtype=bool)
        self.infos = np.zeros((buffer_size, self.max_ep_len), dtype=dict)
        self.ep_length = np.zeros((buffer_size,), dtype=int)

        self.ep_idx = -1
        self.t = 0

    def new_episode(self, observation: np.ndarray) -> None:
        self.ep_idx += 1
        self.t = 0
        self.observations[self.ep_idx][self.t] = observation
        self.ep_length[self.ep_idx] += 1
        self.t += 1

    def add(
        self, action: np.ndarray, observation: np.ndarray, reward: float, done: bool, info: Dict
    ) -> None:
        """
        Store a transition.

        Args:
            observation (np.ndarray): Observation
            action (np.ndarray): Action
            reward (float): Reward
            done (bool): Whether the episode is done
            info (Dict): Info dict
        """
        self.observations[self.ep_idx][self.t] = observation
        self.actions[self.ep_idx][self.t] = action
        self.rewards[self.ep_idx][self.t] = reward
        self.dones[self.ep_idx][self.t] = done
        self.infos[self.ep_idx][self.t] = info
        self.ep_length[self.ep_idx] += 1
        self.t += 1

    def get_current_episode(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.observations[self.ep_idx][: self.t], self.actions[self.ep_idx][: self.t]

    def sample(self, batch_size: int) -> Tuple[np.ndarray]:
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


if __name__ == "__main__":

    import gym

    env = gym.make("Pendulum-v1")

    buffer = EpisodeBuffer(20, env)

    observation = env.reset()
    buffer.new_episode(observation)
    for _ in range(3000):
        observations, actions = buffer.get_current_episode()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        buffer.add(action, observation, reward, done, info)
        if done:
            observation = env.reset()
            buffer.new_episode(observation)

    observations, actions, rewards, done, infos, src_key_padding_mask = buffer.sample(5)
