from typing import Dict
import gymnasium as gym
import numpy as np


class EpisodeBuffer:
    def __init__(self, buffer_size: int, env: gym.Env) -> None:
        max_ep_len = env.spec.max_episode_steps
        observation_shape = env.observation_space.shape
        observation_dtype = env.observation_space.dtype
        action_shape = env.action_space.shape
        action_dtype = env.action_space.dtype
        self.observations = np.zeros((buffer_size, max_ep_len + 1, *observation_shape), dtype=observation_dtype)
        self.actions = np.zeros((buffer_size, max_ep_len + 1, *action_shape), dtype=action_dtype)
        self.rewards = np.zeros((buffer_size, max_ep_len + 1), dtype=np.float32)
        self.terminated = np.zeros((buffer_size, max_ep_len + 1), dtype=bool)
        self.truncated = np.zeros((buffer_size, max_ep_len + 1), dtype=bool)
        self.infos = np.zeros((buffer_size, max_ep_len + 1), dtype=dict)

        self.ep_idx = -1
        self.t = 0

    def new_episode(self, observation: np.ndarray, info: Dict):
        self.ep_idx += 1
        self.t = 0
        self.observations[self.ep_idx][self.t] = observation
        self.infos[self.ep_idx][self.t] = info
        self.t += 1

    def add(
        self, action: np.ndarray, observation: np.ndarray, reward: float, terminated: bool, truncated: bool, info: Dict
    ) -> None:
        """
        Store a transition.

        Args:
            observation (np.ndarray): Observation
            action (np.ndarray): Action
            reward (float): Reward
            terminated (bool): Whether the episode is terminated
            truncated (bool): Whether the episode is truncated
            info (Dict): Info dict
        """
        self.observations[self.ep_idx][self.t] = observation
        self.actions[self.ep_idx][self.t] = action
        self.rewards[self.ep_idx][self.t] = reward
        self.terminated[self.ep_idx][self.t] = terminated
        self.truncated[self.ep_idx][self.t] = truncated
        self.infos[self.ep_idx][self.t] = info
        self.t += 1


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("Pendulum-v1", render_mode="human")

    buffer = EpisodeBuffer(10, env)

    observation, info = env.reset()
    buffer.new_episode(observation, info)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        buffer.add(action, observation, reward, terminated, truncated, info)
        if terminated or truncated:
            observation, info = env.reset()
            buffer.new_episode(observation, info)
