import random

import gym
from gym.envs.registration import EnvSpec


class DummyEnv(gym.Env):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.spec = EnvSpec("DummyEnv-v0")

    def step(self, action):
        assert self.action_space.contains(action)
        observation = self.observation_space.sample()
        reward = random.random()
        done = random.random() < 0.02  # average 1 per 50 steps
        return observation, reward, done, {}

    def reset(self):
        observation = self.observation_space.sample()
        return observation
