import pytest
import torch
from gym import spaces
from gym.wrappers import TimeLimit

from rl_transformer.buffer import EpisodeBuffer
from rl_transformer.wrappers import TorchWrapper
from tests.utils import DummyEnv

SPACES = [
    spaces.Box(-2, 2, (3,)),
    spaces.Discrete(3),
]


@pytest.mark.parametrize("observation_space", SPACES)
@pytest.mark.parametrize("action_space", SPACES)
def test_add(observation_space, action_space):
    env = TorchWrapper(TimeLimit(DummyEnv(observation_space, action_space), max_episode_steps=100))

    buffer = EpisodeBuffer(100, env)

    observation = env.reset()
    value = 1.0
    buffer.new_episode(observation, value)
    for _ in range(3000):
        action = torch.tensor(env.action_space.sample())
        log_prob = 0.0
        observation, reward, done, info = env.step(action)
        value = 1.0
        buffer.add(action, log_prob, observation, value, reward, done, info)
        if done:
            observation = env.reset()
            value = 1.0
            buffer.new_episode(observation, value)


@pytest.mark.parametrize("observation_space", SPACES)
@pytest.mark.parametrize("action_space", SPACES)
def test_get_current_episode(observation_space, action_space):
    env = TorchWrapper(TimeLimit(DummyEnv(observation_space, action_space), max_episode_steps=100))

    buffer = EpisodeBuffer(20, env)

    first_observation = env.reset()
    value = 1.0
    buffer.new_episode(first_observation, value)
    for _ in range(10):
        action = torch.tensor(env.action_space.sample())
        log_prob = 0.0
        observation, reward, done, info = env.step(action)
        value = 1.0
        buffer.add(action, log_prob, observation, value, reward, done, info)

    observations, actions = buffer.get_current_episode()
    assert observations.shape == (11, *env.observation_space.shape)
    assert actions.shape == (10, *env.action_space.shape)
    assert torch.all(observations[0] == first_observation)
    assert torch.all(actions[-1] == action)
