import numpy as np
import pytest
import torch
from gym import spaces
from gym.wrappers import TimeLimit

from rl_transformer.rl_transformer import ActorCriticTransformer
from rl_transformer.utils import get_space_shape
from rl_transformer.wrappers import TorchWrapper
from tests.utils import DummyEnv

SPACES = [
    spaces.Box(-2, 2, (3,)),
    spaces.Discrete(3),
]


@pytest.mark.parametrize("observation_space", SPACES)
@pytest.mark.parametrize("action_space", SPACES)
def test_act(observation_space, action_space):
    env = TorchWrapper(TimeLimit(DummyEnv(observation_space, action_space), max_episode_steps=100))

    ac = ActorCriticTransformer(env, d_model=16)

    observations = torch.tensor(np.array([observation_space.sample() for _ in range(13)]))
    actions = torch.tensor(np.array([action_space.sample() for _ in range(12)]))
    action = ac.act(observations, actions).numpy()
    assert action_space.contains(action)


@pytest.mark.parametrize("observation_space", SPACES)
@pytest.mark.parametrize("action_space", SPACES)
def test_interract(observation_space, action_space):
    env = TorchWrapper(TimeLimit(DummyEnv(observation_space, action_space), max_episode_steps=100))
    ac = ActorCriticTransformer(env, d_model=16)

    for ep_idx in range(3):
        env.reset()
        done = False
        t = 0
        while not done:
            observations = torch.tensor(np.array([observation_space.sample() for _ in range(t + 1)]))
            if t == 0:  # trick for actions to have the right shape whe t=0
                actions = np.zeros((0, *get_space_shape(action_space)))
            else:
                actions = np.array([action_space.sample() for _ in range(t)])
            actions = torch.tensor(actions)
            action, _, _ = ac.act(observations, actions)
            _, _, done, _ = env.step(action)
            t += 1
