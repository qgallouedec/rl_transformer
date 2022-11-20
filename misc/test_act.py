import gym
import torch

from rl_transformer.rl_transformer import ActorCriticTransformer
from rl_transformer.utils import get_space_size

env = gym.make("Pendulum-v1")

ac = ActorCriticTransformer(env, d_model=16)

obs_size = get_space_size(env.observation_space)
action_size = env.action_space.shape[0]
observations = torch.rand((13, obs_size))
actions = torch.rand((12, action_size))
print(ac.act(observations, actions))

env = gym.make("CartPole-v1")

ac = ActorCriticTransformer(env, d_model=16)

obs_size = get_space_size(env.observation_space)
observations = torch.rand((13, obs_size))
actions = torch.randint(0, 2, (12,))
print(ac.act(observations, actions))
