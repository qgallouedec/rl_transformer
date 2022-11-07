import gym
import torch

from rl_transformer.buffer import EpisodeBuffer
from rl_transformer.rl_transformer import ActorCriticTransformer

env = gym.make("Pendulum-v1")

buffer = EpisodeBuffer(20, env)
ac = ActorCriticTransformer(env, d_model=16)

observation = env.reset()
buffer.new_episode(observation)
for _ in range(3000):
    observations, actions = buffer.get_current_episode()
    action = ac.act(torch.from_numpy(observations), torch.from_numpy(actions)).numpy()
    observation, reward, done, info = env.step(action)
    buffer.add(action, observation, reward, done, info)
    if done:
        observation = env.reset()
        buffer.new_episode(observation)

observations, actions, rewards, done, infos, src_key_padding_mask = buffer.sample(5)
observations = torch.from_numpy(observations)
actions = torch.from_numpy(actions)
src_key_padding_mask = torch.from_numpy(src_key_padding_mask)
actions, values, log_prob = ac.forward(observations, actions, src_key_padding_mask)

print(actions)
